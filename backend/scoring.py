"""Scoring: v1 weighted preference score + lawful landlord rules check. v2 (LightGBM) plugs in here."""
import math, numpy as np, pandas as pd

LAST_FUNNEL = {}

def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))

def passes_rules(renter: dict, l: pd.Series) -> tuple[bool, str]:
    """Landlord side: lawful gates only. Returns (ok, reason_if_not)."""
    income = float(renter.get("annual_income", 0))
    if income < l.min_income_multiple * l.rent * 12 and not renter.get("has_guarantor"):
        return False, "income"
    if renter.get("has_guarantor") and not l.guarantor_accepted and income < l.min_income_multiple * l.rent * 12:
        return False, "guarantor"
    if l.referencing_required and not renter.get("passes_referencing", True):
        return False, "referencing"
    if renter.get("has_pets") and l.pet_fee < 0:
        return False, "pets"
    return True, ""

def preference_score(renter: dict, l: pd.Series, work_lat=None, work_lon=None) -> float:
    """0..1 how much this renter should like this listing (v1 hand weights)."""
    budget = float(renter["budget"])
    over = max(0.0, (l.rent - budget) / budget)
    price = math.exp(-6 * over) if over > 0 else 1.0 - 0.15 * (budget - l.rent) / budget  # cheaper than budget is fine, slightly less exciting
    beds = 1.0 if l.bedrooms == renter["bedrooms"] else (0.6 if abs(l.bedrooms - renter["bedrooms"]) == 1 else 0.15)
    if work_lat is not None:
        km = haversine_km(l.lat, l.lon, work_lat, work_lon)
        commute = math.exp(-km / max(2.0, float(renter.get("max_commute_km", 8))))
    else:
        commute = 0.7
    w = renter.get("weights", {})
    wt, ws, wg, we = (float(w.get(k, 1)) for k in ("transport", "safety", "green", "energy"))
    epc_pts = {"A": 1, "B": 0.9, "C": 0.75, "D": 0.5, "E": 0.3, "F": 0.15, "G": 0.05}.get(l.epc, 0.5)
    nbhd = (wt * l.transport_score / 10 + ws * l.safety_score / 10 + wg * l.environment_score / 10 + we * epc_pts) / max(1e-6, wt + ws + wg + we)
    size = min(1.0, l.tfarea / (35 + 20 * max(1, renter["bedrooms"])))
    return float(0.35 * price + 0.2 * beds + 0.2 * commute + 0.15 * nbhd + 0.1 * size)

def rank(renter: dict, listings: pd.DataFrame, swiped_ids: set, work_lat=None, work_lon=None, liked: pd.DataFrame | None = None, model=None) -> pd.DataFrame:
    """Only homes the renter can actually get: within budget, right size, within reach, and passing the
    landlord's rules. Relax size and distance only when the strict deck is thin, never budget or rules."""
    df = listings[~listings.id.isin(swiped_ids)].copy()
    budget = float(renter["budget"]); beds = int(renter["bedrooms"]); max_km = float(renter.get("max_commute_km", 8))
    funnel = [("London homes scanned", int(len(listings)))]
    if renter.get("area_lat") is not None:
        df["area_km"] = [haversine_km(r.lat, r.lon, renter["area_lat"], renter["area_lon"]) for r in df.itertuples()]
        rad = float(renter.get("radius_km", 3))
        in_area = df[df.area_km <= rad]
        df = in_area if len(in_area) >= 5 else df[df.area_km <= rad * 2]
        funnel.append((f"within {rad:g} km of {str(renter.get('area_postcode','')).upper()}", int(len(df))))
    df = df[df.rent <= budget * 1.05]; funnel.append(("within your budget", int(len(df))))
    ht = renter.get("home_type") or "any"
    if ht in ("flat", "house"):
        df = df[df.propertytype == "F"] if ht == "flat" else df[df.propertytype != "F"]; funnel.append((f"{ht}s only", int(len(df))))
    fu = renter.get("furnished_pref") or "any"
    if fu in ("yes", "no"):
        df = df[df.furnished == (fu == "yes")]; funnel.append(("furnished" if fu == "yes" else "unfurnished", int(len(df))))
    if renter.get("min_area"):
        df = df[df.tfarea >= float(renter["min_area"])]; funnel.append((f"at least {int(renter['min_area'])} m²", int(len(df))))
    if renter.get("min_epc"):
        df = df[df.epc <= renter["min_epc"]]; funnel.append((f"EPC {renter['min_epc']} or better", int(len(df))))
    size = df[df.bedrooms == beds]
    df = size if len(size) >= 5 else df[df.bedrooms.between(beds, beds + 1)]; funnel.append(("the right size", int(len(df))))
    if renter.get("move_in") and "available_from" in df.columns:
        want = pd.Timestamp(renter["move_in"]); avail = pd.to_datetime(df.available_from, errors="coerce")
        ok_date = (avail <= want + pd.Timedelta(days=21)) & (avail >= want - pd.Timedelta(days=60))
        df = df[ok_date | avail.isna()] if ok_date.sum() >= 5 else df
        funnel.append(("available for your move-in", int(len(df))))
    if work_lat is not None:
        df["commute_km"] = [round(haversine_km(r.lat, r.lon, work_lat, work_lon), 1) for r in df.itertuples()]
        near = df[df.commute_km <= max_km]
        df = near if len(near) >= 5 else df[df.commute_km <= max_km * 1.5]
        funnel.append((f"within {max_km:g} km of {renter.get('work_postcode')}", int(len(df))))
    else:
        df["commute_km"] = np.nan
    ok = [passes_rules(renter, r) for r in df.itertuples()]
    df["eligible"] = [o for o, _ in ok]; df["block_reason"] = [r for _, r in ok]
    df = df[df.eligible]; funnel.append(("whose landlord would accept you", int(len(df))))
    LAST_FUNNEL[id(renter) if False else "last"] = funnel
    if df.empty:
        return df
    if model is not None:
        from backend.model import predict
        df["score"] = predict(model, renter, df, work_lat, work_lon, liked)
    else:
        df["score"] = [preference_score(renter, r, work_lat, work_lon) for r in df.itertuples()]
    if liked is not None and len(liked) >= 2:  # taste adaptation from this renter's own likes
        bonus = df.borough.isin(liked.borough).astype(float) * 0.05 + df.propertytype.isin(liked.propertytype).astype(float) * 0.03
        bonus += (1 - (df.tfarea - liked.tfarea.mean()).abs() / max(20.0, liked.tfarea.mean())).clip(0, 1) * 0.04
        df["score"] = df["score"] + bonus
    if "live" in df.columns:  # homes listed live in this session go first so the two-phone demo connects
        df["score"] = df["score"] + df.live.fillna(False).astype(float) * 1.0
    return df.sort_values("score", ascending=False)

def max_rent_for(renter: dict, multiple: float = 2.75) -> float:
    """Rent a typical landlord's income rule allows, before any guarantor."""
    return float(renter.get("annual_income", 0)) / (12 * multiple)

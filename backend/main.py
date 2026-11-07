import json, pathlib, sqlite3, time, requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from backend import scoring

ROOT = pathlib.Path(__file__).resolve().parents[1]
LISTINGS = pd.read_csv(ROOT / "data/processed/listings.csv")
META = json.load(open(ROOT / "data/processed/meta.json"))
DB = ROOT / "chairmatch.sqlite"
MODEL = None
try:
    from backend.model import load as load_model
    MODEL = load_model(ROOT / "data/processed/ranker.txt")
except Exception as e:
    print("ranker not loaded, using v1 scoring:", e)

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS renters(id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, profile TEXT);
    CREATE TABLE IF NOT EXISTS swipes(id INTEGER PRIMARY KEY AUTOINCREMENT, renter_id INT, listing_id INT, liked INT, matched INT, reason TEXT, ts REAL);
    """)

app = FastAPI(title="ChairMatch")

class Profile(BaseModel):
    name: str = "You"
    budget: float
    bedrooms: int
    work_postcode: str | None = None
    max_commute_km: float = 8
    annual_income: float = 40000
    has_guarantor: bool = False
    passes_referencing: bool = True
    has_pets: bool = False
    weights: dict = {}

def geocode(pc: str):
    try:
        r = requests.get(f"https://api.postcodes.io/postcodes/{pc.replace(' ', '')}", timeout=5).json()["result"]
        return r["latitude"], r["longitude"]
    except Exception:
        return None, None

@app.get("/api/meta")
def meta():
    return {**META, "ranker": "lightgbm" if MODEL is not None else "v1-weighted", "boroughs": sorted(LISTINGS.borough.unique().tolist())}

@app.post("/api/renters")
def create_renter(p: Profile):
    prof = p.model_dump(); prof["work_lat"], prof["work_lon"] = geocode(p.work_postcode) if p.work_postcode else (None, None)
    with db() as c:
        cur = c.execute("INSERT INTO renters(created, profile) VALUES(?,?)", (time.time(), json.dumps(prof)))
        rid = cur.lastrowid
    swiped = set(); ranked = scoring.rank(prof, LISTINGS, swiped, prof.get("work_lat"), prof.get("work_lon"), None, MODEL)
    return {"renter_id": rid, "profile": prof, "n_available": int(len(ranked)), "max_rent_income_rule": round(scoring.max_rent_for(prof))}

def _renter(rid):
    with db() as c:
        row = c.execute("SELECT profile FROM renters WHERE id=?", (rid,)).fetchone()
    if not row: raise HTTPException(404, "renter not found")
    return json.loads(row["profile"])

def _swiped(rid):
    with db() as c:
        rows = c.execute("SELECT listing_id, liked FROM swipes WHERE renter_id=?", (rid,)).fetchall()
    return {r["listing_id"] for r in rows}, {r["listing_id"] for r in rows if r["liked"]}

@app.get("/api/renters/{rid}/deck")
def deck(rid: int, n: int = 10):
    prof = _renter(rid); swiped, liked_ids = _swiped(rid)
    liked = LISTINGS[LISTINGS.id.isin(liked_ids)]
    ranked = scoring.rank(prof, LISTINGS, swiped, prof.get("work_lat"), prof.get("work_lon"), liked, MODEL)
    cols = [c for c in ["id","borough","ward","outcode","lat","lon","type_name","propertytype","bedrooms","tfarea","numberrooms","epc","age_band","rent","transport_score","safety_score","environment_score","imd_decile","housing_barriers_decile","pet_fee","furnished","min_tenancy_months","min_income_multiple","referencing_required","guarantor_accepted","area_label","score","eligible","block_reason","homes_approved_600m","schemes_approved_600m","applications_since_2024_600m","nearest_big_scheme_km","nearest_big_scheme_homes","nearest_big_scheme_desc","private_rented_pct","social_rented_pct","owned_pct","ptal_level","borough_typical_rent","rent_vs_typical_pct","borough_rent_growth_pct","demand_tightness","demand_tightness_pct"] if c in ranked.columns]
    if ranked.empty: return []
    cols = cols + (["commute_km"] if "commute_km" in ranked.columns else [])
    out = ranked.head(n)[cols].copy()
    return json.loads(out.to_json(orient="records"))

class Swipe(BaseModel):
    listing_id: int
    liked: bool

@app.post("/api/renters/{rid}/swipes")
def swipe(rid: int, s: Swipe):
    prof = _renter(rid); l = LISTINGS[LISTINGS.id == s.listing_id]
    if l.empty: raise HTTPException(404, "listing not found")
    ok, reason = scoring.passes_rules(prof, l.iloc[0])
    matched = bool(s.liked and ok)
    with db() as c:
        c.execute("INSERT INTO swipes(renter_id, listing_id, liked, matched, reason, ts) VALUES(?,?,?,?,?,?)", (rid, s.listing_id, int(s.liked), int(matched), reason, time.time()))
    return {"matched": matched, "landlord_ok": ok, "block_reason": reason}

@app.get("/api/renters/{rid}/matches")
def matches(rid: int):
    with db() as c:
        rows = c.execute("SELECT listing_id, matched, reason, liked FROM swipes WHERE renter_id=? AND liked=1", (rid,)).fetchall()
    ids = [r["listing_id"] for r in rows]; info = {r["listing_id"]: dict(r) for r in rows}
    out = LISTINGS[LISTINGS.id.isin(ids)].copy()
    out["matched"] = out.id.map(lambda i: bool(info[i]["matched"])); out["block_reason"] = out.id.map(lambda i: info[i]["reason"])
    return json.loads(out.to_json(orient="records"))

@app.get("/api/stats")
def stats():
    with db() as c:
        r = c.execute("SELECT COUNT(DISTINCT renter_id) renters, COUNT(*) swipes, SUM(liked) likes, SUM(matched) matches FROM swipes").fetchone()
    return dict(r)

app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="static")

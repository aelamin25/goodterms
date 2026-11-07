import json, pathlib, sqlite3, time, requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd, numpy as np
from backend import scoring
from backend import live

ROOT = pathlib.Path(__file__).resolve().parents[1]
LISTINGS = pd.read_csv(ROOT / "data/processed/listings.csv")
_rng = np.random.default_rng(7)
LISTINGS["available_from"] = [(pd.Timestamp("2026-11-07") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in _rng.integers(0, 75, len(LISTINGS))]
LISTINGS["live"] = False; LISTINGS["verified"] = (LISTINGS.id % 4 != 0); LISTINGS["landlord_name"] = ""
PHOTOS = ["1502672260266-1c1ef2d93688", "1522708323590-d24dbb6b0267", "1484154218962-a197022b5858", "1493809842364-78817add7ffb", "1560448204-e02f11c3d0e2", "1554995207-c18c203602cb", "1600607687939-ce8a6c25118c", "1600210492486-724fe5c67fb0", "1588854337115-1c67d9247e4d", "1505693416388-ac5ce068fe85", "1540518614846-7eded433c457", "1513694203232-719a280e022f", "1567767292278-a4f21aa2d36e", "1536376072261-38c75010e6c9", "1502005229762-cf1b2da7c5d6", "1556911220-bff31c812dba", "1565182999561-18d7dc61c393", "1555041469-a586c61ea9bc", "1507089947368-19c1da9775ae", "1595526114035-0d45ed16cfbf"]
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
live.init(LISTINGS, DB, scoring)
app.include_router(live.router)
def listings(): return live.STATE["listings"]
def photos_for(lid: int, k: int = 3):
    n = len(PHOTOS); return [f"https://images.unsplash.com/photo-{PHOTOS[(lid * 7 + j * 11) % n]}?w=900&q=70&auto=format&fit=crop" for j in range(k)]

class Profile(BaseModel):
    name: str = "You"
    budget: float
    bedrooms: int
    move_in: str = ""
    area_postcode: str | None = None
    radius_km: float = 3
    household: str = ""
    job: str = ""
    lifestyle: list = []
    wants: list = []
    dealbreakers: list = []
    work_postcode: str | None = None
    max_commute_km: float = 8
    annual_income: float = 40000
    has_guarantor: bool = False
    passes_referencing: bool = True
    has_pets: bool = False
    weights: dict = {}

def geocode(pc: str):
    try:
        q = pc.replace(' ', '')
        r = requests.get(f"https://api.postcodes.io/postcodes/{q}", timeout=5).json().get("result")
        if not r:
            r = requests.get(f"https://api.postcodes.io/outcodes/{q}", timeout=5).json().get("result")
        return r["latitude"], r["longitude"]
    except Exception:
        return None, None
scoring.geocode = geocode

@app.get("/api/meta")
def meta():
    return {**META, "ranker": "lightgbm" if MODEL is not None else "v1-weighted", "boroughs": sorted(LISTINGS.borough.unique().tolist())}

@app.post("/api/renters")
def create_renter(p: Profile):
    prof = p.model_dump(); prof["work_lat"], prof["work_lon"] = geocode(p.work_postcode) if p.work_postcode else (None, None)
    prof["area_lat"], prof["area_lon"] = geocode(p.area_postcode) if p.area_postcode else (None, None)
    with db() as c:
        cur = c.execute("INSERT INTO renters(created, profile) VALUES(?,?)", (time.time(), json.dumps(prof)))
        rid = cur.lastrowid
    swiped = set(); ranked = scoring.rank(prof, listings(), swiped, prof.get("work_lat"), prof.get("work_lon"), None, MODEL)
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
    liked = listings()[listings().id.isin(liked_ids)]
    ranked = scoring.rank(prof, listings(), swiped, prof.get("work_lat"), prof.get("work_lon"), liked, MODEL)
    cols = [c for c in ["id","borough","ward","outcode","lat","lon","type_name","propertytype","bedrooms","tfarea","numberrooms","epc","age_band","rent","transport_score","safety_score","environment_score","imd_decile","housing_barriers_decile","pet_fee","furnished","min_tenancy_months","min_income_multiple","referencing_required","guarantor_accepted","area_label","score","eligible","block_reason","homes_approved_600m","schemes_approved_600m","applications_since_2024_600m","nearest_big_scheme_km","nearest_big_scheme_homes","nearest_big_scheme_desc","private_rented_pct","social_rented_pct","owned_pct","ptal_level","borough_typical_rent","rent_vs_typical_pct","borough_rent_growth_pct","demand_tightness","demand_tightness_pct","available_from","live","verified","landlord_name"] if c in ranked.columns]
    if ranked.empty: return []
    cols = cols + (["commute_km"] if "commute_km" in ranked.columns else [])
    out = ranked.head(n)[cols].copy()
    recs = json.loads(out.to_json(orient="records"))
    for r in recs: r["photos"] = photos_for(r["id"])
    return recs

class Swipe(BaseModel):
    listing_id: int
    liked: bool
    liked_feature: str = ""

@app.post("/api/renters/{rid}/swipes")
def swipe(rid: int, s: Swipe):
    prof = _renter(rid); l = listings()[listings().id == s.listing_id]
    if l.empty: raise HTTPException(404, "listing not found")
    ok, reason = scoring.passes_rules(prof, l.iloc[0])
    matched = bool(s.liked and ok)
    with db() as c:
        c.execute("INSERT INTO swipes(renter_id, listing_id, liked, matched, reason, ts) VALUES(?,?,?,?,?,?)", (rid, s.listing_id, int(s.liked), int(matched), reason, time.time()))
    if s.liked: live.record_interest(s.listing_id, rid, s.liked_feature, matched)
    return {"matched": matched, "landlord_ok": ok, "block_reason": reason}

@app.get("/api/renters/{rid}/matches")
def matches(rid: int):
    with db() as c:
        rows = c.execute("SELECT listing_id, matched, reason, liked FROM swipes WHERE renter_id=? AND liked=1", (rid,)).fetchall()
    ids = [r["listing_id"] for r in rows]; info = {r["listing_id"]: dict(r) for r in rows}
    out = listings()[listings().id.isin(ids)].copy()
    out["matched"] = out.id.map(lambda i: bool(info[i]["matched"])); out["block_reason"] = out.id.map(lambda i: info[i]["reason"])
    return json.loads(out.to_json(orient="records"))

@app.get("/api/stats")
def stats():
    with db() as c:
        r = c.execute("SELECT COUNT(DISTINCT renter_id) renters, COUNT(*) swipes, SUM(liked) likes, SUM(matched) matches FROM swipes").fetchone()
    return dict(r)

app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="static")

"""Live two-sided layer: landlords list a home with rules; renters' likes become interests the landlord
sees in real time; landlord offers a viewing; renter sees it. All state in SQLite, polled by both apps."""
import json, time, sqlite3, pathlib
import pandas as pd, numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
STATE = {"listings": None, "db": None, "scoring": None}

def init(listings: pd.DataFrame, db_path: pathlib.Path, scoring):
    STATE["listings"] = listings; STATE["db"] = db_path; STATE["scoring"] = scoring
    with sqlite3.connect(db_path) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS landlords(id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, name TEXT, listing_id INT, rules TEXT);
        CREATE TABLE IF NOT EXISTS interests(id INTEGER PRIMARY KEY AUTOINCREMENT, landlord_id INT, listing_id INT, renter_id INT, liked_feature TEXT, status TEXT, slot TEXT, ts REAL, UNIQUE(listing_id, renter_id));
        """)

def db():
    c = sqlite3.connect(STATE["db"]); c.row_factory = sqlite3.Row; return c

class Listing(BaseModel):
    name: str = "Landlord"
    postcode: str
    bedrooms: int = 1
    rent: float
    available_from: str = ""
    min_income_multiple: float = 2.5
    referencing_required: bool = True
    guarantor_accepted: bool = True
    pet_fee: int = -1
    min_tenancy_months: int = 12
    furnished: bool = True
    verified: bool = True

@router.post("/api/landlords")
def create_landlord(l: Listing):
    """Create a live listing: borrow attributes from the nearest real dwelling of that size near the postcode."""
    L = STATE["listings"]; sc = STATE["scoring"]
    lat, lon = sc.geocode(l.postcode)
    if lat is None: raise HTTPException(400, "postcode not found")
    pool = L[L.bedrooms == l.bedrooms] if (L.bedrooms == l.bedrooms).sum() > 20 else L
    d = np.hypot((pool.lat - lat) * 110.57, (pool.lon - lon) * 69.4)
    base = pool.loc[d.idxmin()].to_dict()
    new_id = int(L.id.max()) + 1
    row = {**base, "id": new_id, "lat": lat, "lon": lon, "rent": int(l.rent), "bedrooms": l.bedrooms,
           "min_income_multiple": l.min_income_multiple, "referencing_required": l.referencing_required, "guarantor_accepted": l.guarantor_accepted,
           "pet_fee": l.pet_fee, "min_tenancy_months": l.min_tenancy_months, "furnished": l.furnished, "available_from": l.available_from or base.get("available_from", ""),
           "live": True, "verified": l.verified, "landlord_name": l.name}
    STATE["listings"] = pd.concat([L, pd.DataFrame([row])], ignore_index=True)
    with db() as c:
        cur = c.execute("INSERT INTO landlords(created, name, listing_id, rules) VALUES(?,?,?,?)", (time.time(), l.name, new_id, l.model_dump_json()))
        lid = cur.lastrowid
    return {"landlord_id": lid, "listing_id": new_id, "listing": json.loads(pd.Series(row).to_json())}

def record_interest(listing_id: int, renter_id: int, liked_feature: str, matched: bool):
    with db() as c:
        row = c.execute("SELECT id FROM landlords WHERE listing_id=?", (listing_id,)).fetchone()
        if not row: return
        c.execute("INSERT OR IGNORE INTO interests(landlord_id, listing_id, renter_id, liked_feature, status, slot, ts) VALUES(?,?,?,?,?,?,?)",
                  (row["id"], listing_id, renter_id, liked_feature, "matched" if matched else "liked_unqualified", "", time.time()))

@router.get("/api/landlords/{lid}/interested")
def interested(lid: int):
    with db() as c:
        rows = c.execute("""SELECT i.*, r.profile FROM interests i JOIN renters r ON r.id=i.renter_id WHERE i.landlord_id=? ORDER BY i.ts DESC""", (lid,)).fetchall()
    out = []
    for r in rows:
        p = json.loads(r["profile"]); p.pop("work_lat", None); p.pop("work_lon", None)
        out.append({"interest_id": r["id"], "renter_id": r["renter_id"], "status": r["status"], "slot": r["slot"], "liked_feature": r["liked_feature"], "ts": r["ts"], "profile": p})
    return out

class Decision(BaseModel):
    action: str  # offer_viewing | pass
    slot: str = "Sat 11:00"

@router.post("/api/interests/{iid}")
def decide(iid: int, d: Decision):
    with db() as c:
        c.execute("UPDATE interests SET status=?, slot=? WHERE id=?", ("viewing_offered" if d.action == "offer_viewing" else "passed", d.slot, iid))
    return {"ok": True}

@router.get("/api/renters/{rid}/inbox")
def inbox(rid: int):
    with db() as c:
        rows = c.execute("SELECT i.*, l.name landlord_name FROM interests i JOIN landlords l ON l.id=i.landlord_id WHERE i.renter_id=?", (rid,)).fetchall()
    return [dict(r) for r in rows]

@router.get("/api/live")
def live_listings():
    L = STATE["listings"]
    if "live" not in L.columns: return []
    return json.loads(L[L.live == True].to_json(orient="records"))

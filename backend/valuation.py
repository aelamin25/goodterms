"""Home valuation and rent estimate. Value: LightGBM trained on 1.28m real London sales (Land Registry x EPC),
temporal holdout 2024 median error 10.6%. Rent: value x borough-and-size yield anchored to ONS rents."""
import json, pathlib, numpy as np, pandas as pd, lightgbm as lgb
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/processed"; RAW = ROOT / "data/raw"
MODEL = lgb.Booster(model_file=str(OUT / "price_model.txt"))
CATS = json.load(open(OUT / "price_model_cats.json"))
YIELDS = json.load(open(OUT / "rent_yields.json"))
METRICS = json.load(open(OUT / "price_model_metrics.json"))
FEATS = METRICS["model"]["features"]
T_NOW = (pd.Timestamp("2026-11-07") - pd.Timestamp("2010-01-01")).days / 365.25
_iod = pd.read_csv(RAW / "iod2025_file7.csv"); _c = lambda p: next(x for x in _iod.columns if x.startswith(p))
NBHD = _iod[[_c("LSOA code"), _c("Index of Multiple Deprivation (IMD) Score"), _c("Crime Score"), _c("Living Environment Score")]]
NBHD.columns = ["lsoa21", "imd", "crime", "env"]; NBHD = NBHD.set_index("lsoa21")
PTAL = pd.read_csv(RAW / "ptal_lsoa2011.csv")[["LSOA2011", "AvPTAI2015"]].set_index("LSOA2011").AvPTAI2015
EPC_MID = {"A": 95, "B": 85, "C": 75, "D": 62, "E": 48, "F": 30, "G": 12}

def frame(rows: pd.DataFrame) -> pd.DataFrame:
    """rows need: lat, lon, tfarea, numberrooms, epc or CURRENT_ENERGY_EFFICIENCY, propertytype, age_band, borough, outcode, lsoa21"""
    d = pd.DataFrame(index=rows.index)
    d["t"] = T_NOW; d["lat"] = rows.lat; d["lon"] = rows.lon; d["tfarea"] = rows.tfarea; d["numberrooms"] = rows.numberrooms.fillna(rows.tfarea / 25).clip(1, 12)
    d["CURRENT_ENERGY_EFFICIENCY"] = rows["CURRENT_ENERGY_EFFICIENCY"] if "CURRENT_ENERGY_EFFICIENCY" in rows else rows.epc.map(EPC_MID).fillna(62)
    nb = NBHD.reindex(rows.lsoa21.values); d["imd"] = nb.imd.values; d["crime"] = nb.crime.values; d["env"] = nb.env.values; d["ptai"] = PTAL.reindex(rows.lsoa21.values).values
    for k in ["imd", "crime", "env", "ptai"]: d[k] = d[k].fillna(d[k].median() if d[k].notna().any() else 0)
    d["propertytype"] = rows.propertytype; d["duration"] = rows.duration if "duration" in rows else np.where(rows.propertytype == "F", "L", "F")
    d["CONSTRUCTION_AGE_BAND"] = rows.age_band.map(lambda a: a if str(a).startswith("England") else f"England and Wales: {a}")
    d["borough"] = rows.borough; d["outcode"] = rows.outcode
    for k, cats in CATS.items(): d[k] = pd.Categorical(d[k], categories=cats)
    return d[FEATS]

def value(rows: pd.DataFrame) -> np.ndarray:
    return np.exp(MODEL.predict(frame(rows)))

def rent(rows: pd.DataFrame) -> np.ndarray:
    v = value(rows); out = []
    for val, b, bed in zip(v, rows.borough, rows.bedrooms):
        y = YIELDS.get(f"{b}|{int(bed)}") or YIELDS.get(f"{b}|{min(int(bed)+1,5)}") or YIELDS.get(f"{b}|1")
        out.append(val * y["annual_yield"] / 12 if y else np.nan)
    return np.array(out)

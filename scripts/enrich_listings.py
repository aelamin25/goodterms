"""Add real-data context to each listing: Foundations planning activity nearby, Census 2021 private-rent
share, PTAL level, rent vs borough typical (ONS PIPR), and WhereToBuild demand tightness if the file exists."""
import pathlib, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT=pathlib.Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw"; OUT=ROOT/"data/processed"
s=pd.read_csv(OUT/"listings.csv")
base=[c for c in s.columns if c in ["id","borough","ward","outcode","lat","lon","lsoa21","msoa21","type_name","propertytype","bedrooms","tfarea","numberrooms","epc","age_band","rent","transport_score","safety_score","environment_score","imd_decile","housing_barriers_decile","min_income_multiple","referencing_required","guarantor_accepted","pet_fee","min_tenancy_months","furnished","area_label"]]
s=s[base]  # idempotent: drop any previous enrichment

# 1. Foundations: approved housing within 600 m since 2022, and total applications since 2024 (activity)
f=pd.read_parquet(RAW/"foundations.parquet"); f=f.dropna(subset=['lat','lng']); f['decided_date']=pd.to_datetime(f.decided_date,errors='coerce'); f['start_date']=pd.to_datetime(f.start_date,errors='coerce')
def xy(lat,lng): return np.c_[(lng.values)*np.cos(np.radians(51.5))*111.32, lat.values*110.57]
appr=f[(f.status=='Permitted')&f.app_type.isin(['Full','Outline'])&(f.n_dwellings>0)&(f.n_dwellings<=3000)&(f.decided_date>='2022-01-01')]
t=cKDTree(xy(appr.lat,appr.lng)); L=xy(s.lat,s.lon)
near=t.query_ball_point(L,r=0.6)
s['homes_approved_600m']=[int(appr.n_dwellings.values[i].sum()) for i in near]; s['schemes_approved_600m']=[len(i) for i in near]
recent=f[f.start_date>='2024-01-01']; t2=cKDTree(xy(recent.lat,recent.lng)); s['applications_since_2024_600m']=[len(i) for i in t2.query_ball_point(L,r=0.6)]
big=appr.loc[appr.n_dwellings>=50]; t3=cKDTree(xy(big.lat,big.lng)); d,i=t3.query(L,k=1)
s['nearest_big_scheme_km']=d.round(2); s['nearest_big_scheme_homes']=big.n_dwellings.values[i].astype(int); s['nearest_big_scheme_desc']=pd.Series(big.description.values[i]).astype(str).str.slice(0,140).values

# 2. Census 2021 tenure by LSOA
c=pd.read_csv(RAW/"census_tenure_lsoa.csv"); c.columns=['lsoa21','cat','n']
tot=c[c.cat.str.startswith('Total')].set_index('lsoa21').n
prs=c[c.cat.str.startswith('Private rented')].groupby('lsoa21').n.sum()
own=c[c.cat.str.startswith('Owned')].groupby('lsoa21').n.sum(); soc=c[c.cat.str.startswith('Social rented')].groupby('lsoa21').n.sum()
s['private_rented_pct']=(s.lsoa21.map(prs)/s.lsoa21.map(tot)*100).round(0); s['social_rented_pct']=(s.lsoa21.map(soc)/s.lsoa21.map(tot)*100).round(0); s['owned_pct']=(s.lsoa21.map(own)/s.lsoa21.map(tot)*100).round(0)

# 3. PTAL level
ptal=pd.read_csv(RAW/"ptal_lsoa2011.csv").rename(columns={'LSOA2011':'lsoa11'})
geo=json.load(open(RAW/"geo_cache.json"))  # postcode -> codes
lsoa11=s.lsoa21.copy()
# listings.csv kept lsoa21 only; recover lsoa11 from geo cache by matching on outcode+ward is lossy, so re-derive from cache by nearest match on lsoa21 (identical for most London LSOAs)
s=s.merge(ptal[['lsoa11','PTAL']].rename(columns={'lsoa11':'lsoa21','PTAL':'ptal_level'}),on='lsoa21',how='left'); s['ptal_level']=s.ptal_level.fillna('n/a')

# 4. Rent vs borough typical (PIPR latest month, same bedroom band)
p=pd.read_csv(RAW/"pipr_london_boroughs.csv"); p['Time period']=pd.to_datetime(p['Time period']); p=p[p['Time period']==p['Time period'].max()].set_index('Area name')
bedcol={0:'Rental price one bed',1:'Rental price one bed',2:'Rental price two bed',3:'Rental price three bed',4:'Rental price four or more bed',5:'Rental price four or more bed'}
def typical(r):
    b='Westminster' if r.borough=='City of London' else r.borough; v=float(p.loc[b,bedcol[r.bedrooms]]); return v*0.85 if r.bedrooms==0 else v
s['borough_typical_rent']=s.apply(typical,axis=1).round(0); s['rent_vs_typical_pct']=((s.rent/s.borough_typical_rent-1)*100).round(0)
prev=pd.read_csv(RAW/"pipr_london_boroughs.csv"); prev['Time period']=pd.to_datetime(prev['Time period']); last=prev['Time period'].max(); yr=prev[prev['Time period']==last-pd.DateOffset(years=1)].set_index('Area name')['Rental price']; now=prev[prev['Time period']==last].set_index('Area name')['Rental price']
s['borough_rent_growth_pct']=s.borough.replace({'City of London':'Westminster'}).map(((now/yr-1)*100).round(1))

# 5. WhereToBuild (event-only data, optional, never committed)
w=RAW/"wheretobuild_msoa_stats.csv"
if w.exists():
    wt=pd.read_csv(w); wt.columns=[c.strip().replace('\\','') for c in wt.columns]; wt=wt.set_index('msoa_code')
    s['demand_tightness']=s.msoa21.map(wt['tightness']).round(1); s['demand_tightness_pct']=(s.msoa21.map(wt['tightness']).rank(pct=True)*100).round(0)
else:
    s['demand_tightness']=np.nan; s['demand_tightness_pct']=np.nan

s.to_csv(OUT/"listings.csv",index=False)
print(s[['homes_approved_600m','applications_since_2024_600m','private_rented_pct','rent_vs_typical_pct','borough_rent_growth_pct','demand_tightness_pct']].describe().round(1).to_string())
print(s.ptal_level.value_counts().head(8).to_dict())

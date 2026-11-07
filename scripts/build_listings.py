"""Build data/processed/listings.csv: ~N real London dwellings with modelled asking rents,
geocodes, neighbourhood features and generated landlord rules."""
import glob, json, pathlib, sys, time, requests, numpy as np, pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw"; OUT=ROOT/"data/processed"; OUT.mkdir(exist_ok=True)
N_TARGET=int(sys.argv[1]) if len(sys.argv)>1 else 6000
rng=np.random.default_rng(42)

# 1. load homes
frames=[]
for f in sorted(glob.glob(str(RAW/"hpm_*.csv"))):
    b=pathlib.Path(f).stem[4:].replace('_',' ')
    if b=='Camden 2024': continue
    d=pd.read_csv(f, usecols=['year','propertytype','duration','price','postcode','lad23cd','lmk_key','tfarea','numberrooms','CURRENT_ENERGY_EFFICIENCY','CONSTRUCTION_AGE_BAND'])
    d['borough']=b; frames.append(d)
h=pd.concat(frames,ignore_index=True)
h=h[(h.year>=2008)&(h.tfarea.between(18,400))&(h.numberrooms.between(1,12))].dropna(subset=['postcode'])
h=h.sort_values('year').drop_duplicates('lmk_key',keep='last')
print('dwellings',len(h),'boroughs',h.borough.nunique())
# stratified sample: cap per borough, proportional to sqrt(count) to keep inner London visible
w=h.groupby('borough').size(); alloc=(np.sqrt(w)/np.sqrt(w).sum()*N_TARGET).round().astype(int)
s=pd.concat([g.sample(min(len(g),alloc[b]),random_state=42) for b,g in h.groupby('borough')]).reset_index(drop=True)

# 2. bedrooms from habitable rooms + floor area
beds=(s.numberrooms-1).clip(0,5).astype(int)
beds=np.where(s.tfarea<38,0,np.where((beds==0)&(s.tfarea>=38),1,beds))
beds=np.minimum(beds, (s.tfarea/18).astype(int).clip(1,5))  # no 4-bed at 50 m2
s['bedrooms']=np.where(s.tfarea<38,0,np.maximum(beds,1))

# 3. rents from ONS PIPR (latest month) borough x bedrooms x type, scaled by floor area
p=pd.read_csv(RAW/"pipr_london_boroughs.csv"); p['Time period']=pd.to_datetime(p['Time period']); p=p[p['Time period']==p['Time period'].max()]
p=p.set_index('Area name'); rent_month=p['Time period'].iloc[0].strftime('%b %Y')
bedcol={0:'Rental price one bed',1:'Rental price one bed',2:'Rental price two bed',3:'Rental price three bed',4:'Rental price four or more bed',5:'Rental price four or more bed'}
typecol={'D':'Rental price detached','S':'Rental price semidetached','T':'Rental price terraced','F':'Rental price flat maisonette'}
def base_rent(r):
    bor='Westminster' if r.borough=='City of London' else r.borough
    row=p.loc[bor]; base=float(row[bedcol[r.bedrooms]]); 
    if r.bedrooms==0: base*=0.85
    tadj=float(row[typecol[r.propertytype]])/float(row['Rental price'])
    return base*(0.6+0.4*tadj)
s['rent_base']=s.apply(base_rent,axis=1)
med=s.groupby(['borough','bedrooms']).tfarea.transform('median')
s['rent']=(s.rent_base*(s.tfarea/med)**0.35*rng.normal(1,0.07,len(s))).round(-1).astype(int)

# 4. geocode via postcodes.io bulk
pcs=s.postcode.str.replace(' ','').str.upper().unique().tolist()
cache=RAW/'geo_cache.json'; geo=json.load(open(cache)) if cache.exists() else {}
pcs=[x for x in pcs if x not in geo]
for i in range(0,len(pcs),100):
    for attempt in range(3):
        try:
            r=requests.post('https://api.postcodes.io/postcodes',json={'postcodes':pcs[i:i+100]},timeout=30); r.raise_for_status(); break
        except Exception as e: time.sleep(2)
    for q in r.json()['result']:
        res=q['result']
        if res: geo[q['query']]={'lat':res['latitude'],'lon':res['longitude'],'lsoa11':res['codes']['lsoa11'],'lsoa21':res['codes']['lsoa21'],'msoa21':res['codes']['msoa21'],'ward':res['admin_ward'],'outcode':res['outcode']}
    print('geocoded',min(i+100,len(pcs)),'/',len(pcs),flush=True)
json.dump(geo,open(cache,'w'))
g=pd.DataFrame.from_dict(geo,orient='index'); s=s.join(g,on=s.postcode.str.replace(' ','').str.upper()).dropna(subset=['lat'])
s['lat']=s.lat+rng.normal(0,0.0012,len(s)); s['lon']=s.lon+rng.normal(0,0.002,len(s))  # ~150 m jitter

# 5. neighbourhood features
ptal=pd.read_csv(RAW/"ptal_lsoa2011.csv").rename(columns={'LSOA2011':'lsoa11'}); s=s.merge(ptal[['lsoa11','AvPTAI2015','PTAL']],on='lsoa11',how='left')
iod=pd.read_csv(RAW/"iod2025_file7.csv")
def col(prefix): return next(c for c in iod.columns if c.startswith(prefix))
keep={col('LSOA code'):'lsoa21',col('Index of Multiple Deprivation (IMD) Decile'):'imd_decile',col('Crime Decile'):'crime_decile',col('Living Environment Decile'):'environment_decile',col('Barriers to Housing and Services Decile'):'housing_barriers_decile'}
iod=iod[list(keep)].rename(columns=keep); s=s.merge(iod,on='lsoa21',how='left')
s['ptai']=s.AvPTAI2015.fillna(s.AvPTAI2015.median()); s['transport_score']=(s.ptai.rank(pct=True)*10).round(1)
s['safety_score']=s.crime_decile.fillna(5).astype(float); s['environment_score']=s.environment_decile.fillna(5).astype(float)
s['epc']=pd.cut(s.CURRENT_ENERGY_EFFICIENCY,[-1,20,38,54,68,80,91,200],labels=list('GFEDCBA')).astype(str)

# 6. landlord rules (lawful only)
n=len(s)
s['min_income_multiple']=rng.choice([2.5,2.75,3.0],n,p=[0.3,0.3,0.4])
s['referencing_required']=rng.random(n)<0.9
s['guarantor_accepted']=rng.random(n)<0.7
s['pet_fee']=np.where(rng.random(n)<0.8, rng.choice([0,25,40,60],n), -1)  # -1 = pets refused with reason
s['min_tenancy_months']=rng.choice([6,12],n,p=[0.3,0.7])
s['furnished']=rng.random(n)<0.55

s['id']=np.arange(1,n+1); s['type_name']=s.propertytype.map({'D':'Detached house','S':'Semi-detached house','T':'Terraced house','F':'Flat'})
s['area_label']=s.outcode+' · '+s.ward+', '+s.borough
s['age_band']=s.CONSTRUCTION_AGE_BAND.astype(str).str.replace('England and Wales: ','')
cols=['id','borough','ward','outcode','lat','lon','lsoa21','msoa21','type_name','propertytype','bedrooms','tfarea','numberrooms','epc','age_band','rent','transport_score','safety_score','environment_score','imd_decile','housing_barriers_decile','min_income_multiple','referencing_required','guarantor_accepted','pet_fee','min_tenancy_months','furnished','area_label']
s[cols].to_csv(OUT/"listings.csv",index=False)
json.dump({'rent_month':rent_month,'n_listings':int(n),'source_dwellings':int(len(h))},open(OUT/"meta.json",'w'))
print('wrote',n,'listings; rent month',rent_month); print(s.groupby('bedrooms').rent.median())

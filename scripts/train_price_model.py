"""Sale-price model on real London transactions (Land Registry x EPC, UCL linkage), temporal holdout.
Train 2010-2022, validate 2023, test 2024. Then derive a rent estimate anchored to ONS borough rents."""
import glob, json, pathlib, time, numpy as np, pandas as pd, lightgbm as lgb
ROOT=pathlib.Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw"; OUT=ROOT/"data/processed"
t0=time.time()
cols=['year','dateoftransfer','propertytype','duration','price','postcode','tfarea','numberrooms','CURRENT_ENERGY_EFFICIENCY','CONSTRUCTION_AGE_BAND','lmk_key']
frames=[]
for f in sorted(glob.glob(str(RAW/"hpm_*.csv"))):
    d=pd.read_csv(f,usecols=cols); d['borough']=pathlib.Path(f).stem[4:].replace('_',' '); frames.append(d)
h=pd.concat(frames,ignore_index=True)
h=h[(h.year>=2010)&(h.price.between(50_000,5_000_000))&(h.tfarea.between(15,400))].copy()
h['date']=pd.to_datetime(h.dateoftransfer,errors='coerce'); h=h.dropna(subset=['date'])
h['t']=(h.date-pd.Timestamp('2010-01-01')).dt.days/365.25
print('rows',len(h),'load',round(time.time()-t0),'s')

# postcode -> lat/lon/lsoa via the geocode cache plus a sector-level fallback built from it
geo=json.load(open(RAW/"geo_cache.json"))
g=pd.DataFrame.from_dict(geo,orient='index')[['lat','lon','lsoa21']]; g.index.name='pc'
h['pc']=h.postcode.str.replace(' ','').str.upper()
h=h.join(g,on='pc')
# fallback: outward code centroid from what we have cached
h['outcode']=h.postcode.str.split(' ').str[0]
oc=h.dropna(subset=['lat']).groupby('outcode')[['lat','lon']].median()
miss=h.lat.isna(); h.loc[miss,'lat']=h.loc[miss,'outcode'].map(oc.lat); h.loc[miss,'lon']=h.loc[miss,'outcode'].map(oc.lon)
h=h.dropna(subset=['lat'])
print('geocoded rows',len(h),'exact',int((~miss).sum()),'outcode-level',int(miss.sum()))

# neighbourhood: IMD + PTAL where we have the LSOA, else outcode medians
iod=pd.read_csv(RAW/"iod2025_file7.csv"); c=lambda p: next(x for x in iod.columns if x.startswith(p))
iod=iod[[c('LSOA code'),c('Index of Multiple Deprivation (IMD) Score'),c('Crime Score'),c('Living Environment Score')]]; iod.columns=['lsoa21','imd','crime','env']
h=h.merge(iod,on='lsoa21',how='left')
ptal=pd.read_csv(RAW/"ptal_lsoa2011.csv")[['LSOA2011','AvPTAI2015']].rename(columns={'LSOA2011':'lsoa21','AvPTAI2015':'ptai'}); h=h.merge(ptal,on='lsoa21',how='left')
for k in ['imd','crime','env','ptai']:
    h[k]=h[k].fillna(h.groupby('outcode')[k].transform('median'))

cat_cols=['propertytype','duration','CONSTRUCTION_AGE_BAND','borough','outcode']
for k in cat_cols: h[k]=h[k].astype('category')
feats=['t','lat','lon','tfarea','numberrooms','CURRENT_ENERGY_EFFICIENCY','imd','crime','env','ptai']+cat_cols
y=np.log(h.price)
tr=h.year<=2022; va=h.year==2023; te=h.year==2024
params=dict(objective='regression',learning_rate=0.05,num_leaves=255,min_data_in_leaf=40,feature_fraction=0.85,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=8)
dtr=lgb.Dataset(h.loc[tr,feats],y[tr],categorical_feature=cat_cols); dva=lgb.Dataset(h.loc[va,feats],y[va],reference=dtr)
m=lgb.train(params,dtr,num_boost_round=3000,valid_sets=[dva],callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
print('trained',m.best_iteration,'trees',round(time.time()-t0),'s')
def metrics(mask,name):
    p=np.exp(m.predict(h.loc[mask,feats],num_iteration=m.best_iteration)); a=h.loc[mask,'price'].values
    ape=np.abs(p-a)/a; r2=1-np.sum((np.log(p)-np.log(a))**2)/np.sum((np.log(a)-np.log(a).mean())**2)
    return dict(name=name,n=int(mask.sum()),median_ape=round(float(np.median(ape))*100,1),mean_ape=round(float(ape.mean())*100,1),within_10pct=round(float((ape<=.1).mean())*100,1),within_20pct=round(float((ape<=.2).mean())*100,1),r2_log=round(float(r2),3))
# baseline: borough x type x year median price per m2 from the training set, applied to area
base=h[tr].assign(ppm=lambda d:d.price/d.tfarea).groupby(['borough','propertytype'],observed=True).ppm.median()
def baseline(mask,name):
    d=h[mask]; p=d.set_index(['borough','propertytype']).index.map(base).values*d.tfarea.values; a=d.price.values
    ape=np.abs(p-a)/a; return dict(name=name,median_ape=round(float(np.nanmedian(ape))*100,1),within_20pct=round(float(np.nanmean(ape<=.2))*100,1))
res={'model':{'trees':int(m.best_iteration),'features':feats,'n_train':int(tr.sum()),'train_years':'2010-2022','validate':'2023','test':'2024'},
     'validation_2023':metrics(va,'validation 2023'),'test_2024':metrics(te,'test 2024'),
     'baseline_borough_type_ppm':{'validation_2023':baseline(va,'baseline 2023'),'test_2024':baseline(te,'baseline 2024')}}
imp=sorted(zip(feats,m.feature_importance('gain')),key=lambda x:-x[1]); tot=sum(v for _,v in imp); res['importance_gain_pct']={k:round(v/tot*100,1) for k,v in imp}
print(json.dumps({k:res[k] for k in ['validation_2023','test_2024','baseline_borough_type_ppm']},indent=1)); print(res['importance_gain_pct'])

# refit on everything through 2024 for deployment, same number of trees
dall=lgb.Dataset(h[feats],y,categorical_feature=cat_cols); mfull=lgb.train(params,dall,num_boost_round=m.best_iteration)
mfull.save_model(str(OUT/"price_model.txt"))
# category vocabularies so the API can encode new rows identically
json.dump({k:list(h[k].cat.categories) for k in cat_cols},open(OUT/"price_model_cats.json",'w'))
# yield anchors: ONS rent (borough x bedrooms, latest month) / model value of a typical dwelling of that size in that borough
p=pd.read_csv(RAW/"pipr_london_boroughs.csv"); p['Time period']=pd.to_datetime(p['Time period']); p=p[p['Time period']==p['Time period'].max()].set_index('Area name')
bedcol={0:'Rental price one bed',1:'Rental price one bed',2:'Rental price two bed',3:'Rental price three bed',4:'Rental price four or more bed',5:'Rental price four or more bed'}
h['bedrooms']=np.where(h.tfarea<38,0,np.minimum((h.numberrooms-1).clip(1,5),(h.tfarea/18).astype(int).clip(1,5)))
h['t']=(pd.Timestamp('2026-11-07')-pd.Timestamp('2010-01-01')).days/365.25  # value today
h['value_now']=np.exp(mfull.predict(h[feats]))
yields={}
for (b,bed),grp in h[h.year>=2018].groupby(['borough','bedrooms'],observed=True):
    bb='Westminster' if b=='City of London' else b
    rent=float(p.loc[bb,bedcol[int(bed)]])*(0.85 if bed==0 else 1)
    yields[f"{b}|{int(bed)}"]=dict(annual_yield=round(rent*12/float(grp.value_now.median()),4),ons_rent=rent,median_value=int(grp.value_now.median()),n=int(len(grp)))
json.dump(yields,open(OUT/"rent_yields.json",'w'))
res['yield_examples']={k:yields[k] for k in list(yields)[:4]}
json.dump(res,open(OUT/"price_model_metrics.json",'w'),indent=1)
print('saved; total',round(time.time()-t0),'s')

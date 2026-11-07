"""Pull the 33 London borough CSVs out of the UCL house-price-per-sqm zip using HTTP range
requests (no 1.3 GB download). Writes data/raw/hpm_<Borough>.csv"""
import json, struct, zlib, subprocess, pathlib, sys
URL="https://data.london.gov.uk/download/epo9w/e4a4f81e-d922-4d14-b6f7-9a4d8c09e9bc/hpm_la_2024.zip"
RAW=pathlib.Path(__file__).resolve().parents[1]/"data/raw"
LONDON=['Barking_and_Dagenham','Barnet','Bexley','Brent','Bromley','Camden','City_of_London','Croydon','Ealing','Enfield','Greenwich','Hackney','Hammersmith_and_Fulham','Haringey','Harrow','Havering','Hillingdon','Hounslow','Islington','Kensington_and_Chelsea','Kingston_upon_Thames','Lambeth','Lewisham','Merton','Newham','Redbridge','Richmond_upon_Thames','Southwark','Sutton','Tower_Hamlets','Waltham_Forest','Wandsworth','Westminster']
def rng(a,b): return subprocess.run(['curl','-s','-m','600','--retry','3','-r',f'{a}-{b}',URL],capture_output=True,check=True).stdout
entries=json.load(open(RAW/"hpm_zip_index.json"))
for name,comp,csz,usz,off in entries:
    b=name.split('/')[-1].replace('_link_26122024.csv','')
    if b not in LONDON: continue
    out=RAW/f"hpm_{b}.csv"
    if out.exists() and out.stat().st_size>1000: print('skip',b); continue
    hdr=rng(off,off+29); fl,el=struct.unpack('<HH',hdr[26:30]); start=off+30+fl+el
    raw=zlib.decompress(rng(start,start+csz-1),-15) if comp==8 else rng(start,start+csz-1)
    out.write_bytes(raw); print('got',b,round(usz/1e6,1),'MB',flush=True)
print('done')

"""Validate fixed source inputs and render an all-window registration overview."""
import itertools
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from pyproj import Geod
from common import ROOT,read_json,write_json,sha256,utc_now

maps=read_json(ROOT/'project/data/map_manifest_512.json');styles=read_json(ROOT/'project/data/style_manifest_512.json')
assert len(maps)==16 and len({m['map_id'] for m in maps})==16
assert len({m['region_id'] for m in maps})==16
assert len(styles)==3 and all(s['is_public_domain'] for s in styles)
assert len({m['content_sha256'] for m in maps})==16
geod=Geod(ellps='WGS84');distances=[];rows=[]
for a,b in itertools.combinations(maps,2):
 d=geod.inv(a['center_lon'],a['center_lat'],b['center_lon'],b['center_lat'])[2]
 assert d>10000,'Overlapping or adjacent window centers'
 distances.append((d,a['map_id'],b['map_id']))
canvas=Image.new('RGB',(4*300,4*332),'white');draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for i,m in enumerate(maps):
 assert sha256(ROOT/m['content_path'])==m['content_sha256']
 assert sha256(ROOT/m['prior_path'])==m['prior_sha256']
 assert sha256(ROOT/m['raw_path'])==m['sha256']
 g=np.load(ROOT/m['prior_path']);mask=np.asarray(Image.open(ROOT/m['mask_path']))>0
 assert g.shape==(512,512) and np.isfinite(g).all() and 0<=g.min()<=g.max()<=1 and mask.any()
 x=(i%4)*300;y=(i//4)*332
 draw.text((x+3,y+3),f'{m["map_id"]}: {m["region_id"]}',font=font,fill='black')
 draw.text((x+3,y+22),m['split'],font=font,fill='black')
 canvas.paste(Image.open(ROOT/m['content_path']).resize((288,288)),(x,y+44))
 rows.append({k:m[k] for k in ['map_id','region_id','split','category','center_lat','center_lon','road_segments','junction_count','road_pixel_fraction','content_sha256']})
for s in styles:
 assert sha256(ROOT/s['raw_path'])==s['sha256']
 assert sha256(ROOT/s['image_path'])==s['image_sha256']
canvas.save(ROOT/'project/data/all_source_maps.png')
pd.DataFrame(rows).to_csv(ROOT/'project/data/source_summary.csv',index=False)
minimum=min(distances)
write_json(ROOT/'provenance/data_audit.json',{'checked_utc':utc_now(),'passed':True,'map_count':16,'development_maps':4,'test_maps':12,
 'style_count':3,'minimum_center_separation_m':minimum[0],'closest_pair':list(minimum[1:]),'hashes_verified':True,
 'note':'Fixed, geographically separated convenience windows; no random geographical sampling or independent style sampling claimed.'})
print('All input hashes, 16 distinct maps, three CC0 style records and geographic separation checked.')

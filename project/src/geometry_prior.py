"""Registered road rendering and source-only geometric importance.

The renderer uses symbolic widths, not physical road widths. Source OSM ways
tagged as bridges/tunnels or nonzero layers are excluded from this planar task.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from pyproj import CRS,Transformer
from scipy.ndimage import distance_transform_edt,binary_erosion
from shapely.geometry import LineString,box
from common import ROOT,read_json,write_json,sha256,relative,utc_now

ROAD_WIDTHS={'primary':8.0,'secondary':7.0,'tertiary':6.0,'residential':5.0,
             'unclassified':5.0,'living_street':5.0,'pedestrian':4.0,'service':4.0}

def importance_field(road_mask,junction_mask,sigma_road=3.,sigma_junction=8.,kappa=1.):
    if not road_mask.any():raise ValueError('Road-free input is outside study scope')
    gr=np.exp(-distance_transform_edt(~road_mask)**2/(2*sigma_road**2))
    gj=np.exp(-distance_transform_edt(~junction_mask)**2/(2*sigma_junction**2)) if junction_mask.any() else np.zeros_like(gr)
    return ((gr+kappa*gj)/(1+kappa)).astype(np.float32)

def render_map(row,resolution,outdir):
    side=row['render_square_side_m'];half=side/2
    projection=f'+proj=aeqd +lat_0={row["center_lat"]} +lon_0={row["center_lon"]} +datum=WGS84 +units=m +no_defs'
    crs=CRS.from_proj4(projection)
    trans=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    bounds=box(-half,-half,half,half)
    def xy(lon,lat):
        x,y=trans.transform(lon,lat)
        return ((x+half)/side*resolution,(half-y)/side*resolution)
    raw=read_json(ROOT/row['raw_path']); roads=[];neighbors=defaultdict(set);node_xy={};excluded=[]
    for way in raw['elements']:
        tags=way.get('tags',{});hw=tags.get('highway')
        if hw not in ROAD_WIDTHS:continue
        if tags.get('bridge','no') not in ('no','false','0') or tags.get('tunnel','no') not in ('no','false','0') or tags.get('layer','0') not in ('0','0.0'):
            excluded.append(way['id']);continue
        geo=way.get('geometry',[])
        if len(geo)<2:continue
        pts=[trans.transform(p['lon'],p['lat']) for p in geo]
        clip=LineString(pts).intersection(bounds)
        if clip.is_empty:continue
        segments=[clip] if clip.geom_type=='LineString' else list(getattr(clip,'geoms',[]))
        for segment in segments:
            if segment.geom_type!='LineString':continue
            coords=[((x+half)/side*resolution,(half-y)/side*resolution) for x,y in segment.coords]
            roads.append({'osm_id':way['id'],'highway':hw,'pixels':coords,'width_px':ROAD_WIDTHS[hw]*resolution/512})
        nodes=way.get('nodes',[])
        for nid,p in zip(nodes,geo):node_xy[nid]=xy(p['lon'],p['lat'])
        for a,b in zip(nodes,nodes[1:]):neighbors[a].add(b);neighbors[b].add(a)
    junctions=[(nid,*node_xy[nid]) for nid,adj in neighbors.items() if len(adj)>=3 and 0<=node_xy[nid][0]<resolution and 0<=node_xy[nid][1]<resolution]
    ss=4; high=Image.new('L',(resolution*ss,resolution*ss));d=ImageDraw.Draw(high)
    for road in roads:
        pts=[(x*ss,y*ss) for x,y in road['pixels']];width=max(1,round(road['width_px']*ss))
        d.line(pts,fill=255,width=width,joint='curve')
        radius=width/2
        for x,y in pts:d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=255)
    alpha=np.asarray(high.resize((resolution,resolution),Image.Resampling.LANCZOS)).astype(np.float32)/255
    mask=alpha>=.5
    jmask=np.zeros_like(mask)
    for _,x,y in junctions:jmask[min(resolution-1,int(y)),min(resolution-1,int(x))]=True
    prior=importance_field(mask,jmask,3*resolution/512,8*resolution/512)
    bg=np.array([246,242,232],dtype=np.float32);fg=np.array([62,68,81],dtype=np.float32)
    content=np.rint(bg[None,None,:]*(1-alpha[:,:,None])+fg[None,None,:]*alpha[:,:,None]).astype(np.uint8)
    boundary=mask & ~binary_erosion(mask)
    outdir.mkdir(parents=True,exist_ok=True)
    Image.fromarray(content).save(outdir/'content.png')
    Image.fromarray(mask.astype(np.uint8)*255).save(outdir/'road_mask.png')
    Image.fromarray(boundary.astype(np.uint8)*255).save(outdir/'boundary.png')
    Image.fromarray(jmask.astype(np.uint8)*255).save(outdir/'junction_mask.png')
    np.save(outdir/'prior.npy',prior)
    # This source visualization is a diagnostic, never an empirical result.
    heat=np.stack([prior,np.zeros_like(prior),1-prior],axis=-1)
    Image.fromarray((255*heat).astype(np.uint8)).save(outdir/'prior.png')
    overlay=Image.fromarray(content);od=ImageDraw.Draw(overlay)
    for _,x,y in junctions:od.ellipse((x-3,y-3,x+3,y+3),outline=(255,0,0),width=1)
    overlay.save(outdir/'registration.png')
    geom={'projection':projection,'projection_wkt':crs.to_wkt(),'resolution':resolution,'extent_m':[-half,-half,half,half],
          'pixel_transform':'x=(easting+side/2)/side*resolution; y=(side/2-northing)/side*resolution',
          'road_widths_at_512':ROAD_WIDTHS,'roads':roads,'junctions_osm_id_x_y':junctions,'excluded_grade_separated_way_ids':excluded,
          'sigma_road_px':3*resolution/512,'sigma_junction_px':8*resolution/512,'kappa':1}
    write_json(outdir/'geometry.json',geom)
    return {**row,'resolution':resolution,'projection':projection,'road_segments':len(roads),'junction_count':len(junctions),
            'road_pixel_fraction':float(mask.mean()),'content_path':relative(outdir/'content.png'),
            'mask_path':relative(outdir/'road_mask.png'),'boundary_path':relative(outdir/'boundary.png'),
            'prior_path':relative(outdir/'prior.npy'),'geometry_path':relative(outdir/'geometry.json'),
            'content_sha256':sha256(outdir/'content.png'),'prior_sha256':sha256(outdir/'prior.npy')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--resolution',type=int,default=512);p.add_argument('--available-only',action='store_true');args=p.parse_args()
    rows=read_json(ROOT/'project/data/source_windows.json')['windows'];manifest=[]
    for row in rows:
        if args.available_only and not (ROOT/row['raw_path']).exists():continue
        record=render_map(row,args.resolution,ROOT/f'project/data/rendered/{args.resolution}/{row["map_id"]}')
        manifest.append(record);print(row['map_id'],record['road_segments'],record['junction_count'],flush=True)
    write_json(ROOT/f'project/data/map_manifest_{args.resolution}.json',manifest)
    style_path=ROOT/'project/data/source_styles.json'
    if style_path.exists():
        styles=[]
        for style in read_json(style_path):
            dest=ROOT/f'project/data/rendered/{args.resolution}/styles/{style["style_id"]}.png';dest.parent.mkdir(parents=True,exist_ok=True)
            Image.open(ROOT/style['raw_path']).convert('RGB').resize((args.resolution,args.resolution),Image.Resampling.BICUBIC).save(dest)
            styles.append({**style,'resolution':args.resolution,'image_path':relative(dest),'image_sha256':sha256(dest)})
        write_json(ROOT/f'project/data/style_manifest_{args.resolution}.json',styles)
    write_json(ROOT/f'project/data/render_config_{args.resolution}.json',{'created_utc':utc_now(),'resolution':args.resolution,'widths_at_512':ROAD_WIDTHS,'supersampling':4,'extent_m':1200,'map_count':len(manifest)})

if __name__=='__main__':main()

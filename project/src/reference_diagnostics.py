"""Unchanged-source and same-VAE endpoints; run only after protocol freeze."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc,time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import AutoencoderKL
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from common import ROOT,read_json,write_json,sha256,relative,utc_now
from evaluate import ImageEvaluator,boundary_metrics,METRIC_SPEC

if not (ROOT/'configs/protocol_frozen.json').exists():raise RuntimeError('Freeze formal protocol first')
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
maps=read_json(ROOT/'project/data/map_manifest_512.json');styles=read_json(ROOT/'project/data/style_manifest_512.json')
out=ROOT/'project/results/reference_diagnostics';out.mkdir(parents=True,exist_ok=True)
vae=AutoencoderKL.from_pretrained(str(ROOT/'models/sd15'),subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval().requires_grad_(False)
vae.enable_slicing();reconstructions=[];checks=[]
with torch.no_grad():
 for m in maps:
  x=ToTensor()(Image.open(ROOT/m['content_path']).convert('RGB')).unsqueeze(0).to('cuda',dtype=torch.bfloat16)
  z=vae.encode(x*2.-1.)['latent_dist'].mean*vae.config.scaling_factor
  y=(vae.decode(z/vae.config.scaling_factor)[0]*.5+.5).clamp(0,1)
  p=out/(m['map_id']+'_vae.png');save_image(y,p)
  reconstructions.append({'map_id':m['map_id'],'path':relative(p),'sha256':sha256(p)})
  original=ROOT/f'project/results/baseline_deterministic/{m["map_id"]}_vae.png'
  if original.exists():
   same=sha256(original)==sha256(p);checks.append({'map_id':m['map_id'],'matches_upstream_vae_png':same})
   if not same:raise RuntimeError('Standalone VAE reconstruction differs from upstream diagnostic')
del vae,x,z,y;gc.collect();torch.cuda.empty_cache()
evaluator=ImageEvaluator();rows=[]
for m in maps:
 boundary=np.asarray(Image.open(ROOT/m['boundary_path']))>0
 for method,p in [('Source',ROOT/m['content_path']),('VAE',out/(m['map_id']+'_vae.png'))]:
  edge=boundary_metrics(np.asarray(Image.open(p).convert('RGB')),boundary)
  for style in styles:
   metrics=evaluator.pair(p,ROOT/m['content_path'],ROOT/style['image_path'])
   rows.append({'map_id':m['map_id'],'region_id':m['region_id'],'split':m['split'],'style_id':style['style_id'],'method':method,
                'output_path':relative(p),'output_sha256':sha256(p),**edge,**metrics})
pd.DataFrame(rows).to_csv(out/'per_image_metrics.csv',index=False)
write_json(out/'manifest.json',{'created_utc':utc_now(),'kind':'source/compression diagnostic, not a competing style-transfer method',
 'vae_model':'same pinned SD1.5 VAE, BF16, posterior mean, identical encoding/decoding arithmetic',
 'upstream_checks':checks,'reconstructions':reconstructions,'metric_spec':METRIC_SPEC,'row_count':len(rows)})
print('Finished source/VAE diagnostics:',len(rows),'map-style-method rows',flush=True)

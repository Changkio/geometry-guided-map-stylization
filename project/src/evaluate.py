"""Independent image-level boundary, LPIPS and VGG Gram evaluation."""
import os
from common import ROOT,read_json,write_json,sha256,relative,utc_now
os.environ.setdefault('TORCH_HOME',str(ROOT/'models/evaluation'))
import argparse,time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F
from torchvision.models import vgg19,VGG19_Weights
from torchvision.transforms import ToTensor
import lpips

METRIC_SPEC={
    'version':'boundary-v1_lpips-alex-v01_vgg19-v1',
    'canny_low':50,'canny_high':150,'canny_aperture':3,'canny_L2gradient':True,
    'gaussian_kernel':3,'gaussian_sigma':1.0,'roi_radius_at_512':12,
    'tolerances_at_512':[1,2,3],'primary_tolerance_at_512':2,
    'lpips_network':'alex','lpips_version':'0.1','lpips_input':'full-resolution RGB scaled to [-1,1]',
    'style_network':'VGG19_Weights.IMAGENET1K_V1','style_layers':[1,6,11,20,29],
    'style_resolution':256,'style_preprocessing':'bilinear antialiased resize, ImageNet RGB normalization',
    'gram_definition':'FF^T/(H*W); layer mean of squared entrywise differences (equiv. Frobenius squared / C^2)',
    'interpretation':'boundary/appearance proxies, not topology or human preference',
}

def boundary_metrics(output_rgb,reference_boundary,tolerances=(1,2,3),roi_radius=12):
    ref=reference_boundary.astype(bool)
    if not ref.any():raise ValueError('Reference boundary is empty')
    gray=cv2.cvtColor(output_rgb,cv2.COLOR_RGB2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),1.0)
    edges=cv2.Canny(gray,50,150,apertureSize=3,L2gradient=True)>0
    distance_ref=distance_transform_edt(~ref)
    roi=distance_ref<=roi_radius
    pred=edges & roi
    result={'predicted_edge_count':int(pred.sum()),'reference_boundary_count':int(ref.sum()),'roi_pixels':int(roi.sum())}
    distance_pred=distance_transform_edt(~pred) if pred.any() else None
    for tol in tolerances:
        if not pred.any():precision=recall=f1=0.
        else:
            precision=float((distance_ref[pred]<=tol).mean())
            recall=float((distance_pred[ref]<=tol).mean())
            f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        result.update({f'boundary_precision_{tol}':precision,f'boundary_recall_{tol}':recall,f'boundary_f1_{tol}':f1})
    return result

class ImageEvaluator:
    def __init__(self,device='cuda'):
        self.device=device
        self.vgg=vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:30].eval().requires_grad_(False).to(device)
        self.lpips=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).to(device)
        self.mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
        self.std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
        self.style_cache={}
    def tensor(self,path):
        return ToTensor()(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
    @torch.no_grad()
    def grams(self,image):
        x=F.interpolate(image,size=(256,256),mode='bilinear',align_corners=False,antialias=True)
        x=(x-self.mean)/self.std;grams=[]
        for i,layer in enumerate(self.vgg):
            x=layer(x)
            if i in METRIC_SPEC['style_layers']:
                f=x.flatten(2);grams.append(f@f.transpose(1,2)/f.shape[-1])
        return grams
    @torch.no_grad()
    def pair(self,output,content,style):
        x=self.tensor(output);c=self.tensor(content)
        if str(style) not in self.style_cache:self.style_cache[str(style)]=self.grams(self.tensor(style))
        gg=self.grams(x);gs=self.style_cache[str(style)]
        return {'lpips_content':float(self.lpips(x*2-1,c*2-1).item()),
                'gram_style_distance':float(torch.stack([(a-b).square().mean() for a,b in zip(gg,gs)]).mean().item())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--result-root',required=True);p.add_argument('--output');p.add_argument('--device',default='cuda');a=p.parse_args()
    base=ROOT/a.result_root;out=ROOT/a.output if a.output else base/'evaluation';out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();torch.set_num_threads(4)
    evaluator=ImageEvaluator(a.device);rows=[]
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or not (folder/'result.json').exists():continue
        r=read_json(folder/'result.json')
        if r['status']!='success':continue
        resolved=read_json(folder/'resolved_config.json');cfg=resolved['config'];m=resolved['map'];s=resolved['style']
        output=ROOT/r['output_path']
        if sha256(output)!=r['output_sha256']:raise ValueError('Output checksum changed')
        rgb=np.asarray(Image.open(output).convert('RGB'))
        boundary=np.asarray(Image.open(ROOT/m['boundary_path']))>0
        if rgb.shape[:2]!=boundary.shape:raise ValueError('Output/reference dimensions differ')
        if cfg['resolution']!=512:raise ValueError('Metric parameters must be explicitly re-frozen at other resolutions')
        metrics=boundary_metrics(rgb,boundary)
        metrics.update(evaluator.pair(output,ROOT/m['content_path'],ROOT/s['image_path']))
        rows.append({**r,**metrics,'split':m['split'],'region_id':m['region_id'],'category':m['category'],
                     'steps':cfg['steps'],'content_mode':cfg['content_mode'],'content_weight':cfg['content_weight'],'beta':cfg.get('beta',0)})
        print(r['run_id'],'F1',round(metrics['boundary_f1_2'],4),'style',round(metrics['gram_style_distance'],4),flush=True)
    pd.DataFrame(rows).to_csv(out/'per_image_metrics.csv',index=False)
    write_json(out/'metric_spec.json',{**METRIC_SPEC,'executed_utc':utc_now(),'device':a.device,'opencv':cv2.__version__,'evaluation_seconds_including_loading':time.perf_counter()-start,'image_count':len(rows),
               'script_sha256':sha256(Path(__file__))})

if __name__=='__main__':main()

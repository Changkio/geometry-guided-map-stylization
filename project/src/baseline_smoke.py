"""Run the untouched upstream notebook's optimize route on licensed inputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,json,argparse
from pathlib import Path
import torch
from accelerate.utils import set_seed
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from common import ROOT,read_json,write_json,sha256,utc_now
sys.path.insert(0,str(ROOT/'project/third_party/AttentionDistillation'))
from pipeline_sd import ADPipeline
from utils import load_image,Controller

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
args=argparse.ArgumentParser();args.add_argument('--output',default='project/results/baseline_deterministic');args=args.parse_args()
out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
model=ROOT/'models/sd15'
pipe=ADPipeline.from_pretrained(str(model),scheduler=DDIMScheduler.from_pretrained(str(model),subfolder='scheduler'),
                torch_dtype=torch.bfloat16,safety_checker=None,local_files_only=True)
pipe.classifier=pipe.unet
maps={m['map_id']:m for m in read_json(ROOT/'project/data/map_manifest_512.json')}
styles={s['style_id']:s for s in read_json(ROOT/'project/data/style_manifest_512.json')}
content=load_image(ROOT/maps['dev01']['content_path'],(512,512))
style=load_image(ROOT/styles['hokusai']['image_path'],(512,512))
set_seed(42)
start=time.perf_counter()
images=pipe.optimize(lr=.05,batch_size=1,iters=1,width=512,height=512,weight=.25,
    controller=Controller((10,16)),style_image=style,content_image=content,
    mixed_precision='bf16',num_inference_steps=16,enable_gradient_checkpoint=False)
torch.cuda.synchronize();elapsed=time.perf_counter()-start
save_image(images,out/'output.png')
for mid,row in maps.items():
    if row['split']!='development':continue
    img=load_image(ROOT/row['content_path'],(512,512))
    rec=pipe.latent2image(pipe.image2latent(img))
    save_image(rec,out/f'{mid}_vae.png')
write_json(out/'report.json',{'status':'success','time_utc':utc_now(),'route':'untouched upstream ADPipeline.optimize',
    'upstream_commit':'142800c389bb4dff41e7f79cb9c2e88c981bbec2','steps':16,'resolution':512,'seed':42,
    'precision':'bf16','lr':.05,'weight':.25,'inner_iterations':1,'layers':[10,16],
    'elapsed_seconds_including_initialization':elapsed,'output_sha256':sha256(out/'output.png'),
    'shared_unet_classifier':pipe.classifier is pipe.unet,'all_weights_frozen':all(not p.requires_grad for p in pipe.unet.parameters()),
    'model_gradients_absent':all(p.grad is None for p in pipe.unet.parameters()),'image_finite':bool(torch.isfinite(images).all())})
print('UPSTREAM SMOKE SUCCESS',elapsed,flush=True)

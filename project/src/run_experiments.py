"""Manifest-driven serial AD runs with complete configurations and failures."""
import os
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,json,platform,sys,threading,time,traceback
from pathlib import Path
import numpy as np
import psutil
import torch
from PIL import Image
from diffusers import DDIMScheduler
from accelerate.utils import set_seed
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from common import ROOT,read_json,write_json,sha256,relative,utc_now
from map_pipeline import MapADPipeline

class MemorySampler:
    def __init__(self):self.stop=threading.Event();self.peak=0;self.min_available=2**64
    def sample(self):
        process=psutil.Process()
        while not self.stop.is_set():
            self.peak=max(self.peak,process.memory_info().rss)
            self.min_available=min(self.min_available,psutil.virtual_memory().available)
            self.stop.wait(.1)
    def __enter__(self):self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()

def image_tensor(path,resolution):
    im=Image.open(path).convert('RGB')
    if im.size!=(resolution,resolution):raise ValueError(f'Unexpected image size {im.size}')
    return ToTensor()(im).unsqueeze(0)

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--only',nargs='*');args=parser.parse_args()
    spec=read_json(ROOT/args.config);res=spec['common']['resolution']
    maps={m['map_id']:m for m in read_json(ROOT/f'project/data/map_manifest_{res}.json')}
    styles={s['style_id']:s for s in read_json(ROOT/f'project/data/style_manifest_{res}.json')}
    model=ROOT/spec.get('model_path','models/sd15')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for formal model runs')
    load_start=time.perf_counter()
    pipe=MapADPipeline.from_pretrained(str(model),scheduler=DDIMScheduler.from_pretrained(str(model),subfolder='scheduler'),
                    safety_checker=None,torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.initialize_research(spec['common'].get('precision','bf16'),tuple(spec['common'].get('layers',[10,16])))
    torch.cuda.synchronize();load_time=time.perf_counter()-load_start
    model_provenance=read_json(ROOT/'provenance/model_download.json')
    code_hashes={relative(ROOT/'project/src'/name):sha256(ROOT/'project/src'/name) for name in ['map_pipeline.py','geometry_losses.py','run_experiments.py','common.py']}
    environment={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,
                 'gpu':torch.cuda.get_device_name(),'load_seconds':load_time}
    outroot=ROOT/spec['output_root'];outroot.mkdir(parents=True,exist_ok=True)
    write_json(outroot/'environment.json',environment)
    if spec.get('warmup_steps',0):
        warmed=set();warmup_records=[]
        for job in spec['jobs']:
            mode=job['content_mode']
            if mode in warmed:continue
            cfg={**spec['common'],**job,'steps':spec['warmup_steps']}
            mr=maps[job['map_id']];sr=styles[job['style_id']]
            content=image_tensor(ROOT/mr['content_path'],res);style=image_tensor(ROOT/sr['image_path'],res)
            prior=torch.from_numpy(np.load(ROOT/mr['prior_path']))
            set_seed(cfg.get('seed',42));warm_start=time.perf_counter()
            im,diag=pipe.optimize_map(content,style,prior,cfg)
            torch.cuda.synchronize()
            warmup_records.append({'mode':mode,'steps':cfg['steps'],'seconds':time.perf_counter()-warm_start,'diagnostics':diag})
            del im;warmed.add(mode)
        write_json(outroot/'warmup.json',warmup_records)
    failures=0
    for job in spec['jobs']:
        if args.only and job['run_id'] not in args.only:continue
        mid=job['map_id'];sid=job['style_id'];config={**spec['common'],**job};name=config['run_id']
        outdir=outroot/name
        if (outdir/'result.json').exists() and read_json(outdir/'result.json').get('status')=='success':
            existing=read_json(outdir/'resolved_config.json')
            if existing['config']!=config:raise RuntimeError('Resume configuration differs: '+name)
            if existing['code_hashes']!=code_hashes:raise RuntimeError('Resume generation source differs: '+name)
            print('SKIP completed',name,flush=True);continue
        if outdir.exists():
            attempt=1
            while (outroot/f'{name}.failed_attempt_{attempt}').exists():attempt+=1
            outdir.rename(outroot/f'{name}.failed_attempt_{attempt}')
        outdir.mkdir(parents=True)
        maprow=maps[mid];stylerow=styles[sid]
        write_json(outdir/'resolved_config.json',{'config':config,'map':maprow,'style':stylerow,
               'model':model_provenance,'code_hashes':code_hashes,'start_utc':utc_now()})
        write_json(outdir/'result.json',{'status':'running','start_utc':utc_now(),'run_id':name})
        print('START',name,config['steps'],config['content_mode'],flush=True)
        content=image_tensor(ROOT/maprow['content_path'],res);style=image_tensor(ROOT/stylerow['image_path'],res)
        prior=torch.from_numpy(np.load(ROOT/maprow['prior_path']))
        set_seed(config.get('seed',42))
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.perf_counter()
        try:
            with (outdir/'steps.jsonl').open('w',encoding='utf-8') as log,MemorySampler() as memory:
                def report(row):
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                    if row['step']%10==0:print(name,row['step'],round(row['loss'],5),flush=True)
                image,diagnostics=pipe.optimize_map(content,style,prior,config,report,config.get('check_equivalence',False))
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                alloc=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
                save_image(image,outdir/'output.png')
            write_json(outdir/'diagnostics.json',diagnostics)
            record={'status':'success','run_id':name,'map_id':mid,'style_id':sid,'method':config['method'],'end_utc':utc_now(),
                    'generation_seconds':elapsed,'peak_allocated_vram_bytes':alloc,'peak_reserved_vram_bytes':reserved,
                    'peak_process_ram_bytes':memory.peak,'minimum_system_available_ram_bytes':memory.min_available,
                    'output_sha256':sha256(outdir/'output.png'),'output_path':relative(outdir/'output.png')}
            write_json(outdir/'result.json',record)
            with (outroot/'runs.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(record)+'\n')
            print('DONE',name,round(elapsed,2),'seconds',round(alloc/2**30,2),'GiB',flush=True)
            del image
        except Exception as error:
            failures+=1
            (outdir/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
            write_json(outdir/'result.json',{'status':'failed','run_id':name,'end_utc':utc_now(),'error':repr(error),'elapsed_seconds':time.perf_counter()-start})
            print('FAILED',name,repr(error),flush=True)
            pipe.cache.clear();gc.collect();torch.cuda.empty_cache()
            if spec.get('stop_on_failure',True):raise
    print('FINISHED; failures:',failures,flush=True)

if __name__=='__main__':main()

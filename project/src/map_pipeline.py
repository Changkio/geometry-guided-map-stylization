"""A small instrumented extension of the pinned official optimize route.

Original files are unchanged. AD, uniform Q and QK are imported directly from
the pinned mother repository. Only geometry Q changes the content objective.
"""
import sys
from pathlib import Path
import time
import numpy as np
import torch
from accelerate import Accelerator
from common import ROOT
sys.path.insert(0,str(ROOT/'project/third_party/AttentionDistillation'))
from pipeline_sd import ADPipeline
from utils import Controller,DataCache,register_attn_control
from losses import ad_loss,q_loss,qk_loss
from geometry_losses import geometry_q_loss,projected_prior

class GridCache(DataCache):
    def __init__(self):
        super().__init__();self.grids=[];self.current_grid=None
    def clear(self):
        super().clear();self.grids.clear();self.current_grid=None
    def add(self,q,k,v,out):
        if self.current_grid is None:raise RuntimeError('No actual spatial-grid observation')
        if self.current_grid[0]*self.current_grid[1]!=q.shape[-2]:raise RuntimeError('Actual grid/token mismatch')
        super().add(q,k,v,out);self.grids.append(self.current_grid)

class MapADPipeline(ADPipeline):
    def initialize_research(self,precision='bf16',layers=(10,16)):
        if getattr(self,'research_initialized',False):
            if self.research_precision!=precision or list(self.controller.self_layers)!=list(range(*layers)):
                raise ValueError('A loaded pipeline cannot change precision/layers')
            return
        self.classifier=self.unet
        self.accelerator=Accelerator(mixed_precision=precision,gradient_accumulation_steps=1)
        self.init(False)
        self.cache=GridCache();self.controller=Controller(self_layers=layers)
        self.grid_hooks=[]
        def observe(module,args):
            if not args or not isinstance(args[0],torch.Tensor) or args[0].ndim!=4:
                raise RuntimeError('Expected spatial Transformer2D input')
            self.cache.current_grid=tuple(args[0].shape[-2:])
        for module in self.classifier.modules():
            if module.__class__.__name__=='Transformer2DModel':
                self.grid_hooks.append(module.register_forward_pre_hook(observe))
        register_attn_control(self.classifier,self.controller,self.cache)
        self.research_initialized=True;self.research_precision=precision

    def optimize_map(self,content,style,prior,config,step_callback=None,check_equivalence=False):
        self.initialize_research(config.get('precision','bf16'),tuple(config.get('layers',[10,16])))
        width=config['resolution'];steps=config['steps'];weight=config['content_weight'];beta=config.get('beta',0)
        mode=config['content_mode'];lr=config.get('lr',.05);iters=config.get('inner_iterations',1)
        style_latent=self.image2latent(style)
        # Same (otherwise unused) random allocation as the original optimize.
        latents=torch.randn((1,4,width//8,width//8),device=self.device)
        null=self.encode_prompt('',self.device,1,False)[0]
        content_latent=self.image2latent(content)
        latents=content_latent.clone().detach().float().requires_grad_()
        self.scheduler.set_timesteps(steps)
        optimizer=torch.optim.Adam([latents],lr=lr)
        optimizer=self.accelerator.prepare(optimizer)
        prior=prior.to(device=self.device,dtype=torch.float32)
        diagnostics={'shared_unet_classifier':self.classifier is self.unet,
                     'all_model_parameters_frozen':all(not p.requires_grad for m in [self.unet,self.vae,self.text_encoder] for p in m.parameters()),
                     'selected_layers':self.controller.self_layers,'equivalence':None,'latent_gradient_max':0.,'latent_gradient_nonzero':False}
        watched=[(n,p,p.detach().clone()) for n,p in list(self.unet.named_parameters())[::100]]
        for i,t in enumerate(self.scheduler.timesteps):
            with torch.no_grad():
                qs,ks,vs,outs=self.extract_feature(style_latent,t,null)
                qc,kc,vc,oc=self.extract_feature(content_latent,t,null)
            for j in range(iters):
                optimizer.zero_grad()
                q,k,v,out=self.extract_feature(latents,t,null)
                grids=list(self.cache.grids)
                sl=ad_loss(q,ks,vs,out,scale=config.get('attention_scale',1.))
                if mode=='q':cl=q_loss(q,qc)
                elif mode=='qk':cl=qk_loss(q,k,qc,kc)
                elif mode=='geometry_q':cl=geometry_q_loss(q,qc,grids,prior,beta,q_loss)
                else:raise ValueError(f'Unknown content mode: {mode}')
                loss=sl+weight*cl
                if check_equivalence and i==1 and j==0:
                    orig=sl+weight*q_loss(q,qc)
                    fallback=sl+weight*geometry_q_loss(q,qc,grids,prior,0,q_loss)
                    ga=torch.autograd.grad(orig,latents,retain_graph=True)[0]
                    ga_repeat=torch.autograd.grad(orig,latents,retain_graph=True)[0]
                    gb=torch.autograd.grad(fallback,latents,retain_graph=True)[0]
                    diagnostics['equivalence']={'loss_absolute_error':float((orig-fallback).abs()),
                         'latent_gradient_max_absolute_error':float((ga-gb).abs().max()),
                         'baseline_repeat_gradient_max_absolute_error':float((ga-ga_repeat).abs().max()),
                         'reference_gradient_max':float(ga.abs().max()),'passed':bool(torch.equal(ga,gb) and torch.equal(orig,fallback))}
                    del ga,ga_repeat,gb,orig,fallback
                self.accelerator.backward(loss)
                grad=latents.grad
                if not torch.isfinite(loss) or grad is None or not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite loss/latent gradient')
                gmax=float(grad.detach().abs().max())
                diagnostics['latent_gradient_max']=max(diagnostics['latent_gradient_max'],gmax)
                diagnostics['latent_gradient_nonzero'] |= gmax>0
                optimizer.step()
                if not torch.isfinite(latents).all():raise FloatingPointError('Nonfinite latent update')
                if step_callback:
                    step_callback({'step':i,'inner':j,'timestep':int(t),'loss':float(loss.detach()),'style_loss':float(sl.detach()),
                                   'content_loss':float(cl.detach()),'latent_grad_max':gmax,'grids':grids})
            diagnostics['observed_token_grids']=grids
            # Do not retain a previous generated graph into the next iteration.
            self.cache.clear()
            del q,k,v,out,qs,ks,vs,outs,qc,kc,vc,oc,sl,cl,loss,grad
        images=self.latent2image(latents.detach())
        diagnostics['sampled_parameter_tensors_unchanged']=all(torch.equal(p,ref) for _,p,ref in watched)
        diagnostics['sampled_parameter_names']=[n for n,_,_ in watched]
        diagnostics['no_model_parameter_gradients']=all(p.grad is None for m in [self.unet,self.vae,self.text_encoder] for p in m.parameters())
        diagnostics['total_inner_updates']=steps*iters
        self.cache.clear()
        return images,diagnostics

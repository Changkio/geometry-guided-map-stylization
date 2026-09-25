"""Explicit FP32 uniform-weight control without editing the pinned running generator.

The existing geometry implementation receives an all-zero field and beta one:
its normalized FP32 path then gives every token equal weight. Lambda remains
0.25. This isolates accumulation/arithmetic from the geometry redistribution.
"""
import sys
from pathlib import Path
import torch
from common import ROOT,read_json,write_json,sha256,utc_now
from map_pipeline import MapADPipeline
import run_experiments

class UniformFP32Pipeline(MapADPipeline):
    def optimize_map(self,content,style,prior,config,step_callback=None,check_equivalence=False):
        if config.get('prior_override')!='zero_for_uniform_fp32_control':raise ValueError('Missing explicit precision-control setting')
        if config['content_mode']!='geometry_q' or config['beta']!=1 or config['content_weight']!=.25:
            raise ValueError('Precision control must use beta1, zero prior, and lambda0.25')
        output,diagnostics=super().optimize_map(content,style,torch.zeros_like(prior),config,step_callback,check_equivalence)
        diagnostics['precision_control']={'prior_override':'all zero before token projection','uniform_token_weights':1,
            'coefficient':.25,'error_and_reduction_dtype':'float32','wrapper_sha256':sha256(Path(__file__))}
        return output,diagnostics

if __name__=='__main__':
    config=ROOT/'configs/precision_control.json';frozen=read_json(ROOT/'configs/protocol_frozen.json')
    for name in ['configs/precision_control.json','project/src/precision_control.py']:
        if sha256(ROOT/name)!=frozen['sha256'][name]:raise RuntimeError('Precision-control source changed after freeze')
    spec=read_json(config);out=ROOT/spec['output_root'];out.mkdir(parents=True,exist_ok=True)
    execution={'created_utc':utc_now(),'config_sha256':sha256(config),'wrapper_sha256':sha256(Path(__file__)),
        'purpose':'Uniform FP32 reduction control; no development selection, no geometry information in weights',
        'implementation':'Explicit subclass supplied as runner pipeline class; unchanged generation core files'}
    record=out/'execution_wrapper.json'
    if record.exists() and read_json(record)['wrapper_sha256']!=execution['wrapper_sha256']:raise RuntimeError('Cannot resume a changed wrapper')
    write_json(record,execution)
    run_experiments.MapADPipeline=UniformFP32Pipeline
    sys.argv=[sys.argv[0],'--config','configs/precision_control.json']
    run_experiments.main()

"""Paired map-level analysis of the separately frozen uniform FP32 control."""
import numpy as np
import pandas as pd
from common import ROOT,read_json,write_json,sha256,utc_now
from analyze import checked_table,METRICS

frozen=read_json(ROOT/'configs/protocol_frozen.json')
for name in ['configs/precision_control.json','project/src/precision_control.py','project/src/analyze_precision.py']:
    if sha256(ROOT/name)!=frozen['sha256'][name]:raise RuntimeError('Precision-control protocol changed: '+name)
control,spec=checked_table('project/results/precision_control','configs/precision_control.json')
main,_=checked_table('project/results/formal_test','configs/formal_test.json')
if len(control)!=36 or set(control.method)!={'Q32'} or set(control.split)!={'test'}:raise RuntimeError('Incomplete precision-control matrix')
data=pd.concat([main[main.method.isin(['B0','A1'])],control],ignore_index=True)
permap=data.groupby(['map_id','method'])[METRICS].mean().reset_index()
out=ROOT/'project/results/precision_control/analysis';out.mkdir(exist_ok=True)
permap.to_csv(out/'per_map_metrics.csv',index=False)
boot=np.random.default_rng(20260925).integers(0,12,size=(10000,12));rows=[];summary=[]
for metric in METRICS:
    pivot=permap.pivot(index='map_id',columns='method',values=metric).sort_index()
    if pivot.isna().any().any() or len(pivot)!=12:raise ValueError('Unbalanced precision comparison')
    for method in ['B0','Q32','A1']:
        values=pivot[method].to_numpy();lo,hi=np.quantile(values[boot].mean(1),[.025,.975])
        summary.append({'method':method,'metric':metric,'mean':values.mean(),'ci_low':lo,'ci_high':hi,'n_maps':12})
    for a,b in [('Q32','B0'),('A1','Q32')]:
        values=(pivot[a]-pivot[b]).to_numpy();lo,hi=np.quantile(values[boot].mean(1),[.025,.975])
        rows.append({'comparison':a+' - '+b,'metric':metric,'difference_mean':values.mean(),'ci_low':lo,'ci_high':hi,'n_maps':12})
pd.DataFrame(rows).to_csv(out/'paired_comparisons.csv',index=False)
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
write_json(out/'manifest.json',{'created_utc':utc_now(),'n_images':36,'n_maps':12,'no_tuning':True,
    'control_csv_sha256':sha256(ROOT/'project/results/precision_control/evaluation/per_image_metrics.csv'),
    'interpretation':'Q32-B0 measures numerical-path changes; A1-Q32 compares spatial weights within the same normalized FP32 implementation. Both use lambda0.25.'})
print(pd.DataFrame(rows).query("metric in ['boundary_f1_2','gram_style_distance']").to_string(index=False))

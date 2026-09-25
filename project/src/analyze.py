"""Development-only selection, protocol freezing, and map-level test analysis."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,read_json,write_json,sha256,relative,utc_now

METRICS=['boundary_f1_1','boundary_f1_2','boundary_f1_3','boundary_precision_2','boundary_recall_2',
         'lpips_content','gram_style_distance','generation_seconds','peak_allocated_vram_bytes','peak_reserved_vram_bytes','peak_process_ram_bytes']
METHODS=['B0','B1','A1','AT','B2']

def choose_candidate(candidates,baseline_gram):
    """Select without test inputs; make the style-mismatch fallback explicit."""
    if candidates.empty:raise ValueError('No development candidates')
    eligible=candidates[candidates.gram_style_distance<=1.05*baseline_gram]
    mismatch=eligible.empty
    if mismatch:
        ranked=candidates.assign(style_gap=(candidates.gram_style_distance-baseline_gram).abs())
        winner=ranked.sort_values(['style_gap','content_weight','beta']).iloc[0]
    else:
        winner=eligible.sort_values(['boundary_f1_2','content_weight','beta'],ascending=[False,True,True]).iloc[0]
    return winner,mismatch

def checked_table(result_root,config):
    base=ROOT/result_root;df=pd.read_csv(base/'evaluation/per_image_metrics.csv')
    spec=read_json(ROOT/config);jobs=spec['jobs'];expected={j['run_id'] for j in jobs}
    if len(expected)!=len(jobs):raise ValueError('Duplicate expected run IDs')
    if df.run_id.duplicated().any():raise ValueError('Duplicate evaluated run IDs')
    if set(df.run_id)!=expected:raise ValueError(f'Incomplete matrix: {len(df)}/{len(expected)}')
    if not np.isfinite(df[METRICS].to_numpy()).all():raise ValueError('Missing or nonfinite metric')
    for job in jobs:
        cfg=read_json(base/job['run_id']/'resolved_config.json')['config']
        if cfg!={**spec['common'],**job}:raise ValueError('Resolved config differs from search specification')
        result=read_json(base/job['run_id']/'result.json')
        if result['status']!='success' or sha256(ROOT/result['output_path'])!=result['output_sha256']:raise ValueError('Changed output')
    return df,spec

def select_and_freeze():
    target=ROOT/'configs/protocol_frozen.json'
    if target.exists():raise ValueError('Protocol already frozen; never overwrite after test outputs')
    df,spec=checked_table('project/results/development_search','configs/development_search.json')
    if set(df.split)!={'development'}:raise ValueError('Selection must use development only')
    permap=df.groupby(['method','map_id'])[METRICS].mean().reset_index()
    table=permap.groupby('method')[METRICS].mean().reset_index()
    settings=df.groupby('method')[['content_weight','beta','content_mode']].first().reset_index()
    table=table.merge(settings,on='method',validate='one_to_one')
    b0=table[table.method=='B0'].iloc[0];threshold=float(1.05*b0.gram_style_distance)
    selection={};table['style_eligible']=table.gram_style_distance<=threshold
    for family in ['B1','A1','AT','B2']:
        mask=table.method.str.startswith('A1' if family=='AT' else family)
        if family=='B1':mask=mask | (table.method=='B0')
        if family=='A1':mask=mask & (table.content_weight==.25)
        winner,mismatch=choose_candidate(table[mask].copy(),b0.gram_style_distance)
        selection[family]={'selected_development_method':str(winner.method),'content_mode':str(winner.content_mode),
                           'content_weight':float(winner.content_weight),'beta':float(winner.beta),
                           'style_mismatch_no_eligible_candidate':bool(mismatch),
                           'development_mean_boundary_f1_2':float(winner.boundary_f1_2),
                           'development_mean_gram':float(winner.gram_style_distance)}
    analysis=ROOT/'project/results/development_search/analysis';analysis.mkdir(exist_ok=True)
    permap.to_csv(analysis/'per_map_metrics.csv',index=False);table.to_csv(analysis/'all_candidates.csv',index=False)
    write_json(analysis/'selection.json',{'created_utc':utc_now(),'baseline_gram':float(b0.gram_style_distance),
                'eligibility_threshold_gram':threshold,'selection':selection,'map_count':int(df.map_id.nunique()),'style_count':int(df.style_id.nunique())})
    maps=read_json(ROOT/'project/data/map_manifest_512.json');styles=read_json(ROOT/'project/data/style_manifest_512.json')
    tests=[m for m in maps if m['split']=='test']
    if len(tests)!=12 or len(styles)!=3:raise ValueError('Expected 12 fixed test maps and three styles')
    formal={'status':'FROZEN_BEFORE_TEST_OUTPUTS','common':spec['common'],'model_path':spec['model_path'],
            'output_root':'project/results/formal_test','warmup_steps':4,'stop_on_failure':True,'jobs':[]}
    selected={'B0':{'content_mode':'q','content_weight':.25,'beta':0},**selection}
    for m,s,k in itertools.product(tests,styles,METHODS):
        config={key:selected[k][key] for key in ['content_mode','content_weight','beta']}
        formal['jobs'].append({'run_id':f'{m["map_id"]}_{s["style_id"]}_{k}_200','map_id':m['map_id'],'style_id':s['style_id'],'method':k,**config})
    write_json(ROOT/'configs/formal_test.json',formal)
    records=['configs/development_protocol.json','configs/development_search.json','configs/formal_test.json',
             'project/data/source_windows.json','project/data/source_styles.json','project/data/map_manifest_512.json',
             'project/data/style_manifest_512.json','project/data/render_config_512.json','project/src/evaluate.py',
             'project/src/analyze.py','project/src/geometry_prior.py','project/src/geometry_losses.py','project/src/map_pipeline.py',
             'project/src/run_experiments.py','project/src/common.py','provenance/model_download.json',
             'configs/precision_control.json','project/src/precision_control.py','project/src/analyze_precision.py',
             'project/results/development_search/evaluation/per_image_metrics.csv','project/results/development_search/analysis/selection.json']
    frozen={'status':'FROZEN_BEFORE_ANY_TEST_OUTPUT','frozen_utc':utc_now(),'selection':selection,
            'formal_image_count':len(formal['jobs']),'unit_of_replication':'geographic map window; average the same three fixed styles within each map',
            'separate_precision_control':{'image_count':36,'method':'Q32','config':'configs/precision_control.json','no_tuning':True,
                'purpose':'Uniform spatial weighting through the same FP32 normalized loss path; compare Q32-B0 for precision and A1-Q32 for geometry at lambda0.25'},
            'statistics':{'method':'paired map-level percentile bootstrap','replicates':10000,'confidence':.95,'seed':20260925,
                          'comparisons':['A1 - B0','A1 - B1','A1 - B2','AT - B0','AT - B1','AT - B2'],'primary_metric':'boundary_f1_2','p_values':'not computed',
                          'generalization':'descriptive for a fixed convenience sample; styles are fixed, not independent replicates'},
            'qualitative':{'main_pairs':[['test01','hokusai'],['test03','cezanne'],['test04','vangogh']],
                           'main_crop_xyxy':[192,192,320,320],'all_outputs':'include complete contact sheets',
                           'failure_case_rule':'add the map/style with smallest A1-minus-B1 boundary F1; label as post-hoc diagnostic, not representative'},
            'failure_policy':'Preserve each failed attempt. Retry computational failures under identical settings. Never drop a low-quality successful output.',
            'optional_relative_query_term':'not integrated or empirically evaluated; excluded from the reported method',
            'sha256':{p:sha256(ROOT/p) for p in records}}
    if (ROOT/formal['output_root']).exists() and list((ROOT/formal['output_root']).glob('*/output.png')):raise ValueError('Test outputs already exist')
    write_json(target,frozen)
    print(json.dumps(selection,indent=2));print('Frozen',len(formal['jobs']),'test jobs')

def aggregate():
    frozen=read_json(ROOT/'configs/protocol_frozen.json')
    for p in ['configs/formal_test.json','project/src/evaluate.py','project/src/geometry_prior.py','project/src/geometry_losses.py',
              'project/src/map_pipeline.py','project/src/run_experiments.py','project/src/common.py']:
        if sha256(ROOT/p)!=frozen['sha256'][p]:raise ValueError('Frozen protocol source changed: '+p)
    df,spec=checked_table('project/results/formal_test','configs/formal_test.json')
    if set(df.split)!={'test'}:raise ValueError('Unexpected split')
    counts=df.groupby(['map_id','method']).style_id.nunique()
    if not (counts==3).all() or len(counts)!=12*len(METHODS):raise ValueError('Unbalanced test matrix')
    out=ROOT/'project/results/formal_test/analysis';out.mkdir(exist_ok=True)
    permap=df.groupby(['map_id','region_id','method'])[METRICS].mean().reset_index()
    permap.to_csv(out/'per_map_metrics.csv',index=False)
    n=12;boot=np.random.default_rng(20260925).integers(0,n,size=(10000,n));summary=[];comparisons=[]
    for metric in METRICS:
        pivot=permap.pivot(index='map_id',columns='method',values=metric).sort_index()
        for method in METHODS:
            values=pivot[method].to_numpy();ci=np.quantile(values[boot].mean(1),[.025,.975])
            summary.append({'metric':metric,'method':method,'mean':values.mean(),'map_sd':values.std(ddof=1),
                            'ci_low':ci[0],'ci_high':ci[1],'n_maps':n,'n_styles_per_map':3})
        for proposed,reference in itertools.product(['A1','AT'],['B0','B1','B2']):
            diff=(pivot[proposed]-pivot[reference]).to_numpy();ci=np.quantile(diff[boot].mean(1),[.025,.975])
            comparisons.append({'metric':metric,'comparison':proposed+' - '+reference,'difference_mean':diff.mean(),
                                'ci_low':ci[0],'ci_high':ci[1],'positive_maps':int((diff>0).sum()),'negative_maps':int((diff<0).sum()),'n_maps':n})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    pivot=df.pivot(index=['map_id','style_id'],columns='method',values='boundary_f1_2')
    delta=(pivot.A1-pivot.B1).sort_values(kind='stable');worst=delta.index[0]
    write_json(out/'analysis_manifest.json',{'created_utc':utc_now(),'n_images':len(df),'n_maps':12,'n_styles':3,
                'input_sha256':sha256(ROOT/'project/results/formal_test/evaluation/per_image_metrics.csv'),
                'protocol_sha256':sha256(ROOT/'configs/protocol_frozen.json'),'analysis_script_sha256':sha256(Path(__file__)),
                'worst_pair_A1_minus_B1':{'map_id':worst[0],'style_id':worst[1],'delta':float(delta.iloc[0])},
                'ci_interpretation':'95% percentile intervals over 10,000 resampled geographic maps; the three styles are averaged within each map. No population probability sampling is claimed.'})
    print(pd.DataFrame(summary).query("metric in ['boundary_f1_2','gram_style_distance','lpips_content']").to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select-freeze','aggregate']);a=p.parse_args()
    select_and_freeze() if a.action=='select-freeze' else aggregate()

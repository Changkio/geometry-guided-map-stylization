"""Export measured, provenance-linked LaTeX tables; never manufacture empty results."""
from pathlib import Path
import pandas as pd
from common import ROOT,read_json,write_json,sha256,relative,utc_now

OUT=ROOT/'project/paper/draft/tables'
METHODS=['B0','B1','A1','AT','B2']

def table(name,caption,label,columns,header,rows):
    text='\n'.join([r'\begin{table}[t]',r'\caption{'+caption+'}',r'\label{'+label+r'}\centering\small',
        r'\begin{tabular}{@{}'+columns+r'@{}}',r'\toprule',header+r' \\',r'\midrule',
        *[row+' '+chr(92)*2 for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (OUT/(name+'.tex')).write_text(text,encoding='utf-8')

def main():
    paths={'summary':'project/results/formal_test/analysis/summary.csv',
           'pairs':'project/results/formal_test/analysis/paired_comparisons.csv',
           'images':'project/results/formal_test/evaluation/per_image_metrics.csv',
           'selection':'project/results/development_search/analysis/selection.json',
           'diagnostics':'project/results/reference_diagnostics/per_image_metrics.csv',
           'precision_summary':'project/results/precision_control/analysis/summary.csv',
           'precision_pairs':'project/results/precision_control/analysis/paired_comparisons.csv'}
    for p in paths.values():
        if not (ROOT/p).exists():raise RuntimeError('Required measured source is absent: '+p)
    OUT.mkdir(exist_ok=True)
    summary=pd.read_csv(ROOT/paths['summary']);pairs=pd.read_csv(ROOT/paths['pairs'])
    images=pd.read_csv(ROOT/paths['images']);diagnostics=pd.read_csv(ROOT/paths['diagnostics'])
    selection=read_json(ROOT/paths['selection']);frozen=read_json(ROOT/'configs/protocol_frozen.json')
    if len(images)!=180 or images.run_id.duplicated().any():raise ValueError('Expected 180 distinct formal outputs')
    if not (summary.n_maps==12).all():raise ValueError('Wrong statistical unit')
    def value(method,metric):return float(summary[(summary.method==method)&(summary.metric==metric)].iloc[0]['mean'])
    rows=[]
    for m in METHODS:
        vals=[f'{value(m,k):.4f}' for k in ['boundary_f1_2','lpips_content']]
        vals += [f'{value(m,"gram_style_distance"):.2f}',f'{value(m,"generation_seconds"):.1f}',
                 f'{images.loc[images.method==m,"peak_allocated_vram_bytes"].max()/2**30:.2f}']
        rows.append(' & '.join([m,*vals]))
    table('main_results',r'Test means over 12 maps and three fixed styles. B0/B1 and A1/AT have identical selected settings and byte-identical images; repeated rows are not independent methods. F1 uses two-pixel tolerance. Time is seconds per image; VRAM is the maximum allocated peak over 36 images (GiB).',
          'tab:results','lrrrrr',r'Method & F1 $\uparrow$ & LPIPS $\downarrow$ & Gram $\downarrow$ & Time & VRAM',rows)
    rows=[]
    for comparison in ['A1 - B0','A1 - B1','AT - B1','AT - B2']:
        vals=[]
        for metric,digits in [('boundary_f1_2',4),('gram_style_distance',2)]:
            r=pairs[(pairs.comparison==comparison)&(pairs.metric==metric)].iloc[0]
            vals.append(f'{r.difference_mean:+.{digits}f} [{r.ci_low:+.{digits}f}, {r.ci_high:+.{digits}f}]')
        rows.append(' & '.join([comparison.replace(' - ',r'$-$'),*vals]))
    table('paired_results',r'Paired map-level mean differences and 95\% percentile bootstrap intervals (10,000 resamples; $n=12$ maps). Positive F1 and negative Gram differences favor the first method for the respective proxy. Intervals describe a small convenience sample; styles are kept together within each map.',
          'tab:paired','lrr',r'Comparison & $\Delta$ boundary F1 [95\% CI] & $\Delta$ Gram [95\% CI]',rows)
    rows=['B0 & 0.25 & 0 & Reference']
    for m in ['B1','A1','AT','B2']:
        s=selection['selection'][m]
        beta='---' if m=='B2' else f'{s["beta"]:g}'
        rows.append(' & '.join([m,f'{s["content_weight"]:g}',beta,'No' if s['style_mismatch_no_eligible_candidate'] else 'Yes']))
    table('selected_settings',r'Development-selected settings, frozen before test generation. The coefficient is $\lambda_Q$ except for B2, which uses $\lambda_{QK}$. Eligible means development Gram discrepancy at most 1.05 times B0; ``No'' denotes the predeclared closest-style fallback, not a style-matched comparison.',
          'tab:selected','lrrl',r'Method & Coefficient & $\beta$ & Style eligible',rows)
    refs=diagnostics[diagnostics.split=='test'].groupby(['method','map_id'])[['boundary_f1_2','lpips_content','gram_style_distance']].mean().groupby('method').mean()
    rows=[]
    for m in ['Source','VAE']:
        r=refs.loc[m];rows.append(f'{m} & {r.boundary_f1_2:.4f} & {r.lpips_content:.4f} & {r.gram_style_distance:.2f}')
    table('reference_endpoints',r'Unchanged-source and VAE reconstruction diagnostics on the same test maps and references. These are low-stylization endpoints, not competing style-transfer methods.',
          'tab:reference','lrrr',r'Endpoint & Boundary F1 & LPIPS & Gram discrepancy',rows)
    precision=pd.read_csv(ROOT/paths['precision_summary']);rows=[]
    for m in ['B0','Q32','A1']:
        vals=[float(precision[(precision.method==m)&(precision.metric==k)].iloc[0]['mean']) for k in ['boundary_f1_2','gram_style_distance','generation_seconds']]
        rows.append(f'{m} & {vals[0]:.4f} & {vals[1]:.2f} & {vals[2]:.1f}')
    table('precision_results',r'Numerical-path control on the same twelve test maps and three styles. All rows use $\lambda_Q=0.25$. Q32 uses uniform weights through the normalized FP32 implementation, whereas B0 uses the original reduction. A1 versus Q32 isolates spatial weighting within this common numerical path.',
          'tab:precision','lrrr',r'Method & Boundary F1 & Gram discrepancy & Time (s)',rows)
    write_json(OUT/'table_manifest.json',{'created_utc':utc_now(),'source_script_sha256':sha256(Path(__file__)),
        'sources':{k:{'path':p,'sha256':sha256(ROOT/p)} for k,p in paths.items()},
        'protocol_sha256':sha256(ROOT/'configs/protocol_frozen.json'),
        'tables':{p.name:sha256(p) for p in sorted(OUT.glob('*.tex'))},'rounding':'display only; full precision retained in source CSVs'})
    print('Exported five measured tables')

if __name__=='__main__':main()

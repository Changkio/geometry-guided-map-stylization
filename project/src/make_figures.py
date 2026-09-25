"""Build manuscript figures from retained source inputs and measured tables."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from common import ROOT,read_json,write_json,sha256,relative,utc_now

OUT=ROOT/'project/paper/draft/figures'
STYLE={'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':7.5,'axes.labelsize':7.5,
       'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'pdf.fonttype':42,
       'ps.fonttype':42,'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False}
COLORS={'B0':'#444444','B1':'#0072B2','A1':'#D55E00','AT':'#007E63','B2':'#8A4F91'}
MARKERS={'B0':'o','B1':'s','A1':'^','AT':'D','B2':'v'}
METHODS=['B0','B1','A1','AT','B2']

def export(fig,name,provenance):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/(name+'.pdf'),dpi=400,facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=300,facecolor='white')
    write_json(OUT/(name+'.provenance.json'),{'created_utc':utc_now(),'source_script':relative(Path(__file__)),
        'script_sha256':sha256(Path(__file__)),'size_inches':list(fig.get_size_inches()),'matplotlib':matplotlib.__version__,
        'files':{suffix:sha256(OUT/(name+'.'+suffix)) for suffix in ['pdf','png']},**provenance})
    plt.close(fig)

def box(ax,x,y,w,h,text,color='#E9EEF3',edge='#25364A'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.025',facecolor=color,edgecolor=edge,linewidth=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7.2,color='#25364A')
def arrow(ax,p,q,color='#25364A',style='-'):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=8,color=color,linewidth=.85,linestyle=style))

def source_figures():
    fig,ax=plt.subplots(figsize=(4.8,2.95));fig.subplots_adjust(.01,.015,.99,.99)
    ax.set(xlim=(0,10.5),ylim=(0,6.3));ax.axis('off')
    blue=COLORS['B1'];orange=COLORS['A1'];green=COLORS['AT']
    box(ax,.15,4.85,2.05,.9,'Source map\nand style\nreference','#E8F2F8',blue)
    box(ax,.15,3.2,2.05,.9,'Registered\nroad vectors','#FBEBE4',orange)
    box(ax,3.05,4.85,3,.9,'Frozen diffusion\nfeature extraction')
    box(ax,3.05,3.2,3,.9,'Importance field\nand max projection','#FBEBE4',orange)
    box(ax,6.85,4.85,2.95,.9,'Unchanged style\ndistillation loss','#E8F2F8',blue)
    box(ax,6.85,3.2,2.95,.9,'Normalized geometry-\nweighted query loss','#FBEBE4',orange)
    box(ax,6.85,1.28,2.95,.9,'Combined objective\nupdate latent only','#E7F3EF',green)
    box(ax,3.05,1.28,3,.9,'Current image latent\ninitialized from source','#E7F3EF',green)
    box(ax,.15,1.28,2.05,.9,'Decode final\nimage latent')
    for p,q,c in [((2.2,5.3),(3.05,5.3),blue),((2.2,3.65),(3.05,3.65),orange),
                  ((6.05,5.3),(6.85,5.3),blue),((6.05,3.65),(6.85,3.65),orange),
                  ((6.05,4.95),(6.85,4.0),'#25364A'),((9.45,2.55),(9.45,2.18),blue),
                  ((8.2,3.2),(8.2,2.18),orange),((4.1,4.48),(4.1,4.85),green),((3.05,1.73),(2.2,1.73),green)]:arrow(ax,p,q,c)
    ax.plot([9.82,10.15,10.15,9.45],[5.3,5.3,2.55,2.55],color=blue,lw=.85)
    ax.plot([3.03,2.64,2.64,4.1],[2,2,4.48,4.48],color=green,lw=.85)
    arrow(ax,(6.85,1.73),(6.05,1.73),green,'--')
    ax.text(5.2,.47,'Solid: inputs / features     Dashed: latent update',ha='center',fontsize=6.8)
    export(fig,'overview',{'kind':'implementation schematic','not_an_empirical_result':True,'optional_relation_term':False})
    m=next(x for x in read_json(ROOT/'project/data/map_manifest_512.json') if x['map_id']=='dev02')
    prior=np.load(ROOT/m['prior_path']);content=Image.open(ROOT/m['content_path'])
    pooled=[prior.reshape(n,512//n,n,512//n).max(axis=(1,3)) for n in [32,64]]
    fig,axes=plt.subplots(1,4,figsize=(4.8,1.57));fig.subplots_adjust(.008,.22,.992,.83,.10)
    titles=['(a) Source map','(b) Prior, 512²','(c) Token grid, 32²','(d) Token grid, 64²']
    for ax,title in zip(axes,titles):ax.set_title(title,fontsize=6.8,pad=4);ax.set_xticks([]);ax.set_yticks([])
    axes[0].imshow(content)
    for ax,values in zip(axes[1:],[prior,*pooled]):im=ax.imshow(values,cmap='cividis',vmin=0,vmax=1,interpolation='nearest')
    cax=fig.add_axes([.34,.135,.5,.045]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.5,1]);cb.ax.tick_params(length=2,pad=1,labelsize=6)
    export(fig,'geometry_prior',{'kind':'source-only registered diagnostic','map_id':m['map_id'],'prior_sha256':sha256(ROOT/m['prior_path']),
             'content_sha256':sha256(ROOT/m['content_path']),'projection':'exact non-overlapping max blocks at power-of-two sizes; equal to adaptive max pooling here',
             'color_scale':[0,1],'image_adjustments':'none; displayed at native aspect ratio'})

def result_figures():
    frozen=read_json(ROOT/'configs/protocol_frozen.json')
    dev=pd.read_csv(ROOT/'project/results/development_search/analysis/all_candidates.csv')
    permap=pd.read_csv(ROOT/'project/results/formal_test/analysis/per_map_metrics.csv')
    df=pd.read_csv(ROOT/'project/results/formal_test/evaluation/per_image_metrics.csv')
    summary=pd.read_csv(ROOT/'project/results/formal_test/analysis/summary.csv')
    references=pd.read_csv(ROOT/'project/results/reference_diagnostics/per_image_metrics.csv')
    precision=pd.read_csv(ROOT/'project/results/precision_control/evaluation/per_image_metrics.csv')
    image_lookup=pd.concat([df,precision],ignore_index=True)
    hashes=df.pivot(index=['map_id','style_id'],columns='method',values='output_sha256')
    display_groups=[]
    for method in METHODS:
        match=next((group for group in display_groups if hashes[method].equals(hashes[group[0]])),None)
        if match is None:display_groups.append([method])
        else:match.append(method)
    source={'development_csv_sha256':sha256(ROOT/'project/results/development_search/analysis/all_candidates.csv'),
            'test_csv_sha256':sha256(ROOT/'project/results/formal_test/evaluation/per_image_metrics.csv'),
            'protocol_sha256':sha256(ROOT/'configs/protocol_frozen.json'),'n_maps':12,'n_fixed_styles':3}
    source['reference_csv_sha256']=sha256(ROOT/'project/results/reference_diagnostics/per_image_metrics.csv')
    source['precision_csv_sha256']=sha256(ROOT/'project/results/precision_control/evaluation/per_image_metrics.csv')
    source['byte_identical_display_groups']=display_groups
    fig,axes=plt.subplots(1,2,figsize=(4.8,2.6),layout='constrained')
    for family in ['B0','B1','A1','B2']:
        sub=dev[dev.method.eq('B0') if family=='B0' else dev.method.str.startswith(family)]
        axes[0].scatter(sub.gram_style_distance,sub.boundary_f1_2,c=COLORS[family],marker=MARKERS[family],s=22,label=family if family!='A1' else 'Geometry Q')
    threshold=1.05*float(dev.loc[dev.method=='B0','gram_style_distance'].iloc[0]);axes[0].axvline(threshold,color='#555555',ls=':',lw=.9)
    axes[0].set(title='(a) All development settings',xlabel='Gram discrepancy (log scale) ↓',ylabel='Boundary F1 (2 px) ↑',ylim=(0,1.04),xscale='log')
    axes[0].legend(loc='best',frameon=False)
    for group in display_groups:
        method=group[0]
        vals=permap[permap.method==method]
        axes[1].scatter(vals.gram_style_distance,vals.boundary_f1_2,c=COLORS[method],marker=MARKERS[method],s=10,alpha=.45)
        axes[1].scatter(vals.gram_style_distance.mean(),vals.boundary_f1_2.mean(),c=COLORS[method],marker=MARKERS[method],s=42,edgecolor='white',linewidth=.5,label='/'.join(group))
    for method,marker,color in [('Source','X','#999999'),('VAE','P','#AA7700')]:
        vals=references[(references.split=='test')&(references.method==method)].groupby('map_id')[['gram_style_distance','boundary_f1_2']].mean()
        axes[1].scatter(vals.gram_style_distance.mean(),vals.boundary_f1_2.mean(),c=color,marker=marker,s=36,label=method)
    axes[1].set(title='(b) Held-out map windows',xlabel='Gram discrepancy (log scale) ↓',ylabel='Boundary F1 (2 px) ↑',ylim=(0,1.04),xscale='log')
    axes[1].legend(loc='best',frameon=False,ncol=2,columnspacing=.7,handletextpad=.3)
    export(fig,'structure_style',{**source,'aggregation':'left: mean over four maps after averaging styles; right: 12 map points and larger equal-map means',
                                  'development_threshold':'vertical dotted line =1.05 times B0 development Gram; no connecting lines or hidden candidates',
                                  'reference_endpoints':'unchanged source and same-VAE reconstruction; means only; not competing stylization methods','x_scale':'logarithmic'})
    fig,axes=plt.subplots(1,2,figsize=(4.8,2.35),layout='constrained')
    pivot=permap.pivot(index='map_id',columns='method',values='boundary_f1_2').sort_index()
    for i,comparison in enumerate([('A1','B0'),('A1','B1'),('AT','B1'),('AT','B2')]):
        a,b=comparison;diff=(pivot[a]-pivot[b]).to_numpy()
        axes[0].scatter(np.full(len(diff),i)+np.linspace(-.15,.15,len(diff)),diff,s=12,c=COLORS[a],marker=MARKERS[a],alpha=.7)
        axes[0].plot([i-.24,i+.24],[diff.mean()]*2,color='#222222',lw=1.2)
    axes[0].axhline(0,color='#666666',lw=.75);axes[0].set_xticks(range(4),['A1−B0','A1−B1','AT−B1','AT−B2'],rotation=25)
    axes[0].set(title='(a) Paired map differences',ylabel='Difference in boundary F1')
    for group in display_groups:
        method=group[0]
        vals=[summary[(summary.method==method)&(summary.metric==f'boundary_f1_{t}')]['mean'].iloc[0] for t in [1,2,3]]
        axes[1].plot([1,2,3],vals,color=COLORS[method],marker=MARKERS[method],label='/'.join(group),lw=1,markersize=4)
    axes[1].set(title='(b) Tolerance sensitivity',xlabel='Boundary tolerance (pixels)',ylabel='Mean boundary F1',xticks=[1,2,3],ylim=(0,1.04))
    axes[1].legend(frameon=False,ncol=2,loc='best')
    export(fig,'paired_and_tolerance',{**source,'unit':'one point per map after averaging its styles','paired_horizontal_bars':'map-level means',
             'jitter':'fixed symmetric offsets for visibility; no stochastic jitter','tolerance_values':[1,2,3]})
    maps={m['map_id']:m for m in read_json(ROOT/'project/data/map_manifest_512.json')};styles={s['style_id']:s for s in read_json(ROOT/'project/data/style_manifest_512.json')}
    def panel(pairs,name,crop=None,main=False,method_groups=None):
        groups=method_groups or [[m] for m in METHODS]
        columns=2+len(groups)
        fig,axes=plt.subplots(len(pairs),columns,figsize=(4.8,len(pairs)*(4.8/columns+.08)+.22),squeeze=False)
        fig.subplots_adjust(.055,.035,.995,.88,.03,.13)
        for ri,(mid,sid) in enumerate(pairs):
            paths=[ROOT/maps[mid]['content_path'],ROOT/styles[sid]['image_path']]
            paths += [ROOT/image_lookup[(image_lookup.map_id==mid)&(image_lookup.style_id==sid)&(image_lookup.method==group[0])].iloc[0].output_path for group in groups]
            for ci,p in enumerate(paths):
                im=Image.open(p).convert('RGB')
                if crop and ci!=1:im=im.crop(tuple(crop))
                axes[ri,ci].imshow(im);axes[ri,ci].set_xticks([]);axes[ri,ci].set_yticks([])
                for spine in axes[ri,ci].spines.values():spine.set_visible(False)
                if ri==0:axes[ri,ci].set_title(['Source','Style',*['/'.join(g) for g in groups]][ci],fontsize=6.6,pad=3)
                if ci==0:axes[ri,ci].set_ylabel(mid,fontsize=6.4)
                if main and ci==0:axes[ri,ci].add_patch(Rectangle((192,192),128,128,fill=False,lw=.65,edgecolor='#D55E00'))
        export(fig,name,{**source,'pairs':pairs,'crop_xyxy':crop,'displayed_method_groups':groups,'image_adjustments':'none; common display resizing only; style reference is shown whole even in crop panels',
                         'selection':'pre-frozen main pairs' if main or crop else 'complete matrix or explicitly labeled failure diagnostic'})
    pairs=frozen['qualitative']['main_pairs'];panel(pairs,'qualitative_main',main=True,method_groups=display_groups)
    panel(pairs,'qualitative_crops',crop=frozen['qualitative']['main_crop_xyxy'],method_groups=display_groups)
    worst=read_json(ROOT/'project/results/formal_test/analysis/analysis_manifest.json')['worst_pair_A1_minus_B1']
    panel([[worst['map_id'],worst['style_id']]],'failure_case',method_groups=display_groups)
    for sid in styles:panel([(mid,sid) for mid in sorted(maps) if maps[mid]['split']=='test'],'all_test_'+sid)
    for sid in styles:panel([(mid,sid) for mid in sorted(maps) if maps[mid]['split']=='test'],'all_precision_'+sid,method_groups=[['B0'],['Q32'],['A1']])
    panel(pairs,'precision_crops',crop=frozen['qualitative']['main_crop_xyxy'],method_groups=[['B0'],['Q32'],['A1']])
    write_json(OUT/'result_figures_manifest.json',{'created_utc':utc_now(),'sources':source,'methods':METHODS,
              'native_output_resolution':512,'no_selective_image_adjustment':True,'full_contact_sheets_included':True,
              'byte_identical_display_groups':display_groups,'precision_control_contact_sheets_included':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['source','results','all'],default='all');a=p.parse_args()
    with plt.rc_context(STYLE):
        if a.stage in ['source','all']:source_figures()
        if a.stage in ['results','all']:result_figures()

"""Normalized geometry Q and optional sparse relative Q losses."""
import torch
import torch.nn.functional as F

def projected_prior(prior,grid_hw):
    if prior.ndim==2:prior=prior[None,None]
    if prior.ndim!=4:raise ValueError('Prior requires [B,1,H,W]')
    return F.adaptive_max_pool2d(prior.float(),grid_hw).flatten(2)[:,0,:]

def geometry_q_loss(qs,qcs,grids,prior,beta,uniform_loss_fn):
    if not (len(qs)==len(qcs)==len(grids)):raise ValueError('Mismatched feature grids')
    # Exact mother-method reduction at the defining ablation endpoint, including
    # low-precision reduction behavior. No query-vector normalization is used.
    if beta==0:return uniform_loss_fn(qs,qcs)
    result=qs[0].new_zeros((),dtype=torch.float32)
    for q,qc,hw in zip(qs,qcs,grids):
        if hw[0]*hw[1]!=q.shape[-2]:raise ValueError('Observed grid does not match query tokens')
        g=projected_prior(prior,hw).to(q.device)
        w=1+beta*g
        error=(q.float()-qc.detach().float()).abs().mean(dim=(1,3))
        result=result+((w*error).sum(dim=-1)/w.sum(dim=-1)).mean()
    return result

def deduplicate_pairs(pairs,grid_hw,image_hw,device):
    if not pairs:return torch.empty((0,2),device=device,dtype=torch.long)
    points=torch.as_tensor(pairs,device=device,dtype=torch.float32).reshape(-1,2,2)
    gh,gw=grid_hw;ih,iw=image_hw
    xs=(points[:,:,0]*gw/iw).floor().long().clamp(0,gw-1)
    ys=(points[:,:,1]*gh/ih).floor().long().clamp(0,gh-1)
    ids=(ys*gw+xs).sort(dim=-1).values
    ids=ids[ids[:,0]!=ids[:,1]]
    return torch.unique(ids,dim=0)

def relative_q_loss(qs,qcs,grids,pair_groups,image_hw):
    total=qs[0].sum()*0
    for q,qc,hw in zip(qs,qcs,grids):
        groups=[]
        for pairs in pair_groups:
            indices=deduplicate_pairs(pairs,hw,image_hw,q.device)
            if indices.numel():
                a,b=indices.unbind(-1)
                delta=q[:,:,a,:].float()-q[:,:,b,:].float()
                reference=qc[:,:,a,:].detach().float()-qc[:,:,b,:].detach().float()
                groups.append((delta-reference).abs().mean())
        if groups:total=total+torch.stack(groups).mean()
    return total

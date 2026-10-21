import numpy as np
import torch
from .backbone import CUTOFF,RBF_COUNT,rbf_features,cosine_cutoff
def collate_shells(records,scaler,device='cpu'):
    if not records:raise ValueError('Empty shell batch')
    n=max(len(r['z']) for r in records);b=len(records)
    z=np.zeros((b,n),np.int64);pos=np.zeros((b,n,3),np.float32)
    role=np.zeros((b,n),np.int64);copies=np.zeros((b,n),np.int64)
    species=np.full((b,n),-1,np.int64);core=np.zeros((b,n),bool);mask=np.zeros((b,n),bool)
    for i,r in enumerate(records):
        k=len(r['z']);mask[i,:k]=True
        z[i,:k]=r['z'];pos[i,:k]=r['pos'];role[i,:k]=r['role']
        copies[i,:k]=r['copies'];species[i,:k]=r['component_ids'];core[i,:k]=r['core']
    desc=scaler.transform(np.stack([r['desc_raw'] for r in records]))
    return {key:torch.as_tensor(value,device=device) for key,value in
        dict(z=z,pos=pos,role=role,mask=mask,desc=desc,copies=copies,core=core,component_ids=species).items()}
def legacy_encode(model,batch,block_unlike=False):
    z,pos,role,mask=batch['z'],batch['pos'],batch['role'],batch['mask']
    rel=pos[:,:,None,:]-pos[:,None,:,:];dist=torch.linalg.norm(rel+1e-12,dim=-1)
    direction=rel/dist[...,None].clamp(min=1e-6)
    valid=mask[:,:,None]&mask[:,None,:]&(dist<CUTOFF)&(dist>1e-3)
    if block_unlike:raise ValueError('Only the final original encoder path is released')
    ri,rj=role[:,:,None],role[:,None,:]
    edge=torch.where((ri==0)&(rj==0),0,torch.where((ri==1)&(rj==1),1,2)).long()
    h=model.atom_emb(z)+model.role_emb(role);v=torch.zeros((*h.shape,3),device=h.device,dtype=h.dtype)
    rbf=rbf_features(dist,cutoff=CUTOFF,num_rbf=RBF_COUNT);cut=cosine_cutoff(dist,cutoff=CUTOFF)
    for msg,upd,norm in zip(model.messages,model.updates,model.norms):
        dh,dv=msg(h,v,direction,rbf,edge,valid,cut);h=h+dh;v=v+dv
        dh,dv=upd(h,v);h=norm(h+dh);v=v+dv
    return h,v

def readout_features(model,h,v,batch,core_pool=True):
    role,mask,copies,core=batch['role'],batch['mask'],batch['copies'],batch['core']
    def mean(values,m):return (values*m[...,None]).sum(1)/m.sum(1).clamp(min=1)[...,None]
    s=mask&(role==0);heavy=s&(batch['z']>1);hs=mean(h,s)
    solvent=torch.stack([mean(h,mask&(copies==k)) for k in (1,2,3)]).mean(0)
    vn=torch.linalg.vector_norm(v,dim=-1);vv=torch.stack([mean(vn,mask&(copies==k)) for k in (1,2,3)]).mean(0)
    size=(h*heavy[...,None]).sum(1)/heavy.sum(1).clamp(min=1).sqrt()[...,None]
    hc=mean(h,heavy&core) if core_pool else torch.zeros_like(hs)
    side=mean(h,heavy&~core) if core_pool else torch.zeros_like(hs)
    f=torch.cat([hs,size,hc,side,solvent,hs*solvent,(hs-solvent).abs(),mean(vn,s),vv],-1)
    return torch.cat([f,model.head.desc_net(batch['desc'])],-1)

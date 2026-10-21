import numpy as np
from rdkit import Chem
def can(s):return Chem.MolToSmiles(Chem.MolFromSmiles(s))

def basis(x):
 x=np.asarray(x);return np.stack([x*(1-x)*(2*x-1)**j for j in range(3)],axis=-1)

def pure_feature(f):return np.concatenate([(f[...,:896]+f[...,896:1792])/2,(f[...,3584:4480]+f[...,4480:5376])/2],axis=-1)

def distances(P,centroids,sd):
 a=P/sd;b=centroids/sd;square=np.maximum(np.sum(a*a,axis=1)[:,None]-2*a@b.T+np.sum(b*b,axis=1)[None,:],0);return np.sqrt(square/a.shape[1]).min(axis=1)

def guard_and_scale(raw,s13,s29,features,meta,gates,pairs):
 x=np.array(meta['x_A']);unc=np.abs(s13-s29)[(x>0)&(x<1)].mean();dist=float(distances(pure_feature(features[None,:]),np.asarray(gates['centroids']),np.asarray(gates['centroid_sd']))[0]);outside=(gates['seed_threshold'] is not None and unc>gates['seed_threshold']) or (gates['distance_threshold'] is not None and dist>gates['distance_threshold'])
 pred=raw.copy();fallback='none'
 if outside:
  a,b=can(meta['A']),can(meta['B']);reverse=a>b
  if reverse:a,b=b,a
  key=a+'||'+b
  if gates['fallback']=='pair_mean_if_pair_seen_else_zero' and key in pairs:
   co=np.array(pairs[key]);co=co*np.array([1,-1,1]) if reverse else co;pred=basis(x)@co;fallback='pair_mean'
  else:pred=np.zeros(len(x));fallback='zero'
 return pred*gates['scale'],dict(seed_disagreement=float(unc),feature_distance=dist,outside=bool(outside),fallback=fallback)

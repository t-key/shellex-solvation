"""Frozen paper estimands. All energies kcal/mol, equal solute weighting."""
import numpy as np
import pandas as pd
DISPLAY={'shellex':'ShellEx','shellex_no_fallback':'ShellEx (no fallback)','zero_excess':'Zero excess','pair_mean_train10640':'Pair mean (train10640; unseen=zero)','pair_mean_train9800':'Pair mean (train9800; unseen=global)','fingerprint_2d':'2D (frozen train9800)','UNIFAC_original':'UNIFAC original','direct_COSMO':'Direct COSMO','shellex_narrow':'ShellEx-narrow','shellex_narrow_motifs':'ShellEx-narrow + motifs'}
def ci(a):
    a=np.asarray(a);n=len(a)
    if n<2:return np.nan,np.nan
    draws=np.random.default_rng(20261014).integers(0,n,(1000,n))
    return tuple(np.quantile(a[draws].mean(1),[.025,.975]))
def summary(d,subset,extra=None):
    rows=[]
    for m,g in d.groupby('method',sort=True):
        gm=g.groupby('solute',sort=True).MAE.mean();lo,hi=ci(gm)
        rows.append(dict(subset=subset,method=DISPLAY.get(m,m),method_key=m,curves=g.curve_id.nunique(),solutes=len(gm),solute_balanced_MAE=gm.mean(),curve_mean_MAE=g.MAE.mean(),CI95_low=lo,CI95_high=hi,unit='kcal/mol',bootstrap_unit='solute',bootstrap_replicates=1000,bootstrap_seed=20261014,**(extra or {})))
    return rows

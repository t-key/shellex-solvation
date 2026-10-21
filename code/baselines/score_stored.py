"""UNIFAC/direct-COSMO and other fixed vector scoring; no chemistry rerun."""
import numpy as np
def curve_mae(x_A,reference,predicted):
    x=np.asarray(x_A);y=np.asarray(reference);p=np.asarray(predicted)
    if x.shape!=y.shape or y.shape!=p.shape or not np.isfinite(p).all():raise ValueError('Curve shapes/finiteness')
    inside=(x>1e-10)&(x<1-1e-10)
    if not inside.any():raise ValueError('No interior composition points')
    return float(np.abs(p-y)[inside].mean())
def applicable_mae(record,x_A,target):
    if record['state']!='applicable':return None
    return curve_mae(x_A,target,record['predicted_excess'])

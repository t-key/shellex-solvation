"""Frozen 2D train9800 fingerprint/ridge baseline; never refits."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from shellex.gates import basis
def predict(bundle,solute,a,b,x):
    solute,a,b=[Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in [solute,a,b]];reverse=a>b
    if reverse:a,b=b,a
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=256)
    fs,fa,fb=[gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(float) for s in [solute,a,b]]
    raw=np.r_[fs,fa,fb,fa*fb,fs*(fa+fb)/2];z=np.r_[(raw-bundle['scaler'].mean_)/bundle['scaler'].scale_,1.]
    co=bundle['model'].coef_.reshape(3,len(z))@z
    if reverse:co*=np.array([1,-1,1])
    return basis(np.asarray(x))@co

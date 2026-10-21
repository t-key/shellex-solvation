"""Final zero-excess and training-only pair coefficient baselines."""
import numpy as np
from rdkit import Chem
from shellex.gates import basis
def canonical(s):return Chem.MolToSmiles(Chem.MolFromSmiles(s))
def zero_excess(x):return np.zeros_like(np.asarray(x,dtype=float))
def pair_mean(x,a,b,coefficients):
    a,b=canonical(a),canonical(b);co=np.array(coefficients.get(min(a,b)+'||'+max(a,b),[0.,0.,0.]))
    if a>b:co=co*np.array([1,-1,1])
    return basis(np.asarray(x))@co

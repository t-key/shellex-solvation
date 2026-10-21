"""Opt-in feature cache builder for training reproduction; not run in assembly."""
from pathlib import Path
import sys,argparse
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new cache path')
    import numpy as np,pandas as pd
    from shellex.predictor import ShellEx
    d=pd.read_csv(ROOT/'data/labels/calculated_curves.csv');d=d[d.partition.isin(['train','validation'])].sort_values('curve_id');model=ShellEx(device=a.device);ids=[];features=[]
    for r in d.itertuples():
        f,_=model.feature(r.solute,r.solvent_A,r.solvent_B);ids.append(r.curve_id);features.append(f.numpy())
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,curve_ids=np.array(ids),features=np.stack(features))
if __name__=='__main__':main()

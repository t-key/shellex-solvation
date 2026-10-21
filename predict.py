"""Predict a new neutral solute and binary-solvent excess curve."""
import argparse,json,sys,os
from pathlib import Path
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--solute',required=True);p.add_argument('--solvent_a',required=True);p.add_argument('--solvent_b',required=True);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--config',default=str(ROOT/'config.yaml'));p.add_argument('--output',type=Path)
    args=p.parse_args();from shellex.predictor import ShellEx
    result=ShellEx(args.config,args.device).predict(args.solute,args.solvent_a,args.solvent_b);text=json.dumps(result,indent=2,allow_nan=False)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    else:print(text)
if __name__=='__main__':main()

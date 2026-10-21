"""Opt-in final unweighted ShellEx head training, never runs on import.

Retains final two-seed architecture/hyperparameters and validation-only early
stopping. Gate thresholds and paper deployment remain frozen and are not reset.
Training is not part of reproduce_tables.py and was not run for release assembly.
"""
from pathlib import Path
import sys,json,argparse
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import numpy as np,pandas as pd,torch
from rdkit import Chem
from shellex.head import Head
from shellex.gates import basis
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--features',type=Path,required=True,help='NPZ with curve_ids and features[n,7168] in canonical solvent ordering');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Refusing overwrite; choose a new training-output folder')
    args.output.mkdir(parents=True)
    d=pd.read_csv(ROOT/'data/labels/calculated_curves.csv');index=d.set_index('curve_id');order=json.loads((ROOT/'data/splits/training_order.json').read_text());val=sorted(d[d.partition=='validation'].curve_id)
    assert len(order)==10640 and len(val)==3120
    z=np.load(args.features,allow_pickle=False);f={cid:torch.tensor(x,dtype=torch.float32) for cid,x in zip(z['curve_ids'].tolist(),z['features'])}
    def data(ids):
        X=torch.stack([f[c] for c in ids]);B=[];Y=[];solutes=[]
        for cid in ids:
            r=index.loc[cid];a,b=[Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in [r.solvent_A,r.solvent_B]];x=np.array(json.loads(r.x_A));x=1-x if a>b else x
            B.append(basis(x[1:-1]));Y.append(json.loads(r.excess_kcal_mol)[1:-1]);solutes.append(Chem.MolToSmiles(Chem.MolFromSmiles(r.solute)))
        return X,torch.tensor(np.array(B),dtype=torch.float32),np.array(Y),solutes
    X,bt,Y,_=data(order);XV,bv,YV,ss=data(val);mu=X.mean(0);sd=X.std(0,unbiased=False);sd[sd<1e-4]=1
    xt=((X-mu)/sd).to(args.device);xv=((XV-mu)/sd).to(args.device);bt=bt.to(args.device);bv=bv.to(args.device);yt=torch.tensor(Y,dtype=torch.float32,device=args.device);yv=torch.tensor(YV,dtype=torch.float32,device=args.device)
    names=sorted(set(ss));vw=torch.tensor([1/(len(names)*ss.count(s)) for s in ss],dtype=torch.float32,device=args.device);scale=max(.05,float(Y.std()))
    for seed in [13,29]:
        torch.manual_seed(seed);np.random.seed(seed);model=Head().to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);best=None;bad=0;history=[];folder=args.output/f'seed{seed}';folder.mkdir()
        for epoch in range(400):
            model.train();opt.zero_grad(set_to_none=True);pred=torch.einsum('nkj,nj->nk',bt,model(xt));loss=((pred-yt)/scale).square().mean(1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            if epoch%10==0 or epoch==399:
                model.eval()
                with torch.no_grad():p=torch.einsum('nkj,nj->nk',bv,model(xv));score=float(((p-yv).abs().mean(1)*vw).sum())
                history.append(dict(epoch=epoch,loss=float(loss.detach()),solute_balanced_validation_MAE=score))
                if best is None or score<best-1e-12:
                    best=score;bad=0;torch.save(dict(state=model.state_dict(),feature_mean=mu,feature_sd=sd,seed=seed,epoch=epoch,validation_MAE=score),folder/'best.pt')
                else:bad+=1
                (folder/'history.json').write_text(json.dumps(history,indent=2))
                if bad>=8:break
    print('Training finished. This does not replace the frozen paper models or gates.')
if __name__=='__main__':main()

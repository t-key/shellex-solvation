"""Final ShellEx inference. No training data, QM surfaces or original paths."""
from pathlib import Path
import json
import numpy as np
import torch
from .geometry import FeatureEngine,TrainingDescriptorScaler,config_weights
from .backbone import FieldModel
from .head import Head
from .features import collate_shells,legacy_encode,readout_features
from .gates import basis,guard_and_scale,can

class ShellEx:
    def __init__(self,config=None,device='auto'):
        self.root=Path(__file__).resolve().parents[2]
        self.config_path=Path(config).resolve() if config else self.root/'config.yaml'
        # JSON is a strict subset of YAML; this avoids an extra YAML dependency.
        self.cfg=json.loads(self.config_path.read_text(encoding='utf-8-sig'))
        self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        torch.set_num_threads(self.cfg['cpu_threads'])
        self.enc=[];self.heads=[]
        for name in self.cfg['backbones']:
            path=self.root/name
            if not path.exists():raise FileNotFoundError(f'Inherited backbone missing: {path}. See models/LICENSE-WEIGHTS.md and install the complete public release.')
            cp=torch.load(path,map_location='cpu',weights_only=False)
            m=FieldModel(True).to(self.device).eval();m.load_state_dict(cp['state'],strict=True);m.requires_grad_(False)
            sc=TrainingDescriptorScaler();sc.mean=cp['descriptor_mean'];sc.scale=cp['descriptor_scale'];self.enc.append((m,sc))
        for name in self.cfg['heads']:
            cp=torch.load(self.root/name,map_location='cpu',weights_only=False);h=Head().to(self.device).eval();h.load_state_dict(cp['state']);h.requires_grad_(False)
            self.heads.append((h,cp['feature_mean'].to(self.device),cp['feature_sd'].to(self.device)))
        self.rules=json.loads((self.root/self.cfg['gates']).read_text());self.pairs=json.loads((self.root/self.cfg['pair_coefficients']).read_text())
        self.engine=FeatureEngine(tuple(range(self.cfg['poses'])),cache=self.root/self.cfg['cache'])
        # Known release identities retain the exact original cached ETKDG/MMFF
        # coordinates. Regenerating flexible molecules in another RDKit version
        # can select a different conformer despite the same seed.
        frozen=self.root/'data/molecules/frozen_conformers.json'
        if frozen.exists():
            import hashlib
            self.engine.provenance['frozen_conformers_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest()
            for smiles,g in json.loads(frozen.read_text()).items():
                self.engine.geometry[smiles]=(np.asarray(g['z'],np.int64),np.asarray(g['pos'],np.float32));self.engine.conformer_QC[smiles]=g['QC']

    def feature(self,solute,a,b):
        from rdkit import rdBase
        for s in map(can,[solute,a,b]):
            if s not in self.engine.geometry and rdBase.rdkitVersion!='2026.03.6':
                raise RuntimeError('New-molecule reproducibility requires RDKit2026.03.6; existing release structures use frozen conformers. Install environment/requirements.txt.')
        q,reverse=self.engine.query(solute,a,b);combined=[]
        free=torch.cuda.mem_get_info()[0]/1024**3 if self.device.startswith('cuda') else 0
        size=min(16 if free>5 else 2,max(1,1800//max(len(r['z']) for r in q['records']))) if self.device.startswith('cuda') else 2
        with torch.inference_mode():
            for model,sc in self.enc:
                fs=[];offset=0
                while offset<len(q['records']):
                    try:
                        batch=collate_shells(q['records'][offset:offset+size],sc,self.device)
                        h,v=legacy_encode(model,batch);f=readout_features(model,h,v,batch,True);fs.append(f.cpu());offset+=size
                        del batch,h,v,f
                    except torch.cuda.OutOfMemoryError:
                        if size==1:raise
                        size=max(1,size//2);torch.cuda.empty_cache()
                f=torch.cat(fs);ia=[i for i,r in enumerate(q['records']) if r['pose']<4 and r['config']==(0,0,0)];ib=ia if q['A']==q['B'] else [i for i,r in enumerate(q['records']) if r['pose']<4 and r['config']==(1,1,1)]
                w=torch.tensor(config_weights(q['records'],.5),dtype=f.dtype);mean=w@f;var=(w@f.square()-mean.square()).clamp(min=0)
                combined.append(torch.cat([f[ia].mean(0),f[ib].mean(0),mean,var]))
        feature=torch.cat(combined);assert feature.shape==(7168,) and torch.isfinite(feature).all()
        return feature,reverse

    def predict(self,solute,a,b):
        if can(a)==can(b):
            raise ValueError('ShellEx requires two distinct solvents; identical solvents have zero mixing excess.')
        feature,reverse=self.feature(solute,a,b);x=np.array(self.cfg['x_A']);f=feature.to(self.device)
        with torch.inference_mode():co=np.array([h((f-mu)/sd).cpu().numpy() for h,mu,sd in self.heads])
        if reverse:co=co*np.array([1,-1,1])
        s13=basis(x)@co[0];s29=basis(x)@co[1];raw=(s13+s29)/2
        deployed,flags=guard_and_scale(raw,s13,s29,feature.numpy(),dict(A=a,B=b,x_A=x.tolist()),self.rules,self.pairs)
        coeff=co.mean(0)
        if flags['fallback']=='zero':coeff=np.zeros(3)
        elif flags['fallback']=='pair_mean':
            ac,bc=can(a),can(b);coeff=np.array(self.pairs[min(ac,bc)+'||'+max(ac,bc)])*np.array([1,-1 if ac>bc else 1,1])
        coeff=coeff*self.rules['scale'];assert np.max(abs(basis(x)@coeff-deployed))<1e-6
        return dict(model='ShellEx',solute=can(solute),solvent_A=can(a),solvent_B=can(b),temperature_K=self.cfg['temperature_K'],x_A=x.tolist(),coefficients_c0_c1_c2=coeff.tolist(),excess_kcal_mol=deployed.tolist(),no_fallback_coefficients=co.mean(0).tolist(),no_fallback_excess_kcal_mol=raw.tolist(),seed_coefficients=co.tolist(),**flags,scope='mixing excess only; absolute solubility requires pure-solvent endpoints',device=self.device)

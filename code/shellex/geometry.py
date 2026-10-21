from pathlib import Path
import json,hashlib,pickle,itertools
import numpy as np
from rdkit import Chem,rdBase
from rdkit.Chem import AllChem,Crippen,Descriptors,Lipinski,rdMolDescriptors
HERE=Path(__file__).resolve().parent
def stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % (2**31 - 1)

def rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=4)
    a /= np.linalg.norm(a)
    w, x, y, z = a
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=np.float32)
def descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    return np.array([Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
                     Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
                     Lipinski.RingCount(mol), rdMolDescriptors.CalcFractionCSP3(mol), Chem.GetFormalCharge(mol)], np.float32)

def pair_desc(solute, solvent):
    a, b = descriptors(solute), descriptors(solvent)
    return np.concatenate([a[:7], b[:7], np.abs(a[:2] - b[:2])]).astype(np.float32)
def make_pose(solute, solvent, cache, copies, contact=2.6, pose_seed=0):
    """Exactly K rigid copies. K=1 is a prefix of K=3; no equilibrium claim."""
    za, pa = cache[solute]; zb, pb = cache[solvent]
    zs, ps, roles = [za], [pa], [np.zeros(len(za), dtype=np.int64)]
    all_pos = pa.copy()
    rng = np.random.default_rng(stable_seed(f'{solute}|{solvent}|{pose_seed}'))
    orthogonal = rotation(stable_seed(f'{solute}|{solvent}|axes|{pose_seed}'))
    for k in range(copies):
        direction = orthogonal[:, k % 3] * (-1 if k >= 3 else 1)
        rotated = pb @ rotation(int(rng.integers(0, 2**30))).T
        # At this lower bound, every displacement has a nonnegative axial component,
        # making distance monotonic along the translation ray.
        lo = float((all_pos @ direction).max() - (rotated @ direction).min())
        hi = lo + contact + 1.0
        for _ in range(24):
            mid = (lo + hi) / 2
            trial = rotated + direction * mid
            minimum = np.linalg.norm(trial[:, None] - all_pos[None], axis=-1).min()
            if minimum < contact: lo = mid
            else: hi = mid
        placed = (rotated + direction * hi).astype(np.float32)
        zs.append(zb); ps.append(placed); roles.append(np.ones(len(zb), dtype=np.int64))
        all_pos = np.vstack([all_pos, placed])
    return np.concatenate(zs), np.vstack(ps).astype(np.float32), np.concatenate(roles)
def mixed_pose(solute,components,cache,pose_seed):
    za,pa=cache[solute]
    if len(set(components))==1:
        z,p,r=make_pose(solute,components[0],cache,3,pose_seed=pose_seed)
        lengths=[len(za)]+[len(cache[components[0]][0])]*3
        ids=np.concatenate([np.full(n,k) for k,n in enumerate(lengths)])
        return z,p,r,ids
    key=f'{solute}|mixed|'+ '|'.join(components)+f'|{pose_seed}'
    rng=np.random.default_rng(stable_seed(key)); axes=rotation(stable_seed(key+'|axes'))
    zs=[za];ps=[pa];roles=[np.zeros(len(za),dtype=np.int64)];copies=[np.zeros(len(za),dtype=np.int64)]
    all_pos=pa.copy()
    for k,s in enumerate(components):
        zb,pb=cache[s]; direction=axes[:,k]
        rotated=pb@rotation(int(rng.integers(0,2**30))).T
        lo=float((all_pos@direction).max()-(rotated@direction).min());hi=lo+3.6
        for _ in range(24):
            mid=(lo+hi)/2;trial=rotated+direction*mid
            if np.linalg.norm(trial[:,None]-all_pos[None],axis=-1).min()<2.6:lo=mid
            else:hi=mid
        placed=(rotated+direction*hi).astype(np.float32)
        zs.append(zb);ps.append(placed);roles.append(np.ones(len(zb),dtype=np.int64));copies.append(np.full(len(zb),k+1,dtype=np.int64))
        all_pos=np.vstack([all_pos,placed])
    return np.concatenate(zs),np.vstack(ps).astype(np.float32),np.concatenate(roles),np.concatenate(copies)
NAMES=['MW_solute','LogP_solute','TPSA_solute','HDon_solute','HAcc_solute','RotB_solute','Rings_solute','MW_solv','LogP_solv','TPSA_solv','HDon_solv','HAcc_solv','RotB_solv','Rings_solv','Delta_MW','Delta_LogP','largest_conjugated_component_atoms','heteroaromatic','nonaromatic_carbon_fraction']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(s):
 m=Chem.MolFromSmiles(s)
 if m is None:raise ValueError('Invalid SMILES')
 if Chem.GetFormalCharge(m)!=0:raise ValueError('ShellEx neutral-solute/solvent scope')
 return Chem.MolToSmiles(m,isomericSmiles=True)

def core_and_extras(m):
 idx={a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()};adj={}
 for bond in m.GetBonds():
  if bond.GetIsConjugated() or bond.GetIsAromatic():
   a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();idx.update([a,b]);adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
 remaining=set(adj);sizes=[]
 while remaining:
  stack=[remaining.pop()];n=0
  while stack:
   a=stack.pop();n+=1
   for b in adj[a]:
    if b in remaining:remaining.remove(b);stack.append(b)
  sizes.append(n)
 ext=np.array([max(sizes,default=0),any(a.GetIsAromatic() and a.GetAtomicNum()!=6 for a in m.GetAtoms()),sum(a.GetAtomicNum()==6 and not a.GetIsAromatic() for a in m.GetAtoms())/max(m.GetNumHeavyAtoms(),1)],dtype=np.float32)
 return np.array([i in idx for i in range(m.GetNumAtoms())],bool),ext

def bond_graph(s):
 m=Chem.MolFromSmiles(s);core,_=core_and_extras(m)
 nodes=[]
 for a in m.GetAtoms():
  nodes.append([a.GetAtomicNum()/100,a.GetDegree()/4,a.GetTotalNumHs()/4,a.GetFormalCharge()/3,a.GetIsAromatic(),a.IsInRing(),a.GetMass()/250,a.GetHybridization()==Chem.HybridizationType.SP,a.GetHybridization()==Chem.HybridizationType.SP2,a.GetHybridization()==Chem.HybridizationType.SP3,a.GetIsotope()>0,a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED,a.GetAtomicNum() in (7,8,9,15,16,17,35,53)])
 edges=[];attrs=[]
 for b in m.GetBonds():
  typ=b.GetBondType();st=b.GetStereo()
  f=[typ==Chem.BondType.SINGLE,typ==Chem.BondType.DOUBLE,typ==Chem.BondType.TRIPLE,typ==Chem.BondType.AROMATIC,b.GetIsConjugated(),b.IsInRing(),st==Chem.BondStereo.STEREOZ,st==Chem.BondStereo.STEREOE]
  for i,j in [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),(b.GetEndAtomIdx(),b.GetBeginAtomIdx())]:edges.append((i,j));attrs.append(f)
 return dict(nodes=np.asarray(nodes,np.float32),edge_index=np.asarray(edges,np.int64).reshape(-1,2).T,edge_attr=np.asarray(attrs,np.float32).reshape(-1,8),core=core)

def config_weights(records,x):
 if not 0<=float(x)<=1:raise ValueError('Mole fraction outside [0,1]')
 poses=len({r['pose'] for r in records})
 if not poses:raise ValueError('No geometric poses')
 if all(r['config']==(0,0,0) for r in records):return np.ones(len(records))/len(records)
 w=np.array([x**r['config'].count(0)*(1-x)**r['config'].count(1)/poses for r in records],float)
 if abs(w.sum()-1)>1e-10:raise ValueError('Configuration probability normalization')
 return w

class FeatureEngine:
 def __init__(self,poses=tuple(range(16)),cache=None):
  self.poses=tuple(poses)
  if len(set(self.poses))!=len(self.poses) or not self.poses:raise ValueError('Distinct nonempty pose IDs required')
  self.cache=Path(cache or HERE.parents[1]/'results/prediction_cache');self.cache.mkdir(parents=True,exist_ok=True);self.geometry={};self.conformer_QC={}
  self.provenance={'engine_sha256':sha(__file__),'rdkit_version':rdBase.rdkitVersion,'descriptors':NAMES,'geometry':'three rigid contact copies; minimum2.6A; ETKDG/MMFF','targets_read':False,'transform_fit':False}
 def ensure_geometry(self,s):
  if s in self.geometry:return
  key=hashlib.sha256(json.dumps({'smiles':s,'source':self.provenance},sort_keys=True).encode()).hexdigest();path=self.cache/('molecule_'+key+'.pkl')
  if path.exists():
   with path.open('rb') as f:item=pickle.load(f)
   if item['smiles']!=s or item['provenance']!=self.provenance:raise ValueError('Molecule cache provenance mismatch')
  else:
   m=Chem.AddHs(Chem.MolFromSmiles(s));p=AllChem.ETKDGv3();p.randomSeed=stable_seed(s);p.useRandomCoords=True
   if AllChem.EmbedMolecule(m,p)!=0:raise ValueError('Geometry embedding failed '+s)
   if not AllChem.MMFFHasAllMoleculeParams(m):raise ValueError('Missing MMFF parameters '+s)
   status=int(AllChem.MMFFOptimizeMolecule(m,maxIters=1000))
   if status!=0:raise ValueError('MMFF did not converge '+s+' flag '+str(status))
   z=np.array([a.GetAtomicNum() for a in m.GetAtoms()],np.int64);pos=m.GetConformer().GetPositions().astype(np.float32);pos-=pos.mean(0)
   item=dict(smiles=s,z=z,pos=pos,provenance=self.provenance,QC={'atom_count':len(z),'embedding_seed':stable_seed(s),'MMFF_status':status,'geometry_method':'single ETKDG/MMFF conformer, separate from COSMO QM labels'})
   if not np.isfinite(pos).all():raise ValueError('Nonfinite coordinates')
   with path.with_suffix('.tmp').open('wb') as f:pickle.dump(item,f)
   path.with_suffix('.tmp').replace(path)
  expected=np.array([a.GetAtomicNum() for a in Chem.AddHs(Chem.MolFromSmiles(s)).GetAtoms()])
  if not np.array_equal(item['z'],expected):raise ValueError('Conformer atom mapping mismatch')
  self.geometry[s]=(item['z'],item['pos']);self.conformer_QC[s]=item['QC']
 def query(self,solute,a,b):
  solute,a,b=map(canonical,[solute,a,b]);reverse=a>b
  if reverse:a,b=b,a
  key=json.dumps(dict(solute=solute,A=a,B=b,poses=self.poses,provenance=self.provenance),sort_keys=True);path=self.cache/('shell_'+hashlib.sha256(key.encode()).hexdigest()+'.pkl')
  if path.exists():
   with path.open('rb') as f:data=pickle.load(f)
   if data['key']!=key:raise ValueError('Query cache mismatch')
   return data,reverse
  for s in [solute,a,b]:self.ensure_geometry(s)
  mh=Chem.AddHs(Chem.MolFromSmiles(solute));core,ext=core_and_extras(mh)
  raw=np.stack([np.concatenate([pair_desc(solute,s),ext]) for s in [a,b]]).astype(np.float32)
  if raw.shape!=(2,19) or not np.isfinite(raw).all():raise ValueError('Descriptor contract failed')
  records=[];configs=[(0,0,0)] if a==b else list(itertools.product(range(2),repeat=3))
  for pose in self.poses:
   for cfg in configs:
    components=[[a,b][k] for k in cfg];z,pos,role,copies=mixed_pose(solute,components,self.geometry,pose)
    species=np.array([-1]+list(cfg),np.int64)[copies]
    expected_z=np.concatenate([self.geometry[s][0] for s in [solute]+components])
    if not np.array_equal(z,expected_z) or not np.array_equal(role,(copies>0).astype(np.int64)):raise ValueError('Copy-to-atom mapping failed')
    distance=np.linalg.norm(pos[:,None,:]-pos[None,:,:],axis=-1);different=copies[:,None]!=copies[None,:];minimum=float(distance[different].min())
    if minimum<2.599 or not np.isfinite(pos).all():raise ValueError('Invalid inter-molecular contact')
    unlike=different&(copies[:,None]>0)&(copies[None,:]>0)&(species[:,None]!=species[None,:])
    if a==b:unlike[:]=False
    records.append(dict(z=z,pos=pos,role=role,copies=copies,core=np.r_[core,np.zeros(len(z)-len(core),bool)],component_ids=species,desc_raw=raw[list(cfg)].mean(0),config=cfg,pose=pose,QC={'minimum_contact_A':minimum,'unlike_directed_edges_within_6A':int((unlike&(distance<6)&(distance>1e-3)).sum()),'n_atoms':len(z)}))
  data=dict(key=key,solute=solute,A=a,B=b,raw_pair_descriptors=raw,graphs=[bond_graph(s) for s in [solute,a,b]],records=records,provenance=self.provenance,conformer_QC={s:self.conformer_QC[s] for s in {solute,a,b}})
  with path.with_suffix('.tmp').open('wb') as f:pickle.dump(data,f)
  path.with_suffix('.tmp').replace(path)
  return data,reverse

class TrainingDescriptorScaler:
 def fit(self,raw,partition,training_ids):
  if partition!='train' or not training_ids:raise ValueError('Explicit nonempty training partition/IDs required')
  a=np.asarray(raw,np.float32)
  if a.ndim!=2 or a.shape[1]!=19 or len(a)!=len(training_ids) or not np.isfinite(a).all():raise ValueError('Training scaler input failed')
  self.mean=a.mean(0);self.scale=a.std(0);self.scale[self.scale<1e-6]=1
  self.training_id_sha256=hashlib.sha256(json.dumps(list(training_ids)).encode()).hexdigest();return self
 def transform(self,raw):
  return ((np.asarray(raw,np.float32)-self.mean)/self.scale).astype(np.float32)

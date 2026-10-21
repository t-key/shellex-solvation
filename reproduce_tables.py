"""Reaggregate fixed stored predictions/errors. Never calls a model or QM."""
from pathlib import Path
import sys,json,argparse
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
import numpy as np,pandas as pd
from evaluate.statistics import summary,ci,DISPLAY
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'results/reproduced');ap.add_argument('--check-report',type=Path,default=ROOT/'REPRODUCTION_CHECK.csv');args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
    labels=pd.read_csv(ROOT/'data/labels/calculated_curves.csv');measured=pd.read_csv(ROOT/'data/measured/measured_curves.csv')
    measured=pd.concat([measured,pd.read_csv(ROOT/'data/measured/out_of_scope_indomethacin.csv')],ignore_index=True)
    vectors=read('data/predictions/shellex_predictions.json');vd={r['curve_id']:r for r in vectors}
    comp=pd.read_csv(ROOT/'data/predictions/computed_curve_metrics.csv');meas=pd.read_csv(ROOT/'data/predictions/measured_curve_metrics.csv')
    un={r['curve_id']:r for r in read('data/predictions/measured_pred_unifac.json')};direct={r['curve_id']:r for r in read('data/predictions/measured_pred_cosmors.json')}
    unrows=[]
    for r in measured.to_dict('records'):
        u=un[r['curve_id']]
        if u['state']!='applicable':continue
        x=np.array(json.loads(r['x_A']));inside=(x>1e-10)&(x<1-1e-10);y=np.array(json.loads(r['excess_kcal_mol']));yp=np.array(u['predicted_excess'])
        unrows.append(dict(curve_id=r['curve_id'],solute=r['solute'],method='UNIFAC_original',MAE=np.abs(yp-y)[inside].mean(),points=inside.sum()))
    meas=pd.concat([meas,pd.DataFrame(unrows)],ignore_index=True)
    subsets=read('data/splits/measured_subsets.json');subsets={'excluding_indomethacin148':set(subsets['measured148']),'common132':set(subsets['common132']),'all151':set(subsets['all151'])}
    methods=['zero_excess','pair_mean_train10640','fingerprint_2d','shellex','shellex_no_fallback','UNIFAC_original','direct_COSMO','shellex_narrow','shellex_narrow_motifs']
    outputs={};paired=[];fig3=[];per=[];matched=[];pairrows=[]
    for subset,ids in subsets.items():
        d=meas[meas.curve_id.isin(ids)];fig3+=summary(d[d.method.isin(methods)],subset)
        w=d.pivot(index=['curve_id','solute'],columns='method',values='MAE')
        for cand in ['shellex','shellex_no_fallback']:
            for ref in methods[:3]+['UNIFAC_original']:
                z=w[[cand,ref]].dropna();gm=z.groupby('solute',sort=True).mean();delta=gm[cand]-gm[ref];lo,hi=ci(delta)
                paired.append(dict(subset=subset,candidate=DISPLAY[cand],reference=DISPLAY[ref],candidate_key=cand,reference_key=ref,curves=len(z),solutes=len(gm),candidate_MAE=gm[cand].mean(),reference_MAE=gm[ref].mean(),delta_candidate_minus_reference=delta.mean(),CI95_low=lo,CI95_high=hi,improvement_fraction=1-gm[cand].mean()/gm[ref].mean(),bootstrap_unit='solute',bootstrap_replicates=1000,bootstrap_seed=20261014))
        applicable=set(d[d.method=='UNIFAC_original'].curve_id);matched+=summary(d[d.curve_id.isin(applicable)&d.method.isin(['UNIFAC_original','shellex','shellex_no_fallback','direct_COSMO'])],subset,{'subset_restriction':'UNIFAC-applicable curves only'})
        for (solute,m),g in d[d.method.isin(methods)].groupby(['solute','method'],sort=True):
            lo,hi=ci(g.sort_values('curve_id').MAE);per.append(dict(subset=subset,solute=solute,method=DISPLAY[m],method_key=m,curves=len(g),MAE=g.MAE.mean(),CI95_low=lo,CI95_high=hi,bootstrap_unit='curve within solute',bootstrap_replicates=1000,bootstrap_seed=20261014))
        pairrows+=summary(d[d.method.isin(['pair_mean_train9800','pair_mean_train10640'])],subset)
    fig2=[]
    for (source,axis),d in comp.groupby(['source','axis'],sort=True):fig2+=summary(d[d.method.isin(methods[:5])],source,{'axis':axis,'axis_label':('new structural families / '+axis if source=='new_structural' else axis),'test_only':True})
    d=comp[comp.source=='new_structural'];fig2+=summary(d[d.method.isin(methods[:5])],'new_structural_all',{'axis':'new_structure_family','axis_label':'new structural families (all 240 sealed curves)','test_only':True})
    for source,d in comp.groupby('source'):pairrows+=summary(d[d.method.isin(['pair_mean_train9800','pair_mean_train10640'])],source)
    outputs.update(MEASURED_PAIRED_VS_BASELINES=pd.DataFrame(paired),UNIFAC_MATCHED_COMPARISON=pd.DataFrame(matched),FIG3_MEASURED_MAE=pd.DataFrame(fig3),FIG3_PER_SOLUTE_MAE=pd.DataFrame(per),FIG2_COMPUTED_AXIS_MAE=pd.DataFrame(fig2),PAIR_MEAN_TABLE_VALUES=pd.DataFrame(pairrows))
    # Reconstruct grid extrema from fixed teacher labels and saved ShellEx curves.
    fam=pd.read_csv(ROOT/'data/molecules/solutes.csv').set_index('id').family.to_dict();osc={'anthracene','fluorene','dibenzothiophene','dibenzofuran','biphenyl','naphthalene','benzothiophene'}
    original=labels[~labels.curve_id.str.startswith('ADD_')].copy();teacher={};mapcurves=[]
    for r in original.to_dict('records'):
        cid=r['curve_id'];solute,pair=cid.split('__');y=np.array(json.loads(r['excess_kcal_mol']));x=np.array(json.loads(r['x_A']));teacher[cid]=dict(pair_id=pair,solute_id=solute,family=fam[solute],bonus=min(0,float(y.min())),penalty=max(0,float(y.max())),optimum_x_A=float(x[np.where(np.isclose(y,y.min(),atol=1e-12,rtol=0))[0][0]]))
        if r['partition']!='test':continue
        v=vd[cid];t=teacher[cid];z=dict(curve_id=cid,pair_id=pair,solute_id=solute,family=fam[solute],axis=r['axis'],teacher_bonus=t['bonus'],teacher_penalty=t['penalty'],teacher_optimum_x_A=t['optimum_x_A'])
        for tag,field in [('ShellEx','guarded_scaled_excess'),('ShellEx_raw','raw_excess')]:
            a=np.array(v[field]);z[tag+'_bonus']=min(0,float(a.min()));z[tag+'_penalty']=max(0,float(a.max()));z[tag+'_optimum_x_A']=float(x[np.where(np.isclose(a,a.min(),atol=1e-12,rtol=0))[0][0]])
        mapcurves.append(z)
    mc=pd.DataFrame(mapcurves);full=pd.DataFrame(teacher.values());pairmeta=pd.read_csv(ROOT/'results/SHELLEX_PAIR_RANKING.csv').set_index('pair_id');maps=[]
    for pair,g in mc.groupby('pair_id',sort=True):
        r=pairmeta.loc[pair];z=dict(pair_id=pair,solvent_A=r.solvent_A,solvent_B=r.solvent_B,OSC_category=r.OSC_category)
        for group,gg,tt in [('all100',g,full[full.pair_id==pair]),('OSC35',g[g.family.isin(osc)],full[(full.pair_id==pair)&full.family.isin(osc)])]:
            z[group+'_test_n']=len(gg)
            for field in ['teacher_bonus','teacher_penalty','teacher_optimum_x_A','ShellEx_bonus','ShellEx_penalty','ShellEx_optimum_x_A','ShellEx_raw_bonus','ShellEx_raw_penalty','ShellEx_raw_optimum_x_A']:z[group+'_'+field]=gg[field].median()
            z[group+'_teacher_full_bonus']=tt.bonus.median()
            for tag in ['teacher','ShellEx','ShellEx_raw']:z[group+'_'+tag+'_multiplier']=np.exp(-z[group+'_'+tag+'_bonus']/.5925)
        maps.append(z)
    maps=pd.DataFrame(maps);corr=[]
    for group in ['all100','OSC35']:
        for method in ['ShellEx','ShellEx_raw']:
            for ref,interpretation in [('teacher_bonus','matched test solutes per pair; primary'),('teacher_full_bonus','full100/35 versus partial test solutes; descriptive only')]:
                z=maps[[group+'_'+ref,group+'_'+method+'_bonus']].dropna();a=z.iloc[:,0].to_numpy();b=z.iloc[:,1].to_numpy();ols=np.linalg.lstsq(np.c_[a,np.ones(len(a))],b,rcond=None)[0]
                corr.append(dict(group=group,method=method,teacher_comparator=ref,pairs=len(z),Spearman=np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1],zero_intercept_slope=a@b/(a@a),OLS_slope=ols[0],OLS_intercept=ols[1],interpretation=interpretation))
        for tag in ['teacher','ShellEx','ShellEx_raw']:maps[group+'_'+tag+'_rank']=maps[group+'_'+tag+'_bonus'].rank(method='min')
    outputs.update(SHELLEX_MAP_TEST_CURVES=mc,SHELLEX_PAIR_RANKING=maps,SHELLEX_MAP_RANK_CORRELATION=pd.DataFrame(corr))
    checks=[]
    for name,d in outputs.items():
        target=pd.read_csv(ROOT/'results'/(name+'.csv'));d.to_csv(out/(name+'.csv'),index=False,float_format='%.15g')
        # Match rows by all identifier columns, never by implementation group order.
        keys=[c for c in ['subset','solute','axis','method_key','candidate_key','reference_key','curve_id','pair_id','group','teacher_comparator','method'] if c in d and c in target and not pd.api.types.is_numeric_dtype(d[c])]
        if keys:d=d.sort_values(keys).reset_index(drop=True);target=target.sort_values(keys).reset_index(drop=True)
        if d.shape!=target.shape:raise AssertionError((name,d.shape,target.shape))
        for c in d:
            if c not in target or not pd.api.types.is_numeric_dtype(d[c]) or pd.api.types.is_bool_dtype(d[c]):continue
            a=d[c].to_numpy(dtype=float);b=target[c].to_numpy(dtype=float);err=float(np.nanmax(abs(a-b))) if len(a) else 0.;passed=bool(np.allclose(a,b,atol=0.0005,rtol=0,equal_nan=True))
            checks.append(dict(file=name+'.csv',column=c,rows=len(d),max_abs_difference=err,tolerance=0.0005,matches_to_third_decimal=bool(np.array_equal(np.round(a,3),np.round(b,3),equal_nan=True)),state='passed' if passed else 'failed'))
    # Independent stored ShellEx vector-to-error check, without any model call.
    maximum=0.
    lookup=comp[comp.method=='shellex'].set_index('curve_id').MAE.to_dict();lookup.update(meas[meas.method=='shellex'].set_index('curve_id').MAE.to_dict())
    for r in labels.to_dict('records')+measured.to_dict('records'):
        if r['curve_id'] not in vd:continue
        x=np.array(json.loads(r['x_A']));inside=(x>1e-10)&(x<1-1e-10);y=np.array(json.loads(r['excess_kcal_mol']));yp=np.array(vd[r['curve_id']]['guarded_scaled_excess']);maximum=max(maximum,abs(abs(y-yp)[inside].mean()-lookup[r['curve_id']]))
    assert maximum<1e-10,maximum
    checks.append(dict(file='stored_ShellEx_vectors',column='curve_MAE',rows=len(vd),max_abs_difference=maximum,tolerance=1e-10,matches_to_third_decimal=True,state='passed'))
    check=pd.DataFrame(checks);args.check_report.parent.mkdir(parents=True,exist_ok=True);check.to_csv(args.check_report,index=False);assert check.state.eq('passed').all()
    print(json.dumps(dict(state='passed',tables=len(outputs),checks=len(check),stored_prediction_MAE_max_error=maximum,output=str(out),new_inference=0,new_training=0)))
if __name__=='__main__':main()

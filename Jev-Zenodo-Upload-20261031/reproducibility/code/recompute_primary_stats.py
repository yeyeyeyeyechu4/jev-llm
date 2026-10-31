import json, csv, pathlib, numpy as np
from sklearn.metrics import roc_auc_score
ROOT=pathlib.Path(r'D:/Demo/proj'); B=ROOT/'paper'
V4=ROOT/'results_v4'; RJ=ROOT/'results_jev_replication'; RL=[ROOT/'results_gpt61_r1',ROOT/'results_gpt61_r2',ROOT/'results_llm_rep3']
DS={'nhamcs':'ht_nhamcs_final.csv','mimic_ed':'ht_mimic_ed_final.csv','icu_xlsx':'ht_icu_xlsx_final.csv'}
def rowscsv(ds):
    out=[]
    with open(V4/DS[ds],encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['scored'].lower()=='true': out.append(r)
    return out
def jl(p): return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def boot(vals,n=2000,seed=20261015):
    rng=np.random.default_rng(seed); v=np.asarray(vals)
    out=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)]
    return [float(np.percentile(out,2.5)),float(np.percentile(out,97.5))]
def calib(cor,conf):
    conf=np.array(conf,float); cor=np.array(cor,float); ok=~np.isnan(conf); conf=conf[ok]; cor=cor[ok]
    bins=[]
    for b in range(10):
        lo=b/10; hi=(b+1)/10
        msk=((conf>=lo)&(conf<hi)) if b<9 else ((conf>=lo)&(conf<=hi))
        bins.append(dict(n=int(msk.sum()),conf=float(conf[msk].mean()) if msk.sum() else 0.0,acc=float(cor[msk].mean()) if msk.sum() else 0.0))
    ece=float(sum(b['n']/len(conf)*abs(b['acc']-b['conf']) for b in bins if b['n']))
    return dict(n=int(len(conf)),ece=ece,auc=float(roc_auc_score(cor,conf)),brier=float(np.mean((conf-cor)**2)),mean_conf=float(conf.mean()),obs_acc=float(cor.mean()),bins=bins)
out={}
for ds in DS:
    rs=rowscsv(ds); ids=[r['sample_id'] for r in rs]; truth={r['sample_id']:r['true_label'] for r in rs}
    jruns=[]; lruns=[]; jms=[]; lms=[]; jcost=[]; lcost=[]; jconf=[]; lconf=[]; jcorr=[]; lcorr=[]
    for i in range(1,4):
        jf=jl(RJ/('jev_'+ds+'_form_r'+str(i)+'.jsonl')); lf=jl(RL[i-1]/('llm_gpt-6.1-sol_'+ds+'_form.jsonl'))
        jruns.append({x['sample_id']:(x['sample_id'] in truth and x['pred']==truth[x['sample_id']] and bool(x['ok'])) for x in jf})
        lruns.append({x['sample_id']:(x['sample_id'] in truth and x['pred']==truth[x['sample_id']] and bool(x['ok'])) for x in lf})
        for x in jf:
            if str(x.get('err','')).strip()=='':
                jms.append(float(x['ms'])); jcost.append(float(x['cost'])); jconf.append(float(x['conf']) if x.get('conf') is not None else np.nan); jcorr.append(1 if (x['sample_id'] in truth and x['pred']==truth[x['sample_id']] and bool(x['ok'])) else 0)
        for x in lf:
            if str(x.get('err','')).strip()=='':
                lms.append(float(x['ms'])); lcost.append(float(x['cost'])); lconf.append(float(x['conf']) if x.get('conf') is not None else np.nan); lcorr.append(1 if (x['sample_id'] in truth and x['pred']==truth[x['sample_id']] and bool(x['ok'])) else 0)
    jm=np.array([np.mean([jruns[k].get(s,False) for k in range(3)]) for s in ids])
    lm=np.array([np.mean([lruns[k].get(s,False) for k in range(3)]) for s in ids])
    def arm(m,ms,cost):
        acc=float(m.mean()); n=len(m)
        return dict(k=int(round(acc*n)),n=n,acc=acc,ci=boot(m),cost_per_1000=float(np.mean(cost)*1000),ms=[float(x) for x in ms])
    diff=float((jm-lm).mean()); bci=boot(jm-lm)
    bonly=int(np.sum((jm>=0.5)&(lm<0.5))); conly=int(np.sum((jm<0.5)&(lm>=0.5)))
    rng2=np.random.default_rng(20261015); obs=abs(diff); cnt=0
    for _ in range(10000):
        s=np.mean((jm-lm)*(2*rng2.integers(0,2,len(jm))-1))
        if abs(s)>=obs-1e-15: cnt+=1
    out[ds]=dict(jev_form=arm(jm,jms,jcost),llm_form=arm(lm,lms,lcost),pair=dict(diff=diff,boot=bci,p=float((cnt+1)/10001),b_only=bonly,c_only=conly,n=len(jm)),calib=dict(jev_form=calib(jcorr,jconf),llm_form=calib(lcorr,lconf)))
(B/'analysis'/'repeats_stats.json').write_text(json.dumps(out,indent=1,ensure_ascii=False),encoding='utf-8')
print('written')
for k,v in out.items(): print(k,round(v['jev_form']['acc'],4),round(v['llm_form']['acc'],4),round(v['pair']['diff']*100,2),round(v['pair']['p'],3))
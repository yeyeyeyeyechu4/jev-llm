import pandas as pd, json, pathlib, numpy as np, time, lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
ROOT=pathlib.Path(r'D:/Demo/proj'); t0=time.time()
b=pd.read_csv(ROOT/'data_bench'/'mimic_ed_bench.csv')
ed=pd.read_pickle(ROOT/'data_bench'/'mimic_ed_reduced.pkl')
ed=ed[ed['disposition'].notna()].reset_index(drop=True)
recs=[dict(json.loads(x)) for x in b.fields_json]
inter=sorted(set.intersection(*[set(r) for r in recs]))
print('intersection',len(inter))
def norm(v):
    s=str(v).strip().lower()
    try: return ('n',round(float(s),6))
    except Exception: return ('s',s)
src={}
for i,row in enumerate(ed[inter].itertuples(index=False,name=None)):
    k=tuple(norm(v) for v in row)
    src.setdefault(k,[]).append(i)
matched=[]; amb=0
for r in recs:
    k=tuple(norm(r.get(c,'')) for c in inter)
    v=src.get(k,[])
    if not v: matched.append(None); continue
    if len(v)>1: amb+=1
    want='Admit' if r==recs[0] else None
    lab=b.iloc[len(matched)].label
    want='Admit' if lab=='admitted' else 'Discharge'
    pick=next((x for x in v if ed.loc[x,'disposition']==want), v[0])
    matched.append(int(pick))
print('matched',sum(1 for x in matched if x is not None),'amb',amb,'time',round(time.time()-t0,1))
eval_rows=[x for x in matched if x is not None]
print('unique eval',len(set(eval_rows)))
exclude={'disposition','esi','dep_name','n_surgeries','cxr_count','ekg_count','headct_count','otherimg_count','n_admissions'}
feats=[c for c in ed.columns if c not in exclude]
mask=np.ones(len(ed),bool); mask[eval_rows]=False
mask &= ed['disposition'].isin(['Admit','Discharge']).values
def frame(df):
    X=pd.DataFrame(index=df.index)
    for c in feats:
        s=df[c]
        if s.dtype==object or str(s.dtype).startswith('string'):
            X[c]=s.astype('category').cat.codes.astype(float).replace(-1,np.nan)
        else: X[c]=pd.to_numeric(s,errors='coerce')
    return X
Xtr=frame(ed.loc[mask].reset_index(drop=True)); ytr=(ed.loc[mask,'disposition']=='Admit').astype(int)
Xev=frame(ed.loc[eval_rows].reset_index(drop=True))
yev=(ed.loc[eval_rows,'disposition']=='Admit').astype(int).values
names=['f'+str(i) for i in range(Xtr.shape[1])]; Xtr.columns=names; Xev.columns=names
print('train',Xtr.shape,'eval',Xev.shape,'pos',round(ytr.mean(),3),round(yev.mean(),3))
params=dict(objective='binary',learning_rate=0.06,num_leaves=31,min_data_in_leaf=40,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4,seed=20261015)
model=lgb.train(params,lgb.Dataset(Xtr,label=ytr),num_boost_round=300)
p=model.predict(Xev)
def ece(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p); e=0.0
    for k in range(bins):
        lo=k/bins; hi=(k+1)/bins; m=(p>=lo)&(p<hi) if k<bins-1 else (p>=lo)&(p<=hi)
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)
res={'n_train':int(Xtr.shape[0]),'n_eval':int(len(yev)),'acc':float(accuracy_score(yev,p>0.5)),'auc':float(roc_auc_score(yev,p)),'brier':float(brier_score_loss(yev,p)),'ece':ece(yev,p)}
print(json.dumps(res,indent=1),'elapsed',round(time.time()-t0,1))
pathlib.Path(r'D:/Demo/proj/_work/ed_gbdt.json').write_text(json.dumps(res,indent=1),encoding='utf-8')
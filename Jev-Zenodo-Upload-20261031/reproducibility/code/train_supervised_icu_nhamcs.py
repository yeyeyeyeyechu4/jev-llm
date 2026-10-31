import pandas as pd, numpy as np, json, pathlib, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
print("sklearn", sklearn.__version__)
ROOT=pathlib.Path(r"D:\Demo\proj")
def ece(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p); e=0.0
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins
        m=(p>=lo)&(p<hi) if b<bins-1 else (p>=lo)&(p<=hi)
        if m.sum()==0: continue
        e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)
def numeric_frame(df, cols):
    out={}
    for c in cols:
        s=df[c]
        if str(s.dtype).startswith(("object","str","string")) or s.dtype.name=="category":
            codes=s.astype("category").cat.codes.astype(float)
            codes[codes<0]=np.nan
            out[c]=codes
        else:
            out[c]=pd.to_numeric(s,errors="coerce")
    return pd.DataFrame(out,index=df.index)
res={}
# ICU
icu=pd.read_excel(ROOT/"data"/"external_datasets"/"icu-sepsis-dataset-xlsx"/"ICU Sepsis Dataset.xlsx").reset_index(drop=True)
ib=pd.read_csv(ROOT/"data_bench"/"icu_xlsx_bench.csv")
eval_rows={int(s.split("_")[-1]) for s in ib.sample_id}
icu["_idx"]=icu.index; icu["_is_eval"]=icu["_idx"].isin(eval_rows)
train=icu[(~icu["_is_eval"]) & icu["Mortalite"].isin([0.0,1.0])].copy()
ev=icu[icu["_is_eval"] & icu["Mortalite"].isin([0.0,1.0])].copy()
feat=[c for c in icu.columns if c not in ("Variable","Mortalite","Length_of_stay_in_intensive_care","_idx","_is_eval")]
Xtr=numeric_frame(train,feat); Xev=numeric_frame(ev,feat)
ytr=(train["Mortalite"]==1).astype(int).values; yev=(ev["Mortalite"]==1).astype(int).values
print("ICU train",len(Xtr),"eval",len(Xev),"pos",round(ytr.mean(),3),round(yev.mean(),3))
gb=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.06,max_leaf_nodes=31,random_state=20261015).fit(Xtr,ytr)
p=gb.predict_proba(Xev)[:,1]
res["ICU GBDT"]=dict(n_train=int(len(Xtr)),n_eval=int(len(Xev)),acc=float(accuracy_score(yev,p>0.5)),auc=float(roc_auc_score(yev,p)),brier=float(brier_score_loss(yev,p)),ece=ece(yev,p))
lr=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=3000))]).fit(Xtr,ytr)
p2=lr.predict_proba(Xev)[:,1]
res["ICU LR"]=dict(n_train=int(len(Xtr)),n_eval=int(len(Xev)),acc=float(accuracy_score(yev,p2>0.5)),auc=float(roc_auc_score(yev,p2)),brier=float(brier_score_loss(yev,p2)),ece=ece(yev,p2))
dm=DummyClassifier(strategy="most_frequent").fit(Xtr,ytr)
res["ICU majority"]=dict(acc=float(accuracy_score(yev,dm.predict(Xev))))
# NHAMCS
nh=pd.read_csv(ROOT/"data"/"external_datasets"/"nhamcs-2018-22"/"nhamcs_data_2018_22.csv")
nb=pd.read_csv(ROOT/"data_bench"/"nhamcs_bench.csv")
cols=["visit_month","day_of_week","arrival_time","age","sex","residence","insurance"]
def key_from_fields(fj, cs):
    fj=json.loads(fj) if isinstance(fj,str) else fj
    d=dict(fj)
    return tuple(str(d.get(c,"")).strip() for c in cs)
bkeys=[key_from_fields(x,cols) for x in nb.fields_json]
src={}
for i,row in nh.iterrows():
    k=tuple(str(row[c]).strip() for c in cols); src.setdefault(k,[]).append(i)
matched=[]; miss=0; dup=0
for k in bkeys:
    v=src.get(k,[])
    if len(v)==0: miss+=1; matched.append(None)
    else:
        if len(v)>1: dup+=1
        matched.append(v[0])
print("NH match missing",miss,"dup",dup)
keep=[k for k,i in enumerate(matched) if i is not None]
ev_rows=[matched[k] for k in keep]
lab2int={"acuity_1_resuscitation":1,"acuity_2_emergent":2,"acuity_3_urgent":3,"acuity_4_semi_urgent":4,"acuity_5_non_urgent":5}
yev3=np.array([lab2int[str(nb.iloc[k].label)] for k in keep])
nh_eval=nh.loc[ev_rows].copy()
nh_tr=nh.drop(index=ev_rows)
nh_tr=nh_tr[nh_tr["target_triage_acuity"].isin([1,2,3,4,5])]
ytr3=nh_tr["target_triage_acuity"].astype(int).values
feat2=[c for c in nh.columns if c not in ("target_triage_acuity","wait_time_minutes","intervention_iv_fluids")]
Xtr2=numeric_frame(nh_tr,feat2); Xev2=numeric_frame(nh_eval,feat2)
print("NH train",len(Xtr2),"eval",len(Xev2))
gb2=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.06,max_leaf_nodes=31,random_state=20261015).fit(Xtr2,ytr3)
pred=gb2.predict(Xev2)
res["NHAMCS GBDT"]=dict(n_train=int(len(Xtr2)),n_eval=int(len(Xev2)),acc=float((pred==yev3).mean()))
dm2=DummyClassifier(strategy="most_frequent").fit(Xtr2,ytr3)
res["NHAMCS majority"]=dict(acc=float((dm2.predict(Xev2)==yev3).mean()))
print(json.dumps(res,indent=1))
pathlib.Path(r"D:\Demo\proj\_work\supervised_full.json").write_text(json.dumps(res,indent=1),encoding="utf-8")

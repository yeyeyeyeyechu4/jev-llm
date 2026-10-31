"""Cheap baselines and saturation diagnosis for every benchmark.

Runs after the audit gate: majority class, a logistic regression on the numeric
fields only, and TF-IDF+LR on the serialised text.  A dataset whose numeric LR
reaches ~1.0 is linearly separable and its accuracy cannot rank methods.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def numeric_frame(bench: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(json.loads(text)) for text in bench["fields_json"]]
    frame = pd.DataFrame(rows)
    return pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce") for c in frame.columns})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="nhamcs,icu_csv,icu_xlsx,mimic_ed")
    ap.add_argument("--out", default="results_bench/baselines.json")
    args = ap.parse_args()

    results = {}
    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        path = ROOT / "data_bench" / (name + "_bench.csv")
        if not path.exists():
            continue
        bench = pd.read_csv(path)
        groups = bench["group_id"].astype(str).to_numpy()
        classes = sorted(set(bench["label"]))
        binary = len(classes) == 2
        y = (bench["label"] == classes[-1]).astype(int) if binary else bench["label"]
        entry = {"n": int(len(bench)),
                 "classes": {str(k): int(v) for k, v in bench["label"].value_counts().items()},
                 "majority": float(bench["label"].value_counts(normalize=True).max()),
                 "binary": binary, "positive_label": classes[-1] if binary else None}
        cv = GroupKFold(n_splits=5)
        num = numeric_frame(bench)
        num = num.loc[:, num.notna().mean() > 0.5].fillna(0.0)
        num = num[[c for c in num.columns if num[c].nunique() > 1]]
        entry["n_numeric_features"] = int(num.shape[1])
        if num.shape[1] > 0:
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
            pred = cross_val_predict(lr, num.to_numpy(), y, cv=cv, groups=groups)
            entry["numericLR_acc"] = float(accuracy_score(y, pred))
            if binary:
                prob = cross_val_predict(lr, num.to_numpy(), y, cv=cv, groups=groups,
                                         method="predict_proba")[:, 1]
                entry["numericLR_auc"] = float(roc_auc_score(y, prob))
        tok = make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True),
                            LogisticRegression(max_iter=3000, class_weight="balanced"))
        pred = cross_val_predict(tok, bench["state_text"], y, cv=cv, groups=groups)
        entry["tfidfLR_acc"] = float(accuracy_score(y, pred))
        if binary:
            prob = cross_val_predict(tok, bench["state_text"], y, cv=cv, groups=groups,
                                     method="predict_proba")[:, 1]
            entry["tfidfLR_auc"] = float(roc_auc_score(y, prob))
        if entry.get("numericLR_acc", 0) >= 0.99:
            entry["saturated"] = True
        results[name] = entry
        print("%-9s n=%3d maj=%.4f  numericLR=%.4f  tfidfLR=%.4f%s" % (
            name, entry["n"], entry["majority"], entry.get("numericLR_acc", float("nan")),
            entry["tfidfLR_acc"], "  <-- SATURATED" if entry.get("saturated") else ""))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()

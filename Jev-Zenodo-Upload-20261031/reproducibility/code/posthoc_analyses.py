"""Post-hoc analyses added after the audit (all use existing data; no new API calls).

B2  confidence calibration: AUC(confidence -> correct), ECE(10 bins), Brier, per arm.
B3  TOST equivalence test for the primary paired difference (bounds +/-5 and +/-3 pp),
    90% paired-bootstrap CI.
B4  held-out supervised baselines on the audit split (train/val/test): TF-IDF + LR and
    numeric-feature LR, hyper-parameter chosen on val, test evaluated once.

Outputs:
  paper/analysis/posthoc_results.json
  paper/analysis/posthoc_report.md
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

BASE = pathlib.Path(r"D:\Demo\proj")
V4 = BASE / "results_v4"
BENCH = BASE / "data_bench"
OUT = BASE / "paper" / "analysis"
RNG_SEED = 20261015
N_BOOT = 2000

FILES = {"triage": "ht_nhamcs_final.csv", "disposition": "ht_mimic_ed_final.csv",
         "icu_mortality": "ht_icu_xlsx_final.csv"}
BENCH_FILES = {"triage": "nhamcs_bench.csv", "disposition": "mimic_ed_bench.csv",
               "icu_mortality": "icu_xlsx_bench.csv"}


def auc_conf(conf, correct):
    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided", method="asymptotic").statistic
    return float(u / (len(pos) * len(neg)))


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.sum():
            e += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def brier(conf, correct):
    return float(np.mean((conf.astype(float) - correct.astype(float)) ** 2))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def paired_bootstrap_ci(a, b, lo=5.0, hi=95.0, seed=RNG_SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = a[idx].mean() - b[idx].mean()
    return float(np.percentile(deltas, lo)), float(np.percentile(deltas, hi))


def main():
    res = {"calibration": {}, "tost": {}, "baselines": {}}

    # ---------- B2 calibration ----------
    for task, fname in FILES.items():
        df = pd.read_csv(V4 / fname)
        s = df[df.scored]
        task_cal = {}
        for arm in ["jev_form", "jev_native", "llm_form", "llm_native", "jev_rebuilt"]:
            conf_col, ok_col = arm + "_conf", arm + "_ok"
            if conf_col not in s:
                continue
            m = s[conf_col].notna() & s[ok_col].notna()
            if m.sum() < 10:
                continue
            conf = s.loc[m, conf_col].to_numpy(float)
            ok = s.loc[m, ok_col].to_numpy(bool)
            task_cal[arm] = {"n": int(m.sum()), "mean_conf": float(conf.mean()),
                             "auc": auc_conf(conf, ok), "ece_10bin": ece(conf, ok),
                             "brier": brier(conf, ok),
                             "observed_accuracy": float(ok.mean())}
        res["calibration"][task] = task_cal

    # ---------- B3 TOST ----------
    for task, fname in FILES.items():
        df = pd.read_csv(V4 / fname)
        s = df[df.scored]
        a = s["jev_form_ok"].fillna(False).to_numpy(bool)
        b = s["llm_form_ok"].fillna(False).to_numpy(bool)
        n = len(s)
        delta = float(a.mean() - b.mean())
        ci90 = paired_bootstrap_ci(a.astype(float), b.astype(float), 5.0, 95.0)
        entry = {"n": n, "delta_pp": 100 * delta, "ci90_pp": [100 * ci90[0], 100 * ci90[1]]}
        for bound in (0.05, 0.03, 0.10):
            within = (ci90[0] > -bound) and (ci90[1] < bound)
            entry[f"tost_equivalence_{int(bound*100)}pp"] = bool(within)
        res["tost"][task] = entry

    # ---------- B4 held-out supervised baselines ----------
    for task, fname in BENCH_FILES.items():
        d = pd.read_csv(BENCH / fname)
        if task == "icu_mortality":
            d = d[d["label"] != "code_3_unverified"].reset_index(drop=True)
        y = d["label"].to_numpy()
        tr = (d.split == "train").to_numpy()
        va = (d.split == "val").to_numpy()
        te = (d.split == "test").to_numpy()
        texts = d.state_text.astype(str).to_numpy()

        def fit_eval_tfidf(C):
            pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                 LogisticRegression(C=C, max_iter=3000))
            pipe.fit(texts[tr], y[tr])
            return float((pipe.predict(texts[va]) == y[va]).mean()), pipe

        best_C, best_val = None, -1.0
        for C in (0.1, 1.0, 10.0):
            val_acc, _ = fit_eval_tfidf(C)
            if val_acc > best_val:
                best_C, best_val = C, val_acc
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                             LogisticRegression(C=best_C, max_iter=3000))
        pipe.fit(texts[tr | va], y[tr | va])
        pred = pipe.predict(texts[te])
        acc_tfidf = float((pred == y[te]).mean())

        # numeric-feature logistic regression
        rows = []
        for fj in d.fields_json:
            pairs = json.loads(fj)
            rows.append({k: v for k, v in pairs})
        num = pd.DataFrame(rows)
        num = num.apply(pd.to_numeric, errors="coerce")
        med = num.loc[tr].median(numeric_only=True)
        num = num.fillna(med).fillna(0.0)
        X = num.to_numpy(float)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
        lr = LogisticRegression(C=1.0, max_iter=3000)
        lr.fit(X[tr], y[tr])
        acc_num = float((lr.predict(X[te]) == y[te]).mean())

        labels = np.unique(y)
        majority = float(pd.Series(y[te]).value_counts().iloc[0] / te.sum())
        k_tf = int(round(acc_tfidf * te.sum()))
        k_num = int(round(acc_num * te.sum()))
        res["baselines"][task] = {
            "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
            "best_C": best_C, "val_accuracy_selected": best_val,
            "tfidf_lr_test_accuracy": acc_tfidf,
            "tfidf_lr_test_wilson95": list(wilson(k_tf, int(te.sum()))),
            "numeric_lr_test_accuracy": acc_num,
            "numeric_lr_test_wilson95": list(wilson(k_num, int(te.sum()))),
            "test_majority_accuracy": majority,
            "n_classes": int(len(labels)),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "posthoc_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown fragment
    lines = ["# Post-hoc analyses (no new API calls)", "",
             "Generated by `paper/analysis/posthoc_analyses.py`.", ""]
    lines += ["## B2. Confidence calibration", "",
              "| Dataset | Arm | n | Mean confidence | Observed accuracy | AUC(conf→correct) | ECE (10 bins) | Brier |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task, arms in res["calibration"].items():
        for arm, v in arms.items():
            lines.append(f"| {task} | {arm} | {v['n']} | {v['mean_conf']:.3f} | {v['observed_accuracy']:.3f} | "
                         f"{v['auc']:.3f} | {v['ece_10bin']:.3f} | {v['brier']:.3f} |")
    lines += ["", "## B3. TOST equivalence (paired bootstrap 90% CI)", "",
              "| Dataset | n | Δ (pp) | 90% CI (pp) | Equivalent within ±5 pp | ±3 pp | ±10 pp |",
              "|---|---:|---:|---|---|---|---|"]
    for task, v in res["tost"].items():
        lines.append(f"| {task} | {v['n']} | {v['delta_pp']:+.1f} | [{v['ci90_pp'][0]:+.1f}, {v['ci90_pp'][1]:+.1f}] | "
                     f"{v['tost_equivalence_5pp']} | {v['tost_equivalence_3pp']} | {v['tost_equivalence_10pp']} |")
    lines += ["", "## B4. Held-out supervised baselines (audit split)", "",
              "| Dataset | n train/val/test | Selected C | TF-IDF+LR test acc | Numeric LR test acc | Test majority |",
              "|---|---|---|---:|---:|---:|"]
    for task, v in res["baselines"].items():
        lines.append(f"| {task} | {v['n_train']}/{v['n_val']}/{v['n_test']} | {v['best_C']} | "
                     f"{v['tfidf_lr_test_accuracy']:.3f} | {v['numeric_lr_test_accuracy']:.3f} | {v['test_majority_accuracy']:.3f} |")
    (OUT / "posthoc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for p, v in res["tost"].items():
        print(p, "delta %.1f pp, CI90 [%.1f, %.1f], TOST5=%s TOST3=%s" %
              (v["delta_pp"], v["ci90_pp"][0], v["ci90_pp"][1],
               v["tost_equivalence_5pp"], v["tost_equivalence_3pp"]))
    for p, v in res["baselines"].items():
        print(p, "tfidf test %.3f numeric %.3f majority %.3f (n_test %d)" %
              (v["tfidf_lr_test_accuracy"], v["numeric_lr_test_accuracy"],
               v["test_majority_accuracy"], v["n_test"]))
    print("wrote", OUT / "posthoc_results.json")


if __name__ == "__main__":
    main()

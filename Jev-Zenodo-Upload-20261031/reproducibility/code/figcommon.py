# -*- coding: utf-8 -*-
"""Shared style, statistics and data loading for the v2 figure set (Jev vs LLM).

Every figure script imports from here so that the numbers, colours and fonts are
defined exactly once.  All data are read-only: ``results_v4`` per-sample CSVs,
``results_c4_full`` full-sample grid-OCR JSONL, ``results_combined`` summaries and
the ``paper/analysis`` JSONs.  Nothing in this module writes to those paths.

Naming convention: the decision model is "Jev" (never "JEV"), the reasoning model
is "LLM", the second reasoning model is "gpt-6".
"""
from __future__ import annotations

import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter  # noqa: F401  (re-exported for scripts)

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
FIG_DIR = Path(__file__).resolve().parent
PAPER = FIG_DIR.parent
ROOT = PAPER.parent
V4 = ROOT / "results_v4"
C4_FULL = ROOT / "results_c4_full"
COMBINED = ROOT / "results_combined"
ANALYSIS = PAPER / "analysis"
ANALYSIS_V9 = PAPER / "build" / "revision_v9" / "analysis_v9.json"

NUMBERS_JSON = ANALYSIS / "numbers.json"
REPEATS_JSON = ANALYSIS / "repeats_stats.json"
POSTHOC_JSON = ANALYSIS / "posthoc_results.json"
FINAL_EXP_JSON = COMBINED / "final_experiments.json"
LAYOUT_FULL_JSON = COMBINED / "layout_full.json"
SUMMARY_JSON = V4 / "summary_v4.json"

DPI = 350
MM = 1 / 25.4
Z95 = 1.959963984540054
BOOT_SEED = 20261015
BOOT_REPS = 2000

# --------------------------------------------------------------------------- #
# palette (paper convention)
# --------------------------------------------------------------------------- #
PALETTE = dict(
    jev="#1f6f78",
    jev_light="#9dc6c9",
    llm="#c45a1a",
    gpt6="#6b4f9e",       # second reasoning LLM (gpt-6); outside the core trio
    baseline="#8a8a8a",
    grid_fill="#d9d9d9",
    grid_edge="#6e6e6e",
    ink="#1a1a1a",
    muted="#555555",
    hairline="#8c8c8c",
    ds_fill="#f4f4f4",
    ds_edge="#5a5a5a",
    serial_fill="#e8f1f2",
    pair_fill="#fbeee4",
    sens_fill="#f7f7f7",
    arrow="#4d4d4d",
)

# --------------------------------------------------------------------------- #
# datasets / task labels
# --------------------------------------------------------------------------- #
DATASETS = [
    dict(
        key="nhamcs",
        csv="ht_nhamcs_final.csv",
        short="NHAMCS",
        task_en="ED triage",
        task_zh="急诊分诊",
        expect_n=300,
        expect_majority=0.5100,
    ),
    dict(
        key="mimic_ed",
        csv="ht_mimic_ed_final.csv",
        short="ED Kaggle copy",
        task_en="ED disposition",
        task_zh="急诊处置",
        expect_n=300,
        expect_majority=0.7033333333333334,
    ),
    dict(
        key="icu_xlsx",
        csv="ht_icu_xlsx_final.csv",
        short="ICU",
        task_en="ICU mortality",
        task_zh="ICU 院内死亡",
        expect_n=290,
        expect_majority=0.5241379310344828,
    ),
]

C2_ARMS = ("jev_form", "llm_form")
C1_ARMS = ("jev_native", "llm_native")
ALL_ARMS = ("jev_native", "jev_form", "jev_rebuilt", "jev_grid", "llm_native", "llm_form")


def mm(*vals: float):
    """Millimetres -> inches."""
    return tuple(v * MM for v in vals)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b_only: int, c_only: int) -> float:
    """Exact two-sided McNemar p (binomial test on discordant pairs, p0=0.5)."""
    disc = b_only + c_only
    if disc == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(min(b_only, c_only), disc, 0.5).pvalue)


def wald_ci(diff: float, b_only: int, c_only: int, n: int) -> tuple[float, float]:
    """Wald 95% CI of a paired proportion difference: SE = sqrt(b+c)/n."""
    if n <= 0:
        return float("nan"), float("nan")
    se = math.sqrt(b_only + c_only) / n
    return diff - Z95 * se, diff + Z95 * se


def bootstrap_ci(a: np.ndarray, b: np.ndarray, seed: int = BOOT_SEED, n_boot: int = BOOT_REPS):
    """Legacy paired bootstrap of the draft (2,000 record-level resamples).

    Reproduces the published Table 2 intervals exactly: ``random.Random`` stream,
    per-resample means, sorted deltas, indices int(0.025*n_boot) and int(0.975*n_boot).
    """
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(bool(a[i]) for i in idx) / n - sum(bool(b[i]) for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def auc_conf(conf: np.ndarray, correct: np.ndarray) -> float:
    """AUC of top-label confidence for predicting a correct decision (Mann-Whitney)."""
    from scipy.stats import mannwhitneyu

    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided", method="asymptotic").statistic
    return float(u / (len(pos) * len(neg)))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over 10 equal-width confidence bins."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.sum():
            e += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def brier_score(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((conf.astype(float) - correct.astype(float)) ** 2))


def calibration_bins(conf: np.ndarray, correct: np.ndarray, bins: int = 10):
    """List of (n, mean_conf, observed_acc) per equal-width bin (drops empty bins)."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            out.append(dict(bin=b, n=int(m.sum()), conf=float(conf[m].mean()), acc=float(correct[m].mean())))
    return out


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%"


def fmt_p(p: float) -> str:
    """p-value formatting used consistently across figures.

    Three decimals above 0.01; four decimals for 0.001 <= p < 0.01 is not needed
    (three decimals already resolves 0.008), but values just above 0.001 keep four
    decimals so that e.g. 0.0016 is not rounded to 0.002.
    """
    if p < 1e-3:
        return "p < 0.001"
    if p < 2e-3:
        return f"p = {p:.4f}"
    return f"p = {p:.3f}"


def fmt_p_short(p: float) -> str:
    if p < 1e-3:
        return "p < 0.001"
    return f"p = {p:.3f}"


def fmt_sci(p: float) -> str:
    """Small p-value, compact (e.g. '2.4e-5')."""
    if p >= 1e-3:
        return fmt_p(p)
    mant, exp = f"{p:.1e}".split("e")
    return f"p = {mant}e{int(exp)}"


# --------------------------------------------------------------------------- #
# figure style / save
# --------------------------------------------------------------------------- #
def set_style(lang: str) -> None:
    sans = ["Microsoft YaHei", "Arial"] if lang == "zh" else ["Arial", "DejaVu Sans"]
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "font.size": 7,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "axes.edgecolor": "#333333",
            "text.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "legend.frameon": False,
            "legend.fontsize": 6.2,
            "legend.handlelength": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def check_fonts(lang: str) -> str:
    family = "Microsoft YaHei" if lang == "zh" else "Arial"
    try:
        path = font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
    except Exception as exc:  # pragma: no cover - host dependent
        raise RuntimeError(f"required font {family!r} not found: {exc}") from exc
    return f"{family} ({path})"


def save(fig: plt.Figure, stem: str, lang: str) -> Path:
    out = FIG_DIR / f"{stem}_{lang}.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    glyph = [str(w.message) for w in caught if "glyph" in str(w.message).lower()]
    if glyph:
        raise RuntimeError(f"missing glyphs while saving {out.name}: {glyph}")
    return out


def panel_label(ax, text: str, **kw) -> None:
    kw.setdefault("fontsize", 7.6)
    kw.setdefault("pad", 3.0)
    ax.set_title(text, loc="left", **kw)


# --------------------------------------------------------------------------- #
# schematic helpers (Fig 1)
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, fc, ec, ls="-", lw=0.8, radius=0.0, z=2):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, p1, p2, color=None, ls="-", lw=0.75, rad=0.0, z=1):
    from matplotlib.patches import FancyArrowPatch

    if color is None:
        color = PALETTE["arrow"]
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=6.5, linewidth=lw,
            linestyle=ls, color=color, shrinkA=0.0, shrinkB=0.0,
            connectionstyle=f"arc3,rad={rad}", zorder=z,
        )
    )


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_scored(ds_meta: dict) -> pd.DataFrame:
    df = pd.read_csv(V4 / ds_meta["csv"])
    df["scored"] = df["scored"].astype(str).str.lower().eq("true")
    scored = df[df.scored].reset_index(drop=True)
    if len(scored) != ds_meta["expect_n"]:
        raise RuntimeError(f"{ds_meta['key']}: expected {ds_meta['expect_n']} scored rows, found {len(scored)}")
    counts = scored["true_label"].value_counts()
    majority = float(counts.max() / len(scored))
    if abs(majority - ds_meta["expect_majority"]) > 5e-5:
        raise RuntimeError(f"{ds_meta['key']}: majority {majority:.6f} != expected {ds_meta['expect_majority']}")
    return df, scored


def _ok_flags(frame: pd.DataFrame, arm: str) -> np.ndarray:
    """Correctness flag; a missing ``*_ok`` (transport failure) counts as incorrect."""
    return np.asarray(frame[arm + "_ok"].astype(object).eq(True), dtype=bool)


def _latency_answered(frame: pd.DataFrame, arm: str) -> np.ndarray:
    """Per-call ``*_ms`` values with transport-failure rows removed (err is null)."""
    m = frame[arm + "_err"].isna() & frame[arm + "_ms"].notna()
    return frame.loc[m, arm + "_ms"].to_numpy(float)


def _conf_array(frame: pd.DataFrame, arm: str):
    m = frame[arm + "_conf"].notna() & frame[arm + "_ok"].notna()
    return frame.loc[m, arm + "_conf"].to_numpy(float), frame.loc[m, arm + "_ok"].to_numpy(bool)


def _confusion(frame: pd.DataFrame, arm: str, task: str):
    """Confusion matrix on transport-failure-free rows; returns (labels, matrix, n)."""
    sub = frame[frame[arm + "_err"].isna()]
    if task == "ED triage":
        t = sub["true_label"].str.extract(r"acuity_(\d)_")[0].astype(int)
        p = sub[arm + "_pred"].str.extract(r"acuity_(\d)_")[0].astype(int)
        labels = [1, 2, 3, 4, 5]
    elif task == "ED disposition":
        t = sub["true_label"].map({"admitted": 1, "discharged": 0})
        p = sub[arm + "_pred"].map({"admitted": 1, "discharged": 0})
        labels = [0, 1]
    else:
        t = sub["true_label"].map({"died": 1, "survived": 0})
        p = sub[arm + "_pred"].map({"died": 1, "survived": 0})
        labels = [0, 1]
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, tr in enumerate(labels):
        for j, pr in enumerate(labels):
            cm[i, j] = int(((t == tr) & (p == pr)).sum())
    return labels, cm, int(len(sub))


def _full_sample_grid_fidelity() -> dict:
    """Cell fidelity of the bordered-grid OCR evaluated on every record (300/300/300)."""
    out = {}
    files = {"nhamcs": "ocr_nhamcs.jsonl", "mimic_ed": "ocr_mimic_ed.jsonl", "icu_xlsx": "ocr_icu_xlsx.jsonl"}
    for key, fname in files.items():
        recs = [json.loads(line) for line in (C4_FULL / fname).read_text(encoding="utf-8").splitlines() if line.strip()]
        total = sum(r["grid_cells_total"] for r in recs)
        rec = sum(r["grid_cells_recovered"] for r in recs)
        out[key] = dict(n=len(recs), recovered=int(rec), total=int(total), rate=float(rec / total))
    return out


def build_data() -> dict:
    """Load and recompute every number plotted by the v2 figures."""
    numbers = load_json(NUMBERS_JSON)
    posthoc = load_json(POSTHOC_JSON)
    final = load_json(FINAL_EXP_JSON)
    layout = load_json(LAYOUT_FULL_JSON)
    grid_ocr = _full_sample_grid_fidelity()
    task_key = {"nhamcs": "triage", "mimic_ed": "disposition", "icu_xlsx": "icu_mortality"}

    datasets: dict[str, dict] = {}
    for meta in DATASETS:
        key = meta["key"]
        df, scored = _read_scored(meta)
        n = len(scored)
        task = meta["task_en"]

        arms = {}
        for arm in ALL_ARMS:
            ok = _ok_flags(scored, arm)
            k = int(ok.sum())
            lo, hi = wilson_ci(k, n)
            cost = scored[arm + "_cost"].astype(float)
            answered = scored[arm + "_ok"].notna()
            arms[arm] = dict(
                k=k, n=n, acc=k / n, ci=(lo, hi),
                answered=int(answered.sum()),
                failures=int((~answered).sum()),
                cost_per_1000=float(cost.sum() / n * 1000.0),
                cost_total=float(cost.sum()),
                ms_answered=_latency_answered(scored, arm),
                ms_all=scored[arm + "_ms"].to_numpy(float),
                ok=ok,
            )

        pair = dict()
        for cond, (arm_a, arm_b) in (("C2", C2_ARMS), ("C1", C1_ARMS)):
            a, b = arms[arm_a]["ok"], arms[arm_b]["ok"]
            b_only, c_only = int((a & ~b).sum()), int((~a & b).sum())
            diff = float(a.mean() - b.mean())
            pair[cond] = dict(
                arm_a=arm_a, arm_b=arm_b, n=n,
                k_a=int(a.sum()), k_b=int(b.sum()), diff=diff,
                b_only=b_only, c_only=c_only,
                p=mcnemar_exact(b_only, c_only),
                wald=wald_ci(diff, b_only, c_only, n),
                boot=bootstrap_ci(a, b),
            )

        calib = {}
        for arm in C2_ARMS:
            conf, correct = _conf_array(scored, arm)
            calib[arm] = dict(
                n=len(conf), mean_conf=float(conf.mean()),
                auc=auc_conf(conf, correct), ece=ece(conf, correct),
                brier=brier_score(conf, correct), obs_acc=float(correct.mean()),
                bins=calibration_bins(conf, correct),
            )

        confusion = {arm: _confusion(scored, arm, task) for arm in C2_ARMS}

        # OCR fidelity, C2 (all records with OCR output) -- canonical numbers.json values
        nb = numbers["datasets"][key]
        fidelity_c2 = {
            "all": nb["ocr"]["c2_cells"],
            "numeric": nb["ocr"]["c2_numeric"],
        }

        # original first-100 grid subset (paired C4 vs C2), recomputed and from numbers.json
        grid = scored[scored["grid_cells_total"].notna()].reset_index(drop=True)
        g_form = _ok_flags(grid, "jev_form")
        g_grid = _ok_flags(grid, "jev_grid")
        c2_only = int((g_form & ~g_grid).sum())   # C2 correct, C4 wrong
        c4_only = int((~g_form & g_grid).sum())   # C4 correct, C2 wrong
        m = len(grid)
        diff_c4_minus_c2 = float(g_grid.mean() - g_form.mean())
        subset = dict(
            n=m, c2_k=int(g_form.sum()), c4_k=int(g_grid.sum()),
            diff_c4_minus_c2=diff_c4_minus_c2, c2_only=c2_only, c4_only=c4_only,
            p=mcnemar_exact(c4_only, c2_only),
            wald=wald_ci(diff_c4_minus_c2, c4_only, c2_only, m),
        )

        datasets[key] = dict(
            meta=meta, n=n, majority=float(nb["majority_baseline"]),
            label_counts={str(k): int(v) for k, v in scored["true_label"].value_counts().items()},
            arms=arms, pair=pair, calib=calib, confusion=confusion,
            fidelity_c2=fidelity_c2,
            fidelity_c4_full=grid_ocr[key],
            subset=subset, subset_pair=nb["c4_subset"]["pair"],
            layout=layout[f"JEV|{key}"],
            task_key=task_key[key],
        )

    repeats = load_json(REPEATS_JSON)
    for key, ds in datasets.items():
        r = repeats[key]
        for arm in ("jev_form", "llm_form"):
            s = r[arm]
            a = ds["arms"][arm]
            a["k"] = s["k"]; a["n"] = s["n"]; a["acc"] = s["acc"]; a["ci"] = (s["ci"][0], s["ci"][1])
            a["cost_per_1000"] = s["cost_per_1000"]
            a["ms_answered"] = np.asarray(s["ms"], dtype=float)
            a["answered"] = s["n"]; a["failures"] = 0
        sp = r["pair"]
        pp = ds["pair"]["C2"]
        pp["diff"] = sp["diff"]; pp["boot"] = (sp["boot"][0], sp["boot"][1]); pp["p"] = sp["p"]
        pp["b_only"] = sp["b_only"]; pp["c_only"] = sp["c_only"]; pp["n"] = sp["n"]
        pp["k_a"] = r["jev_form"]["k"]; pp["k_b"] = r["llm_form"]["k"]
        for arm in ("jev_form", "llm_form"):
            ds["calib"][arm].update(r["calib"][arm])
    return dict(datasets=datasets, numbers=numbers, posthoc=posthoc,
                final=final, layout=layout, grid_ocr=grid_ocr)


# --------------------------------------------------------------------------- #
# cross-checks (recomputed vs canonical JSONs)
# --------------------------------------------------------------------------- #
ARM_TO_NUMBERS = {
    "jev_native": "jev_c1", "llm_native": "llm_c1", "jev_form": "jev_c2",
    "llm_form": "llm_c2", "jev_rebuilt": "jev_c3", "jev_grid": "jev_c4",
}


def _cmp(source: str, name: str, ours, theirs, tol=1e-9) -> dict:
    try:
        diff = float(ours) - float(theirs)
        ok = abs(diff) <= tol
    except Exception:
        diff, ok = float("nan"), False
    return dict(source=source, name=name, ours=ours, theirs=theirs, diff=diff, ok=ok)


def crosschecks(data: dict) -> list[dict]:
    """Compare every recomputed plotted quantity with the canonical JSON sources."""
    num, post, final, layout = data["numbers"], data["posthoc"], data["final"], data["layout"]
    checks: list[dict] = []
    task_key = {"nhamcs": "triage", "mimic_ed": "disposition", "icu_xlsx": "icu_mortality"}
    for key, d in data["datasets"].items():
        nb = num["datasets"][key]
        checks.append(_cmp("numbers.json", f"{key}.majority", d["majority"], nb["majority_baseline"], 1e-12))
        for arm, akey in ARM_TO_NUMBERS.items():
            na = nb["arms"][akey]
            ar = d["arms"][arm]
            checks.append(_cmp("numbers.json", f"{key}.{akey}.correct", ar["k"], na["correct"], 0))
            checks.append(_cmp("numbers.json", f"{key}.{akey}.accuracy", ar["acc"], na["accuracy"], 1e-12))
            checks.append(_cmp("numbers.json", f"{key}.{akey}.wilson_lo", ar["ci"][0], na["wilson_ci95"][0], 1e-9))
            checks.append(_cmp("numbers.json", f"{key}.{akey}.wilson_hi", ar["ci"][1], na["wilson_ci95"][1], 1e-9))
            checks.append(_cmp("numbers.json", f"{key}.{akey}.cost_total", ar["cost_total"], na["cost_total_usd"], 1e-9))
            if arm != "jev_grid":
                # numbers.json divides the C4 (grid) arm cost and latency by answered rows
                checks.append(_cmp("numbers.json", f"{key}.{akey}.cost_per_1000", ar["cost_per_1000"], na["cost_per_1000_usd"], 1e-9))
                ms = ar["ms_all"]
                checks.append(_cmp("numbers.json", f"{key}.{akey}.latency_median", float(np.median(ms)), na["latency_ms"]["median"], 1e-6))
                checks.append(_cmp("numbers.json", f"{key}.{akey}.latency_p95", float(np.percentile(ms, 95)), na["latency_ms"]["p95"], 1e-6))
        for cond, ref_key in (("C2", "c2_primary_jev_vs_llm"), ("C1", "c1_secondary_jev_vs_llm")):
            pr, ref = d["pair"][cond], nb["comparisons"][ref_key]
            checks.append(_cmp("numbers.json", f"{key}.{cond}.delta_pp", pr["diff"] * 100, ref["delta_pp"], 1e-9))
            checks.append(_cmp("numbers.json", f"{key}.{cond}.mcnemar_p", pr["p"], ref["mcnemar_exact_p"], 1e-12))
            checks.append(_cmp("numbers.json", f"{key}.{cond}.b_only", pr["c_only"], ref["discordant_b_only"], 0))
            checks.append(_cmp("numbers.json", f"{key}.{cond}.c_only", pr["b_only"], ref["discordant_a_only"], 0))
            checks.append(_cmp("numbers.json", f"{key}.{cond}.boot_lo", pr["boot"][0] * 100, ref["diff_ci95_bootstrap_legacy"][0] * 100, 1e-9))
            checks.append(_cmp("numbers.json", f"{key}.{cond}.boot_hi", pr["boot"][1] * 100, ref["diff_ci95_bootstrap_legacy"][1] * 100, 1e-9))
        sp, ref = d["subset"], nb["c4_subset"]["pair"]
        checks.append(_cmp("numbers.json", f"{key}.subset.delta_pp", sp["diff_c4_minus_c2"] * 100, ref["delta_pp"], 1e-9))
        checks.append(_cmp("numbers.json", f"{key}.subset.mcnemar_p", sp["p"], ref["mcnemar_exact_p"], 1e-12))
        checks.append(_cmp("numbers.json", f"{key}.subset.ci_lo", sp["wald"][0] * 100, ref["diff_ci95_wald"][0] * 100, 1e-9))
        checks.append(_cmp("numbers.json", f"{key}.subset.ci_hi", sp["wald"][1] * 100, ref["diff_ci95_wald"][1] * 100, 1e-9))
        checks.append(_cmp("numbers.json", f"{key}.subset.n", sp["n"], ref["n"], 0))
        # calibration
        for arm, ref_arm in (("jev_form", "jev_form"), ("llm_form", "llm_form")):
            cal, ref_cal = d["calib"][arm], post["calibration"][task_key[key]][ref_arm]
            checks.append(_cmp("posthoc_results.json", f"{key}.{arm}.calib_n", cal["n"], ref_cal["n"], 0))
            checks.append(_cmp("posthoc_results.json", f"{key}.{arm}.ece", cal["ece"], ref_cal["ece_10bin"], 1e-9))
            checks.append(_cmp("posthoc_results.json", f"{key}.{arm}.auc", cal["auc"], ref_cal["auc"], 1e-9))
            checks.append(_cmp("posthoc_results.json", f"{key}.{arm}.brier", cal["brier"], ref_cal["brier"], 1e-9))
            checks.append(_cmp("posthoc_results.json", f"{key}.{arm}.mean_conf", cal["mean_conf"], ref_cal["mean_conf"], 1e-9))
        # full-sample layout effect
        lay = d["layout"]
        b, c = lay["c2_only"], lay["c4_only"]
        ours_delta = (c - b) / lay["n"] * 100
        checks.append(_cmp("layout_full.json", f"{key}.layout.delta_pp", ours_delta, lay["delta_c4_minus_c2_pp"], 0.06))
        checks.append(_cmp("layout_full.json", f"{key}.layout.mcnemar_p", mcnemar_exact(b, c), lay["mcnemar_p"], 1e-12))
    # final_experiments comparisons: delta_pp must equal (a_only - b_only)/n (rounded to 0.1)
    for comp, vals in final["comparisons"].items():
        recomputed_pp = (vals["a_only"] - vals["b_only"]) / vals["n"] * 100
        checks.append(_cmp("final_experiments.json", f"{comp}.delta_pp", recomputed_pp, vals["delta_pp"], 0.06))
    # grid OCR fidelity: rounds to the manuscript's 45.7 / 93.6 / 51.1%
    for key, target in (("nhamcs", 45.7), ("mimic_ed", 93.6), ("icu_xlsx", 51.1)):
        checks.append(_cmp("manuscript Table 4", f"{key}.grid_fidelity_pp", round(data["grid_ocr"][key]["rate"] * 100, 1), target, 1e-9))
    return checks


def summarise_crosschecks(checks: list[dict]) -> dict:
    bad = [c for c in checks if not c["ok"]]
    return dict(n=len(checks), mismatches=bad)


# --------------------------------------------------------------------------- #
# derived value tables (single source for the figures AND figures_data.md)
# --------------------------------------------------------------------------- #
def task_of(meta: dict, lang: str = "en") -> str:
    return meta["task_zh"] if lang == "zh" else meta["task_en"]


def fig2_accuracy_rows(data: dict) -> list[dict]:
    rows = []
    for meta in DATASETS:
        d = data["datasets"][meta["key"]]
        for arm, who in (("jev_form", "Jev"), ("llm_form", "LLM")):
            a = d["arms"][arm]
            rows.append(dict(task=meta["task_en"], short=meta["short"], who=who, arm=arm,
                             k=a["k"], n=a["n"], acc=a["acc"], ci=a["ci"], majority=d["majority"]))
    return rows


def fig2_forest_rows(data: dict) -> list[dict]:
    rows = []
    for meta in DATASETS:
        d = data["datasets"][meta["key"]]
        p = d["pair"]["C2"]
        rows.append(dict(cond="C2", label="Main comparison", task=meta["task_en"], short=meta["short"],
                         delta_pp=p["diff"] * 100, ci=(p["boot"][0] * 100, p["boot"][1] * 100),
                         b=p["c_only"], c=p["b_only"], p=p["p"], n=p["n"]))
    return rows

def fig3_run_rows(data: dict) -> list[dict]:
    rows = []
    for meta in DATASETS:
        key = meta["key"]
        d = data["datasets"][key]
        rep = data["final"]["jev_repeats"][key]
        llm = data["final"]["primary_llm_repeats"][key]
        gpt = data["final"]["second_llm_c2"][key]
        for i, v in enumerate(rep["per_repeat"], 1):
            rows.append(dict(task=meta["task_en"], series=f"Jev repeat {i}", value=v * 100))
        rows.append(dict(task=meta["task_en"], series="Jev original", value=d["arms"]["jev_form"]["acc"] * 100))
        rows.append(dict(task=meta["task_en"], series="LLM original", value=d["arms"]["llm_form"]["acc"] * 100))
        for i, v in enumerate((llm["r1"], llm["r2"]), 1):
            rows.append(dict(task=meta["task_en"], series=f"LLM repeat {i}", value=v * 100))
        if meta["key"] != "nhamcs":
            # the manuscript no longer reports the gpt-6 triage run; plot its three runs
            v9 = load_json(ANALYSIS_V9)
            v9key = {"mimic_ed": "disposition", "icu_xlsx": "icu_mortality"}[key]
            per_run = v9["part1"]["tasks"][v9key]["gpt6"]["per_run"].values()
            for i, run in enumerate(per_run):
                series = "gpt-6 original" if i == 0 else f"gpt-6 repeat {i}"
                rows.append(dict(task=meta["task_en"], series=series, value=run["accuracy_pct"]))
    return rows


def fig3_forest_rows(data: dict) -> list[dict]:
    """Four paired comparison types per task (A-D), each a genuine paired test.

    A Jev (original run) - LLM (original run): paper/analysis/numbers.json c2_primary_jev_vs_llm
    B Jev (first repeat) - LLM (first repeat):  final_experiments JEVnew_vs_llm6.1r1
    C gpt-6 - LLM (first repeat):              final_experiments gpt6_vs_llm6.1r1
    D gpt-6 - Jev (first repeat):               final_experiments gpt6_vs_JEVnew

    Rows: A and B cover all three tasks; C and D cover disposition and ICU mortality only
    (the gpt-6 triage run is no longer reported in the manuscript).
    """
    final = data["final"]
    rows = []
    labels = [
        ("A", "Jev (original run) − LLM (original run)", "Jev（原始运行）− LLM（原始运行）"),
        ("B", "Jev (first repeat) − LLM (first repeat)", "Jev（第 1 次重复）− LLM（第 1 次重复）"),
        ("C", "gpt-6 − LLM (first repeat)", "gpt-6 − LLM（第 1 次重复）"),
        ("D", "gpt-6 − Jev (first repeat)", "gpt-6 − Jev（第 1 次重复）"),
    ]
    comp_keys = [None, "JEVnew_vs_llm6.1r1", "gpt6_vs_llm6.1r1", "gpt6_vs_JEVnew"]
    for meta in DATASETS:
        d = data["datasets"][meta["key"]]
        for (tag, label_en, label_zh), comp in zip(labels, comp_keys):
            if comp is None:
                pr = d["pair"]["C2"]
                rows.append(dict(task=meta["task_en"], tag=tag, comparison=label_en,
                                 comparison_zh=label_zh, comp="paperC2",
                                 delta_pp=pr["diff"] * 100,
                                 ci=(pr["wald"][0] * 100, pr["wald"][1] * 100), p=pr["p"],
                                 a_only=pr["b_only"], b_only=pr["c_only"], n=pr["n"]))
            else:
                if tag in ("C", "D") and meta["key"] == "nhamcs":
                    continue  # gpt-6 triage is not reported in the manuscript
                v = final["comparisons"][f"{comp}|{meta['key']}"]
                se = math.sqrt(v["a_only"] + v["b_only"]) / v["n"] * 100
                rows.append(dict(task=meta["task_en"], tag=tag, comparison=label_en,
                                 comparison_zh=label_zh, comp=comp,
                                 delta_pp=v["delta_pp"],
                                 ci=(v["delta_pp"] - Z95 * se, v["delta_pp"] + Z95 * se),
                                 p=v["p"], a_only=v["a_only"], b_only=v["b_only"], n=v["n"]))
    return rows


def fig4_cost_rows(data: dict) -> list[dict]:
    rows = []
    for meta in DATASETS:
        d = data["datasets"][meta["key"]]
        for who, arm1, arm2 in (("Jev", "jev_native", "jev_form"), ("LLM", "llm_native", "llm_form")):
            for cond, arm in (("C1", arm1), ("C2", arm2)):
                a = d["arms"][arm]
                rows.append(dict(task=meta["task_en"], who=who, cond=cond, arm=arm,
                                 cost_per_1000=a["cost_per_1000"], acc=a["acc"],
                                 majority=d["majority"]))
    return rows


def fig4_latency_rows(data: dict) -> list[dict]:
    rows = []
    for meta in DATASETS:
        d = data["datasets"][meta["key"]]
        num = data["numbers"]["datasets"][meta["key"]]["arms"]
        for who, arm, nkey in (("Jev", "jev_form", "jev_c2"), ("LLM", "llm_form", "llm_c2")):
            vals = d["arms"][arm]["ms_answered"] / 1000.0
            ref = num[nkey]["latency_ms"]
            rows.append(dict(task=meta["task_en"], who=who, n=len(vals), values=vals,
                             median=float(np.median(vals)), p95=float(np.percentile(vals, 95)),
                             canonical_median=ref["median"] / 1000.0, canonical_p95=ref["p95"] / 1000.0))
    return rows


def fig5_fidelity_rows(data: dict) -> list[dict]:
    rows = []
    for meta in DATASETS:
        d = data["datasets"][meta["key"]]
        num = data["numbers"]["datasets"][meta["key"]]["ocr"]
        rows.append(dict(task=meta["task_en"], short=meta["short"],
                         c2_all=num["c2_cells"]["rate"], c2_numeric=num["c2_numeric"]["rate"],
                         c4_full=d["fidelity_c4_full"]["rate"], c4_subset=num["c4_cells"]["rate"],
                         c2_cells=(num["c2_cells"]["recovered"], num["c2_cells"]["total"]),
                         c2_numeric_cells=(num["c2_numeric"]["recovered"], num["c2_numeric"]["total"]),
                         c4_full_cells=(d["fidelity_c4_full"]["recovered"], d["fidelity_c4_full"]["total"]),
                         c4_subset_cells=(num["c4_cells"]["recovered"], num["c4_cells"]["total"])))
    return rows


def fig5_layout_rows(data: dict) -> list[dict]:
    rows = []
    for meta in DATASETS:
        d = data["datasets"][meta["key"]]
        lay = d["layout"]
        full_ci = wald_ci(lay["delta_c4_minus_c2_pp"] / 100, lay["c4_only"], lay["c2_only"], lay["n"])
        sp = d["subset"]
        rows.append(dict(task=meta["task_en"], short=meta["short"], sample="full", n=lay["n"],
                         delta_pp=lay["delta_c4_minus_c2_pp"], ci=(full_ci[0] * 100, full_ci[1] * 100),
                         p=lay["mcnemar_p"], c2_acc=lay["c2_accuracy"], c4_acc=lay["c4_accuracy"]))
        rows.append(dict(task=meta["task_en"], short=meta["short"], sample="subset", n=sp["n"],
                         delta_pp=sp["diff_c4_minus_c2"] * 100, ci=(sp["wald"][0] * 100, sp["wald"][1] * 100),
                         p=sp["p"], c2_acc=sp["c2_k"] / sp["n"], c4_acc=sp["c4_k"] / sp["n"]))
    return rows


def fig6_rows(data: dict) -> list[dict]:
    rows = []
    for meta in DATASETS:
        d = data["datasets"][meta["key"]]
        for arm, who in (("jev_form", "Jev"), ("llm_form", "LLM")):
            c = d["calib"][arm]
            rows.append(dict(task=meta["task_en"], who=who, n=c["n"], ece=c["ece"], auc=c["auc"],
                             brier=c["brier"], mean_conf=c["mean_conf"], obs_acc=c["obs_acc"],
                             bins=c["bins"]))
    return rows


def figs1_rows(data: dict) -> list[dict]:
    """Confusion matrices with end-to-end row denominators.

    Cells exclude transport-failure rows (they have no prediction), but the row-normalised
    percentages and the row recall use the *scored* class sizes, i.e. a transport failure counts
    as a false negative (manuscript / SI Table S3 convention).
    """
    rows = []
    for meta in DATASETS:
        key = meta["key"]
        d = data["datasets"][key]
        counts = d["label_counts"]

        def class_count(label: int) -> int:
            if key == "nhamcs":
                name = next(name for name in counts if name.startswith(f"acuity_{label}_"))
                return counts[name]
            name = ({0: "discharged", 1: "admitted"} if key == "mimic_ed"
                    else {0: "survived", 1: "died"})[label]
            return counts[name]

        for arm, who in (("jev_form", "Jev"), ("llm_form", "LLM")):
            labels, cm, n = d["confusion"][arm]
            row_n = [class_count(lab) for lab in labels]
            recall = [float(cm[i, i] / row_n[i]) if row_n[i] else float("nan")
                      for i in range(len(labels))]
            rows.append(dict(task=meta["task_en"], short=meta["short"], who=who, labels=labels,
                             cm=cm, n=n, row_n=row_n, recall=recall,
                             task_key=key))
    return rows


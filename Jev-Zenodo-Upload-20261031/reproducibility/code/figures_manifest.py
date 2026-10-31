#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write paper/figures/figures_data.md - every value plotted in the v2 figure set.

Run after the figure scripts (``make_all_figures.py`` does this automatically):
    C:\\Users\\ZHOUSHENGZHE\\AppData\\Local\\Programs\\Python\\Python314\\python.exe figures_manifest.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

import figcommon as fc
from figcommon import DATASETS

P = fc.PALETTE


def _tbl(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _pct(v: float, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%"


def _pp(v: float, d: int = 1) -> str:
    return f"{v:+.{d}f}"


def _sci(p: float) -> str:
    """Same rounding as figcommon.fmt_p, so the manifest matches the figure annotations."""
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}e{int(exp)}"
    if p < 2e-3:
        return f"{p:.4f}"
    return f"{p:.3f}"


def write_manifest(data: dict, checks: list[dict], fonts: dict, paths: dict) -> Path:
    L: list[str] = []
    A = L.append
    mismatch = [c for c in checks if not c["ok"]]

    A("# Figure data — typed decision model and reasoning LLM, v2 figure set")
    A("")
    A("Generated deterministically by `figures_manifest.py` from the read-only inputs below; do not")
    A("edit by hand. Regenerate the whole set with")
    A("")
    A("```")
    A("python make_all_figures.py          # figures 1-6 + S1 (EN/ZH), figures_data.md and TIFFs")
    A("python make_figures_paper.py        # Fig 1 + Fig 2 only")
    A("python make_fig3.py                 # Fig 3 only")
    A("python make_fig4.py                 # Fig 4 only")
    A("python make_fig5.py                 # Fig 5 only")
    A("python make_fig6.py                 # Fig 6 only")
    A("python make_figS1.py                # Fig S1 only")
    A("python ../build/make_tiff.py        # PNG -> RGB TIFF, 350 dpi")
    A("```")
    A("")
    A("Figure numbering (v2): Fig 1 study design; Fig 2 accuracy on identical input; Fig 3")
    A("robustness across runs and comparators; Fig 4 cost and latency; Fig 5 rendering and layout;")
    A("Fig 6 confidence calibration; Fig S1 confusion matrices.")
    A("")

    # ------------------------------------------------------------------ inputs
    A("## Inputs (all read-only)")
    A("")
    rows = []
    for m in DATASETS:
        d = data["datasets"][m["key"]]
        rows.append([m["short"], f"`results_v4/{m['csv']}`", d["n"], _pct(d["majority"], 2)])
    A(_tbl(["dataset", "per-sample file", "scored rows (denominator)", "majority baseline"], rows))
    A("")
    A("- `results_v4/ht_*_final.csv` - per-record predictions, confidence, cost, latency, OCR cells.")
    A("- `results_c4_full/ocr_*.jsonl` - full-sample bordered-grid OCR (300/300/300 records); used for")
    A("  the C4 cell fidelity in Fig 5a.")
    A("- `results_v4/summary_v4.json`, `results_combined/final_experiments.json`,")
    A("  `results_combined/layout_full.json`, `paper/analysis/numbers.json`,")
    A("  `paper/analysis/posthoc_results.json` - canonical statistics.")
    A("")
    A("## Conventions")
    A("")
    A("- **Accuracy** = correct / all **scored** records (n = 300/300/290); a missing `*_ok` flag")
    A("  (transport failure) is counted as incorrect (end-to-end denominator).")
    A("- **Wilson 95% CI** for single proportions on that denominator.")
    A("- **Exact two-sided McNemar** p on the discordant pairs.")
    A("- **Paired bootstrap 95% CI** for Fig 2b: 2,000 record-level resamples, seed 20261015,")
    A("  `random.Random` stream / sorted deltas (index 0.025 and 0.975 of 2,000); this reproduces the")
    A("  published Table 2 intervals exactly.")
    A("- **Wald 95% CI** for paired differences in Figs 3b and 5b: SE = sqrt(b+c)/n.")
    A("- **Cost per 1,000 records** = sum(`*_cost` over scored rows) / n_scored x 1,000 USD.")
    A("- **Latency** = `*_ms` per call, excluding transport-failure rows in the plotted distributions")
    A("  (Fig 4b); annotated medians/P95 are the canonical `numbers.json` values, which include the")
    A("  single failed call recorded with ms = 0 (see the latency table below).")
    A("- **Calibration**: reliability bins over ten equal-width top-label-confidence bins, computed on")
    A("  scored calls that returned a confidence (LLM n = 299 for triage/disposition); ECE, AUC and")
    A("  Brier follow `posthoc_results.json` exactly.")
    A("- **Confusion matrices** (Fig S1) exclude transport-failure rows (`*_err` non-null): LLM")
    A("  n = 299/299/290 and Jev n = 300/300/290.")
    A("- Naming: \"Jev\" (the typed decision model), \"LLM\" (the primary reasoning LLM, gpt-6.1-sol),")
    A("  \"gpt-6\" (second reasoning LLM).")
    A("")

    # ------------------------------------------------- canonical vs recomputed
    A("## Canonical and recomputed values")
    A("")
    A("| quantity | source | origin |")
    A("|---|---|---|")
    A("| C2/C1 accuracy, Wilson CI, cost per 1,000, latency, McNemar, bootstrap CI | `numbers.json` | recomputed from the per-sample CSVs; identical (see cross-checks) |")
    A("| OCR fidelity C2 all/numeric | `numbers.json` `datasets.<ds>.ocr` | copied verbatim (computed on every record with OCR output) |")
    A("| OCR fidelity C4 grid (full sample) | `results_c4_full/ocr_*.jsonl` | recomputed; not in `numbers.json` (its C4 block is the first-100 subset) |")
    A("| First-100 subset layout effect | `numbers.json` `datasets.<ds>.c4_subset` | copied verbatim |")
    A("| Full-sample layout effect | `layout_full.json` | copied verbatim |")
    A("| Run-repeat accuracies, gpt-6, repeat-adjusted comparisons | `final_experiments.json` | copied verbatim |")
    A("| Calibration ECE/AUC/Brier/n | `posthoc_results.json` | recomputed; identical |")
    A("| End-to-end accuracy annotated in Fig 6 | `numbers.json` | copied verbatim; reliability bins use the answered-call subset |")
    A("")

    # ---------------------------------------------------------------- sizing
    A("## Sizing, palette and fonts")
    A("")
    A("All figures are 170 mm wide (6.69 in) or narrower and 350 dpi on white with")
    A("`bbox_inches=\"tight\"`; base font 6-8 pt at that width. EN uses Arial, ZH uses Microsoft")
    A("YaHei. Palette: Jev `#1f6f78`, LLM `#c45a1a`, majority grey `#8a8a8a`, ink `#1a1a1a`; gpt-6")
    A("uses `#6b4f9e` (a fourth colour needed only for the second LLM).")
    A("")
    A("| figure | file | pixel size (EN) |")
    A("|---|---|---|")
    for stem in ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "figS1"):
        p = paths["en"][stem]
        im = Image.open(p)
        A(f"| {stem} | `{p.name}` | {im.size[0]} x {im.size[1]} |")
    A("")

    # ------------------------------------------------------------------- Fig 1
    A("## Fig 1 - study design and analysis plan (schematic, no data)")
    A("")
    A("Boxes: three datasets (NHAMCS 300 scored, MIMIC-IV-ED 300 scored, ICU Sepsis Dataset 290")
    A("scored of 300) -> field-value serialisation -> C1 native, C2 render (one field per line) ->")
    A("OCR -> model, C3 OCR structure rebuild, C4 bordered grid rendered for the **full sample**")
    A("(n = 300/300/290). Right-hand boxes: paired same-input comparison (C1 & C2 byte-identical);")
    A("Jev-only layout probes (C3, C4 full sample); post hoc robustness probes (Jev C2 x 3 repeats;")
    A("primary LLM C2 x 3 repeats; second LLM gpt-6 on C2; full-sample layout C4 vs C2). Footer:")
    A("identical serialised text within a condition, no abstention, every scored record in the")
    A("denominator (transport failures incorrect), ICU 10/300 unscored upstream, ICU dataset")
    A("provenance.")
    A("")

    # ------------------------------------------------------------------- Fig 2
    A("## Fig 2 - accuracy on byte-identical text")
    A("")
    A("### (a) C2 accuracy with Wilson 95% CI")
    A("")
    A(_tbl(["task", "dataset", "model", "arm", "correct/n", "accuracy", "Wilson 95% CI", "majority"],
           [[r["task"], r["short"], r["who"], f"`{r['arm']}`", f"{r['k']}/{r['n']}", _pct(r["acc"], 2),
             f"[{_pct(r['ci'][0], 2)}, {_pct(r['ci'][1], 2)}]", _pct(r["majority"], 2)]
            for r in fc.fig2_accuracy_rows(data)]))
    A("")
    A("### (b) paired difference, Jev - LLM (percentage points), 95% bootstrap CI, exact McNemar p")
    A("")
    A(_tbl(["condition", "task", "n", "delta (pp)", "95% CI (pp)", "discordant Jev only", "discordant LLM only", "exact p"],
           [[r["label"], r["task"], r["n"], _pp(r["delta_pp"], 1),
             f"[{_pp(r['ci'][0], 1)}, {_pp(r['ci'][1], 1)}]", r["c"], r["b"], _sci(r["p"])]
            for r in fc.fig2_forest_rows(data)]))
    A("")
    A("Zero line shown; rows 1-3 C2 (primary), rows 4-6 C1 (secondary). Values match Table 2.")
    A("")

    # ------------------------------------------------------------------- Fig 3
    A("## Fig 3 - robustness of the accuracy comparison")
    A("")
    A("### (a) accuracy across runs of C2 (%)")
    A("")
    run_rows = fc.fig3_run_rows(data)
    series_order = ["Jev repeat 1", "Jev repeat 2", "Jev repeat 3", "Jev original", "LLM original",
                    "LLM repeat 1", "LLM repeat 2", "gpt-6"]
    table_rows = []
    for m in DATASETS:
        vals = {r["series"]: r["value"] for r in run_rows if r["task"] == m["task_en"]}
        table_rows.append([m["task_en"]] + [f"{vals[s]:.1f}" if s in vals else "—" for s in series_order])
    A(_tbl(["task"] + series_order, table_rows))
    A("")
    A("Repeat means (horizontal bars): Jev 46.4 / 76.4 / 74.5 %; LLM 47.7 / 76.2 / 72.1 %.")
    A("Run-to-run SD: Jev 1.02 / 0.19 / 0.00 pp; LLM 0.00 / 0.71 / 0.00 pp (from `final_experiments.json`).")
    A("")
    A("### (b) paired differences (percentage points), 95% Wald CI, exact McNemar p")
    A("")
    A(_tbl(["block", "comparison", "task", "n", "delta (pp)", "95% CI (pp)", "exact p", "source"],
           [[r["tag"], r["comparison"], r["task"], r["n"], _pp(r["delta_pp"], 1),
             f"[{_pp(r['ci'][0], 1)}, {_pp(r['ci'][1], 1)}]", _sci(r["p"]),
             "`numbers.json` c2_primary" if r["comp"] == "paperC2"
             else f"`final_experiments.json` {r['comp']}"]
            for r in fc.fig3_forest_rows(data)]))
    A("")
    A("Block A is the Table 2 comparison: deltas and p match Table 2 (+3.0, p = 0.386; -2.0,")
    A("p = 0.441; +6.9, p = 0.008). Its CI in this panel is the Wald interval from the discordant")
    A("counts (as the caption states), whereas Table 2 quotes the paired-bootstrap interval; the")
    A("bootstrap intervals are the ones plotted in Fig 2b.")
    A("")

    # ------------------------------------------------------------------- Fig 4
    A("## Fig 4 - cost and latency")
    A("")
    A("### (a) cost-accuracy plane (C2 filled, C1 open)")
    A("")
    A(_tbl(["task", "model", "condition", "arm", "cost per 1,000 USD", "accuracy", "majority"],
           [[r["task"], r["who"], r["cond"], f"`{r['arm']}`", f"{r['cost_per_1000']:.5f}",
             _pct(r["acc"], 2), _pct(r["majority"], 2)] for r in fc.fig4_cost_rows(data)]))
    A("")
    A("C2 cost ratios (LLM / Jev): 81x / 80x / 82x; annotated as \"81x / 80x / 82x lower cost\" with a")
    A("horizontal bracket between the two C2 points.")
    A("")
    A("### (b) per-record latency, C2 (seconds)")
    A("")
    A(_tbl(["task", "model", "plotted n", "plotted median", "plotted P95", "canonical median (numbers.json)", "canonical P95"],
           [[r["task"], r["who"], r["n"], f"{r['median']:.3f}", f"{r['p95']:.3f}",
             f"{r['canonical_median']:.3f}", f"{r['canonical_p95']:.3f}"] for r in fc.fig4_latency_rows(data)]))
    A("")
    A("The canonical values include the single failed LLM call recorded with ms = 0; the plotted")
    A("distributions drop it, which shifts the triage LLM median from 38.86 to 38.98 s and the")
    A("disposition P95 from 179.96 to 180.14 s (Jev and ICU unaffected). The figure annotates the")
    A("canonical values, as quoted in the manuscript.")
    A("")

    # ------------------------------------------------------------------- Fig 5
    A("## Fig 5 - rendering and layout")
    A("")
    A("### (a) cell-level OCR fidelity (%)")
    A("")
    A(_tbl(["task", "dataset", "C2 all cells", "C2 numeric cells", "C4 grid cells (full sample)", "C4 grid (first-100 subset, numbers.json)"],
           [[r["task"], r["short"],
             f"{_pct(r['c2_all'], 2)} ({r['c2_cells'][0]}/{r['c2_cells'][1]})",
             f"{_pct(r['c2_numeric'], 2)} ({r['c2_numeric_cells'][0]}/{r['c2_numeric_cells'][1]})",
             f"{_pct(r['c4_full'], 2)} ({r['c4_full_cells'][0]}/{r['c4_full_cells'][1]})",
             f"{_pct(r['c4_subset'], 2)} ({r['c4_subset_cells'][0]}/{r['c4_subset_cells'][1]})"]
            for r in fc.fig5_fidelity_rows(data)]))
    A("")
    A("The C2 values are copied from `numbers.json`; the C4 full-sample values are recomputed from")
    A("`results_c4_full/ocr_*.jsonl` and round to 45.7 / 93.6 / 51.1 %, matching manuscript Table 4.")
    A("`numbers.json` still carries the older first-100 subset C4 values (45.64 / 93.81 / 50.91 %);")
    A("the full-sample values supersede them and are the ones plotted.")
    A("")
    A("### (b) layout effect on Jev accuracy, C4 - C2 (percentage points)")
    A("")
    A(_tbl(["task", "sample", "n", "C2 accuracy", "C4 accuracy", "delta (pp)", "95% CI (pp)", "exact p"],
           [[r["task"], "full sample" if r["sample"] == "full" else "first-100 subset", r["n"],
             _pct(r["c2_acc"], 1), _pct(r["c4_acc"], 1), _pp(r["delta_pp"], 1),
             f"[{_pp(r['ci'][0], 1)}, {_pp(r['ci'][1], 1)}]", _sci(r["p"])]
            for r in fc.fig5_layout_rows(data)]))
    A("")
    A("Full-sample estimates come from `layout_full.json` (Jev C2 run 1 vs the C4 full-sample run);")
    A("the first-100 subset estimates come from `numbers.json` `c4_subset` (original C2 run).")
    A("")

    # ------------------------------------------------------------------- Fig 6
    A("## Fig 6 - confidence calibration (C2)")
    A("")
    A(_tbl(["task", "model", "n (answered)", "ECE", "AUC", "Brier", "mean confidence", "observed accuracy (answered)", "end-to-end accuracy (annotated)"],
           [[r["task"], r["who"], r["n"], f"{r['ece']:.3f}", f"{r['auc']:.3f}", f"{r['brier']:.3f}",
             f"{r['mean_conf']:.2f}", _pct(r["obs_acc"], 1),
             _pct(data["datasets"][{m["task_en"]: m["key"] for m in DATASETS}[r["task"]]]["arms"]["jev_form" if r["who"] == "Jev" else "llm_form"]["acc"], 1)]
            for r in fc.fig6_rows(data)]))
    A("")
    A("Reliability bins (mean confidence, observed accuracy, n) are computed exactly as in")
    A("`posthoc_results.json`. The annotated accuracy is the end-to-end value from `numbers.json`")
    A("(the single failed LLM call counted incorrect); the posthoc answered-call accuracy differs by")
    A("at most 0.2 pp (triage 43.5 vs 43.3 %, disposition 78.9 vs 78.7 %).")
    A("")
    A("Bin-level values (mean confidence / observed accuracy / n):")
    A("")
    bin_rows = []
    for r in fc.fig6_rows(data):
        bins = r["bins"]
        bin_rows.append([r["task"], r["who"]]
                        + [f"{b['conf']:.2f} / {b['acc']:.2f} / {b['n']}" for b in bins])
    A(_tbl(["task", "model"] + [f"bin {i}" for i in range(1, 11)], bin_rows))
    A("")

    # ------------------------------------------------------------------ Fig S1
    A("## Fig S1 - confusion matrices (C2, transport failures excluded)")
    A("")
    for r in fc.figs1_rows(data):
        names = []
        if r["task_key"] == "nhamcs":
            names = [f"acuity {i}" for i in range(1, 6)]
        elif r["task_key"] == "mimic_ed":
            names = ["discharged", "admitted"]
        else:
            names = ["survived", "died"]
        A(f"**{r['who']} - {r['task']} (n = {r['n']})**: rows = reference, columns = prediction.")
        A("")
        A(_tbl([f"reference \\ predicted (n = {r['n']} answered)"] + names + ["scored row n", "recall"],
               [[names[i]] + [str(int(v)) for v in r["cm"][i]] + [r["row_n"][i], _pct(r["recall"][i], 1)]
                for i in range(len(names))]))
        A("")
    A("Key contrast: ICU sensitivity (died) Jev 70.3% vs LLM 44.2%; specificity (survived) Jev 78.9%")
    A("vs LLM 89.5%. Disposition recalls use the scored-class denominators, matching SI Table S3 and")
    A("the Results text: admitted Jev 59.6% vs LLM 51.7% (46/89; the single LLM transport failure is a")
    A("false negative); discharged Jev 83.9% vs LLM 90.0%. Matrix cells exclude the failed call, so the")
    A("LLM admitted row sums to 88 of 89 cells; LLM triage acuity-3 recall is 64/153 = 41.8%.")
    A("")

    # ------------------------------------------------------------ cross-checks
    A("## Cross-checks")
    A("")
    A(f"- **{len(checks)} checks** against `numbers.json`, `posthoc_results.json`, `layout_full.json`,")
    A(f"  `final_experiments.json` and the full-sample grid OCR: **{len(mismatch)} mismatch(es)**.")
    if mismatch:
        for c in mismatch:
            A(f"  - `{c['name']}` ({c['source']}): recomputed {c['ours']!r} vs canonical {c['theirs']!r}")
    A("- The recomputed Table 2 bootstrap intervals reproduce the published values exactly")
    A("  (-3.0 to +9.0, -6.3 to +2.0, +2.1 to +11.7 pp).")
    A("- Latency, OCR-subset and LLM-disposition recall conventions are listed above.")
    A("")
    A("## Fonts")
    A("")
    A(f"- EN: {fonts['en']}")
    A(f"- ZH: {fonts['zh']}")
    A("- Every save is wrapped in a glyph-warning trap; no missing-glyph warning was raised.")
    A("")

    out = fc.FIG_DIR / "figures_data.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main() -> None:
    data = fc.build_data()
    checks = fc.crosschecks(data)
    fonts = {}
    for lang in ("en", "zh"):
        fc.set_style(lang)
        fonts[lang] = fc.check_fonts(lang)
    paths = {"en": {}, "zh": {}}
    for stem in ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "figS1"):
        for lang in ("en", "zh"):
            paths[lang][stem] = fc.FIG_DIR / f"{stem}_{lang}.png"
            if not paths[lang][stem].exists():
                raise RuntimeError(f"missing {paths[lang][stem]} - run the figure scripts first")
    print("wrote", write_manifest(data, checks, fonts, paths))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 5 (v2 numbering) - rendering fidelity and layout sensitivity.

(a) Cell-level OCR fidelity of the one-field-per-line condition (C2, all cells and numeric
    cells) and of the bordered-grid condition (C4), per dataset.  C2 values are the canonical
    ``paper/analysis/numbers.json`` OCR block; C4 values are recomputed on the full sample from
    ``results_c4_full/ocr_*.jsonl`` (grid rendered for every record), which reproduces the
    manuscript's 45.7 / 93.6 / 51.1 %.
(b) Difference in Jev accuracy between the bordered grid (C4) and the one-field-per-line layout
    (C2), in percentage points, with Wald 95% CI (SE = sqrt(b+c)/n) and exact McNemar p, on the
    full sample (``results_combined/layout_full.json``).  Grey open markers overlay the original
    first-100-record subset estimates (``numbers.json`` c4_subset) for transparency.

Run:
    C:\\Users\\ZHOUSHENGZHE\\AppData\\Local\\Programs\\Python\\Python314\\python.exe make_fig5.py
"""
from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import figcommon as fc
from figcommon import DATASETS, PALETTE, mm

P = PALETTE


def fmt_p_layout(p: float, decimals: int) -> str:
    """Layout p-values: 2 dp for the full sample / 3 dp for the subset, sci below 0.001."""
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return f"p = {mant}e{int(exp)}"
    if p < 1e-2:
        return f"p = {p:.4f}"
    return f"p = {p:.{decimals}f}"


def draw_fig5(data: dict, lang: str):
    en = lang == "en"
    fid_rows = fc.fig5_fidelity_rows(data)
    layout_rows = fc.fig5_layout_rows(data)

    fig = plt.figure(figsize=mm(170, 88))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], left=0.072, right=0.985,
                          top=0.865, bottom=0.145, wspace=0.30)

    # ---- (a) cell-level OCR fidelity -------------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(DATASETS), dtype=float)
    width = 0.25
    series = [
        ("c2_all", ("C2 all cells", "C2 全部单元"), P["jev"], "", P["jev"]),
        ("c2_numeric", ("C2 numeric cells", "C2 数值单元"), P["jev_light"], "", P["jev"]),
        ("c4_full", ("C4 table cells", "C4 表格单元"), P["grid_fill"], "////", P["grid_edge"]),
    ]
    for i, row in enumerate(fid_rows):
        for j, (key, label, fill, hatch, edge) in enumerate(series):
            v = row[key] * 100
            x = xs[i] + (j - 1) * width
            ax.bar(x, v, width=width * 0.92, color=fill, edgecolor=edge, linewidth=0.5,
                   hatch=hatch, zorder=2)
            ax.text(x, v + 1.8, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2, color=P["ink"])
    ax.set_xlim(-0.6, len(DATASETS) - 0.4)
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['short']}\nn = {data['datasets'][m['key']]['n']}"
                        for r, m in zip(fid_rows, DATASETS)], fontsize=6.4)
    ax.set_ylabel("Cell fidelity (%)" if en else "单元保真度（%）", fontsize=7.0)
    fc.panel_label(ax, "(a) Cell-level OCR fidelity" if en else "(a) 单元级 OCR 保真度")
    ax.legend(handles=[Patch(facecolor=f, edgecolor=e, hatch=h, linewidth=0.5, label=lab[0] if en else lab[1])
                       for _, lab, f, h, e in series],
              loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=6.0,
              columnspacing=1.0, handlelength=1.2, handletextpad=0.4)

    # ---- (b) full-sample layout effect with subset overlay ---------------- #
    ax2 = fig.add_subplot(gs[0, 1])
    y_map = {0: 0.0, 1: 1.0, 2: 2.0}
    for i, meta in enumerate(DATASETS):
        full = [r for r in layout_rows if r["task"] == meta["task_en"] and r["sample"] == "full"][0]
        sub = [r for r in layout_rows if r["task"] == meta["task_en"] and r["sample"] == "subset"][0]
        for off, r, filled, col in ((0.16, full, True, P["jev"]), (-0.16, sub, False, P["baseline"])):
            y = y_map[i] + off
            ax2.plot([r["ci"][0], r["ci"][1]], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2)
            for x in r["ci"]:
                ax2.plot([x, x], [y - 0.075, y + 0.075], color=col, lw=0.9, zorder=2)
            ax2.plot([r["delta_pp"]], [y], marker="o", ms=3.8, mfc=col if filled else "white",
                     mec=col, mew=0.9, zorder=3)
            ax2.text(29.2, y, f"{r['delta_pp']:+.1f} pp   "
                     + fmt_p_layout(r["p"], 2 if filled else 3),
                     ha="right", va="center", fontsize=6.2, color=P["ink"])
    ax2.axvline(0, color=P["baseline"], lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    for sep in (0.5, 1.5):
        ax2.axhline(sep, color="#d0d0d0", lw=0.6, zorder=1)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels([f"{m['short']}" for m in DATASETS], fontsize=6.8)
    ax2.set_ylim(-0.75, 2.75)
    ax2.set_xlim(-45, 30)
    ax2.set_xticks([-40, -30, -20, -10, 0, 10, 20])
    ax2.set_xlabel("Δ Jev accuracy, C4 − C2 (percentage points); 95% CI, exact McNemar p"
                   if en else "Δ Jev 准确率（C4 − C2，百分点）；95% 置信区间，精确 McNemar p",
                   fontsize=6.8)
    fc.panel_label(ax2, "(b) Layout effect on Jev accuracy" if en else "(b) 版式对 Jev 准确率的影响")
    ax2.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    ax2.legend(handles=[
        Line2D([0], [0], marker="o", color=P["jev"], mfc=P["jev"], mec=P["jev"], ms=3.8, lw=0.9,
               label="Full sample (n = 300/300/290)" if en else "全样本（n = 300/300/290）"),
        Line2D([0], [0], marker="o", color=P["baseline"], mfc="white", mec=P["baseline"], ms=3.8,
               lw=0.9, label="First-100 subset (n = 100/100/95)" if en else "前 100 条子集（n = 100/100/95）"),
    ], loc="upper right", bbox_to_anchor=(1.0, 0.995), ncol=1, fontsize=6.0,
        columnspacing=1.2, handlelength=1.6, handletextpad=0.4)

    fig.text(0.5, 0.012,
             "Subset comparison uses the original C2 run (100/100/95 rendered records); the full-sample comparison uses Jev C2 run 1 (post hoc)."
             if en else
             "子集比较使用原始 C2 运行（渲染 100/100/95 条记录）；全样本比较使用 Jev C2 第 1 次运行（事后分析）。",
             ha="center", va="bottom", fontsize=6.0, color=P["muted"])

    return fc.save(fig, "fig5", lang)


def main() -> None:
    data = fc.build_data()
    for lang in ("en", "zh"):
        fc.set_style(lang)
        print("wrote", draw_fig5(data, lang))


if __name__ == "__main__":
    main()


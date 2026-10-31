#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 3 (v2 numbering) - robustness of the accuracy comparison.

(a) Accuracy across runs on identical inputs: Jev three repeats plus the
    original Jev run; the primary LLM original run plus two repeats; the second reasoning LLM
    (gpt-6), one run on disposition and mortality only (the gpt-6 triage point is no longer
    reported).  Marker type and colour distinguish the run types; a legend names them and
    horizontal bars mark the repeat means.
(b) One forest plot of paired differences in accuracy (percentage points), 10 rows in four
    blocks: three tasks for A and B, disposition and ICU mortality only for the gpt-6 blocks
    C and D.  Every row is a genuine paired test with a Wald 95% CI from the
    discordant counts (SE = sqrt(b+c)/n) and the exact two-sided McNemar p:
      A  Jev (original run) - LLM (original run)   [numbers.json c2_primary_jev_vs_llm]
      B  Jev (first repeat) - LLM (first repeat)   [final_experiments JEVnew_vs_llm6.1r1]
      C  gpt-6              - LLM (first repeat)   [final_experiments gpt6_vs_llm6.1r1]
      D  gpt-6              - Jev (first repeat)    [final_experiments gpt6_vs_JEVnew]
    Block A reproduces Table 2 (+3.0 / -2.0 / +6.9 percentage points).

Run:
    C:\\Users\\ZHOUSHENGZHE\\AppData\\Local\\Programs\\Python\\Python314\\python.exe make_fig3.py
"""
from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figcommon as fc
from figcommon import DATASETS, PALETTE, mm, fmt_p

P = PALETTE

COMPARISONS = [
    ("A", "Jev (original run) − LLM (original run)", "Jev（原始运行）− LLM（原始运行）"),
    ("B", "Jev (first repeat) − LLM (first repeat)", "Jev（第 1 次重复）− LLM（第 1 次重复）"),
    ("C", "gpt-6 − LLM (first repeat)", "gpt-6 − LLM（第 1 次重复）"),
    ("D", "gpt-6 − Jev (first repeat)", "gpt-6 − Jev（第 1 次重复）"),
]
BLOCK_BASE = {"A": 14.4, "B": 9.6, "C": 4.8, "D": 0.0}   # y of the third (ICU) row of each block


def draw_fig3(data: dict, lang: str):
    en = lang == "en"
    run_rows = fc.fig3_run_rows(data)
    run_rows = [r for r in run_rows if not (r.get("series") == "gpt-6" and r.get("task") == DATASETS[0]["task_en"])]
    forest_rows = fc.fig3_forest_rows(data)

    fig = plt.figure(figsize=mm(170, 120))
    ax = fig.add_axes([0.175, 0.690, 0.810, 0.250])    # (a)
    fig.text(0.175, 0.962, "(a) Accuracy across runs" if en else "(a) 各次运行的准确率",
             ha="left", va="center", fontsize=8.0, fontweight="bold", color=P["ink"])
    axb = fig.add_axes([0.360, 0.115, 0.625, 0.415])   # (b)
    fig.text(0.020, 0.565, "(b) Paired differences in accuracy" if en else "(b) 准确率配对差值",
             ha="left", va="center", fontsize=8.0, fontweight="bold", color=P["ink"])

    # ---- (a) accuracy across runs ----------------------------------------- #
    task_names = [m["task_en"] for m in DATASETS]
    xs = {name: i for i, name in enumerate(task_names)}
    styles = {
        "Jev repeat 1": (P["jev"], "o", True, 3.4),
        "Jev repeat 2": (P["jev"], "o", True, 3.4),
        "Jev repeat 3": (P["jev"], "o", True, 3.4),
        "Jev original": (P["jev"], "D", False, 3.9),
        "LLM original": (P["llm"], "s", True, 3.6),
        "LLM repeat 1": (P["llm"], "^", True, 4.0),
        "LLM repeat 2": (P["llm"], "^", True, 4.0),
        "gpt-6 original": (P["gpt6"], "D", True, 3.8),
        "gpt-6 repeat 1": (P["gpt6"], "D", True, 3.8),
        "gpt-6 repeat 2": (P["gpt6"], "D", True, 3.8),
    }
    offsets = {
        "Jev repeat 1": -0.375, "Jev repeat 2": -0.335, "Jev repeat 3": -0.295,
        "Jev original": -0.190, "LLM original": -0.020,
        "LLM repeat 1": 0.140, "LLM repeat 2": 0.180,
        "gpt-6 original": 0.265, "gpt-6 repeat 1": 0.300, "gpt-6 repeat 2": 0.335,
    }
    for r in run_rows:
        col, marker, filled, ms = styles[r["series"]]
        ax.plot([xs[r["task"]] + offsets[r["series"]]], [r["value"]], marker=marker, ms=ms,
                mfc=col if filled else "white", mec=col, mew=0.8, ls="none", zorder=4)
    for meta in DATASETS:
        task = meta["task_en"]
        jev_rep = [r["value"] for r in run_rows if r["task"] == task and r["series"].startswith("Jev repeat")]
        llm_rep = [r["value"] for r in run_rows if r["task"] == task and r["series"].startswith("LLM repeat")]
        ax.plot([xs[task] - 0.405, xs[task] - 0.265], [np.mean(jev_rep)] * 2, color=P["jev"], lw=1.3, zorder=3)
        ax.plot([xs[task] + 0.105, xs[task] + 0.215], [np.mean(llm_rep)] * 2, color=P["llm"], lw=1.3, zorder=3)
    ax.set_ylim(40, 87)
    ax.set_yticks([40, 50, 60, 70, 80])
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([f"{fc.task_of(m, lang)}\n({m['short']})" for m in DATASETS], fontsize=6.6)
    ax.set_ylabel("Accuracy (%)" if en else "准确率（%）", fontsize=7.0)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    handles = [
        Line2D([0], [0], marker="o", color=P["jev"], mfc=P["jev"], mec=P["jev"], ms=3.4, lw=1.3,
               label="Jev repeat run" if en else "Jev 重复运行"),
        Line2D([0], [0], marker="D", color=P["jev"], mfc="white", mec=P["jev"], ms=3.9, lw=0,
               label="Jev original run" if en else "Jev 原始运行"),
        Line2D([0], [0], marker="s", color=P["llm"], mfc=P["llm"], mec=P["llm"], ms=3.6, lw=0,
               label="Primary LLM (gpt-6.1-sol) original" if en else "主 LLM（gpt-6.1-sol）原始运行"),
        Line2D([0], [0], marker="^", color=P["llm"], mfc=P["llm"], mec=P["llm"], ms=4.0, lw=1.3,
               label="Primary LLM repeat run" if en else "主 LLM 重复运行"),
        Line2D([0], [0], marker="D", color=P["gpt6"], mfc=P["gpt6"], mec=P["gpt6"], ms=4.0, lw=0,
               label="Second LLM gpt-6 (3 runs)" if en else "第二对照 gpt-6（3 次运行）"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1,
              fontsize=6.0, labelspacing=0.32, handlelength=1.5, handletextpad=0.5, borderaxespad=0.25)

    # ---- (b) single forest: A/B all tasks, C/D disposition + mortality ----- #
    # gpt-6 comparisons are only reported for disposition and ICU mortality, so blocks
    # C and D have two rows while A and B keep three; the blocks are laid out from the
    # bottom (D) upwards (A) with fixed row/header/block spacing.
    block_datasets = {"A": DATASETS, "B": DATASETS, "C": DATASETS[1:], "D": DATASETS[1:]}
    row_gap, header_off, block_gap = 1.0, 1.35, 1.35
    block_bottom, y_cursor = {}, 0.0
    for tag, _, _ in reversed(COMPARISONS):
        block_bottom[tag] = y_cursor
        y_cursor += (len(block_datasets[tag]) - 1) + header_off + block_gap
    y_top = y_cursor - block_gap + 0.75

    xlim = (-9.0, 30.0)
    axb.set_xlim(*xlim)
    axb.set_ylim(-0.7, y_top)
    axb.set_xticks([-5, 0, 5, 10, 15])
    axb.axvline(0, color=P["baseline"], lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    axb.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    row_labels, row_ys = [], []
    for tag, label_en, label_zh in COMPARISONS:
        block_rows = [r for r in forest_rows if r["tag"] == tag]
        col = P["jev"] if tag in ("A", "B") else P["gpt6"]
        marker = "o" if tag in ("A", "B") else "s"
        metas = block_datasets[tag]
        for j, meta in enumerate(reversed(metas)):
            r = [x for x in block_rows if x["task"] == meta["task_en"]][0]
            y = block_bottom[tag] + j
            row_ys.append(y)
            row_labels.append(fc.task_of(meta, lang))
            axb.plot([r["ci"][0], r["ci"][1]], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2)
            for x in r["ci"]:
                axb.plot([x, x], [y - 0.15, y + 0.15], color=col, lw=0.9, zorder=2)
            axb.plot([r["delta_pp"]], [y], marker=marker, ms=3.6, mfc=col, mec=col, zorder=3)
            axb.text(29.4, y, f"{r['delta_pp']:+.1f} pp    {fmt_p(r['p'])}", ha="right", va="center",
                     fontsize=6.1, color=P["ink"])
        # bold block header, left-aligned in the left margin, above the block's first row
        header_y = block_bottom[tag] + (len(metas) - 1) + header_off
        axb.text(-0.544, header_y, f"{tag}  " + (label_zh if lang == "zh" else label_en),
                 transform=axb.get_yaxis_transform(), ha="left", va="center", fontsize=6.3,
                 fontweight="bold", color=P["ink"], clip_on=False)
    axb.set_yticks(row_ys)
    axb.set_yticklabels(row_labels, fontsize=6.2)
    axb.set_xlabel("Δ accuracy (percentage points); 95% Wald CI from discordant counts; exact McNemar p"
                   if en else "Δ 准确率（百分点）；由不一致对计数计算的 95% Wald 置信区间；精确 McNemar p",
                   fontsize=6.7)
    return fc.save(fig, "fig3", lang)


def main() -> None:
    data = fc.build_data()
    for lang in ("en", "zh"):
        fc.set_style(lang)
        print("wrote", draw_fig3(data, lang))


if __name__ == "__main__":
    main()


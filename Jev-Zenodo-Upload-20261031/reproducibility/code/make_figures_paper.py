#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 1 (study-design schematic) and Figure 2 (single-run accuracy on identical input).

Run from anywhere:
    C:\\Users\\ZHOUSHENGZHE\\AppData\\Local\\Programs\\Python\\Python314\\python.exe make_figures_paper.py

Outputs: fig1_en.png / fig1_zh.png and fig2_en.png / fig2_zh.png, 350 dpi, white background,
bbox_inches="tight".  All numbers are recomputed from the read-only inputs by ``figcommon``;
the only canonical values copied verbatim are those already recorded in
``paper/analysis/numbers.json`` (see ``figures_data.md``).

Figure numbering follows the v2 manuscript: Fig 1 = design, Fig 2 = accuracy on identical
input, Fig 3 = robustness across runs, Fig 4 = cost and latency, Fig 5 = rendering/layout,
Fig 6 = confidence calibration, Fig S1 = confusion matrices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: F401  (used indirectly through figcommon frames)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import figcommon as fc
from figcommon import DATASETS, PALETTE, mm, pct, fmt_p

P = PALETTE


# --------------------------------------------------------------------------- #
# Figure 1 - study design and analysis plan (schematic, no data)
# --------------------------------------------------------------------------- #
def draw_fig1(lang: str):
    en = lang == "en"
    T = {
        "datasets": (
            [("NHAMCS", "ED triage", "n = 300"),
             ("ED Kaggle copy", "ED disposition", "n = 300"),
             ("ICU Sepsis", "ICU mortality", "n = 290")]
            if en else
            [("NHAMCS", "急诊分诊", "n = 300"),
             ("急诊 Kaggle 副本", "急诊处置", "n = 300"),
             ("ICU Sepsis", "ICU 院内死亡", "n = 290")]
        ),
        "serial": ("Text serialisation" if en else "文本序列化",
                   ["field name: value", "same text to every model"] if en
                   else ["“字段名: 取值”文本", "同一份文本提供给所有模型"]),
        "models": ("Models" if en else "模型",
                   ["Jev (decision model)" if en else "Jev（决策模型）",
                    "Primary LLM: gpt-6.1-sol" if en else "主 LLM：gpt-6.1-sol",
                    "Additional LLMs: DeepSeek, GLM" if en else "另外两个 LLM：DeepSeek、GLM",
                    "Second comparator: gpt-6" if en else "第二对照：gpt-6"]),
        "analysis": ("Analyses" if en else "分析",
                     ["accuracy: 3 repeats per model" if en else "准确率：每模型三次重复运行",
                      "error profile" if en else "错误谱",
                      "cost and latency; calibration" if en else "成本与延迟；置信度校准"]),
        "post": ("Post hoc checks" if en else "事后复核",
                 ["Jev and primary LLM, 3 repeats" if en else "Jev 与主 LLM 各 3 次重复",
                  "second LLM (gpt-6), 3 runs" if en else "第二 LLM（gpt-6），3 次运行"]),
        "notes": (
            "The same text was supplied to every model; no abstention option; every scored record in the denominator.",
            "Transport failures counted as incorrect. ICU: 10 of 300 records unscored (outcome code undefined upstream).",
        ) if en else (
            "同一份文本提供给所有模型；不提供弃答选项；所有已评分记录均计入分母。",
            "传输失败计为错误。ICU：300 条中 10 条因结局编码未定义而未计分。",
        ),
    }

    fig = plt.figure(figsize=mm(170, 96))
    ax = fig.add_axes([0.004, 0.004, 0.992, 0.992])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ink, muted = P["ink"], P["muted"]

    def box(x, y, w, h, title, lines, edge=None, fill="white", lw=0.85, fs=7.2):
        edge = ink if edge is None else edge
        fc._box(ax, x, y, w, h, fill, edge, lw=lw)
        ax.text(x + w / 2, y + h - 4.1, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=ink)
        for k, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 8.8 - k * 4.4, line, ha="center", va="center",
                    fontsize=fs - 0.5, color=muted)

    # datasets (left)
    ds_x, ds_w, ds_h = 1.0, 19.0, 12.0
    centres = [78.0, 63.0, 48.0]
    for (name, task, n), cy in zip(T["datasets"], centres):
        fc._box(ax, ds_x, cy - ds_h / 2, ds_w, ds_h, P["ds_fill"], P["ds_edge"], lw=0.75)
        ax.text(ds_x + ds_w / 2, cy + 2.9, name, ha="center", va="center", fontsize=7.3,
                fontweight="bold", color=ink)
        ax.text(ds_x + ds_w / 2, cy - 1.1, task, ha="center", va="center", fontsize=6.7, color=muted)
        ax.text(ds_x + ds_w / 2, cy - 4.6, n, ha="center", va="center", fontsize=6.6, color=muted)

    # text serialisation
    ser_x, ser_w, ser_y, ser_h = 24.5, 20.5, 50.0, 30.0
    box(ser_x, ser_y, ser_w, ser_h, T["serial"][0], T["serial"][1], lw=0.9)
    for cy, ty in zip(centres, [74.0, 65.0, 56.0]):
        fc._arrow(ax, (ds_x + ds_w, cy), (ser_x, ty), rad=0.03)

    # models
    mod_x, mod_w, mod_y, mod_h = 49.5, 24.5, 43.0, 40.0
    box(mod_x, mod_y, mod_w, mod_h, T["models"][0], T["models"][1], lw=0.95)
    fc._arrow(ax, (ser_x + ser_w, 65.0), (mod_x, 65.0))

    # analyses
    an_x, an_w, an_y, an_h = 78.5, 20.5, 58.0, 28.0
    box(an_x, an_y, an_w, an_h, T["analysis"][0], T["analysis"][1], lw=0.85)
    fc._arrow(ax, (mod_x + mod_w, 70.0), (an_x, 72.0))

    # post hoc checks
    po_x, po_w, po_y, po_h = 78.5, 20.5, 22.0, 26.0
    box(po_x, po_y, po_w, po_h, T["post"][0], T["post"][1], edge="#6e6e6e", lw=0.85)
    fc._arrow(ax, (an_x + an_w / 2, an_y), (po_x + po_w / 2, po_y + po_h), ls=(0, (3.0, 1.8)), lw=0.7)

    for k, line in enumerate(T["notes"]):
        ax.text(34.0, 8.0 - k * 4.2, line, ha="center", va="center", fontsize=6.4, color=muted)

    return fc.save(fig, "fig1", lang)


# --------------------------------------------------------------------------- #
# Figure 2 - single-run accuracy on identical input
# --------------------------------------------------------------------------- #
def draw_fig2(data: dict, lang: str):
    en = lang == "en"
    rows_a = fc.fig2_accuracy_rows(data)
    rows_b = fc.fig2_forest_rows(data)

    fig = plt.figure(figsize=mm(170, 100))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.08], left=0.215, right=0.985,
                          top=0.875, bottom=0.150, wspace=0.34, hspace=0.62)
    fig.text(0.215, 0.955, "(a) Repeat-averaged accuracy (primary estimate; identical input)" if en else "(a) 重复均值准确率（主估计；两个模型输入相同）",
             ha="left", va="center", fontsize=8.0, fontweight="bold", color=P["ink"])

    # ---- (a) C2 accuracy bars, one panel per task ------------------------- #
    for c, meta in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, c])
        d = data["datasets"][meta["key"]]
        xs = [0.0, 1.0]
        vals = [d["arms"]["jev_form"]["acc"] * 100, d["arms"]["llm_form"]["acc"] * 100]
        cis = [d["arms"]["jev_form"]["ci"], d["arms"]["llm_form"]["ci"]]
        cols = [P["jev"], P["llm"]]
        for x, v, (lo, hi), col in zip(xs, vals, cis, cols):
            ax.bar(x, v, width=0.62, color=col, edgecolor="white", linewidth=0.5, zorder=2)
            ax.errorbar(x, v, yerr=[[v - lo * 100], [hi * 100 - v]], fmt="none", ecolor="#2b2b2b",
                        elinewidth=0.7, capsize=2.0, capthick=0.7, zorder=3)
            ax.text(x, hi * 100 + 2.0, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2, color=P["ink"])
        mb = d["majority"] * 100
        ax.axhline(mb, color=P["baseline"], lw=0.9, ls=(0, (3.2, 2.0)), zorder=4)
        ax.set_xlim(-0.62, 1.62)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"Jev\n(n = {d['n']})", f"LLM\n(n = {d['n']})"], fontsize=6.2)
        task = fc.task_of(meta, lang)
        fc.panel_label(ax, f"{meta['short']} · {task}")
        if c == 0:
            ax.set_ylabel("Accuracy (%)" if en else "准确率（%）", fontsize=7.0)
        else:
            ax.set_yticklabels([])

    # ---- (b) forest of paired differences --------------------------------- #
    axb = fig.add_subplot(gs[1, :])
    ypos = np.arange(len(rows_b))
    for i, r in enumerate(rows_b):
        axb.plot([r["ci"][0], r["ci"][1]], [i, i], color=P["ink"], lw=0.9, solid_capstyle="butt", zorder=2)
        for x in r["ci"]:
            axb.plot([x, x], [i - 0.13, i + 0.13], color=P["ink"], lw=0.9, zorder=2)
        axb.plot([r["delta_pp"]], [i], marker="s", ms=3.6, color=P["jev"], mec="white", mew=0.5, zorder=3)
        axb.text(17.7, i, f"{r['delta_pp']:+.1f}    {fmt_p(r['p'])}", ha="right", va="center",
                 fontsize=6.4, color=P["ink"])
    axb.axvline(0, color=P["baseline"], lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    axb.axhline(2.5, color="#d0d0d0", lw=0.6, zorder=1)
    axb.set_yticks(ypos)
    labels = []
    for r in rows_b:
        task_short = {"ED triage": ("ED triage", "急诊分诊"),
                      "ED disposition": ("ED disposition", "急诊处置"),
                      "ICU mortality": ("ICU mortality", "ICU 院内死亡")}[r["task"]]
        if lang == "zh" and r["label"] == "Main comparison":
            cond = "主要比较"
        else:
            cond = r["label"].replace("(primary)", "（主要）" if lang == "zh" else "(primary)")
            if lang == "zh":
                cond = cond.replace("(secondary)", "（次要）")
        labels.append(f"{cond} · " + (task_short[1] if lang == "zh" else task_short[0]))
    axb.set_yticklabels(labels, fontsize=6.2)
    axb.set_ylim(len(rows_b) - 0.55, -0.55)
    axb.set_xlim(-12.5, 18.2)
    axb.set_xticks([-10, -5, 0, 5, 10])
    axb.set_xlabel("Δ accuracy, Jev − LLM (percentage points); 95% paired-bootstrap CI"
                   if en else "Δ 准确率（Jev − LLM，百分点）；95% 配对自助法置信区间", fontsize=7.0)
    fc.panel_label(axb, "(b) Paired difference in accuracy" if en else "(b) 准确率配对差值")
    axb.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)

    # shared legend for (a), aligned with the group heading
    handles = [Patch(facecolor=P["jev"], label="Jev"), Patch(facecolor=P["llm"], label="LLM"),
               Line2D([0], [0], color=P["baseline"], ls=(0, (3.2, 2.0)), lw=0.9,
                      label="majority baseline" if en else "多数类基线")]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.985, 1.0),
               fontsize=6.4, columnspacing=1.6, handlelength=1.4)

    return fc.save(fig, "fig2", lang)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    data = fc.build_data()
    checks = fc.crosschecks(data)
    summary = fc.summarise_crosschecks(checks)
    print(f"cross-checks: {summary['n']} checked, {len(summary['mismatches'])} mismatch(es)")
    for m in summary["mismatches"]:
        print("  MISMATCH", m)
    for lang in ("en", "zh"):
        fc.set_style(lang)
        print("font:", fc.check_fonts(lang))
        print("wrote", draw_fig1(lang))
        print("wrote", draw_fig2(data, lang))


if __name__ == "__main__":
    main()




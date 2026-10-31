#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 6 (v2 numbering) - confidence calibration.

Reliability diagrams for the primary condition (C2), one panel per task: observed accuracy
against mean top-label confidence in ten equal-width confidence bins for Jev and the LLM; the
dashed diagonal is perfect calibration.  Bin means are computed on scored calls with a returned
confidence, exactly as in ``paper/analysis/posthoc_results.json`` (LLM n = 299 for triage and
disposition; no failures for Jev or ICU).  Annotations give ECE, AUC and Brier from that block
plus the mean confidence and the end-to-end accuracy from ``paper/analysis/numbers.json``
(the end-to-end denominator counts the single failed LLM call as incorrect).

Run:
    C:\\Users\\ZHOUSHENGZHE\\AppData\\Local\\Programs\\Python\\Python314\\python.exe make_fig6.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figcommon as fc
from figcommon import DATASETS, PALETTE, mm

P = PALETTE


def draw_fig6(data: dict, lang: str):
    en = lang == "en"
    rows = fc.fig6_rows(data)
    fig = plt.figure(figsize=mm(170, 92))
    positions = [(0.045, 0.355, 0.272, 0.565), (0.379, 0.355, 0.272, 0.565), (0.713, 0.355, 0.272, 0.565)]
    letters = ["(a)", "(b)", "(c)"]

    for c, meta in enumerate(DATASETS):
        ax = fig.add_axes(list(positions[c]))
        d = data["datasets"][meta["key"]]
        ax.plot([0, 1], [0, 1], color=P["baseline"], lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
        for arm, who, col in (("jev_form", "Jev", P["jev"]), ("llm_form", "LLM", P["llm"])):
            r = [x for x in rows if x["task"] == meta["task_en"] and x["who"] == who][0]
            bins = r["bins"]
            xs = [b["conf"] for b in bins]
            ys = [b["acc"] for b in bins]
            ns = [b["n"] for b in bins]
            sizes = [2.2 + 2.8 * (n ** 0.5) / (max(ns) ** 0.5) for n in ns]
            ax.plot(xs, ys, color=col, lw=0.8, alpha=0.85, zorder=3)
            ax.scatter(xs, ys, s=[s ** 2 for s in sizes], color=col, edgecolor="white", linewidth=0.4,
                       zorder=4)
        ax.set_xlim(0, 1.03)
        ax.set_ylim(0, 1.03)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        task = fc.task_of(meta, lang)
        fc.panel_label(ax, f"{letters[c]} {task} (n = {d['calib']['jev_form']['n']}/{d['calib']['llm_form']['n']})")
        ax.grid(color="#eeeeee", lw=0.5, zorder=0)
        if c == 0:
            ax.set_ylabel("Observed accuracy" if en else "观测准确率", fontsize=6.9)
        if c == 1:
            ax.set_xlabel("Mean top-label confidence" if en else "平均最高标签置信度", fontsize=6.9)
        # annotations (ECE/AUC/Brier/mean conf/accuracy) in the lower-right corner
        jev = [x for x in rows if x["task"] == meta["task_en"] and x["who"] == "Jev"][0]
        llm = [x for x in rows if x["task"] == meta["task_en"] and x["who"] == "LLM"][0]
        acc_jev = d["arms"]["jev_form"]["acc"] * 100
        acc_llm = d["arms"]["llm_form"]["acc"] * 100
        txt = (
            f"Jev  ECE {jev['ece']:.3f}  AUC {jev['auc']:.3f}  Brier {jev['brier']:.3f}\n"
            f"     conf {jev['mean_conf']:.2f} → acc {acc_jev:.1f}%\n"
            f"LLM  ECE {llm['ece']:.3f}  AUC {llm['auc']:.3f}  Brier {llm['brier']:.3f}\n"
            f"     conf {llm['mean_conf']:.2f} → acc {acc_llm:.1f}%"
        )
        ax.text(0.0, -0.245, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.0, color=P["ink"], linespacing=1.22)

    handles = [
        Line2D([0], [0], marker="o", color=P["jev"], mfc=P["jev"], mec="white", ms=4.2, lw=0.8, label="Jev"),
        Line2D([0], [0], marker="o", color=P["llm"], mfc=P["llm"], mec="white", ms=4.2, lw=0.8, label="LLM"),
        Line2D([0], [0], color=P["baseline"], ls=(0, (3.0, 2.0)), lw=0.8,
               label="Perfect calibration" if en else "完美校准"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3,
               fontsize=6.2, columnspacing=1.6, handlelength=1.6)
    return fc.save(fig, "fig6", lang)


def main() -> None:
    data = fc.build_data()
    for lang in ("en", "zh"):
        fc.set_style(lang)
        print("wrote", draw_fig6(data, lang))


if __name__ == "__main__":
    main()

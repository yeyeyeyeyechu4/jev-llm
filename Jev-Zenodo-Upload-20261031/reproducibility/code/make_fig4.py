#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 4 (v2 numbering) - cost and latency.

(a) Mean accuracy against cost per 1,000 scored records (log x, USD) for Jev and the LLM
    for the primary comparison, with the majority-class baseline and a
    horizontal bracket marking the cost ratio (LLM / Jev).
(b) Per-record wall-clock latency distributions (box + individual points, log y, seconds) on
    C2.  Transport-failure calls are excluded from the distributions (``*_err`` non-null).

Annotated medians / P95 are the canonical values recorded in ``paper/analysis/numbers.json``
(which include the single failed call recorded with ms = 0); the plotted distributions drop
that call, a difference of <= 0.2 s documented in ``figures_data.md``.

Run:
    C:\\Users\\ZHOUSHENGZHE\\AppData\\Local\\Programs\\Python\\Python314\\python.exe make_fig4.py
"""
from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

import figcommon as fc
from figcommon import DATASETS, PALETTE, mm

P = PALETTE


def _jitter(n: int, width: float = 0.34) -> np.ndarray:
    """Deterministic horizontal jitter (golden-ratio sequence, no RNG)."""
    return ((np.arange(n) * 0.6180339887498949) % 1.0 - 0.5) * width


def draw_fig4(data: dict, lang: str):
    en = lang == "en"
    cost_rows = fc.fig4_cost_rows(data)
    lat_rows = fc.fig4_latency_rows(data)

    fig = plt.figure(figsize=mm(170, 130))
    gs = fig.add_gridspec(2, 3, left=0.072, right=0.988, top=0.885, bottom=0.128,
                          wspace=0.27, hspace=0.80)
    fig.text(0.072, 0.962, "(a) Cost–accuracy plane" if en else "(a) 成本–准确率平面",
             ha="left", va="center", fontsize=8.0, fontweight="bold", color=P["ink"])
    fig.text(0.072, 0.495, "(b) Latency per call" if en else "(b) 每次调用延迟",
             ha="left", va="center", fontsize=8.0, fontweight="bold", color=P["ink"])

    # ---- (a) cost-accuracy plane ----------------------------------------- #
    for c, meta in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, c])
        ax.set_xscale("log")
        ax.set_xlim(0.012, 12.0)
        ax.set_ylim(33.0, 100.0)
        ax.set_yticks([40, 50, 60, 70, 80, 90])
        ax.set_xticks([0.03, 0.1, 0.3, 1.0, 3.0])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
        rows = [r for r in cost_rows if r["task"] == meta["task_en"]]
        mb = rows[0]["majority"] * 100
        ax.axhline(mb, color=P["baseline"], lw=0.8, ls=(0, (3.2, 2.0)), zorder=1)
        ax.text(0.98, mb + 1.6, ("majority " if en else "多数类 ") + f"{mb:.1f}%",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=5.7, color=P["baseline"])
        for who, col in (("Jev", P["jev"]), ("LLM", P["llm"])):
            c2 = [r for r in rows if r["who"] == who and r["cond"] == "C2"][0]
            ax.plot([c2["cost_per_1000"]], [c2["acc"] * 100], marker="s", ms=3.6, mfc=col,
                    mec="white", mew=0.4, zorder=5)
        jev_c2 = [r for r in rows if r["who"] == "Jev" and r["cond"] == "C2"][0]
        llm_c2 = [r for r in rows if r["who"] == "LLM" and r["cond"] == "C2"][0]
        ax.annotate("Jev", xy=(jev_c2["cost_per_1000"], jev_c2["acc"] * 100), xytext=(-7, 0),
                    textcoords="offset points", fontsize=6.6, fontweight="bold", color=P["jev"],
                    ha="right", va="center")
        ax.annotate("LLM", xy=(llm_c2["cost_per_1000"], llm_c2["acc"] * 100), xytext=(7, 0),
                    textcoords="offset points", fontsize=6.6, fontweight="bold", color=P["llm"],
                    ha="left", va="center")
        # horizontal bracket between the two C2 points marking the cost ratio
        xj, xl = jev_c2["cost_per_1000"], llm_c2["cost_per_1000"]
        yb = 93.5
        ax.plot([xj, xl], [yb, yb], color="#5a5a5a", lw=0.7, zorder=2)
        for x in (xj, xl):
            ax.plot([x, x], [yb - 1.3, yb + 1.3], color="#5a5a5a", lw=0.7, zorder=2)
        ratio = xl / xj
        ax.text(np.sqrt(xj * xl), yb + 1.8,
                (f"{ratio:.0f}× lower cost" if en else f"成本约为 1/{ratio:.0f}"),
                ha="center", va="bottom", fontsize=6.0, color="#3f3f3f")
        fc.panel_label(ax, f"{meta['short']} · {fc.task_of(meta, lang)}")
        if c == 0:
            ax.set_ylabel("Accuracy (%)" if en else "准确率（%）", fontsize=7.0)
        ax.set_xlabel("Cost per 1,000 records (USD, log)" if en else "每千条记录成本（美元，对数）",
                      fontsize=6.8, labelpad=1.5)


    # ---- (b) latency distributions --------------------------------------- #
    for c, meta in enumerate(DATASETS):
        ax = fig.add_subplot(gs[1, c])
        rows = [r for r in lat_rows if r["task"] == meta["task_en"]]
        jev, llm = rows[0], rows[1]
        for pos, row, col, light in ((0, jev, P["jev"], P["jev_light"]),
                                     (1, llm, P["llm"], "#e6b79c")):
            vals = row["values"]
            jit = _jitter(len(vals))
            ax.scatter(pos + jit, vals, s=1.6, color=col, alpha=0.35, lw=0, zorder=2)
            bp = ax.boxplot([vals], positions=[pos], widths=0.44, showfliers=False,
                            patch_artist=True, zorder=3,
                            boxprops=dict(facecolor=light, edgecolor=col, linewidth=0.8),
                            whiskerprops=dict(color=col, linewidth=0.8),
                            capprops=dict(color=col, linewidth=0.8),
                            medianprops=dict(color=P["ink"], linewidth=1.0))
            for b in bp["boxes"]:
                b.set_alpha(0.85)
        ax.set_yscale("log")
        ax.set_ylim(0.4, 900.0)
        ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(-0.65, 1.65)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Jev\n(n = {jev['n']})", f"LLM\n(n = {llm['n']})"], fontsize=6.2)
        ax.text(0.5, 0.975,
                (f"Jev  median {jev['canonical_median']:.3f} s · P95 {jev['canonical_p95']:.3f} s\n"
                 f"LLM  median {llm['canonical_median']:.2f} s · P95 {llm['canonical_p95']:.2f} s")
                if en else
                (f"Jev  中位数 {jev['canonical_median']:.3f} 秒 · P95 {jev['canonical_p95']:.3f} 秒\n"
                 f"LLM  中位数 {llm['canonical_median']:.2f} 秒 · P95 {llm['canonical_p95']:.2f} 秒"),
                transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=P["ink"],
                linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88))
        fc.panel_label(ax, f"{meta['short']} · {fc.task_of(meta, lang)}")
        if c == 0:
            ax.set_ylabel("Latency per call (s, log)" if en else "每次调用延迟（秒，对数）", fontsize=7.0)

    fig.text(0.53, 0.012,
             "Transport-failure calls excluded (Jev n = 300/300/290, LLM n = 299/299/290). "
             "Box = median and IQR; whiskers = 1.5 × IQR."
             if en else
             "已排除传输失败调用（Jev n = 300/300/290，LLM n = 299/299/290）。"
             "箱体为中位数与四分位距；须线为 1.5 倍四分位距。",
             ha="center", va="bottom", fontsize=6.0, color=P["muted"])

    return fc.save(fig, "fig4", lang)


def main() -> None:
    data = fc.build_data()
    for lang in ("en", "zh"):
        fc.set_style(lang)
        print("wrote", draw_fig4(data, lang))


if __name__ == "__main__":
    main()



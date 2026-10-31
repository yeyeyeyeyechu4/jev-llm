#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure S1 (v2 numbering) - confusion matrices for the primary condition (C2).

One row per task; within a row, left = Jev and right = LLM.  Rows of each matrix are the
reference label and columns the model prediction: triage 5x5 (acuity 1-5), disposition 2x2
(discharged/admitted), ICU 2x2 (survived/died).  Cell text = count and row-normalised
percentage; each row label also gives the row recall.  Transport-failure rows are excluded
(``*_err`` non-null), so the LLM matrices use n = 299/299/290 and the Jev matrices
n = 300/300/290; n is stated in each matrix title.

Run:
    C:\\Users\\ZHOUSHENGZHE\\AppData\\Local\\Programs\\Python\\Python314\\python.exe make_figS1.py
"""
from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import figcommon as fc
from figcommon import DATASETS, PALETTE, mm

P = PALETTE


def _cmap(hex_color: str, name: str):
    return LinearSegmentedColormap.from_list(name, ["#ffffff", hex_color], N=256)


def _tick_names(task_key: str, lang: str) -> list[str]:
    if task_key == "nhamcs":
        return [str(i) for i in range(1, 6)]
    if task_key == "mimic_ed":
        return ["discharged", "admitted"] if lang == "en" else ["出院", "入院"]
    return ["survived", "died"] if lang == "en" else ["存活", "死亡"]


def _row_names(task_key: str, lang: str) -> list[str]:
    if task_key == "nhamcs":
        return [f"acuity {i}" if lang == "en" else f"分级 {i}" for i in range(1, 6)]
    if task_key == "mimic_ed":
        return ["discharged", "admitted"] if lang == "en" else ["出院", "入院"]
    return ["survived", "died"] if lang == "en" else ["存活", "死亡"]


def draw_figS1(data: dict, lang: str):
    en = lang == "en"
    rows = fc.figs1_rows(data)
    fig = plt.figure(figsize=mm(170, 140))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.0, 1.0], left=0.205, right=0.985,
                          top=0.905, bottom=0.260, hspace=0.66, wspace=0.44)

    col_titles = {
        "nhamcs": ("ED triage", "急诊分诊"),
        "mimic_ed": ("ED disposition", "急诊处置"),
        "icu_xlsx": ("ICU mortality", "ICU 院内死亡"),
    }
    for r, meta in enumerate(DATASETS):
        for c, who in enumerate(("Jev", "LLM")):
            ax = fig.add_subplot(gs[r, c])
            row = [x for x in rows if x["task"] == meta["task_en"] and x["who"] == who][0]
            labels, cm, n = row["labels"], row["cm"], row["n"]
            row_n = np.asarray(row["row_n"], dtype=float)
            frac = np.divide(cm, row_n[:, None], out=np.zeros_like(cm, dtype=float),
                             where=row_n[:, None] > 0)
            col_hex = P["jev"] if who == "Jev" else P["llm"]
            ax.imshow(frac, cmap=_cmap(col_hex, f"cm_{who}_{meta['key']}"), vmin=0.0, vmax=1.0,
                      aspect="auto", interpolation="nearest")
            k = len(labels)
            for i in range(k):
                for j in range(k):
                    v, f = int(cm[i, j]), float(frac[i, j])
                    ax.text(j, i, f"{v}\n{f * 100:.1f}%", ha="center", va="center",
                            fontsize=6.0 if k > 2 else 7.2,
                            color="white" if f > 0.55 else P["ink"], linespacing=1.15)
            ax.set_xticks(range(k))
            ax.set_xticklabels(_tick_names(meta["key"], lang), fontsize=6.1 if k > 2 else 7.0)
            ylabels = [f"{rn} ({rc * 100:.1f}%)"
                       for rn, rc in zip(_row_names(meta["key"], lang), row["recall"])]
            ax.set_yticks(range(k))
            ax.set_yticklabels(ylabels, fontsize=6.0 if k > 2 else 6.6)
            ax.tick_params(length=2)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.0)
            ax.tick_params(which="minor", length=0)
            if c == 0:
                ax.set_ylabel("Reference" if en else "参考标签", fontsize=6.8)
            if r == len(DATASETS) - 1:
                ax.set_xlabel("Predicted" if en else "预测", fontsize=6.8, labelpad=1.5)
            task_en, task_zh = col_titles[meta["key"]]
            task_txt = task_zh if lang == "zh" else task_en
            if meta["key"] == "icu_xlsx":
                head = task_txt
            else:
                head = f"{meta['short']} {task_txt}"
            ax.set_title(f"{who} — {head} (n = {n})", loc="left", fontsize=6.6, pad=3.0)

    by_task = {(r["task"], r["who"]): r for r in rows}
    icu_j = by_task[("ICU mortality", "Jev")]["recall"]
    icu_l = by_task[("ICU mortality", "LLM")]["recall"]
    disp_j = by_task[("ED disposition", "Jev")]["recall"]
    disp_l = by_task[("ED disposition", "LLM")]["recall"]
    notes_grey = (
        ["Cells exclude transport-failure rows (non-null *_err): Jev n = 300/300/290, LLM n = 299/299/290.",
         "Percentages and recall use the scored class sizes (a transport failure counts as a false negative),",
         "so the LLM triage acuity-3 row sums to 152/153 and the LLM disposition admitted row to 88/89."]
        if en else
        ["矩阵单元已排除传输失败记录（*_err 非空）：Jev n = 300/300/290，LLM n = 299/299/290。",
         "行百分比与召回率以已评分类别数为分母（传输失败计为假阴性），",
         "故 LLM 分诊 3 级行合计 152/153，LLM 处置入院行合计 88/89。"]
    )
    line_icu = (
        f"ICU mortality — sensitivity (died): Jev {icu_j[1] * 100:.1f}%, LLM {icu_l[1] * 100:.1f}%; "
        f"specificity (survived): Jev {icu_j[0] * 100:.1f}%, LLM {icu_l[0] * 100:.1f}%."
        if en else
        f"ICU 院内死亡——灵敏度（死亡）：Jev {icu_j[1] * 100:.1f}% 对 LLM {icu_l[1] * 100:.1f}%；"
        f"特异度（存活）：Jev {icu_j[0] * 100:.1f}% 对 LLM {icu_l[0] * 100:.1f}%。"
    )
    line_disp = (
        f"Disposition — admitted recall: Jev {disp_j[1] * 100:.1f}%, LLM {disp_l[1] * 100:.1f}%; "
        f"discharged recall: Jev {disp_j[0] * 100:.1f}%, LLM {disp_l[0] * 100:.1f}%."
        if en else
        f"处置去向——入院召回率：Jev {disp_j[1] * 100:.1f}% 对 LLM {disp_l[1] * 100:.1f}%；"
        f"出院召回率：Jev {disp_j[0] * 100:.1f}% 对 LLM {disp_l[0] * 100:.1f}%。"
    )
    for y, line in zip((0.196, 0.166, 0.136), notes_grey):
        fig.text(0.5, y, line, ha="center", va="center", fontsize=6.0, color=P["muted"])
    fig.text(0.5, 0.083, line_icu, ha="center", va="center", fontsize=6.0, color=P["ink"])
    fig.text(0.5, 0.030, line_disp, ha="center", va="center", fontsize=6.0, color=P["ink"])

    return fc.save(fig, "figS1", lang)


def main() -> None:
    data = fc.build_data()
    for lang in ("en", "zh"):
        fc.set_style(lang)
        print("wrote", draw_figS1(data, lang))


if __name__ == "__main__":
    main()




#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the complete v2 figure pipeline (figures 1-6 + S1, EN/ZH, figures_data.md, TIFFs).

    C:\\Users\\ZHOUSHENGZHE\\AppData\\Local\\Programs\\Python\\Python314\\python.exe make_all_figures.py

Each step is a standalone script; this orchestrator only runs them in the required order with
the same interpreter and raises on the first failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    HERE / "make_figures_paper.py",   # Fig 1 + Fig 2, cross-checks
    HERE / "make_fig3.py",            # Fig 3
    HERE / "make_fig4.py",            # Fig 4
    HERE / "make_fig5.py",            # Fig 5
    HERE / "make_fig6.py",            # Fig 6
    HERE / "make_figS1.py",           # Fig S1
    HERE / "figures_manifest.py",     # figures_data.md
    HERE.parent / "build" / "make_tiff.py",
]


def main() -> None:
    for step in STEPS:
        if not step.exists():
            raise SystemExit(f"missing step: {step}")
        print(f"\n=== {step.name} ===", flush=True)
        proc = subprocess.run([sys.executable, str(step)], cwd=str(HERE))
        if proc.returncode != 0:
            raise SystemExit(f"{step.name} failed with exit code {proc.returncode}")
    print("\nAll figures, the data manifest and the TIFFs are up to date.")


if __name__ == "__main__":
    main()

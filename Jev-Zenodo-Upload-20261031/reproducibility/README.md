# Reproducibility package: Jev versus a reasoning LLM for structured clinical record classification

This package contains the code and per-sample derived data needed to reproduce the tables and figures of the manuscript. It contains **no source clinical records and no input text**.

## Contents

- `code/` - sampling/serialisation, inference post-processing, statistical recomputation, supervised baselines, figure scripts.
- `data/` - per-sample predictions, confidences, correctness, latency, token counts and cost for the runs analysed in the paper. NHAMCS, the public ED Kaggle copy and the ICU Sepsis dataset are **not** redistributed; they must be obtained from their original providers.
- `manifest.sha256` - SHA-256 hashes of all files in this package.
- `zenodo.json` - draft Zenodo metadata (DOI to be added after deposition).
- `TRIPOD-LLM_checklist.md` - draft item mapping against the TRIPOD-LLM reporting guideline.

## Environment

- Python 3.14 (pandas 3.0, numpy, scikit-learn 1.9, LightGBM 4.7, matplotlib 3.10).
- Pandoc 3.x + a Word installation are needed only to rebuild the .docx/.pdf files; the figures and statistics can be regenerated with Python alone.

## Reproduction steps

1. Obtain the source data (NHAMCS 2018-22 Kaggle mirror; the public ED Kaggle copy named in the manuscript; the ICU Sepsis Dataset.xlsx) and place them as expected by `code/build_bench_ed_icu.py` (`data/external_datasets/...`).
2. Rebuild the benchmark tables:
   `python code/build_bench_ed_icu.py --datasets nhamcs,icu_xlsx,mimic_ed`
3. Recompute the primary run statistics (repeat means, CIs, permutation p, calibration, selective prediction) from the per-sample JSONL/CSV files:
   `python code/recompute_primary_stats.py`
4. Recompute the supervised baselines:
   `python code/train_supervised_icu_nhamcs.py`
   `python code/train_supervised_ed.py`
5. Regenerate the figures:
   `python code/figcommon.py` (library) then `python code/make_figures_paper.py`, `python code/make_fig3.py`, `python code/make_fig4.py`, `python code/make_fig6.py`.

## Data dictionary (per-sample files)

- `ht_*_final.csv` - one row per scored record: sample_id, true_label, model predictions, confidence, end-to-end correctness, latency (ms), input/output tokens, cost, scored flag, transport-error flag.
- `jev_*_form_r*.jsonl` and `llm_gpt-6.1-sol_*_form.jsonl` - per-run per-record records used for the repeated-run analysis (prediction, confidence, correctness, latency, tokens, cost).

## Notes

- The original LLM run is retained in the data but excluded from the primary estimate because its token counts show a different prompt/configuration; the three-repeated-run mean is the primary estimate.
- Costs are calculated from provider-reported token counts and the prices stated in the manuscript. Reasoning tokens that are billed but not shown in the API usage are not captured, so the reported cost ratio is a lower bound.
- The ED Kaggle copy is a public, de-identified third-party copy. Its upstream provenance and redistribution licence have not been confirmed by the publisher; it is used for non-commercial research only and is not redistributed here.

## Licence

Code: MIT. Derived per-sample data: CC BY 4.0. No source clinical data are included.

## DOI and citation

This package is archived at Zenodo: **10.5281/zenodo.23838134** (https://doi.org/10.5281/zenodo.23838134).

Please cite:

> Zhou, S. Code and per-sample results: Jev versus a reasoning LLM for structured clinical record classification. Zenodo (2026). https://doi.org/10.5281/zenodo.23838134

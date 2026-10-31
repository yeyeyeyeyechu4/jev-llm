# TRIPOD-LLM reporting checklist — final mapping (2026-10-31)

Manuscript: *Jev versus a reasoning LLM for structured clinical record classification*
DOI of this reproducibility package: 10.5281/zenodo.23838134

Item wording is abridged; the mapping should be checked against the official TRIPOD-LLM checklist before submission. Locations refer to the manuscript sections/tables.

| # | Domain / item (abridged) | Reported | Location |
|---|---|---|---|
| 1 | Title identifies an LLM study | Yes | Title |
| 2 | Structured abstract with objectives, methods, results, conclusions | Yes | Abstract |
| 3 | Scientific rationale for comparing LLMs with a typed decision model | Yes | Introduction |
| 4 | Study objectives and intended use | Yes | Introduction (end) |
| 5 | Study design (paired, retrospective, offline) | Yes | Methods: Study design |
| 6 | Data sources and eligibility | Yes | Methods: Datasets; Table 1; Supplementary S18 |
| 7 | Data preparation, serialisation and de-identification | Yes | Methods: Datasets; Supplementary S2 |
| 8 | LLM name, version/alias and access date | Yes | Methods: Models and inference |
| 9 | Prompts, candidate definitions and output format | Yes | Supplementary S2a–S2e |
| 10 | Decoding settings (temperature, token limit, response format) | Yes | Methods: Models and inference; Supplementary S2e |
| 11 | Determinism/reproducibility of LLM outputs | Partly | Results: Between-run consistency; Limitations |
| 12 | Handling of API failures, retries and truncation | Partly | Methods: Cost/statistics; Limitations |
| 13 | Comparator description | Yes | Methods: Models and inference |
| 14 | Reference standard and label mapping | Yes | Methods: Candidate categories; Supplementary S2 |
| 15 | Outcome definition and decision time point | Partly | Methods: Datasets; Limitations |
| 16 | Sample size and justification | Partly | Methods: Study design; Limitations |
| 17 | Missing data and transport failures | Yes | Methods: Statistics; Results; Supplementary S5 |
| 18 | Evaluation metrics (accuracy, sensitivity/specificity, PPV/NPV, calibration, cost, latency) | Yes | Methods: Statistics; Results; Tables 2, S10, S12, S14 |
| 19 | Statistical methods, CIs, permutation tests, repeat runs | Yes | Methods: Statistics; Table 2; Supplementary S19 |
| 20 | Handling of multiplicity and post hoc choices | Yes | Methods: Statistics; Limitations |
| 21 | Record flow and scored denominators | Yes | Table 1; Results; Supplementary S1a |
| 22 | Model performance with uncertainty | Yes | Results; Table 2; Figure 2 |
| 23 | Error analysis (sensitivity/specificity, confusion) | Yes | Results; Figure S1 |
| 24 | Confidence calibration and selective prediction | Yes | Results; Figure 5; Supplementary S6, S13 |
| 25 | Cost and latency | Yes | Results: Cost and latency; Figure 4; Supplementary S14 |
| 26 | Comparisons with additional LLMs | Yes | Results; Supplementary S9, S17 |
| 27 | Robustness to run-to-run variation | Yes | Results: Between-run consistency; Supplementary S8a/S8b |
| 28 | Principal findings and comparison with prior work | Yes | Discussion |
| 29 | Limitations (design, data provenance, sample size, statistics) | Yes | Limitations |
| 30 | Clinical implications and generalisability | Partly | Discussion; Limitations |
| 31 | Registration and protocol | No | Not registered; stated in Limitations |
| 32 | Funding | Yes | Funding |
| 33 | Competing interests | Yes | Competing interests |
| 34 | Data availability | Yes | Data availability (DOI 10.5281/zenodo.23838134) |
| 35 | Code availability | Yes | Code availability (DOI 10.5281/zenodo.23838134) |
| 36 | Ethics approval / exemption | Yes | Ethics |
| 37 | Patient and public involvement | N/A | No patient involvement (secondary data) |
| 38 | Human evaluation of LLM outputs | N/A | Not performed; all outcomes scored against reference labels |

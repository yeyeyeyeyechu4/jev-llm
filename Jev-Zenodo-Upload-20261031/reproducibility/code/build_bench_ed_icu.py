"""Build the ED/ICU benchmark used by the JEV-vs-LLM high-throughput run.

Outputs one row per sample with a code-serialised native table text plus the
ordered field list needed to render the same record as a table image (OCR arm).
Sampling is stratified by label with a fixed seed; the split is a deterministic
hash of the sample id, so the file is reproducible.

Leakage policy (documented in the report):
  * the target column never enters the state
  * columns recorded downstream of the decision are dropped
    (NHAMCS wait_time_minutes / intervention_iv_fluids; ICU icu_los_hours /
    readmission_30day; xlsx Length_of_stay_in_intensive_care)
  * pure identifiers are dropped (subject_id)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from xml.etree import ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external_datasets"
OUT = ROOT / "data_bench"
SAMPLING_SEED = 20261015
NL = chr(10)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "<na>"} else text


def _split_for(sample_id: str) -> str:
    bucket = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 60 else ("val" if bucket < 80 else "test")


def _state_text(pairs: Sequence[Tuple[str, str]]) -> str:
    return NL.join(k + ": " + v for k, v in pairs)


def _stratified_sample(df: pd.DataFrame, label_col: str, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    groups = list(df.groupby(label_col, dropna=False))
    share = {key: len(g) / len(df) for key, g in groups}
    quota = {key: int(share[key] * n) for key, _ in groups}
    remainder = n - sum(quota.values())
    order = sorted(groups, key=lambda kv: -(share[kv[0]] * n - int(share[kv[0]] * n)))
    for key, _ in order[:remainder]:
        quota[key] += 1
    parts = []
    for key, g in groups:
        take = min(quota[key], len(g))
        if take:
            parts.append(g.sample(n=take, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _read_xlsx_rows(path: Path):
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def col_index(ref: str) -> int:
        letters = "".join(ch for ch in ref if ch.isalpha()).upper()
        value = 0
        for char in letters:
            value = value * 26 + ord(char) - ord("A") + 1
        return value - 1

    with zipfile.ZipFile(path) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", ns):
                shared.append("".join(item.itertext()))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: List[List[str]] = []
        for row in sheet.findall(".//x:sheetData/x:row", ns):
            cells: Dict[int, str] = {}
            for cell in row.findall("x:c", ns):
                index = col_index(cell.attrib.get("r", "A1"))
                node = cell.find("x:v", ns)
                raw = "" if node is None else (node.text or "")
                if cell.attrib.get("t") == "s" and raw:
                    raw = shared[int(raw)]
                cells[index] = raw
            if cells:
                rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    if not rows:
        return [], []
    headers = [str(x).strip() or ("column_" + str(i)) for i, x in enumerate(rows[0])]
    return headers, rows[1:]


def _pack(pairs):
    return {"state_text": _state_text(pairs), "n_fields": len(pairs),
            "fields_json": json.dumps(pairs, ensure_ascii=False)}


def build_nhamcs(n: int) -> List[Dict[str, Any]]:
    path = EXT / "nhamcs-2018-22" / "nhamcs_data_2018_22.csv"
    df = pd.read_csv(path, low_memory=False)
    label_col = "target_triage_acuity"
    drop = {label_col, "wait_time_minutes", "intervention_iv_fluids"}
    df = df.dropna(subset=[label_col])
    df = _stratified_sample(df, label_col, n, SAMPLING_SEED)
    level = {"1": "resuscitation", "2": "emergent", "3": "urgent", "4": "semi_urgent", "5": "non_urgent"}
    rows = []
    for original_index, row in df.iterrows():
        pairs = []
        for key, value in row.items():
            if key in drop:
                continue
            text = _as_text(value)
            if text:
                pairs.append((str(key), text))
        sample_id = "nhamcs_%06d" % int(original_index)
        raw = _as_text(row[label_col])
        rows.append({"sample_id": sample_id, "dataset": "nhamcs", "group_id": sample_id,
                     "split": _split_for(sample_id),
                     "label": "acuity_" + raw + "_" + level.get(raw, "unknown"), **_pack(pairs)})
    return rows


def build_icu_csv(n: int) -> List[Dict[str, Any]]:
    path = EXT / "mimic-iv-style-icu-sepsis" / "sepsis_icu_synthetic.csv"
    df = pd.read_csv(path, low_memory=False)
    label_col = "sepsis_label"
    drop = {label_col, "subject_id", "icu_los_hours", "readmission_30day"}
    df = df.dropna(subset=[label_col])
    df = _stratified_sample(df, label_col, n, SAMPLING_SEED)
    rows = []
    for original_index, row in df.iterrows():
        pairs = []
        for key, value in row.items():
            if key in drop:
                continue
            text = _as_text(value)
            if text:
                pairs.append((str(key), text))
        sample_id = "icu_csv_%06d" % int(original_index)
        label = "sepsis_present" if _as_text(row[label_col]) in {"1", "1.0", "true", "yes"} else "sepsis_absent"
        rows.append({"sample_id": sample_id, "dataset": "icu_csv",
                     "group_id": "subject_" + _as_text(row.get("subject_id", "")),
                     "split": _split_for(sample_id), "label": label, **_pack(pairs)})
    return rows


def build_icu_xlsx(n: int) -> List[Dict[str, Any]]:
    path = EXT / "icu-sepsis-dataset-xlsx" / "ICU Sepsis Dataset.xlsx"
    headers, data = _read_xlsx_rows(path)
    drop = {"Mortalite", "Variable", "Length_of_stay_in_intensive_care"}
    records = []
    for index, values in enumerate(data):
        row = {headers[i]: (values[i] if i < len(values) else "") for i in range(len(headers))}
        label = _as_text(row.get("Mortalite"))
        if label in {"0", "0.0", "1", "1.0", "3", "3.0"}:
            records.append({"_i": index, "_label": label, **row})
    frame = pd.DataFrame(records)
    sampled = _stratified_sample(frame, "_label", n, SAMPLING_SEED)
    rows = []
    for _, row in sampled.iterrows():
        pairs = []
        for key, value in row.items():
            if key in {"_i", "_label"} or key in drop:
                continue
            text = _as_text(value)
            if text:
                pairs.append((str(key), text))
        sample_id = "icu_xlsx_%06d" % int(row["_i"])
        raw = _as_text(row["_label"])
        label = {"0": "survived", "1": "died"}.get(raw, "code_" + raw + "_unverified")
        rows.append({"sample_id": sample_id, "dataset": "icu_xlsx", "group_id": sample_id,
                     "split": _split_for(sample_id), "label": label, **_pack(pairs)})
    return rows


def build_mimic_ed(n: int) -> List[Dict[str, Any]]:
    """Real ED visits (MIMIC-IV-ED derived). Label = disposition (Admit/Discharge).

    The task is framed as *triage-time* decision support, so only information
    available when the patient is triaged enters the state: demographics,
    arrival mode/time, triage vitals, active chief complaints, chronic-condition
    history and the previous visit disposition.  Deliberately excluded:
      * disposition          - the label
      * esi                  - the triage decision itself, derived from the same
                               inputs; keeping it turns the task into decoding a
                               code instead of reading the record
      * * _last labs, imaging counts, n_surgeries - recorded during/after the ED
                               course and therefore downstream of disposition
      * n_admissions         - may already include the current admission
    """
    source = ROOT / "data_bench" / "mimic_ed_reduced.pkl"
    frame = pd.read_pickle(source)
    frame = frame[frame["disposition"].notna()]
    demographics = ["age", "gender", "ethnicity", "race", "lang", "maritalstatus",
                    "employstatus", "insurance_status", "dep_name"]
    arrival = ["arrivalmode", "arrivalmonth", "arrivalday", "arrivalhour_bin"]
    vitals = ["triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp", "triage_vital_rr",
              "triage_vital_o2", "triage_vital_o2_device", "triage_vital_temp"]
    history = ["previousdispo"]
    chronic = [c for c in CHRONIC_FLAGS if c in frame.columns]
    complaints = [c for c in frame.columns if c.startswith("cc_")]
    sampled = _stratified_sample(frame, "disposition", n, SAMPLING_SEED)
    rows = []
    for original_index, row in sampled.iterrows():
        pairs = []
        for key in demographics + arrival + vitals + history:
            if key not in row.index:
                continue
            text = _as_text(row[key])
            if text:
                pairs.append((key, text))
        active_cc = [c[3:] for c in complaints if _as_text(row[c]) not in {"", "0", "0.0", "False"}]
        if active_cc:
            pairs.append(("chief_complaints", ", ".join(sorted(active_cc))))
        active_chronic = [c for c in chronic if _as_text(row[c]) not in {"", "0", "0.0", "False"}]
        if active_chronic:
            pairs.append(("chronic_conditions", ", ".join(sorted(active_chronic))))
        sample_id = "mimic_ed_%06d" % int(original_index)
        raw = _as_text(row["disposition"])
        label = "admitted" if raw.lower().startswith("admit") else "discharged"
        rows.append({"sample_id": sample_id, "dataset": "mimic_ed", "group_id": sample_id,
                     "split": _split_for(sample_id), "label": label, **_pack(pairs)})
    return rows


CHRONIC_FLAGS = ["htn", "diabmelnoc", "diabmelwcm", "chfnonhp", "copd", "asthma",
                 "chrkidneydisease", "acutemi", "acutecvd", "septicemia", "pneumonia",
                 "epilepsycnv", "parkinsons", "ms", "anemia", "maligneopls", "breastcancr",
                 "coloncancer", "prostatecan", "thyroiddsor", "hyperlipidem",
                 "substancerelateddisorders", "mooddisorders", "anxietydisorders", "dysrhythmia",
                 "conduction", "shock", "respdistres", "gihemorrhag", "syncope", "fuo"]


BUILDERS = {"nhamcs": build_nhamcs, "icu_csv": build_icu_csv, "icu_xlsx": build_icu_xlsx,
            "mimic_ed": build_mimic_ed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="nhamcs,icu_csv,icu_xlsx")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        rows = BUILDERS[name](args.n)
        path = OUT / (name + "_bench.csv")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(name, "n=%d" % len(rows), dict(Counter(r["label"] for r in rows)),
              dict(Counter(r["split"] for r in rows)), "->", path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "Datasets")
SUPPORTED_EXTENSIONS = (".xes", ".xes.gz")
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _n_cases_from_name(file_name: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*cases", file_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)_[A-Za-z]+\.xes$", file_name)
    return int(match.group(1)) if match else None


def _gt_bose(_: int) -> List[int]:
    return [1199, 2399, 3599, 4799]



def _gt_ceravolo(n_cases: int) -> List[int]:
    return [n_cases // 2]


def _gt_ostovar(_: int) -> List[int]:
    return [999, 1999]


GT_RESOLVERS = {
    "bose": _gt_bose,
    "ceravolo": _gt_ceravolo,
    "ostovar": _gt_ostovar,
}


def _is_supported_log(name: str) -> bool:
    lower_name = name.lower()
    return any(lower_name.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def _dedupe_prefer_gz(paths: List[str]) -> List[str]:
    by_base: Dict[str, str] = {}
    for path in sorted(paths):
        key = path[:-3] if path.lower().endswith(".gz") else path
        current = by_base.get(key)
        if current is None:
            by_base[key] = path
            continue
        if path.lower().endswith(".gz"):
            by_base[key] = path
    return sorted(by_base.values())


def _collect_files_in_folder(folder: str) -> List[str]:
    files: List[str] = []
    if not os.path.isdir(folder):
        return files
    for root, _, names in os.walk(folder):
        parts = [p.lower() for p in root.split(os.sep) if p]
        if "all" in parts:
            continue
        for name in names:
            if _is_supported_log(name):
                files.append(os.path.join(root, name))
    return _dedupe_prefer_gz(files)


def _collect_target_files(base_folder: str) -> List[str]:
    ceravolo_dir = os.path.join(base_folder, "Ceravolo")
    ostovar_dir = os.path.join(base_folder, "Ostovar")
    bose_dir = os.path.join(base_folder, "Bose")

    ceravolo_all = _collect_files_in_folder(ceravolo_dir)
    ceravolo_1000 = [
        path for path in ceravolo_all
        if re.search(r"_1000_", os.path.basename(path), re.IGNORECASE)
    ]

    ostovar_files = _collect_files_in_folder(ostovar_dir)
    bose_files = _collect_files_in_folder(bose_dir)

    return sorted(ceravolo_1000 + ostovar_files + bose_files)


def _parse_list_like(value: object) -> List[int]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if text in ("", "ERROR", "[]"):
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    if isinstance(parsed, (list, tuple)):
        try:
            return [int(x) for x in parsed]
        except (TypeError, ValueError):
            return []
    return []


def _dataset_source_from_path(path: str) -> str:
    norm = path.lower()
    if "\\ceravolo\\" in norm:
        return "ceravolo"
    if "\\ostovar\\" in norm:
        return "ostovar"
    if "\\bose\\" in norm:
        return "bose"
    return "unknown"


def _infer_n_cases_from_log(log_path: str) -> Optional[int]:
    try:
        from cvdrift.preprocessing import load_log

        frame = load_log(log_path)
        for col in ("Case ID", "case:concept:name", "case_id"):
            if col in frame.columns:
                return int(frame[col].nunique())
        for col in frame.columns:
            if "case" in col.lower():
                return int(frame[col].nunique())
    except Exception:
        return None
    return None


def _ground_truth_for_log(log_path: str) -> List[int]:
    source = _dataset_source_from_path(log_path)

    if source == "ostovar":
        return _gt_ostovar(0)

    n_cases = _n_cases_from_name(os.path.basename(log_path))
    if n_cases is None and source == "bose":
        n_cases = _infer_n_cases_from_log(log_path)

    if n_cases is None:
        return []

    if source == "ceravolo":
        return _gt_ceravolo(n_cases)

    if source == "bose":
        return _gt_bose(n_cases)

    return []


def _export_requested_excel(csv_path: str, out_xlsx: str) -> str:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        report = pd.DataFrame(
            columns=[
                "Algorithm name",
                "Log name",
                "Detected points",
                "Actual points (GT)",
                "Runtime (Seconds)",
            ]
        )
    else:
        report = pd.DataFrame(
            {
                "Algorithm name": ["CVDrift"] * len(frame),
                "Log name": frame.get("Log", "").astype(str),
                "Detected points": frame.get("Detected Changepoints", "[]").astype(str),
                "Actual points (GT)": frame.get("Actual Changepoints for Log", "[]").astype(str),
                "Runtime (Seconds)": frame.get("Duration (Seconds)", 0.0),
            }
        )

    out_xlsx = os.path.abspath(out_xlsx)
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    report.to_excel(out_xlsx, index=False)
    return out_xlsx


def _fmt_time(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _append_csv_row(csv_path: str, row: Dict[str, object]) -> None:
    fields = [
        "Algorithm",
        "Log Source",
        "Log",
        "Drift Types",
        "Detected Changepoints",
        "Actual Changepoints for Log",
        "Duration (Seconds)",
        "Duration (hh:mm:ss)",
        "Parameter Settings",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _single_result_row(log_path: str, drift_types: Iterable[str], params: Dict, quiet: bool = False) -> Dict[str, object]:
    from cvdrift.drift_detection import extract_cps, run_pipeline
    from cvdrift.preprocessing import load_log

    start_time = time.time()
    df = load_log(log_path)
    results = run_pipeline(df, drift_types, params=params, quiet=quiet)
    cps = extract_cps(results)
    elapsed = time.time() - start_time
    return {
        "Algorithm": "CVDrift",
        "Log Source": os.path.basename(log_path),
        "Log": os.path.basename(log_path),
        "Drift Types": ", ".join(drift_types),
        "Detected Changepoints": str(cps),
        "Actual Changepoints for Log": "[]",
        "Duration (Seconds)": round(elapsed, 3),
        "Duration (hh:mm:ss)": _fmt_time(elapsed),
        "Parameter Settings": json.dumps(params, default=str),
    }


def _run_batch_local(log_paths: List[str], out_path: str, drift_types: Iterable[str], params: Dict, quiet: bool = False) -> str:
    csv_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print(f"Processing {len(log_paths)} log(s) -> {csv_path}")
    for index, log_path in enumerate(log_paths, start=1):
        print(f"[{index}/{len(log_paths)}] {os.path.basename(log_path)}")
        try:
            row = _single_result_row(log_path, drift_types, params, quiet=quiet)
        except Exception as exc:
            elapsed = 0.0
            row = {
                "Algorithm": "CVDrift",
                "Log Source": os.path.basename(log_path),
                "Log": os.path.basename(log_path),
                "Drift Types": ", ".join(drift_types),
                "Detected Changepoints": "ERROR",
                "Actual Changepoints for Log": "[]",
                "Duration (Seconds)": elapsed,
                "Duration (hh:mm:ss)": _fmt_time(elapsed),
                "Parameter Settings": json.dumps({"error": str(exc)}),
            }
            print(f"  ERROR: {exc}")
        _append_csv_row(csv_path, row)
    print(f"Batch complete. Results: {csv_path}")
    return csv_path

   


def _augment_results_with_ground_truth(csv_path: str, files: List[str]) -> str:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return csv_path

    actual_values: List[str] = []
    dataset_values: List[str] = []
    n_case_values: List[Optional[int]] = []

    for index, _ in frame.iterrows():
        if index < len(files):
            log_path = files[index]
            gt = _ground_truth_for_log(log_path)
            mode = _dataset_source_from_path(log_path)
            n_cases = _n_cases_from_name(os.path.basename(log_path))
            if n_cases is None and mode == "bose":
                n_cases = _infer_n_cases_from_log(log_path)
        else:
            gt = []
            mode = "unknown"
            n_cases = None

        actual_values.append(str(gt))
        dataset_values.append(mode)
        n_case_values.append(n_cases)

    frame["Dataset"] = dataset_values
    frame["n_cases"] = n_case_values
    frame["Actual Changepoints for Log"] = actual_values
    frame.to_csv(csv_path, index=False)
    return csv_path


def evaluate_cps(detected: Iterable[int], ground_truth: Iterable[int], tolerance: int) -> Dict[str, float]:
    detected_sorted = sorted(int(x) for x in detected)
    gt_sorted = sorted(int(x) for x in ground_truth)

    matched = set()
    tp = 0
    for gt in gt_sorted:
        best_index = None
        best_distance = float("inf")
        for index, detected_cp in enumerate(detected_sorted):
            if index in matched:
                continue
            distance = abs(detected_cp - gt)
            if distance <= tolerance and distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None:
            matched.add(best_index)
            tp += 1

    fp = len(detected_sorted) - tp
    fn = len(gt_sorted) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
    }


def evaluate_results_csv(csv_path: str, tol: Optional[int] = None) -> Dict[str, float]:
    frame = pd.read_csv(csv_path)
    frame = frame[~frame["Log"].astype(str).str.startswith("===")].copy().reset_index(drop=True)

    cp_col = "Detected Changepoints"
    if cp_col not in frame.columns:
        raise ValueError(f"No changepoint column found in {csv_path}")

    total_tp = total_fp = total_fn = 0
    rows: List[Dict[str, object]] = []

    for _, row in frame.iterrows():
        file_name = str(row["Log"])
        raw_detected = row[cp_col]
        detected = _parse_list_like(raw_detected)

        raw_actual = row.get("Actual Changepoints for Log", "[]")
        ground_truth = _parse_list_like(raw_actual)
        if ground_truth:
            if "n_cases" in row and not pd.isna(row["n_cases"]):
                n_cases = int(row["n_cases"])
            else:
                n_cases = _n_cases_from_name(file_name)
        else:
            n_cases = _n_cases_from_name(file_name)
            if n_cases is None:
                continue
            row_dataset = str(row.get("Dataset", "")).lower().strip()
            gt_resolver = GT_RESOLVERS.get(row_dataset)
            if gt_resolver is None:
                continue
            ground_truth = gt_resolver(n_cases)

        if n_cases is None:
            continue

        tolerance = tol if tol is not None else max(round(0.10 * n_cases), 10)
        metrics = evaluate_cps(detected, ground_truth, tolerance)
        total_tp += int(metrics["TP"])
        total_fp += int(metrics["FP"])
        total_fn += int(metrics["FN"])

        rows.append(
            {
                "Log": file_name,
                "n_cases": n_cases,
                "GT": str(ground_truth),
                "Detected": str(detected),
                "Tol": tolerance,
                **metrics,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    output_path = os.path.splitext(csv_path)[0] + "_eval.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
    }



def run_experiments(
    out_csv: str,
    files: Optional[List[str]] = None,
    params: Optional[Dict] = None,
    quiet: bool = False,
) -> str:
    files = files or _collect_target_files(DATASETS_DIR)
    if not files:
        raise ValueError(f"No supported dataset files found in: {DATASETS_DIR}")

    csv_path = _run_batch_local(files, out_csv, ["routing"], params=params or {}, quiet=quiet)
    csv_path = _augment_results_with_ground_truth(csv_path, files)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment#2 (Control Flow) on Ceravolo(1000 only), Ostovar, and Bose datasets."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: output/my_results.csv",
    )
    parser.add_argument(
        "--excel",
        default=None,
        help="Output Excel path. Default: output/my_results.xlsx",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output from pipeline.",
    )
    args = parser.parse_args()

    default_out = os.path.join(SCRIPT_DIR, "output", "my_results.csv")
    default_excel = os.path.join(SCRIPT_DIR, "output", "my_results.xlsx")
    out_csv = args.out or default_out
    out_excel = args.excel or default_excel
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    csv_path = run_experiments(
        out_csv=out_csv,
        params={"window_strategy": "mode_window"},
        quiet=args.quiet,
    )
    excel_path = _export_requested_excel(csv_path, out_excel)
    print(f"Finished. Results CSV: {csv_path}")
    print(f"Requested Excel: {excel_path}")


if __name__ == "__main__":
    main()

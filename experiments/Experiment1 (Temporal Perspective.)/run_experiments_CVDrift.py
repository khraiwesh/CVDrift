from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional

import pandas as pd

#usage
# python "run_experiments.py" --out "experiments\Experiment#1 (Temporal Perspective.)\output\my_results.csv"

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SUPPORTED_EXTENSIONS = (".xes", ".xes.gz")
LOG_META_RE = re.compile(r"dataset_(\d+)cases_?(\d+)min_ABD(?:_noisy_(\d+)%)?\.xes", re.IGNORECASE)


def _collect_dataset_files(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(folder):
        for name in names:
            lower_name = name.lower()
            if any(lower_name.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                files.append(os.path.join(root, name))
    return sorted(files)


def _n_cases_from_name(file_name: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*cases", file_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)_[A-Za-z]+\.xes$", file_name)
    return int(match.group(1)) if match else None


def _parse_log_meta(file_name: str) -> Dict[str, Optional[int]]:
    name = os.path.basename(str(file_name))
    match = LOG_META_RE.match(name)
    if not match:
        return {"n_cases": _n_cases_from_name(name), "drift_amount_min": None, "noise_pct": None}
    return {
        "n_cases": int(match.group(1)),
        "drift_amount_min": int(match.group(2)),
        "noise_pct": int(match.group(3)) if match.group(3) is not None else 0,
    }


def _ground_truth(n_cases: int) -> List[int]:
    return [round(0.37 * n_cases), round(0.75 * n_cases)]


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
        raise ValueError(f"Expected '{cp_col}' column in {csv_path}")

    total_tp = total_fp = total_fn = 0
    rows: List[Dict[str, object]] = []

    for _, row in frame.iterrows():
        file_name = str(row["Log"])
        raw_detected = row[cp_col]
        detected: List[int] = []
        if not (pd.isna(raw_detected) or str(raw_detected).strip() in ("", "ERROR", "[]")):
            try:
                detected = [int(x) for x in json.loads(str(raw_detected))]
            except json.JSONDecodeError:
                detected = []

        meta = _parse_log_meta(file_name)
        n_cases = meta["n_cases"]
        if n_cases is None:
            continue

        ground_truth = _ground_truth(n_cases)
        tolerance = tol if tol is not None else round(0.10 * n_cases)
        metrics = evaluate_cps(detected, ground_truth, tolerance)
        total_tp += int(metrics["TP"])
        total_fp += int(metrics["FP"])
        total_fn += int(metrics["FN"])

        rows.append(
            {
                "Log": file_name,
                "n_cases": n_cases,
                "drift_amount_min": meta["drift_amount_min"],
                "noise_pct": meta["noise_pct"],
                "GT": str(ground_truth),
                "Detected": str(detected),
                "Tol": tolerance,
                **metrics,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def _micro_from_group(group: pd.DataFrame) -> pd.Series:
        tp_sum = int(group["TP"].sum())
        fp_sum = int(group["FP"].sum())
        fn_sum = int(group["FN"].sum())
        p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return pd.Series(
            {
                "n_files": int(len(group)),
                "TP": tp_sum,
                "FP": fp_sum,
                "FN": fn_sum,
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "F1": round(f, 4),
            }
        )

    eval_df = pd.DataFrame(rows)

    output_path = os.path.splitext(csv_path)[0] + "_eval.csv"
    eval_df.to_csv(output_path, index=False)

    by_drift = pd.DataFrame()
    by_drift_noise = pd.DataFrame()
    by_noise = pd.DataFrame()

    if not eval_df.empty:
        by_drift = (
            eval_df.dropna(subset=["drift_amount_min"])
            .groupby("drift_amount_min", dropna=True)
            .apply(_micro_from_group)
            .reset_index()
            .rename(columns={"drift_amount_min": "Drift_Amount_min"})
        )
        by_drift.to_csv(os.path.splitext(csv_path)[0] + "_eval_overall_micro.csv", index=False)

        by_drift_noise = (
            eval_df.dropna(subset=["drift_amount_min", "noise_pct"])
            .groupby(["drift_amount_min", "noise_pct"], dropna=True)
            .apply(_micro_from_group)
            .reset_index()
            .rename(columns={"drift_amount_min": "drift_magnitude_min", "noise_pct": "noise_level"})
        )
        by_drift_noise["noise_level"] = by_drift_noise["noise_level"].astype(int).astype(str) + "%"
        by_drift_noise.to_csv(os.path.splitext(csv_path)[0] + "_eval_by_noise_micro.csv", index=False)

        by_noise = (
            eval_df.dropna(subset=["noise_pct"])
            .groupby("noise_pct", dropna=True)
            .apply(_micro_from_group)
            .reset_index()
            .rename(columns={"noise_pct": "noise_level"})
            .sort_values(by="noise_level")
        )
        by_noise["noise_level"] = by_noise["noise_level"].astype(int).astype(str) + "%"

    print("\n=== Evaluation (Micro-Averaged) ===")
    print(f"Overall  -> Precision: {round(precision, 4):.4f} | Recall: {round(recall, 4):.4f} | F1: {round(f1, 4):.4f}")

    if not by_noise.empty:
        print("\nF1 by Noise:")
        print(by_noise[["noise_level", "n_files", "Precision", "Recall", "F1"]].to_string(index=False))

    if not by_drift.empty:
        print("\nF1 by Drift Amount (min):")
        print(by_drift[["Drift_Amount_min", "n_files", "Precision", "Recall", "F1"]].to_string(index=False))

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
    tol: Optional[int] = None,
    quiet: bool = False,
    skip_eval: bool = False,
) -> str:
    from main import run_batch

    if not os.path.isdir(DATASETS_DIR):
        raise FileNotFoundError(f"Datasets folder not found: {DATASETS_DIR}")

    dataset_files = _collect_dataset_files(DATASETS_DIR)
    if not dataset_files:
        raise ValueError(f"No supported dataset files found in: {DATASETS_DIR}")

    try:
        csv_path = run_batch(dataset_files, out_csv, ["duration"], params={}, quiet=quiet)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to output file: {out_csv}. Close the file if it is open in Excel/another app, "
            f"or use a different --out path. Original error: {exc}"
        ) from exc

    if not skip_eval:
        evaluate_results_csv(csv_path, tol=tol)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment#1 (Temporal Perspective) on local Datasets folder.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: output/my_results.csv",
    )
    parser.add_argument("--tol", type=int, default=None, help="Optional tolerance override. Default: 10% of dataset size.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output from pipeline.")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation step.")
    args = parser.parse_args()

    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "my_results.csv")
    out_csv = args.out or default_out
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    csv_path = run_experiments(
        out_csv=out_csv,
        tol=args.tol,
        quiet=args.quiet,
        skip_eval=args.no_eval,
    )
    print(f"Finished. Results CSV: {csv_path}")
    if not args.no_eval:
        print(f"Evaluation CSV: {os.path.splitext(csv_path)[0]}_eval.csv")


if __name__ == "__main__":
    main()

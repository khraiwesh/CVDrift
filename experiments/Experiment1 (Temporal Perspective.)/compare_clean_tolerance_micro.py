from __future__ import annotations

import argparse
import ast
import json
import os
import re
from typing import Dict, Iterable, List

import pandas as pd


def _parse_list_like(value: object) -> List[int]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if text in ("", "[]", "ERROR"):
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


def _n_cases_from_name(file_name: str) -> int | None:
    match = re.search(r"dataset_(\d+)cases", file_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*cases", file_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _gt_37_75(n_cases: int) -> List[int]:
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


def _is_clean_log(log_name: str) -> bool:
    return not re.search(r"noisy|noise", str(log_name), re.IGNORECASE)


def compute_clean_micro_table(csv_path: str, tolerance_pcts: List[int]) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "Log" not in frame.columns or "Detected Changepoints" not in frame.columns:
        raise ValueError("Input CSV must contain 'Log' and 'Detected Changepoints' columns.")

    frame = frame[frame["Log"].astype(str).map(_is_clean_log)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No clean logs found. Expected rows whose Log does not contain 'noisy'/'noise'.")

    rows: List[Dict[str, object]] = []

    for tol_pct in tolerance_pcts:
        total_tp = total_fp = total_fn = 0
        used_logs = 0

        for _, row in frame.iterrows():
            log_name = str(row["Log"])
            n_cases = _n_cases_from_name(log_name)
            if n_cases is None:
                continue

            detected = _parse_list_like(row["Detected Changepoints"])
            gt = _gt_37_75(n_cases)
            tolerance = round((tol_pct / 100.0) * n_cases)

            metrics = evaluate_cps(detected, gt, tolerance)
            total_tp += int(metrics["TP"])
            total_fp += int(metrics["FP"])
            total_fn += int(metrics["FN"])
            used_logs += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "tolerance_pct": int(tol_pct),
                "n_logs": used_logs,
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("tolerance_pct", ascending=False).reset_index(drop=True)


def compare_with_reference(computed_df: pd.DataFrame, reference_csv: str) -> pd.DataFrame:
    if not os.path.exists(reference_csv):
        return pd.DataFrame()

    ref = pd.read_csv(reference_csv)
    if "tolerance_pct" not in ref.columns:
        return pd.DataFrame()

    keep_cols = ["tolerance_pct", "TP", "FP", "FN", "Precision", "Recall", "F1"]
    ref = ref[[c for c in keep_cols if c in ref.columns]].copy()

    merged = computed_df.merge(
        ref,
        on="tolerance_pct",
        how="left",
        suffixes=("_computed", "_reference"),
    )

    for metric in ["TP", "FP", "FN", "Precision", "Recall", "F1"]:
        left = f"{metric}_computed"
        right = f"{metric}_reference"
        if left in merged.columns and right in merged.columns:
            merged[f"{metric}_delta"] = (merged[left] - merged[right]).round(4)

    return merged


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(script_dir, "output", "my_results.csv")
    default_ref = os.path.join(script_dir, "output", "clean_tolerance_comparison_micro.csv")
    default_out = os.path.join(script_dir, "output", "clean_tolerance_comparison_micro_recomputed.csv")
    default_cmp = os.path.join(script_dir, "output", "clean_tolerance_comparison_micro_diff.csv")

    parser = argparse.ArgumentParser(
        description=(
            "Compute clean-only micro F1 using GT at 37% and 75% with tolerance percentages, "
            "and optionally compare with a reference CSV."
        )
    )
    parser.add_argument("--in-csv", default=default_in, help="Input results CSV (default: output/my_results.csv)")
    parser.add_argument("--ref-csv", default=default_ref, help="Reference CSV to compare against")
    parser.add_argument("--out-csv", default=default_out, help="Output CSV for recomputed metrics")
    parser.add_argument("--cmp-csv", default=default_cmp, help="Output CSV for comparison table")
    parser.add_argument("--tolerances", default="5,3,2", help="Comma-separated tolerance percentages")
    args = parser.parse_args()

    tolerance_pcts = [int(x.strip()) for x in args.tolerances.split(",") if x.strip()]
    computed_df = compute_clean_micro_table(args.in_csv, tolerance_pcts)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    computed_df.to_csv(args.out_csv, index=False)
    print("Recomputed clean-only micro table:")
    print(computed_df.to_string(index=False))
    print(f"Saved recomputed metrics to: {args.out_csv}")

    comparison_df = compare_with_reference(computed_df, args.ref_csv)
    if comparison_df.empty:
        print("No comparison table generated (reference file missing or incompatible).")
        return

    comparison_df.to_csv(args.cmp_csv, index=False)
    print("\nComparison vs reference:")
    print(comparison_df.to_string(index=False))
    print(f"Saved comparison table to: {args.cmp_csv}")


if __name__ == "__main__":
    main()

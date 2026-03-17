#!/usr/bin/env python
# ============================================================
# run_experiments_v2.py — OCPA concept-drift wrapper with
#                         improved lifecycle handling for noisy logs
# ============================================================
"""
Improved version of run_experiments.py.

Changes vs. the original (v1):
  1. _infer_start_timestamps() now computes **per-activity median gaps**
     and uses them to:
       a) Clip outlier inter-event gaps at 3× the activity median
          (prevents noise-induced timestamp shifts from distorting durations)
       b) Assign the activity median to the **last event** in each case
          (instead of duration = 0)
  2. _xes_to_ocel_df() passes the Activity column to the improved
     _infer_start_timestamps() so it can group by activity.

These three changes mirror how CVDrift handles single-timestamp logs
(see CVDriftPipeline_v2/pipeline/io.py) and are the main reason CVDrift
is robust to noise while the original OCPA wrapper is not.

Usage is identical to run_experiments.py:

    python run_experiments_OCPA.py --dir "path/to/folder" --out output/OCPA.xlsx
    python run_experiments_OCPA.py --dir "Datasets" --out output/OCPA.xlsx
    python run_experiments_OCPA.py --file "path/to/log.xes" --steps drift
"""

import os
import sys
import ast
import re

# Use the local old OCPA package (same layout that experiments.py relies on
# by having ocpa/ sitting next to it).  This must come before any ocpa import.
_LOCAL_OCPA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ex_concept_drift (OCPA)")
if _LOCAL_OCPA not in sys.path:
    sys.path.insert(0, _LOCAL_OCPA)

import time
import glob
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
from statistics import median as median
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.tools.sm_exceptions as stats_exceptions

# ---- OCPA imports ------------------------------------------------
import ocpa.algo.filtering.log.time_filtering
from ocpa.objects.log.obj import OCEL
import ocpa.algo.feature_extraction.factory as feature_extraction
from ocpa.algo.feature_extraction import time_series

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def avg(x):
    """Average that returns NaN for empty lists."""
    if len(x) == 0:
        return np.nan
    return sum(x) / len(x)

def std_dev(x):
    """Population standard deviation."""
    m = sum(x) / len(x)
    return sum((xi - m) ** 2 for xi in x) / len(x)


LOG_META_RE = re.compile(
    r"dataset_(\d+)cases_?(\d+)min_ABD(?:_noisy_(\d+)%)?\.xes",
    re.IGNORECASE,
)


def _n_cases_from_name(file_name):
    match = re.search(r"(\d+)\s*cases", file_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)_[A-Za-z]+\.xes$", file_name)
    return int(match.group(1)) if match else None


def _parse_log_meta(file_name):
    name = os.path.basename(str(file_name))
    match = LOG_META_RE.match(name)
    if not match:
        return {
            "n_cases": _n_cases_from_name(name),
            "drift_amount_min": None,
            "noise_pct": None,
        }
    return {
        "n_cases": int(match.group(1)),
        "drift_amount_min": int(match.group(2)),
        "noise_pct": int(match.group(3)) if match.group(3) is not None else 0,
    }


def _ground_truth(n_cases):
    return [round(0.37 * n_cases), round(0.75 * n_cases)]


def _parse_detected_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [int(x) for x in value]

    text = str(value).strip()
    if text in ("", "ERROR", "[]"):
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []

    if isinstance(parsed, (list, tuple)):
        return [int(x) for x in parsed]
    return []


def evaluate_cps(detected, ground_truth, tolerance):
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


def evaluate_results_table(results_path, tol=None):
    """Evaluate an OCPA results Excel/CSV file using the CVDrift method."""
    ext = os.path.splitext(results_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        frame = pd.read_excel(results_path)
    else:
        frame = pd.read_csv(results_path)

    rename_map = {
        "log": "Log",
        "detected_changepoints": "Detected Changepoints",
    }
    frame = frame.rename(columns=rename_map)

    cp_col = "Detected Changepoints"
    if "Log" not in frame.columns or cp_col not in frame.columns:
        raise ValueError(
            f"Expected columns 'Log' and '{cp_col}' in {results_path}. "
            f"Found: {list(frame.columns)}"
        )

    total_tp = total_fp = total_fn = 0
    rows = []

    for _, row in frame.iterrows():
        file_name = str(row["Log"])
        detected = _parse_detected_list(row[cp_col])

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

    def _micro_from_group(group):
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
    output_base = os.path.splitext(results_path)[0]
    eval_df.to_csv(output_base + "_eval.csv", index=False)

    by_drift = pd.DataFrame()
    by_noise = pd.DataFrame()
    by_drift_noise = pd.DataFrame()

    if not eval_df.empty:
        by_drift = (
            eval_df.dropna(subset=["drift_amount_min"])
            .groupby("drift_amount_min", dropna=True)
            .apply(_micro_from_group)
            .reset_index()
            .rename(columns={"drift_amount_min": "Drift_Amount_min"})
        )
        by_drift.to_csv(output_base + "_eval_overall_micro.csv", index=False)

        by_drift_noise = (
            eval_df.dropna(subset=["drift_amount_min", "noise_pct"])
            .groupby(["drift_amount_min", "noise_pct"], dropna=True)
            .apply(_micro_from_group)
            .reset_index()
            .rename(columns={"drift_amount_min": "drift_magnitude_min", "noise_pct": "noise_level"})
        )
        by_drift_noise["noise_level"] = by_drift_noise["noise_level"].astype(int).astype(str) + "%"
        by_drift_noise.to_csv(output_base + "_eval_by_noise_micro.csv", index=False)

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
    print(
        f"Overall  -> Precision: {round(precision, 4):.4f} | "
        f"Recall: {round(recall, 4):.4f} | F1: {round(f1, 4):.4f}"
    )

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


# ------------------------------------------------------------------
# Dataset configurations (add your own here)
# ------------------------------------------------------------------

# Base directory for example logs
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_LOGS = os.path.join(_SCRIPT_DIR, "example_logs", "mdl")

DATASET_CONFIGS = {
    "BPI2017": {
        "filename": os.path.join(_EXAMPLE_LOGS, "BPI2017", "BPI2017-Full-MDL.csv"),
        "object_types": ["application", "offer"],
        "sep": ",",
        "timestamp_col": "event_timestamp",
        "start_timestamp_col": "event_start_timestamp",
        "description": "BPI Challenge 2017 — loan application process (MDL format)",
        # Features for the example drift detection step
        "event_features": [
            (avg, (feature_extraction.EVENT_SERVICE_TIME,
                   ("event_start_timestamp", "W_Validate application"))),
        ],
        "exec_features": [
            (sum, (feature_extraction.EXECUTION_IDENTITY, ())),
        ],
        # Large feature set for scalability experiment
        "scalability_features": [
            (avg, (feature_extraction.EXECUTION_THROUGHPUT, ())),
            (sum, (feature_extraction.EXECUTION_IDENTITY, ())),
            (avg, (feature_extraction.EXECUTION_FEATURE, ("event_RequestedAmount",))),
            (std_dev, (feature_extraction.EXECUTION_FEATURE, ("event_RequestedAmount",))),
            (max, (feature_extraction.EXECUTION_FEATURE, ("event_RequestedAmount",))),
            (sum, (feature_extraction.EXECUTION_FEATURE, ("event_RequestedAmount",))),
            (avg, (feature_extraction.EXECUTION_FEATURE, ("event_RequestedAmount",))),
            (avg, (feature_extraction.EXECUTION_FEATURE, ("event_OfferedAmount",))),
            (std_dev, (feature_extraction.EXECUTION_FEATURE, ("event_OfferedAmount",))),
            (max, (feature_extraction.EXECUTION_FEATURE, ("event_OfferedAmount",))),
            (sum, (feature_extraction.EXECUTION_FEATURE, ("event_OfferedAmount",))),
            (avg, (feature_extraction.EXECUTION_FEATURE, ("event_OfferedAmount",))),
            (avg, (feature_extraction.EXECUTION_SERVICE_TIME, ("event_start_timestamp",))),
            (std_dev, (feature_extraction.EXECUTION_SERVICE_TIME, ("event_start_timestamp",))),
            (sum, (feature_extraction.EXECUTION_SERVICE_TIME, ("event_start_timestamp",))),
            (max, (feature_extraction.EXECUTION_SERVICE_TIME, ("event_start_timestamp",))),
            (median, (feature_extraction.EXECUTION_SERVICE_TIME, ("event_start_timestamp",))),
            (avg, (feature_extraction.EXECUTION_AVG_SERVICE_TIME, ("event_start_timestamp",))),
            (max, (feature_extraction.EXECUTION_AVG_SERVICE_TIME, ("event_start_timestamp",))),
            (std_dev, (feature_extraction.EXECUTION_AVG_SERVICE_TIME, ("event_start_timestamp",))),
            (avg, (feature_extraction.EXECUTION_NUM_OBJECT, ())),
            (avg, (feature_extraction.EXECUTION_NUM_OF_STARTING_EVENTS, ())),
            (avg, (feature_extraction.EXECUTION_NUM_OF_EVENTS, ())),
            (avg, (feature_extraction.EXECUTION_UNIQUE_ACTIVITIES, ())),
            (median, (feature_extraction.EXECUTION_THROUGHPUT, ())),
            (max, (feature_extraction.EXECUTION_NUM_OBJECT, ())),
            (sum, (feature_extraction.EXECUTION_NUM_OF_STARTING_EVENTS, ())),
            (sum, (feature_extraction.EXECUTION_NUM_OF_EVENTS, ())),
            (avg, (feature_extraction.EXECUTION_UNIQUE_ACTIVITIES, ())),
            (max, (feature_extraction.EXECUTION_UNIQUE_ACTIVITIES, ())),
            (median, (feature_extraction.EXECUTION_UNIQUE_ACTIVITIES, ())),
            (max, (feature_extraction.EXECUTION_NUM_OF_END_EVENTS, ())),
            (avg, (feature_extraction.EXECUTION_NUM_OF_END_EVENTS, ())),
            (max, (feature_extraction.EXECUTION_LAST_EVENT_TIME_BEFORE, ())),
            (avg, (feature_extraction.EXECUTION_LAST_EVENT_TIME_BEFORE, ())),
            (std_dev, (feature_extraction.EXECUTION_THROUGHPUT, ())),
            (std_dev, (feature_extraction.EXECUTION_LAST_EVENT_TIME_BEFORE, ())),
            (max, (feature_extraction.EXECUTION_THROUGHPUT, ())),
        ],
    },
}


# ------------------------------------------------------------------
# XES → pseudo-OCEL conversion
# ------------------------------------------------------------------

def _is_xes(filename):
    """Check if a file is XES/XML based on extension."""
    return filename.lower().endswith((".xes", ".xes.gz", ".xml"))


def _pair_lifecycle_events(df):
    """
    If a DataFrame contains start/complete lifecycle pairs, merge them
    into single rows where ``event_start_timestamp`` comes from the
    *start* event and ``event_timestamp`` from the *complete* event.

    If no lifecycle column is found (or there are no start/complete
    pairs), the DataFrame is returned unchanged.
    """
    # Detect lifecycle column
    lc_col = None
    for c in df.columns:
        if "lifecycle" in c.lower():
            lc_col = c
            break
    if lc_col is None:
        return df

    vals = df[lc_col].astype(str).str.lower().str.strip()
    has_start = (vals == "start").any()
    has_complete = (vals == "complete").any()
    if not (has_start and has_complete):
        return df

    # Identify key columns
    case_col = None
    for c in ("Case ID", "case:concept:name"):
        if c in df.columns:
            case_col = c
            break
    if case_col is None:
        for c in df.columns:
            if "case" in c.lower():
                case_col = c
                break

    act_col = None
    for c in ("Activity", "concept:name"):
        if c in df.columns:
            act_col = c
            break
    if act_col is None:
        for c in df.columns:
            if "activity" in c.lower():
                act_col = c
                break

    ts_col = "event_timestamp" if "event_timestamp" in df.columns else None
    if ts_col is None:
        for c in df.columns:
            if "timestamp" in c.lower():
                ts_col = c
                break

    if case_col is None or act_col is None or ts_col is None:
        return df

    df[ts_col] = pd.to_datetime(df[ts_col])

    starts = df[vals == "start"].copy()
    completes = df[vals == "complete"].copy()

    # Add a sequential occurrence counter per (case, activity) so that
    # the i-th start of activity A in case X is matched to the i-th
    # complete of activity A in case X.
    starts["_occ"] = starts.groupby([case_col, act_col]).cumcount()
    completes["_occ"] = completes.groupby([case_col, act_col]).cumcount()

    merged = starts.merge(
        completes[[case_col, act_col, "_occ", ts_col]],
        on=[case_col, act_col, "_occ"],
        suffixes=("_start", "_complete"),
        how="inner",
    )

    ts_start = ts_col + "_start"
    ts_complete = ts_col + "_complete"

    merged["event_start_timestamp"] = merged[ts_start]
    merged["event_timestamp"] = merged[ts_complete]

    # Drop helper columns
    drop_cols = [c for c in (ts_start, ts_complete, "_occ", lc_col)
                 if c in merged.columns]
    merged.drop(columns=drop_cols, inplace=True, errors="ignore")

    print(f"  Paired {len(merged)} start/complete lifecycle events "
          f"(from {len(df)} raw rows)")
    return merged.reset_index(drop=True)


def _read_xes_to_df(path):
    """
    Parse a XES file into a flat DataFrame.
    Uses pm4py if available, otherwise falls back to xml.etree.

    If the log contains start/complete lifecycle transitions, they are
    automatically paired so that each row has a distinct
    ``event_start_timestamp`` and ``event_timestamp`` (= completion
    time), enabling proper activity-duration / service-time analysis.

    Returns a DataFrame with columns:
        Case ID, Activity, event_timestamp, event_start_timestamp
    """
    try:
        import pm4py
        log = pm4py.read_xes(path)
        # pm4py returns a DataFrame in newer versions
        if isinstance(log, pd.DataFrame):
            df = log.copy()
        else:
            # Older pm4py returns EventLog object
            rows = []
            for trace in log:
                case_id = trace.attributes.get("concept:name", str(id(trace)))
                for ev in trace:
                    rows.append({
                        "Case ID": str(case_id),
                        "Activity": ev.get("concept:name", ""),
                        "event_timestamp": ev.get("time:timestamp"),
                        "Resource": ev.get("org:resource", ""),
                        "lifecycle:transition": ev.get(
                            "lifecycle:transition", "complete"),
                    })
            df = pd.DataFrame(rows)
            # fall through to normalisation below

        # Normalise column names — runs for BOTH pm4py code paths
        col_map = {}
        for col in df.columns:
            low = col.lower().replace(" ", "_")
            if low in ("case:concept:name", "case_id", "caseid"):
                col_map[col] = "Case ID"
            elif low in ("concept:name", "activity"):
                col_map[col] = "Activity"
            elif low in ("time:timestamp", "timestamp"):
                col_map[col] = "event_timestamp"
            elif low in ("org:resource", "resource"):
                col_map[col] = "Resource"
        df = df.rename(columns=col_map)

        # Drop duplicate columns that can arise when pm4py exposes
        # both "time:timestamp" and "start_timestamp" (both mapped to
        # the same normalised name).
        df = df.loc[:, ~df.columns.duplicated()]

        if "Case ID" not in df.columns:
            # Try the pm4py case column
            for c in df.columns:
                if "case" in c.lower():
                    df.rename(columns={c: "Case ID"}, inplace=True)
                    break

        # Pair start/complete lifecycle events → real durations
        df = _pair_lifecycle_events(df)

        if "event_start_timestamp" not in df.columns:
            if "event_timestamp" in df.columns:
                df["event_start_timestamp"] = df["event_timestamp"]
        return df

    except ImportError:
        pass

    # ---- Lightweight XML fallback (no pm4py) ----
    import xml.etree.ElementTree as ET
    if path.lower().endswith(".xes.gz"):
        import gzip
        with gzip.open(path, "rb") as fh:
            tree = ET.parse(fh)
    else:
        tree = ET.parse(path)
    root = tree.getroot()

    def _local(elem, tag):
        return [c for c in list(elem) if c.tag.endswith(tag)]

    rows = []
    for ti, trace in enumerate(_local(root, "trace")):
        case_id = str(ti)
        for s in [c for c in list(trace) if c.tag.endswith("string")]:
            if s.attrib.get("key") in ("concept:name", "case:concept:name"):
                case_id = s.attrib.get("value", str(ti))
                break
        for ev in _local(trace, "event"):
            act = ts = resource = ""
            lifecycle = "complete"
            for attr in list(ev):
                key = attr.attrib.get("key", "")
                val = attr.attrib.get("value", "")
                if attr.tag.endswith("string") and key in ("concept:name", "activity"):
                    act = val
                if attr.tag.endswith("date") and key in ("time:timestamp", "timestamp"):
                    ts = val
                if attr.tag.endswith("string") and key in ("org:resource", "resource"):
                    resource = val
                if attr.tag.endswith("string") and key == "lifecycle:transition":
                    lifecycle = val
            rows.append({
                "Case ID": case_id,
                "Activity": act,
                "event_timestamp": ts,
                "Resource": resource,
                "lifecycle:transition": lifecycle,
            })
    df = pd.DataFrame(rows)

    # Pair start/complete lifecycle events → real durations
    df = _pair_lifecycle_events(df)

    if "event_start_timestamp" not in df.columns:
        if "event_timestamp" in df.columns:
            df["event_start_timestamp"] = df["event_timestamp"]
    return df


# ==================================================================
# IMPROVED: _infer_start_timestamps  (v2 — CVDrift-style)
# ==================================================================
# Changes vs. v1:
#   1. Computes median inter-event gap **per activity** across the log
#   2. Clips outlier gaps at 3× the activity median (noise robustness)
#   3. Last event in each case gets the activity median (not duration=0)
# ==================================================================

def _compute_median_gaps_per_activity(df, case_col, act_col, ts_col):
    """
    Scan all cases and compute the median consecutive-event gap (seconds)
    grouped by the activity of the *current* event.

    Returns
    -------
    dict[str, float]
        Activity name → median gap in seconds.
    """
    from collections import defaultdict
    gaps = defaultdict(list)

    for _cid, grp in df.groupby(case_col):
        grp = grp.sort_values(ts_col)
        ts_vals = grp[ts_col].values  # numpy datetime64
        act_vals = grp[act_col].values

        for i in range(len(ts_vals) - 1):
            delta = (ts_vals[i + 1] - ts_vals[i]) / np.timedelta64(1, "s")
            if delta >= 0:
                gaps[str(act_vals[i])].append(delta)

    med = {}
    for act, vals in gaps.items():
        if vals:
            vals_sorted = sorted(vals)
            mid = len(vals_sorted) // 2
            med[act] = vals_sorted[mid]
    return med


def _infer_start_timestamps(df, case_col, ts_col, act_col=None):
    """
    Infer ``event_start_timestamp`` for single-timestamp logs using
    **per-activity median gaps** with outlier clipping — the same
    strategy used by CVDrift (pipeline/io.py).

    For each event (except the first in a case):
      - raw_gap = timestamp[i] − timestamp[i−1]  (inter-event gap)
      - If raw_gap > 3 × activity_median  →  use activity_median (clip)
      - Otherwise use raw_gap

    For the first event in each case:
      - event_start_timestamp = event_timestamp  (duration = 0)

    For the last event in each case:
      - event_start_timestamp = event_timestamp − activity_median
        (instead of the v1 approach which left duration = 0)

    Parameters
    ----------
    df : pd.DataFrame
        Must already have ``ts_col`` as datetime.
    case_col : str
        Column identifying the case.
    ts_col : str
        Column with the (single) event timestamp.
    act_col : str or None
        Column with the activity name.  If None, falls back to v1
        behaviour (simple shift, no clipping).

    Returns
    -------
    pd.DataFrame
        df with ``event_start_timestamp`` column added / overwritten.
    """
    df = df.sort_values([case_col, ts_col]).copy()

    # --- Fallback: if no activity column, use the original v1 logic ---
    if act_col is None or act_col not in df.columns:
        df["event_start_timestamp"] = df.groupby(case_col)[ts_col].shift(1)
        df["event_start_timestamp"] = df["event_start_timestamp"].fillna(
            df[ts_col])
        print(f"  Inferred event_start_timestamp (v1 fallback, no activity col)"
              f" — {df[case_col].nunique()} cases")
        return df

    # --- v2: per-activity median gaps with outlier clipping ---
    median_gaps = _compute_median_gaps_per_activity(
        df, case_col, act_col, ts_col)
    print(f"  Per-activity median gaps (seconds):")
    for act, gap in sorted(median_gaps.items()):
        print(f"    {act}: {gap:.1f}s")

    # Compute the raw inter-event gap via shift within each case
    df["_prev_ts"] = df.groupby(case_col)[ts_col].shift(1)
    df["_raw_gap_s"] = (
        (df[ts_col] - df["_prev_ts"]).dt.total_seconds()
    )

    # Mark first and last event in each case
    df["_is_first"] = df["_prev_ts"].isna()
    df["_rank_desc"] = df.groupby(case_col).cumcount(ascending=False)
    df["_is_last"] = df["_rank_desc"] == 0

    # For each event, look up the activity median
    df["_act_median"] = df[act_col].map(median_gaps)
    # Fallback: if an activity has no median (rare), use global median
    global_median = np.median(list(median_gaps.values())) if median_gaps else 0.0
    df["_act_median"] = df["_act_median"].fillna(global_median)

    # Compute the clipped gap
    def _clipped_gap(row):
        if row["_is_first"]:
            return 0.0  # first event: duration = 0
        raw = row["_raw_gap_s"]
        med = row["_act_median"]
        if pd.isna(raw) or raw < 0:
            return med if med > 0 else 0.0
        # Clip at 3× activity median
        if med > 0 and raw > 3.0 * med:
            return med
        return raw

    df["_clipped_gap_s"] = df.apply(_clipped_gap, axis=1)

    # event_start_timestamp = event_timestamp − clipped_gap
    df["event_start_timestamp"] = (
        df[ts_col] - pd.to_timedelta(df["_clipped_gap_s"], unit="s")
    )

    # Last event fix: if the last event currently has duration = 0
    # (because there's no next event to define the gap), assign
    # the activity median as the duration.
    last_mask = df["_is_last"] & (df["_clipped_gap_s"] == 0) & ~df["_is_first"]
    n_last_fixed = last_mask.sum()
    if n_last_fixed > 0:
        df.loc[last_mask, "event_start_timestamp"] = (
            df.loc[last_mask, ts_col]
            - pd.to_timedelta(df.loc[last_mask, "_act_median"], unit="s")
        )

    # For single-event cases (first AND last), keep duration = 0
    # (no information to estimate from)

    # Count clipped outliers for diagnostics
    non_first = ~df["_is_first"]
    n_clipped = (
        non_first
        & (df["_act_median"] > 0)
        & (df["_raw_gap_s"] > 3.0 * df["_act_median"])
    ).sum()
    n_total = non_first.sum()

    # Clean up helper columns
    df.drop(columns=["_prev_ts", "_raw_gap_s", "_is_first", "_rank_desc",
                      "_is_last", "_act_median", "_clipped_gap_s"],
            inplace=True)

    print(f"  Inferred event_start_timestamp (v2: per-activity median + clip)"
          f" — {df[case_col].nunique()} cases")
    print(f"    Outliers clipped: {n_clipped}/{n_total} non-first events"
          f" ({100*n_clipped/n_total:.1f}%)" if n_total > 0 else "")
    if n_last_fixed > 0:
        print(f"    Last-event durations filled: {n_last_fixed}")

    return df


# ==================================================================
# IMPROVED: _xes_to_ocel_df  (v2 — passes activity column)
# ==================================================================

def _xes_to_ocel_df(xes_path, max_rows=None):
    """
    Convert a XES file into a pseudo-OCEL DataFrame.

    Strategy: each Case ID becomes an object of type "case".
    The event_timestamp and event_start_timestamp columns are created
    from the XES timestamp.  An event_activity column mirrors Activity.

    For logs with start/complete lifecycle transitions the two timestamps
    come from the paired events.  For single-timestamp logs (e.g. noisy
    variants) the start timestamp is inferred using the improved v2
    strategy: per-activity median gaps with 3× outlier clipping.

    Returns
    -------
    event_df : pd.DataFrame
        DataFrame ready for OCEL(event_df, ["case"])
    object_types : list[str]
        Always ["case"] for XES-derived logs.
    """
    print(f"  Converting XES → pseudo-OCEL ...")
    df = _read_xes_to_df(xes_path)

    if max_rows:
        df = df.head(max_rows)
        print(f"  (limited to first {max_rows} rows for testing)")

    # Ensure required columns exist
    if "event_timestamp" not in df.columns:
        for c in df.columns:
            if "timestamp" in c.lower():
                df.rename(columns={c: "event_timestamp"}, inplace=True)
                break
    if "event_start_timestamp" not in df.columns:
        df["event_start_timestamp"] = df["event_timestamp"]

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["event_start_timestamp"] = pd.to_datetime(df["event_start_timestamp"])

    # If start == end for ALL rows the log has no lifecycle pairs.
    # Fall back to improved inter-event elapsed time estimation.
    ts_equal = (df["event_timestamp"] == df["event_start_timestamp"]).all()
    if ts_equal:
        # Find the case column
        _case_col = "Case ID" if "Case ID" in df.columns else None
        if _case_col is None:
            for c in df.columns:
                if "case" in c.lower():
                    _case_col = c
                    break

        # IMPROVEMENT (v2): find the activity column so we can do
        # per-activity median gaps with outlier clipping
        _act_col = "Activity" if "Activity" in df.columns else None
        if _act_col is None:
            for c in df.columns:
                if "activity" in c.lower():
                    _act_col = c
                    break

        if _case_col is not None:
            df = _infer_start_timestamps(
                df, _case_col, "event_timestamp", act_col=_act_col)

    # Map Activity → event_activity  (OCPA naming convention)
    if "event_activity" not in df.columns:
        act_col = "Activity" if "Activity" in df.columns else None
        if act_col is None:
            for c in df.columns:
                if "activity" in c.lower():
                    act_col = c
                    break
        if act_col:
            df["event_activity"] = df[act_col]

    # Create pseudo object-type column "case":  each event maps to its case ID
    case_col = "Case ID" if "Case ID" in df.columns else None
    if case_col is None:
        for c in df.columns:
            if "case" in c.lower():
                case_col = c
                break
    if case_col is None:
        raise ValueError(
            "Cannot find a Case ID column in the XES. "
            "Columns found: " + str(list(df.columns)))

    # OCEL expects the object column to contain lists of object IDs
    df["case"] = df[case_col].apply(lambda x: [str(x)])

    # event_id
    df["event_id"] = list(range(len(df)))
    df.index = list(range(len(df)))

    n_cases = df[case_col].nunique()
    print(f"  XES → pseudo-OCEL: {len(df)} events, "
          f"{n_cases} cases, object type = 'case'")
    return df, ["case"]


# ------------------------------------------------------------------
# Load OCEL from CSV or XES
# ------------------------------------------------------------------

def load_ocel(filename, object_types=None, sep=",",
              timestamp_col="event_timestamp",
              start_timestamp_col="event_start_timestamp",
              max_rows=None):
    """
    Load a CSV (MDL) or XES file into an OCPA OCEL object.

    For XES files the log is automatically converted to a pseudo-OCEL
    using a single synthetic object type "case".
    """
    if not os.path.isfile(filename):
        print(f"\nERROR: File not found: {filename}")
        print("Please check the path and try again.")
        sys.exit(1)

    print(f"\nLoading: {filename}")

    # ---- XES path ------------------------------------------------
    if _is_xes(filename):
        event_df, object_types = _xes_to_ocel_df(filename, max_rows)
        ocel = OCEL(event_df, object_types)
        print(f"  Events: {len(event_df)}")
        print(f"  Process executions: {len(ocel.cases)}")
        print(f"  Object types: {object_types}  (synthetic from XES)")
        return ocel

    # ---- CSV / MDL path ------------------------------------------
    if object_types is None:
        raise ValueError(
            "object_types is required for CSV files. "
            "Pass --object-types on the command line.")

    event_df = pd.read_csv(filename, sep=sep)
    if max_rows:
        event_df = event_df[:max_rows]
        print(f"  (limited to first {max_rows} rows for testing)")

    event_df[timestamp_col] = pd.to_datetime(event_df[timestamp_col])
    if start_timestamp_col and start_timestamp_col in event_df.columns:
        event_df[start_timestamp_col] = pd.to_datetime(
            event_df[start_timestamp_col])

    for ot in object_types:
        event_df[ot] = event_df[ot].map(
            lambda x: [y.strip() for y in x.split(",")]
            if isinstance(x, str) else [])

    event_df["event_id"] = list(range(len(event_df)))
    event_df.index = list(range(len(event_df)))
    event_df["event_id"] = event_df["event_id"].astype(float).astype(int)

    ocel = OCEL(event_df, object_types)
    print(f"  Events: {len(event_df)}")
    print(f"  Process executions: {len(ocel.cases)}")
    print(f"  Object types: {object_types}")
    return ocel


# ------------------------------------------------------------------
# Step 1: Drift detection + Granger causality
# ------------------------------------------------------------------

def run_drift_detection(ocel, event_features, exec_features,
                        window_days=7, pen=0.1, p_value=0.05,
                        trim=4, output_dir="."):
    """
    Construct time series, detect change points (PELT), and test
    Granger causality between features.

    Returns
    -------
    dict with keys: time_series, time_index, changepoints,
                    explainable_drifts
    """
    print("\n" + "=" * 60)
    print("  Step 1: Drift Detection + Granger Causality")
    print("=" * 60)

    inclusion_fn = ocpa.algo.filtering.log.time_filtering.events

    s, time_index = time_series.construct_time_series(
        ocel, timedelta(days=window_days),
        event_features, exec_features, inclusion_fn)

    print(f"  Time series keys: {list(s.keys())}")
    print(f"  Length before trim: {len(time_index)}")

    # Guard: adapt trim to available length so we never end up empty
    # Need at least 3 points after trimming for meaningful CPD
    n_pts = len(time_index)
    effective_trim = trim
    min_remaining = 3
    if n_pts - 2 * trim < min_remaining:
        effective_trim = max(0, (n_pts - min_remaining) // 2)
        print(f"  WARNING: series too short ({n_pts}) for trim={trim}, "
              f"using trim={effective_trim} "
              f"(keeping {n_pts - 2*effective_trim} points)")

    # Trim edges to reduce start/end recording artefacts
    if effective_trim > 0:
        for k in s.keys():
            s[k] = s[k][effective_trim:-effective_trim]
        time_index = time_index[effective_trim:-effective_trim]
    print(f"  Length after trim: {len(time_index)}")

    if len(time_index) < 3:
        print("  ERROR: time series too short after trim "
              f"({len(time_index)} points). "
              "Try a smaller --window-days or load more data.")
        return {
            "time_series": s, "time_index": time_index,
            "changepoints": {}, "explainable_drifts": [],
        }

    # Change-point detection (PELT on normalized series)
    loc = {}
    for k in s.keys():
        arr = s[k]
        # Guard: skip empty / all-NaN / all-zero series
        if len(arr) == 0:
            loc[k] = []
            continue
        max_val = np.nanmax(arr)
        if not np.isfinite(max_val) or max_val == 0:
            print(f"  WARNING: feature {k} has max=0 or all-NaN, skipping CPD")
            loc[k] = []
            continue
        loc[k] = [bp for bp in rpt.Pelt().fit(
            arr / max_val).predict(pen=pen)
            if bp != len(arr)]
    print(f"  Changepoints: {loc}")

    # --- Map changepoint positions to dates and MEDIAN case IDs ---
    # Collect all unique CP positions across features
    all_cp_positions = sorted(
        {pos for cps in loc.values() for pos in cps})

    median_case_ids = []  # list of (position, median_case_id) tuples

    if all_cp_positions:
        print(f"\n  --- Changepoint → Date & Median Case-ID ---")
        # Get the underlying event DataFrame
        # OCPA stores it as ocel.log (a DataFrame directly)
        try:
            event_df = ocel.log
            if not isinstance(event_df, pd.DataFrame):
                event_df = event_df.log  # fallback for other OCPA versions
        except AttributeError:
            event_df = None

        for pos in all_cp_positions:
            if pos < len(time_index):
                cp_date = time_index[pos]
                # The previous window boundary
                prev_date = time_index[pos - 1] if pos > 0 else time_index[0]
                feats_here = [
                    str(k) for k, cps in loc.items() if pos in cps]
                print(f"\n  Changepoint at position {pos}:")
                print(f"    Window date:  {cp_date}")
                print(f"    Features:     {feats_here}")

                # Find case IDs in the drift window [prev_date, cp_date]
                if event_df is not None:
                    ts_col = None
                    for c in ("event_timestamp", "time:timestamp",
                              "timestamp"):
                        if c in event_df.columns:
                            ts_col = c
                            break
                    # Find the object-type / case column
                    case_col = None
                    for c in event_df.columns:
                        if c in ("Case ID", "case"):
                            case_col = c
                            break
                    if ts_col and case_col:
                        ts_vals = pd.to_datetime(event_df[ts_col])
                        mask = (ts_vals >= prev_date) & (ts_vals < cp_date)
                        # Extract case IDs (may be stored as lists for OCEL)
                        case_vals = event_df.loc[mask, case_col]
                        case_ids = set()
                        for v in case_vals:
                            if isinstance(v, list):
                                case_ids.update(str(x) for x in v)
                            else:
                                case_ids.add(str(v))
                        case_ids = sorted(case_ids)

                        # --- Compute MEDIAN case ID ---
                        # Extract numeric parts from case IDs for median
                        import re as _re
                        numeric_ids = []
                        for cid in case_ids:
                            nums = _re.findall(r'\d+', cid)
                            if nums:
                                numeric_ids.append(int(nums[-1]))
                        if numeric_ids:
                            numeric_ids.sort()
                            mid = len(numeric_ids) // 2
                            median_val = numeric_ids[mid]
                            median_case_ids.append((pos, median_val))
                            print(f"    Cases in window: {len(case_ids)}  "
                                  f"→ Median Case ID: {median_val}")
                        elif case_ids:
                            # Fallback: take the middle element alphabetically
                            mid_case = case_ids[len(case_ids) // 2]
                            median_case_ids.append((pos, mid_case))
                            print(f"    Cases in window: {len(case_ids)}  "
                                  f"→ Median Case ID: {mid_case}")
                        else:
                            print(f"    No cases found in window")
                    else:
                        print(f"    (Could not find timestamp/case columns "
                              f"to map case IDs)")
            else:
                print(f"\n  Changepoint at position {pos}: "
                      f"beyond time_index (endpoint)")

    # Granger causality tests
    explainable_drifts = []
    for feat_1 in s.keys():
        for feat_2 in s.keys():
            if feat_1 == feat_2:
                continue
            for d in loc.get(feat_1, []):
                for d_ in loc.get(feat_2, []):
                    if d_ < d:
                        try:
                            res = grangercausalitytests(
                                pd.DataFrame({feat_1: s[feat_1],
                                              feat_2: s[feat_2]}),
                                [d - d_])
                        except (ValueError, stats_exceptions.InfeasibleTestError):
                            continue
                        pv = res[d - d_][0]["ssr_ftest"][1]
                        if pv <= p_value:
                            explainable_drifts.append(
                                (feat_1, feat_2, d, d_, pv))

    print(f"\n  Explainable drifts found: {len(explainable_drifts)}")
    for drift in explainable_drifts:
        print(f"    {drift[0]} <- {drift[1]}  "
              f"(lags {drift[2]-drift[3]}, p={drift[4]:.4f})")

    # --- Visualization ---
    data_df = pd.DataFrame({k: s[k] for k in s.keys()})
    viz_df = pd.concat([pd.DataFrame({"date": time_index}), data_df], axis=1)
    viz_df.set_index("date", inplace=True)

    plt.clf()
    sns.set_style("darkgrid")
    sns.lineplot(data=viz_df)
    out_path = os.path.join(output_dir, "time_series_all.png")
    plt.savefig(out_path, dpi=300)
    print(f"  Saved: {out_path}")

    # Individual feature plots
    for feat in s.keys():
        plt.clf()
        feat_df = pd.DataFrame({feat: s[feat]})
        fviz = pd.concat([pd.DataFrame({"date": time_index}), feat_df], axis=1)
        fviz.set_index("date", inplace=True)
        sns.set_style("darkgrid")
        sns.lineplot(data=fviz)
        feat_name = feat[1][0] if isinstance(feat, tuple) else str(feat)
        out_path = os.path.join(output_dir, f"time_series_{feat_name}.png")
        plt.savefig(out_path, dpi=300)
        print(f"  Saved: {out_path}")

    return {
        "time_series": s,
        "time_index": time_index,
        "changepoints": loc,
        "explainable_drifts": explainable_drifts,
        "median_case_ids": median_case_ids,
    }


# ------------------------------------------------------------------
# Step 2: Inclusion function comparison
# ------------------------------------------------------------------

def run_inclusion_comparison(ocel, window_days=7, output_dir="."):
    """Compare time series under different inclusion functions."""
    print("\n" + "=" * 60)
    print("  Step 2: Inclusion Function Comparison")
    print("=" * 60)

    inclusion_functions = [
        ocpa.algo.filtering.log.time_filtering.start,
        ocpa.algo.filtering.log.time_filtering.end,
        ocpa.algo.filtering.log.time_filtering.contained,
        ocpa.algo.filtering.log.time_filtering.spanning,
        ocpa.algo.filtering.log.time_filtering.events,
    ]

    feat_to_s = {}
    for f_in in inclusion_functions:
        st = time.time()
        s, time_index = time_series.construct_time_series(
            ocel, timedelta(days=window_days),
            [(avg, (feature_extraction.EVENT_SERVICE_TIME, ()))],
            [(avg, (feature_extraction.EXECUTION_THROUGHPUT, ()))],
            f_in)
        elapsed = time.time() - st
        print(f"  {f_in.__name__}: {elapsed:.2f}s")
        for feat in s.keys():
            feat_to_s.setdefault(feat, []).append(
                (f_in.__name__, s[feat], time_index))

    for feat in feat_to_s.keys():
        plt.clf()
        sns.set(rc={"figure.figsize": (24, 8)})
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.titlesize"] = 14
        data_df = pd.DataFrame(
            {name: vals for (name, vals, _) in feat_to_s[feat]})
        viz_df = pd.concat(
            [pd.DataFrame({"date": feat_to_s[list(feat_to_s.keys())[0]][0][2]}),
             data_df], axis=1)
        viz_df.set_index("date", inplace=True)
        sns.set_style("darkgrid")
        plot_ = sns.lineplot(data=viz_df)
        for i, label in enumerate(plot_.get_xticklabels()):
            label.set_visible(i % 2 == 0)
        feat_name = feat[1][0] if isinstance(feat, tuple) else str(feat)
        plot_.set_title(f"Inclusion Functions — {feat_name}")
        plot_.set_ylabel(
            "Avg throughput (s)" if "throughput" in feat_name
            else "Avg #objects/event")
        plot_.set_xlabel("Date")
        out_path = os.path.join(
            output_dir, f"inclusion_comparison_{feat_name}.png")
        plt.savefig(out_path, dpi=600)
        print(f"  Saved: {out_path}")


# ------------------------------------------------------------------
# Step 3: Window-size comparison
# ------------------------------------------------------------------

def run_window_comparison(ocel, window_array=None, output_dir="."):
    """Compare time series under different window sizes."""
    print("\n" + "=" * 60)
    print("  Step 3: Window-Size Comparison")
    print("=" * 60)

    if window_array is None:
        window_array = [28, 6, 1, 7]

    feat_to_s = {}
    for w in window_array:
        s, time_index = time_series.construct_time_series(
            ocel, timedelta(days=w), [],
            [(avg, (feature_extraction.EXECUTION_THROUGHPUT, ()))],
            ocpa.algo.filtering.log.time_filtering.start)
        for feat in s.keys():
            feat_to_s.setdefault(feat, []).append(
                (f"{w} days", s[feat], time_index))

    for w in window_array:
        s, time_index = time_series.construct_time_series(
            ocel, timedelta(days=w),
            [(avg, (feature_extraction.EVENT_SERVICE_TIME, ()))],
            [], ocpa.algo.filtering.log.time_filtering.events)
        for feat in s.keys():
            feat_to_s.setdefault(feat, []).append(
                (f"{w} days", s[feat], time_index))

    for feat in feat_to_s.keys():
        data_df = pd.DataFrame()
        plt.clf()
        sns.set(rc={"figure.figsize": (24, 8)})
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.titlesize"] = 14
        for (w, s_vals, ti) in feat_to_s[feat]:
            if w == "1 days":
                data_df["date"] = ti
                data_df.set_index("date", inplace=True)
        for (w, s_vals, ti) in feat_to_s[feat]:
            for i in range(len(s_vals)):
                if s_vals[i] > 0.001:
                    data_df.loc[ti[i], w] = s_vals[i]
        sns.set_style("darkgrid")
        plot_ = sns.lineplot(data=data_df)
        for i, label in enumerate(plot_.get_xticklabels()):
            label.set_visible(i % 2 == 0)
        feat_name = feat[1][0] if isinstance(feat, tuple) else str(feat)
        plot_.set_title(f"Window Sizes — {feat_name}")
        plot_.set_ylabel(
            "Avg throughput (s)" if "throughput" in feat_name
            else "Avg #objects/event")
        plot_.set_xlabel("Date")
        out_path = os.path.join(
            output_dir, f"window_comparison_{feat_name}.png")
        plt.savefig(out_path, dpi=600)
        print(f"  Saved: {out_path}")


# ------------------------------------------------------------------
# Step 4: Scalability experiment
# ------------------------------------------------------------------

def run_scalability(ocel, feature_array, p_value=0.01,
                    window_days=7, pen=0.05, output_dir="."):
    """Measure runtime as a function of the number of features."""
    print("\n" + "=" * 60)
    print("  Step 4: Scalability Experiment")
    print("=" * 60)

    times = {}
    explainable_drifts = []

    for l in range(1, len(feature_array)):
        n_feat = l + 1
        # Time-series extraction
        st = time.time()
        s, time_index = time_series.construct_time_series(
            ocel, timedelta(days=window_days), [],
            feature_array[:n_feat],
            ocpa.algo.filtering.log.time_filtering.start)
        extraction_time = time.time() - st

        # Change-point detection
        st = time.time()
        loc = {}
        for k in s.keys():
            arr = s[k]
            if len(arr) == 0:
                loc[k] = []
                continue
            max_val = np.nanmax(arr)
            if not np.isfinite(max_val) or max_val == 0:
                loc[k] = []
                continue
            loc[k] = [bp for bp in rpt.Pelt().fit(
                arr / max_val).predict(pen=pen)]
        detection_time = time.time() - st

        # Granger causality
        st = time.time()
        for feat_1 in s.keys():
            for feat_2 in s.keys():
                if feat_1 == feat_2:
                    continue
                for d in loc.get(feat_1, []):
                    for d_ in loc.get(feat_2, []):
                        if d_ < d:
                            try:
                                res = grangercausalitytests(
                                    pd.DataFrame({feat_1: s[feat_1],
                                                  feat_2: s[feat_2]}),
                                    [d - d_])
                            except (ValueError,
                                    stats_exceptions.InfeasibleTestError):
                                continue
                            pv = res[d - d_][0]["ssr_ftest"][1]
                            if pv <= p_value:
                                explainable_drifts.append(
                                    (feat_1, feat_2, d, d_, pv))
        correlation_time = time.time() - st

        total = extraction_time + detection_time + correlation_time
        times[n_feat] = [total, extraction_time,
                         detection_time, correlation_time]
        print(f"  {n_feat} features: total={total:.2f}s  "
              f"(extract={extraction_time:.2f}, "
              f"detect={detection_time:.2f}, "
              f"granger={correlation_time:.2f})")

    print(f"\n  Explainable drifts: {len(explainable_drifts)}")

    # Plot
    plt.clf()
    time_df = pd.DataFrame.from_dict(
        times, orient="index",
        columns=["Total Time", "Extraction Time",
                 "Concept Drift Detection", "Granger Causality"])
    sns.set_style("darkgrid")
    plot_ = sns.lineplot(data=time_df)
    plot_.set_title("Runtime for Different Number of Features")
    plot_.set_ylabel("Runtime (s)")
    plot_.set_xlabel("Number of Features")
    out_path = os.path.join(output_dir, "scalability_runtimes.png")
    plt.savefig(out_path, dpi=600)
    print(f"  Saved: {out_path}")

    return times


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Batch processing helpers
# ------------------------------------------------------------------

def _collect_log_files(folder):
    """Recursively collect supported event-log paths, sorted."""
    extensions = (".xes", ".xes.gz", ".csv", ".mxml")
    files = []
    for root, _dirs, entries in os.walk(folder):
        for entry in entries:
            if any(entry.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, entry))
    files.sort()
    return files


def _run_single_file(filepath, args):
    """
    Run drift detection on a single file and return a dict of results.
    Returns: dict with keys log, median_case_ids, changepoints,
             n_explainable, elapsed, error
    """
    log_name = os.path.basename(filepath)
    is_xes = _is_xes(filepath)
    config = {
        "filename": filepath,
        "object_types": args.object_types if not is_xes else ["case"],
        "sep": args.sep,
        "timestamp_col": "event_timestamp",
        "start_timestamp_col": "event_start_timestamp",
        "event_features": [],
        "exec_features": [
            (avg, (feature_extraction.EXECUTION_AVG_SERVICE_TIME, ("event_start_timestamp",))),
            (avg, (feature_extraction.EXECUTION_SERVICE_TIME, ("event_start_timestamp",))),
        ],
    }

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(filepath))
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    try:
        ocel = load_ocel(
            filename=config["filename"],
            object_types=config["object_types"],
            sep=config.get("sep", ","),
            timestamp_col=config.get("timestamp_col", "event_timestamp"),
            start_timestamp_col=config.get("start_timestamp_col",
                                            "event_start_timestamp"),
            max_rows=args.max_rows,
        )
        result = run_drift_detection(
            ocel,
            event_features=config.get("event_features", []),
            exec_features=config.get("exec_features", []),
            window_days=args.window_days,
            pen=args.pen,
            p_value=args.p_value,
            output_dir=output_dir,
        )
        elapsed = time.time() - t0
        median_ids = result.get("median_case_ids", [])
        n_cps = sum(len(v) for v in result["changepoints"].values())
        n_expl = len(result.get("explainable_drifts", []))
        return {
            "log": log_name,
            "detected_changepoints": [m[1] for m in median_ids],
            "n_changepoints": n_cps,
            "n_explainable": n_expl,
            "elapsed": round(elapsed, 2),
            "error": None,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ERROR on {log_name}: {exc}")
        return {
            "log": log_name,
            "detected_changepoints": [],
            "n_changepoints": 0,
            "n_explainable": 0,
            "elapsed": round(elapsed, 2),
            "error": str(exc),
        }


def batch_run(folder, args, out_excel):
    """
    Process all event logs in *folder* one by one, collect the median
    case ID at each detected changepoint, and write results to Excel.
    """
    log_files = _collect_log_files(folder)
    if not log_files:
        print(f"No supported log files found in {folder}")
        return

    print(f"\nFound {len(log_files)} log file(s) in {folder}")
    print(f"Results will be saved to {out_excel}\n")

    rows = []
    for idx, fpath in enumerate(log_files, 1):
        log_name = os.path.basename(fpath)
        print("\n" + "=" * 60)
        print(f"[{idx}/{len(log_files)}] Processing: {log_name}")
        print("=" * 60)

        res = _run_single_file(fpath, args)
        rows.append(res)
        print(f"  -> Detected median case IDs: {res['detected_changepoints']}")
        print(f"  -> Duration: {res['elapsed']}s")

    # --- Build Excel output ---
    out_df = pd.DataFrame(rows)
    out_df.to_excel(out_excel, index=False, sheet_name="Results")
    print(f"\n{'=' * 60}")
    print(f"Batch complete. {len(rows)} files processed.")
    print(f"Results saved to: {out_excel}")
    print(f"{'=' * 60}")
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Run OCPA-based concept drift experiments (v2 — improved lifecycle)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preconfigured datasets:
  %(prog)s --dataset BPI2017

Single file:
  %(prog)s --file "path/to/log.xes" --steps drift

Batch mode (process all files in a folder → Excel):
  %(prog)s --dir "path/to/folder" --out results.xlsx
  %(prog)s --dir "path/to/folder" --out results.xlsx --pen 0.05

Custom CSV:
  %(prog)s --file "path/to/data.csv" --object-types application offer
        """,
    )
    parser.add_argument(
        "--dataset", default=None,
        choices=list(DATASET_CONFIGS.keys()),
        help="Use a preconfigured dataset (see DATASET_CONFIGS in the file).",
    )
    parser.add_argument(
        "--file", default=None,
        help="Path to an MDL CSV or XES file (overrides --dataset). "
             "XES files are automatically converted to pseudo-OCEL.",
    )
    parser.add_argument(
        "--dir", default=None,
        help="Folder of event logs — batch mode. Processes every "
             "XES/CSV file and saves results to Excel.",
    )
    parser.add_argument(
        "--eval-file", default=None,
        help="Evaluate an existing OCPA results file (.xlsx or .csv) and write CVDrift-style evaluation CSVs.",
    )
    parser.add_argument(
        "--out", default="ocpa_results.xlsx",
        help="Output Excel path for batch mode (default: ocpa_results.xlsx).",
    )
    parser.add_argument(
        "--object-types", nargs="+", default=None,
        help="Object-type column names in the CSV (required for CSV; "
             "auto-detected as ['case'] for XES).",
    )
    parser.add_argument(
        "--sep", default=",",
        help="CSV separator (default: comma).",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Load only first N rows (for quick testing).",
    )
    parser.add_argument(
        "--window-days", type=int, default=7,
        help="Window size in days for time-series construction (default: 7).",
    )
    parser.add_argument(
        "--pen", type=float, default=0.1,
        help="PELT penalty parameter (default: 0.1).",
    )
    parser.add_argument(
        "--p-value", type=float, default=0.05,
        help="Granger causality significance level (default: 0.05).",
    )
    parser.add_argument(
        "--tol", type=int, default=None,
        help="Optional tolerance override for evaluation. Default: 10%% of dataset size.",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation CSV generation in batch mode.",
    )
    parser.add_argument(
        "--steps", nargs="+",
        choices=["drift", "inclusion", "window", "scalability", "all"],
        default=["all"],
        help="Which experiment steps to run (default: all).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output plots (default: same as input file).",
    )
    args = parser.parse_args()

    if args.eval_file:
        evaluate_results_table(os.path.abspath(args.eval_file), tol=args.tol)
        return

    # ================================================================
    # BATCH MODE: --dir
    # ================================================================
    if args.dir:
        out_excel = os.path.abspath(args.out)
        batch_run(args.dir, args, out_excel)
        if not args.no_eval:
            evaluate_results_table(out_excel, tol=args.tol)
        return

    # ================================================================
    # SINGLE FILE MODE (original behaviour)
    # ================================================================
    if args.file:
        # Custom file — XES or CSV
        is_xes = _is_xes(args.file)
        if not is_xes and not args.object_types:
            parser.error("--object-types is required when using a CSV --file.")
        config = {
            "filename": args.file,
            "object_types": args.object_types if not is_xes else ["case"],
            "sep": args.sep,
            "timestamp_col": "event_timestamp",
            "start_timestamp_col": "event_start_timestamp",
            "event_features": [
                (avg, (feature_extraction.EVENT_NUM_OF_OBJECTS, ())),
            ],
            "exec_features": [
                (avg, (feature_extraction.EXECUTION_THROUGHPUT, ())),
            ],
            "scalability_features": [
                (avg, (feature_extraction.EXECUTION_THROUGHPUT, ())),
                (sum, (feature_extraction.EXECUTION_IDENTITY, ())),
                (avg, (feature_extraction.EXECUTION_NUM_OBJECT, ())),
                (avg, (feature_extraction.EXECUTION_NUM_OF_EVENTS, ())),
                (avg, (feature_extraction.EXECUTION_UNIQUE_ACTIVITIES, ())),
            ],
        }
        if is_xes:
            print(f"  Detected XES file — object types auto-set to ['case']")
    elif args.dataset:
        config = DATASET_CONFIGS[args.dataset]
    else:
        # Default to BPI2017 if available
        config = DATASET_CONFIGS.get("BPI2017")
        if config is None:
            parser.error("Specify --dataset, --file, or --dir.")

    # ---- Output directory ----------------------------------------
    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(config["filename"]))
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load OCEL -----------------------------------------------
    ocel = load_ocel(
        filename=config["filename"],
        object_types=config["object_types"],
        sep=config.get("sep", ","),
        timestamp_col=config.get("timestamp_col", "event_timestamp"),
        start_timestamp_col=config.get("start_timestamp_col",
                                        "event_start_timestamp"),
        max_rows=args.max_rows,
    )

    steps = set(args.steps)
    run_all = "all" in steps

    # ---- Step 1: Drift detection + Granger causality -------------
    if run_all or "drift" in steps:
        run_drift_detection(
            ocel,
            event_features=config.get("event_features", []),
            exec_features=config.get("exec_features", []),
            window_days=args.window_days,
            pen=args.pen,
            p_value=args.p_value,
            output_dir=output_dir,
        )

    # ---- Step 2: Inclusion function comparison -------------------
    if run_all or "inclusion" in steps:
        run_inclusion_comparison(
            ocel,
            window_days=args.window_days,
            output_dir=output_dir,
        )

    # ---- Step 3: Window-size comparison --------------------------
    if run_all or "window" in steps:
        run_window_comparison(
            ocel,
            output_dir=output_dir,
        )

    # ---- Step 4: Scalability experiment --------------------------
    if run_all or "scalability" in steps:
        scalability_feats = config.get("scalability_features")
        if scalability_feats and len(scalability_feats) >= 2:
            run_scalability(
                ocel,
                feature_array=scalability_feats,
                p_value=0.01,
                window_days=args.window_days,
                pen=0.05,
                output_dir=output_dir,
            )
        else:
            print("\n  [SKIP] Scalability: "
                  "not enough features configured.")

    print("\n" + "=" * 60)
    print("  All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

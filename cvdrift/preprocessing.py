from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from cvdrift.pipeline.io import read_xes_to_dataframe
from cvdrift.pipeline.preprocessing import prepare_event_log_dual, prepare_seq_log
from cvdrift.pipeline.series_arrival import series_arrival_case_indexed
from cvdrift.pipeline.series_duration import series_duration_case_indexed
from cvdrift.pipeline.series_routing import add_next_act, build_routing_pairs_from_elog, series_routing_case_indexed

DEFAULT_PARAMS: Dict[str, Any] = {
    "CASE_COL": "Case ID",
    "ACT_COL": "Activity",
    "START_COL": "Start Timestamp",
    "END_COL": "Complete Timestamp",
    "RES_COL": "Resource",
    "tz": "UTC",
    "candidate_windows": [15, 20, 30, 50, 100, 200, 300, 400, 500, 600, 1000, 1500, 2000, 3000, 5000],
    "duration_stat": "median",
    "duration_per_case": "median",
    "routing_stat": "mean",
    "routing_min_count": None,
    "arrival_stat": "median",
    "arrival_max_gap_hours": 4.0,
    "arrival_same_day_only": True,
    "knee_policy": "before",
    "window_strategy": "cv_perpair",
    "pen_scale": 3.0,
    "cpd_model": "l2",
    "min_cp_distance": 10,
    "min_effect_size": 0.15,
    "min_n_points": 10,
}


def merge_params(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    if overrides:
        params.update(overrides)
    return params


def params_for_drift(drift_type: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = merge_params(overrides)
    drift_type = drift_type.lower().strip()
    if drift_type == "routing":
        # Keep historical routing defaults unless caller explicitly overrides.
        if not overrides or "pen_scale" not in overrides:
            params["pen_scale"] = 7
        if not overrides or "min_effect_size" not in overrides:
            params["min_effect_size"] = 0.15
    elif drift_type == "duration":
        params["pen_scale"] = 5
        params["min_effect_size"] = 0
    return params


def load_log(path: str, sep: str = ",") -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith((".xes", ".xes.gz", ".xml")):
        return read_xes_to_dataframe(path, include_resource=True)
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def preprocess(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = merge_params(params)
    logs = prepare_event_log_dual(
        df,
        merged["CASE_COL"],
        merged["ACT_COL"],
        merged["START_COL"],
        merged["END_COL"],
        merged["RES_COL"],
        tz=merged["tz"],
    )
    elog_seq = prepare_seq_log(df, merged["CASE_COL"], merged["ACT_COL"], merged["START_COL"], tz=merged["tz"])
    seq_with_next = add_next_act(elog_seq)
    n_cases = int(elog_seq[".case"].nunique())
    case_to_orig = elog_seq.groupby(".case")[".orig_case"].first().to_dict() if ".orig_case" in elog_seq.columns else {}
    return {
        "logs": logs,
        "elog_seq": elog_seq,
        "seq_with_next": seq_with_next,
        "n_cases": n_cases,
        "case_to_orig": case_to_orig,
    }


def preparation(
    df: pd.DataFrame,
    drift_type: str,
    params: Optional[Dict[str, Any]] = None,
    preprocessed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    drift_type = drift_type.lower().strip()
    if drift_type not in {"duration", "routing", "arrival"}:
        raise ValueError(f"Unsupported drift type: {drift_type}")

    merged = params_for_drift(drift_type, overrides=params)
    shared = preprocessed or preprocess(df, merged)

    builders = {
        "duration": _build_duration_series,
        "routing": _build_routing_series,
        "arrival": _build_arrival_series,
    }
    series_bundle, routing_meta = builders[drift_type](shared, merged)

    result = {
        "drift_type": drift_type,
        "series_bundle": series_bundle,
        "params": merged,
        "config": {
            "pen_scale": merged["pen_scale"],
            "cpd_model": merged["cpd_model"],
            "min_cp_distance": merged["min_cp_distance"],
            "min_effect_size": merged["min_effect_size"],
            "min_n_points": merged["min_n_points"],
            "duration_per_case": merged["duration_per_case"],
            "duration_roll_stat": merged["duration_stat"],
            "routing_roll_stat": merged["routing_stat"],
            "arrival_roll_stat": merged["arrival_stat"],
        },
        "n_cases": shared["n_cases"],
    }
    if routing_meta is not None:
        result["routing_meta"] = routing_meta
    return result


def _build_duration_series(shared: Dict[str, Any], params: Dict[str, Any]):
    elog_dur = shared["logs"].elog_dur
    n_cases = shared["n_cases"]
    activities = elog_dur[".act"].dropna().astype(str).unique().tolist() if len(elog_dur) > 0 else []

    bundle = []
    for activity in activities:
        values, times, cases, orig_cases = series_duration_case_indexed(elog_dur, activity, n_cases, how=params["duration_per_case"])
        bundle.append(
            {
                "label": f"duration::{activity}",
                "activity": str(activity),
                "values": values,
                "times": times,
                "cases": cases,
                "orig_cases": orig_cases,
                "n_valid": int(np.sum(np.isfinite(values))),
            }
        )
    return bundle, None


def _build_routing_series(shared: Dict[str, Any], params: Dict[str, Any]):
    seq_with_next = shared["seq_with_next"]
    n_cases = shared["n_cases"]
    case_to_orig = shared["case_to_orig"]

    routing_min_count = params["routing_min_count"]
    if routing_min_count is None:
        routing_min_count = max(10, int(np.sqrt(n_cases)))

    routing_pairs = build_routing_pairs_from_elog(seq_with_next, min_count=routing_min_count)
    routing_min_mean_p = None
    if len(routing_pairs) > 0:
        mean_ps = []
        for _, row in routing_pairs.iterrows():
            values, _, _, _ = series_routing_case_indexed(seq_with_next, str(row["from"]), str(row["to"]), n_cases, case_to_orig)
            if np.isfinite(values).any():
                mean_ps.append(float(np.nanmean(values)))
        if mean_ps:
            non_deterministic = [value for value in mean_ps if value < 0.95]
            routing_min_mean_p = max(0.01, min(0.30, float(np.percentile(non_deterministic, 5)))) if non_deterministic else 0.01

    bundle = []
    for _, row in routing_pairs.iterrows():
        from_act = str(row["from"])
        to_act = str(row["to"])
        values, times, cases, orig_cases = series_routing_case_indexed(seq_with_next, from_act, to_act, n_cases, case_to_orig)
        n_valid = int(np.sum(np.isfinite(values)))
        bundle.append(
            {
                "label": f"routing::{from_act}->{to_act}",
                "from": from_act,
                "to": to_act,
                "values": values,
                "times": times,
                "cases": cases,
                "orig_cases": orig_cases,
                "n_valid": n_valid,
                "mean_p": float(np.nanmean(values)) if n_valid > 0 else 0.0,
            }
        )

    return bundle, {
        "routing_min_count": int(routing_min_count),
        "routing_min_mean_p": routing_min_mean_p,
        "routing_pairs": routing_pairs,
    }


def _build_arrival_series(shared: Dict[str, Any], params: Dict[str, Any]):
    values, times, cases, orig_cases = series_arrival_case_indexed(
        shared["elog_seq"],
        shared["n_cases"],
        max_gap_hours=params["arrival_max_gap_hours"],
        same_day_only=params["arrival_same_day_only"],
        case_to_orig=shared["case_to_orig"],
    )
    return (
        [
            {
                "label": "arrival::inter_arrival",
                "values": values,
                "times": times,
                "cases": cases,
                "orig_cases": orig_cases,
                "n_valid": int(np.sum(np.isfinite(values))),
            }
        ],
        None,
    )

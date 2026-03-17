from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np

from cvdrift.pipeline.window_selection import choose_window_size_stability


def select_window(prep_result: Dict) -> Dict:
    drift_type = prep_result["drift_type"]
    params = prep_result["params"]
    bundle = prep_result["series_bundle"]

    win_sel = {
        "activity_duration": [],
        "routing_probability": [],
        "arrival_time": [],
        "meta": {
            "n_cases": prep_result["n_cases"],
            "candidate_windows": list(params["candidate_windows"]),
            "knee_policy": params["knee_policy"],
            "arrival_max_gap_hours": params["arrival_max_gap_hours"],
            "arrival_same_day_only": params["arrival_same_day_only"],
            "window_strategy": params.get("window_strategy", "cv_perpair"),
        },
    }
    if "routing_meta" in prep_result:
        win_sel["meta"].update(prep_result["routing_meta"])

    if drift_type == "duration":
        _select_duration_windows(bundle, params, win_sel)
    elif drift_type == "routing":
        _select_routing_windows(bundle, params, win_sel, prep_result.get("routing_meta", {}))
    else:
        _select_arrival_window(bundle, params, win_sel)

    if params.get("window_strategy", "cv_perpair") == "mode_window":
        _apply_mode_window(win_sel, drift_type)
    return win_sel


def _select_duration_windows(bundle, params, win_sel):
    for item in bundle:
        if item["n_valid"] == 0:
            item["window_status"] = "no_data"
            win_sel["activity_duration"].append(
                {
                    "activity": item["activity"],
                    "note": "no_data",
                    "reason": "No duration data.",
                    "n_cases_with_data": 0,
                }
            )
            continue

        result = choose_window_size_stability(
            x=item["values"],
            candidate_windows=list(params["candidate_windows"]),
            stat=params["duration_stat"],
            min_num_windows=2,
            fail_mode="return",
            knee_policy=params["knee_policy"],
        )

        if result.status != "ok":
            item["window_status"] = result.status
            win_sel["activity_duration"].append(
                {
                    "activity": item["activity"],
                    "note": result.status,
                    "reason": result.reason,
                    "n_cases_with_data": item["n_valid"],
                }
            )
            continue

        item.update(window_status="ok", chosen_window=int(result.chosen_window), chosen_cv=float(result.chosen_cv))
        win_sel["activity_duration"].append(
            {
                "activity": item["activity"],
                "n_cases_with_data": item["n_valid"],
                "chosen_window": int(result.chosen_window),
                "chosen_cv": float(result.chosen_cv),
                "details": result.all_results.copy(),
            }
        )


def _select_routing_windows(bundle, params, win_sel, routing_meta):
    routing_min_mean_p = routing_meta.get("routing_min_mean_p")
    for item in bundle:
        if item["n_valid"] == 0:
            item["window_status"] = "no_data"
            win_sel["routing_probability"].append(
                {
                    "from": item["from"],
                    "to": item["to"],
                    "note": "no_data",
                    "reason": "No data.",
                    "n_cases_with_data": 0,
                }
            )
            continue

        mean_p = item.get("mean_p", 0.0)
        if routing_min_mean_p is not None and np.isfinite(mean_p) and mean_p < float(routing_min_mean_p):
            item["window_status"] = "filtered_rare_route"
            win_sel["routing_probability"].append(
                {
                    "from": item["from"],
                    "to": item["to"],
                    "note": "filtered_rare_route",
                    "reason": f"mean_p={mean_p:.6f} < {routing_min_mean_p}",
                    "n_cases_with_data": item["n_valid"],
                }
            )
            continue

        result = choose_window_size_stability(
            x=item["values"],
            candidate_windows=list(params["candidate_windows"]),
            stat=params["routing_stat"],
            min_num_windows=2,
            fail_mode="return",
            knee_policy=params["knee_policy"],
        )

        if result.status != "ok":
            item["window_status"] = result.status
            win_sel["routing_probability"].append(
                {
                    "from": item["from"],
                    "to": item["to"],
                    "note": result.status,
                    "reason": result.reason,
                    "n_cases_with_data": item["n_valid"],
                }
            )
            continue

        item.update(window_status="ok", chosen_window=int(result.chosen_window), chosen_cv=float(result.chosen_cv))
        win_sel["routing_probability"].append(
            {
                "from": item["from"],
                "to": item["to"],
                "mean_p": mean_p,
                "n_cases_with_data": item["n_valid"],
                "chosen_window": int(result.chosen_window),
                "chosen_cv": float(result.chosen_cv),
                "details": result.all_results.copy(),
            }
        )


def _select_arrival_window(bundle, params, win_sel):
    item = bundle[0]
    if item["n_valid"] == 0:
        item["window_status"] = "no_data"
        win_sel["arrival_time"].append(
            {
                "note": "no_data",
                "reason": "No valid arrivals.",
                "n_cases_with_data": 0,
            }
        )
        return

    result = choose_window_size_stability(
        x=item["values"],
        candidate_windows=list(params["candidate_windows"]),
        stat=params["arrival_stat"],
        min_num_windows=2,
        fail_mode="return",
        knee_policy=params["knee_policy"],
    )

    if result.status != "ok":
        item["window_status"] = result.status
        win_sel["arrival_time"].append(
            {
                "note": result.status,
                "reason": result.reason,
                "n_cases_with_data": item["n_valid"],
            }
        )
        return

    item.update(window_status="ok", chosen_window=int(result.chosen_window), chosen_cv=float(result.chosen_cv))
    win_sel["arrival_time"].append(
        {
            "n_cases_with_data": item["n_valid"],
            "mean_arrival_sec": float(np.nanmean(item["values"])),
            "median_arrival_sec": float(np.nanmedian(item["values"])),
            "chosen_window": int(result.chosen_window),
            "chosen_cv": float(result.chosen_cv),
            "details": result.all_results.copy(),
        }
    )


def _apply_mode_window(win_sel, drift_type):
    key = {
        "duration": "activity_duration",
        "routing": "routing_probability",
        "arrival": "arrival_time",
    }[drift_type]
    items = win_sel[key]
    windows = [item["chosen_window"] for item in items if "chosen_window" in item]
    if not windows:
        return
    counts = Counter(windows)
    max_count = max(counts.values())
    mode_window = min(window for window, count in counts.items() if count == max_count)
    for item in items:
        if "chosen_window" in item:
            item["chosen_window"] = mode_window
    win_sel["meta"]["mode_window"] = mode_window

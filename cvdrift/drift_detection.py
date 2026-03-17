from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from cvdrift.pipeline.runner import detect_drifts_duration_and_routing
from cvdrift.preprocessing import params_for_drift, preparation, preprocess
from cvdrift.window_selection import select_window



def run_pipeline(df: pd.DataFrame, drift_types: Iterable[str], params: Optional[Dict] = None, quiet: bool = False) -> Dict:
    base_params = dict(params or {})
    shared = preprocess(df, base_params)
    results = {}

    for drift_type in drift_types:
        drift_params = params_for_drift(drift_type, base_params)
        prep = preparation(df, drift_type, params=drift_params, preprocessed=shared)
        win_sel = select_window(prep)
        if not quiet:
            print(f"\n--- {drift_type.upper()} ---")
            print(f"Series: {len(prep['series_bundle'])}")
            print(f"pen_scale={prep['config']['pen_scale']} | min_effect_size={prep['config']['min_effect_size']}")

        try:
            detection = detect_drifts_duration_and_routing(
                df=df,
                case_col=drift_params["CASE_COL"],
                act_col=drift_params["ACT_COL"],
                start_col=drift_params["START_COL"],
                end_col=drift_params["END_COL"],
                res_col=drift_params["RES_COL"],
                tz=drift_params.get("tz", "UTC"),
                window_selection=win_sel,
                duration_per_case=prep["config"]["duration_per_case"],
                duration_roll_stat=prep["config"]["duration_roll_stat"],
                routing_roll_stat=prep["config"]["routing_roll_stat"],
                arrival_roll_stat=prep["config"]["arrival_roll_stat"],
                step=None,
                pen_scale=prep["config"]["pen_scale"],
                cpd_model=prep["config"]["cpd_model"],
                min_cp_distance=prep["config"]["min_cp_distance"],
                min_effect_size=prep["config"]["min_effect_size"],
                min_n_points=prep["config"]["min_n_points"],
                plot=False,
            )
        except ValueError as exc:
            detection = None
            if not quiet:
                print(f"Skipped {drift_type}: {exc}")

        results[drift_type] = {
            "preparation": prep,
            "window_selection": win_sel,
            "detection": detection,
        }
    return results



def extract_cps(results: Dict) -> list[int]:
    cps: list[int] = []
    for result in results.values():
        detection = result.get("detection")
        if detection is None:
            continue
        for key, column in (("duration_consensus", "consensus_case"), ("consensus_drifts", "consensus_case"), ("arrival_drifts", "cp_case")):
            frame = detection.get(key)
            if frame is not None and len(frame) > 0 and column in frame.columns:
                cps.extend(int(value) for value in frame[column])
    return sorted(set(cps))

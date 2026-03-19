from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List

from cvdrift.pipeline.io import get_event_log
from cvdrift.drift_detection import extract_cps, run_pipeline
from cvdrift.preprocessing import DEFAULT_PARAMS, load_log
from experiments.Experiment1.run_experiments_CVDrift import evaluate_results_csv

ALGORITHM_NAME = "CVDriftSuggested"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")



def _collect_logs(folder: str) -> List[str]:
    extensions = (".xes", ".xes.gz", ".csv", ".mxml")
    paths: List[str] = []
    for root, _, files in os.walk(folder):
        for name in files:
            if any(name.lower().endswith(ext) for ext in extensions):
                paths.append(os.path.join(root, name))
    return sorted(paths)



def _fmt_time(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"



def _append_csv(csv_path: str, row: Dict[str, object]) -> None:
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



def _single_row(log_path: str, drift_types: Iterable[str], params: Dict, quiet: bool = False) -> Dict[str, object]:
    start_time = time.time()
    df = load_log(log_path)
    results = run_pipeline(df, drift_types, params=params, quiet=quiet)
    cps = extract_cps(results)
    elapsed = time.time() - start_time
    return {
        "Algorithm": ALGORITHM_NAME,
        "Log Source": os.path.basename(log_path),
        "Log": os.path.basename(log_path),
        "Drift Types": ", ".join(drift_types),
        "Detected Changepoints": str(cps),
        "Actual Changepoints for Log": "[]",
        "Duration (Seconds)": round(elapsed, 3),
        "Duration (hh:mm:ss)": _fmt_time(elapsed),
        "Parameter Settings": json.dumps(params, default=str),
    }



def run_batch(log_paths: List[str], out_path: str, drift_types: Iterable[str], params: Dict, quiet: bool = False) -> str:
    csv_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print(f"Processing {len(log_paths)} log(s) -> {csv_path}")
    for index, log_path in enumerate(log_paths, start=1):
        print(f"[{index}/{len(log_paths)}] {os.path.basename(log_path)}")
        try:
            row = _single_row(log_path, drift_types, params, quiet=quiet)
        except Exception as exc:
            elapsed = 0.0
            row = {
                "Algorithm": ALGORITHM_NAME,
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
        _append_csv(csv_path, row)
    print(f"Batch complete. Results: {csv_path}")
    return csv_path



def main() -> None:
    parser = argparse.ArgumentParser(description="CVDrift suggested clean runner")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", help="Single event-log file.")
    source.add_argument("--files", nargs="+", help="Multiple event-log files.")
    source.add_argument("--dir", help="Folder with logs.")

    parser.add_argument("--drift", nargs="+", choices=["duration", "routing", "arrival"], default=["duration", "routing", "arrival"])
    parser.add_argument("--window-strategy", choices=["cv_perpair", "mode_window"], default="cv_perpair")
    parser.add_argument("--out", default=None, help="Output CSV path for single or batch mode.")
    parser.add_argument("--gt-mode", choices=list(GT_MODES.keys()), default="samira")
    parser.add_argument("--tol", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    params["window_strategy"] = args.window_strategy

    if args.file:
        row = _single_row(args.file, args.drift, params, quiet=args.quiet)
        print(f"Detected changepoints: {row['Detected Changepoints']}")
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            _append_csv(args.out, row)
            print(f"Saved to: {args.out}")
        return

    if args.files:
        missing = [path for path in args.files if not os.path.isfile(path)]
        if missing:
            print(f"ERROR: not found: {missing}", file=sys.stderr)
            raise SystemExit(1)
        out_path = args.out or os.path.join(OUTPUT_DIR, f"results_{datetime.now():%Y%m%d_%H%M%S}.csv")
        csv_path = run_batch(args.files, out_path, args.drift, params, quiet=args.quiet)
        if not args.no_eval:
            evaluate_results_csv(csv_path, gt_mode=args.gt_mode, tol=args.tol)
        return

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"ERROR: folder not found: {args.dir}", file=sys.stderr)
            raise SystemExit(1)
        logs = _collect_logs(args.dir)
        out_path = args.out or os.path.join(OUTPUT_DIR, f"batch_{datetime.now():%Y%m%d_%H%M%S}.csv")
        csv_path = run_batch(logs, out_path, args.drift, params, quiet=args.quiet)
        if not args.no_eval:
            evaluate_results_csv(csv_path, gt_mode=args.gt_mode, tol=args.tol)
        return

    df = get_event_log(sep=",")
    results = run_pipeline(df, args.drift, params=params, quiet=args.quiet)
    print(f"Detected changepoints: {extract_cps(results)}")


if __name__ == "__main__":
    main()

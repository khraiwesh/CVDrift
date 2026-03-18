import importlib.util
from pathlib import Path
import pandas as pd
import ast

BASE = Path(r"C:\Users\samira\OneDrive - GJU\Desktop\PhD Progress -Submissions\Dougakn\Concept Drift\cvdrift-suggested\experiments\Experiment#2 (Control Flow Prespective)")
EVAL_PATH = BASE / "evaluate.py"
RUN_EXP_PATH = BASE / "run_experiments.py"

OSTOVAR_SRC = Path(r"C:\Users\samira\OneDrive - GJU\Desktop\PhD Progress -Submissions\Dougakn\Concept Drift\CVDriftPipeline_v2\OstovarScale7.csv")
CERAVOLO_SRC = Path(r"C:\Users\samira\OneDrive - GJU\Desktop\PhD Progress -Submissions\Dougakn\Concept Drift\CVDriftPipeline_v2\Ceravolo_Scale7_Final.ods")

TMP_DIR = BASE / "output" / "tmp_two_file_eval"
TMP_DIR.mkdir(parents=True, exist_ok=True)

spec_eval = importlib.util.spec_from_file_location("eval_exp2", EVAL_PATH)
ev = importlib.util.module_from_spec(spec_eval)
spec_eval.loader.exec_module(ev)

spec_run = importlib.util.spec_from_file_location("run_exp_eval", RUN_EXP_PATH)
run_exp = importlib.util.module_from_spec(spec_run)
spec_run.loader.exec_module(run_exp)


def normalize_ostovar() -> Path:
    df = pd.read_csv(OSTOVAR_SRC)
    # Minimal schema adaptation for evaluate/run_experiments scorer compatibility.
    if "Routing CPs" in df.columns:
        df["Detected Changepoints"] = df["Routing CPs"]
    if "GT " in df.columns:
        df["Actual Changepoints for Log"] = df["GT "]
    if "Algorithm" not in df.columns:
        df["Algorithm"] = "CVDrift"
    else:
        df["Algorithm"] = "CVDrift"
    if "Log Source" not in df.columns:
        df["Log Source"] = "Ostovar"
    df["n_cases"] = 3000

    def _to_list_literal(value):
        if isinstance(value, list):
            return str([int(x) for x in value])
        if pd.isna(value):
            return "[]"
        text = str(value).strip()
        if text in ("", "[]", "ERROR", "nan"):
            return "[]"
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return str([int(x) for x in parsed])
        except Exception:
            pass
        # Fall back to extracting numbers from comma/space-separated text.
        nums = []
        for token in text.replace("[", " ").replace("]", " ").replace(",", " ").split():
            if token.lstrip("-").isdigit():
                nums.append(int(token))
        return str(nums)

    df["Detected Changepoints"] = df["Detected Changepoints"].map(_to_list_literal)
    df["Actual Changepoints for Log"] = df["Actual Changepoints for Log"].map(_to_list_literal)

    out = TMP_DIR / "ostovar_as_eval.csv"
    df.to_csv(out, index=False)
    return out


def normalize_ceravolo() -> Path:
    df = pd.read_excel(CERAVOLO_SRC, engine="odf")
    # Keep only columns needed by evaluate scorer and map to single-approach naming.
    keep = [c for c in ["Algorithm", "Log Source", "Log", "Detected Changepoints", "Actual Changepoints for Log"] if c in df.columns]
    df = df[keep].copy()
    df["Algorithm"] = "CVDrift"
    df["n_cases"] = 1000

    def _to_list_literal(value):
        if isinstance(value, list):
            return str([int(x) for x in value])
        if pd.isna(value):
            return "[]"
        text = str(value).strip()
        if text in ("", "[]", "ERROR", "nan"):
            return "[]"
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return str([int(x) for x in parsed])
        except Exception:
            pass
        nums = []
        for token in text.replace("[", " ").replace("]", " ").replace(",", " ").split():
            if token.lstrip("-").isdigit():
                nums.append(int(token))
        return str(nums)

    df["Detected Changepoints"] = df["Detected Changepoints"].map(_to_list_literal)
    df["Actual Changepoints for Log"] = df["Actual Changepoints for Log"].map(_to_list_literal)

    out = TMP_DIR / "ceravolo_as_eval.csv"
    df.to_csv(out, index=False)
    return out


ost_csv = normalize_ostovar()
cer_csv = normalize_ceravolo()
combo_csv = TMP_DIR / "combined_two_files_as_eval.csv"
pd.concat([pd.read_csv(ost_csv), pd.read_csv(cer_csv)], ignore_index=True).to_csv(combo_csv, index=False)

for tag, path in [("OSTOVAR", ost_csv), ("CERAVOLO", cer_csv), ("COMBINED", combo_csv)]:
    out_dir = TMP_DIR / f"result_{tag.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.evaluate(csv_path=str(path), out_path=str(out_dir), lag_window=200)
    summary = run_exp.evaluate_results_csv(str(path), tol=200)
    print(f"{tag} path: {path}")
    print(f"{tag} TP={summary['TP']} FP={summary['FP']} FN={summary['FN']} Precision={summary['Precision']} Recall={summary['Recall']} F1={summary['F1']}")

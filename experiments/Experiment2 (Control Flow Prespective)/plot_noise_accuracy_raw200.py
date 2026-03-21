"""
Create a second set of noise-accuracy plots computed directly from raw
`algorithm_results_no_xes.csv` using lag window 200 and GT changepoints.

Outputs are written to:
  F1_Category_Results_RAW200/
"""

import os
import re
from statistics import harmonic_mean
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cdrift.utils.helpers import readCSV_Lists
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

matplotlib.rcParams.update({"font.size": 11})

# Configuration
LAG_WINDOW = 200
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "algorithm_results_with_CVDrift.csv")
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "accuracy_noise_ostovar_raw200.png")
OUT_DIR = SCRIPT_DIR

# Legacy variable names for compatibility
RAW_CSV_PATH = INPUT_CSV

shorter_names = {
    "Zheng DBSCAN": "RINV",
    "ProDrift": "ProDrift",
    "Maaradji Runs": "ProDrift",
    "Bose J": "J-Measure",
    "Bose WC": "Window Count",
    "Earth Mover's Distance": "EMD",
    "Process Graph Metrics": "PGM",
    "Martjushev ADWIN J": "ADWIN J",
    "Martjushev ADWIN WC": "ADWIN WC",
    "LCDD": "LCDD",
    "PELT_ModeWindow": "PELT_ModeWindow",
    "PELT_CV_PerPair": "PELT_CV_PerPair",
    "CVDrift_Unified": "CVDrift",
    "CVDrift": "CVDrift",
}

used_parameters = {
    "J-Measure": ["Window Size", "SW Step Size"],
    "Window Count": ["Window Size", "SW Step Size"],
    "ADWIN J": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
    "ADWIN WC": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
    "ProDrift": ["Window Size", "SW Step Size"],
    "EMD": ["Window Size", "SW Step Size"],
    "PGM": ["Min Adaptive Window", "Max Adaptive Window", "P-Value"],
    "RINV": ["MRID", "Epsilon"],
    "LCDD": ["Complete-Window Size", "Detection-Window Size", "Stable Period"],
    "PELT_ModeWindow": ["pen_scale"],
    "PELT_CV_PerPair": ["pen_scale"],
    "CVDrift": ["Algorithm"],
}

metric_styles = {
    "Precision": {"color": "#DB444B", "marker": "s", "linestyle": "-", "label": "Precision"},
    "Recall": {"color": "#006BA2", "marker": "^", "linestyle": "-", "label": "Recall"},
    "F1": {"color": "#EBB434", "marker": "o", "linestyle": "--", "label": "F1-Score"},
}


def map_row_to_noise_level(row):
    logname = row["Log"]
    source = row["Log Source"]
    if source == "Ceravolo":
        match = re.search(r"noise([0-9]*)_", logname)
        return match.group(1) if match else "0"
    if source == "Ostovar":
        last = logname.split("_")[-1]
        return last if last.isnumeric() else "0"
    if source == "Bose":
        return None
    return None


def get_noiseless_logs(df: pd.DataFrame) -> set:
    logs = zip(df["Log Source"], df["Log"])
    return {
        log for source, log in logs if
        (source == "Ostovar" and not (log.endswith("_2") or log.endswith("_5"))) or
        (source == "Ceravolo" and log.split("_")[2] == "noise0") or
        source == "Bose"
    }


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Log Source"] = df["Log Source"].str.strip().str.title()
    df = df[df["Algorithm"] != "Martjushev ADWIN WC"]
    df["Algorithm"] = df["Algorithm"].map(lambda alg: shorter_names.get(alg, alg))

    atomic_logs = {
        log
        for _, (source, log) in df[["Log Source", "Log"]].iterrows()
        if (source.lower() == "ostovar" and log.lower().startswith("atomic"))
        or log.lower().startswith("bose")
        or (source.lower() == "ceravolo" and set(log.lower().split("_")[-1]) != {"i", "o", "r"})
    }
    df = df[df["Log"].isin(atomic_logs)]
    df["Noise Level"] = df.apply(map_row_to_noise_level, axis=1)
    return df


def fast_get_tp_fp(detected: list, known: list, lag: int) -> Tuple[int, int]:
    if len(detected) == 0 or len(known) == 0:
        return 0, len(detected)

    big = 1e9
    cost = np.full((len(detected), len(known)), big)
    for i, d_cp in enumerate(detected):
        for j, k_cp in enumerate(known):
            dist = abs(d_cp - k_cp)
            if dist <= lag:
                cost[i, j] = dist

    row_idx, col_idx = linear_sum_assignment(cost)
    tp = sum(1 for r, c in zip(row_idx, col_idx) if cost[r, c] < big)
    fp = len(detected) - tp
    return tp, fp


def compute_f1_from_rows(rows: pd.DataFrame, lag_window: int) -> Tuple[float, float, float]:
    tps = 0
    positives = 0
    detected_total = 0

    for _, row in rows.iterrows():
        actual_cp = row["Actual Changepoints for Log"]
        detected_cp = row["Detected Changepoints"]
        tp, _ = fast_get_tp_fp(detected_cp, actual_cp, lag_window)
        tps += tp
        positives += len(actual_cp)
        detected_total += len(detected_cp)

    precision = tps / detected_total if detected_total > 0 else np.nan
    recall = tps / positives if positives > 0 else np.nan

    if np.isnan(precision) or np.isnan(recall) or (precision == 0 and recall == 0):
        f1 = np.nan
    else:
        f1 = harmonic_mean([precision, recall])

    return precision, recall, f1


def find_best_params(alg_df: pd.DataFrame, param_names: list, lag_window: int):
    best_f1 = -1
    best_params = None
    for params, group in alg_df.groupby(by=param_names):
        _, _, f1 = compute_f1_from_rows(group, lag_window)
        if not np.isnan(f1) and f1 > best_f1:
            best_f1 = f1
            best_params = params
    return best_params


def make_param_filter(alg: str, bp):
    params = used_parameters.get(alg)
    if params is None or bp is None:
        return lambda sub_df: sub_df.iloc[0:0]
    if not isinstance(bp, tuple):
        bp = (bp,)

    def _filter(sub_df):
        mask = pd.Series(True, index=sub_df.index)
        for pname, pval in zip(params, bp):
            mask &= sub_df[pname] == pval
        return sub_df[mask]

    return _filter


def compute_metric_tables():
    print("Reading raw results and preprocessing ...")
    df = readCSV_Lists(RAW_CSV_PATH)
    df = preprocess(df)

    algorithms = sorted(df["Algorithm"].unique())
    datasets = ["Ostovar", "Ceravolo", "Bose"]

    noiseless_logs = get_noiseless_logs(df)
    df_noiseless = df[df["Log"].isin(noiseless_logs)]

    print("Finding best params on noiseless logs ...")
    best_params: Dict[str, object] = {}
    for alg in tqdm(algorithms, desc="Best params"):
        alg_df_noiseless = df_noiseless[df_noiseless["Algorithm"] == alg]
        params = used_parameters.get(alg)
        if params is None:
            continue
        best_params[alg] = find_best_params(alg_df_noiseless, params, LAG_WINDOW)

    def best_param_rows(alg: str, sub_df: pd.DataFrame) -> pd.DataFrame:
        return make_param_filter(alg, best_params.get(alg))(sub_df)

    rows_dataset_noise = []
    rows_overall_noise = []
    all_noise_levels = sorted(df["Noise Level"].dropna().unique(), key=lambda x: int(x))

    for alg in algorithms:
        if alg not in best_params:
            continue
        alg_df = df[df["Algorithm"] == alg]
        best_rows = best_param_rows(alg, alg_df)

        for ds in datasets:
            ds_rows = best_rows[best_rows["Log Source"] == ds]
            if ds_rows.empty:
                continue

            p, r, f1 = compute_f1_from_rows(ds_rows, LAG_WINDOW)
            rows_dataset_noise.append(
                {"Algorithm": alg, "Dataset": ds, "Noise Level": "All", "Precision": p, "Recall": r, "F1": f1}
            )

            for noise_level in sorted(ds_rows["Noise Level"].dropna().unique(), key=lambda x: int(x)):
                nl_rows = ds_rows[ds_rows["Noise Level"] == noise_level]
                p, r, f1 = compute_f1_from_rows(nl_rows, LAG_WINDOW)
                rows_dataset_noise.append(
                    {
                        "Algorithm": alg,
                        "Dataset": ds,
                        "Noise Level": int(noise_level),
                        "Precision": p,
                        "Recall": r,
                        "F1": f1,
                    }
                )

        for noise_level in all_noise_levels:
            if noise_level == "0":
                nl_rows = best_rows[(best_rows["Noise Level"] == "0") | (best_rows["Noise Level"].isna())]
            else:
                nl_rows = best_rows[best_rows["Noise Level"] == noise_level]

            if nl_rows.empty:
                rows_overall_noise.append(
                    {"Algorithm": alg, "Noise Level": int(noise_level), "Precision": np.nan, "Recall": np.nan, "F1": np.nan}
                )
                continue

            p, r, f1 = compute_f1_from_rows(nl_rows, LAG_WINDOW)
            rows_overall_noise.append(
                {
                    "Algorithm": alg,
                    "Noise Level": int(noise_level),
                    "Precision": p,
                    "Recall": r,
                    "F1": f1,
                }
            )

    table_dataset_noise = pd.DataFrame(rows_dataset_noise)
    table_overall_noise = pd.DataFrame(rows_overall_noise)

    table_dataset_noise.to_csv(os.path.join(OUT_DIR, "f1_per_dataset_noise_raw200.csv"), index=False)
    table_overall_noise.to_csv(os.path.join(OUT_DIR, "f1_overall_per_noise_level_raw200.csv"), index=False)

    return table_dataset_noise, table_overall_noise


def plot_dataset(table_dataset_noise: pd.DataFrame, algorithms: list, dataset: str, noise_levels: list, figsize: Tuple[float, float], filename: str):
    n_algs = len(algorithms)
    ncols = int(np.ceil(n_algs / 2))
    nrows = 2

    df_ds = table_dataset_noise[table_dataset_noise["Dataset"] == dataset].copy()
    df_ds = df_ds[df_ds["Noise Level"] != "All"].copy()
    df_ds["Noise Level"] = df_ds["Noise Level"].astype(int)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, alg in enumerate(algorithms):
        ax = axes[idx]
        alg_data = df_ds[df_ds["Algorithm"] == alg].sort_values("Noise Level")

        if alg_data.empty:
            ax.set_title(alg, fontweight="bold", fontsize=11)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue

        for metric, style in metric_styles.items():
            vals = alg_data[metric].values
            nls = alg_data["Noise Level"].values
            vals_plot = np.where(np.isnan(vals), 0, vals)
            ax.plot(
                nls,
                vals_plot,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=5,
                label=style["label"],
                zorder=3,
            )
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                ax.scatter(nls[nan_mask], vals_plot[nan_mask], marker="x", color="red", s=40, zorder=4)

        ax.set_title(alg, fontweight="bold", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(noise_levels)
        ax.grid(True, alpha=0.3)

        if idx % ncols == 0:
            ax.set_ylabel("Performance", fontsize=11)
        if idx >= ncols:
            ax.set_xlabel("Noise Level", fontsize=11)

    for idx in range(n_algs, nrows * ncols):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(f"Accuracy vs Noise Level — {dataset} Dataset (raw, lag=200)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(table_dataset_noise: pd.DataFrame, algorithms: list):
    n_algs = len(algorithms)
    ncols = int(np.ceil(n_algs / 2))
    nrows = 2

    df_clean = table_dataset_noise[table_dataset_noise["Noise Level"] != "All"].copy()
    df_clean["Noise Level"] = df_clean["Noise Level"].astype(int)

    fig, all_axes = plt.subplots(
        nrows,
        ncols * 2 + 1,
        figsize=(ncols * 6.4 + 1, nrows * 3),
        gridspec_kw={"width_ratios": [1] * ncols + [0.15] + [1] * ncols},
    )

    for r_idx in range(nrows):
        sep_idx = r_idx * (ncols * 2 + 1) + ncols
        all_axes.flatten()[sep_idx].set_visible(False)

    datasets_config = [
        ("Ostovar", [0, 2, 5], range(0, ncols)),
        ("Ceravolo", [0, 5, 10, 15, 20], range(ncols + 1, ncols * 2 + 1)),
    ]

    for dataset, noise_levels, col_range in datasets_config:
        col_list = list(col_range)
        for idx, alg in enumerate(algorithms):
            row = idx // ncols
            col = col_list[idx % ncols]
            ax = all_axes[row, col]

            alg_data = df_clean[(df_clean["Algorithm"] == alg) & (df_clean["Dataset"] == dataset)].sort_values("Noise Level")

            if alg_data.empty:
                ax.set_title(alg, fontweight="bold", fontsize=10)
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_ylim(-0.05, 1.05)
                ax.set_xticks(noise_levels)
                ax.grid(True, alpha=0.3)
                continue

            for metric, style in metric_styles.items():
                vals = alg_data[metric].values
                nls = alg_data["Noise Level"].values
                vals_plot = np.where(np.isnan(vals), 0, vals)
                ax.plot(
                    nls,
                    vals_plot,
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.5,
                    markersize=4,
                    label=style["label"],
                    zorder=3,
                )

            ax.set_title(alg, fontweight="bold", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(noise_levels)
            ax.grid(True, alpha=0.3)

            if col == col_list[0]:
                ax.set_ylabel("Performance", fontsize=10)
            if row == 1:
                ax.set_xlabel("Noise Level", fontsize=10)

        for idx in range(n_algs, nrows * ncols):
            row = idx // ncols
            col = col_list[idx % ncols]
            all_axes[row, col].set_visible(False)

    fig.text(0.25, 1.02, "Ostovar Dataset\n(Noise = additional non-fitting events)", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.75, 1.02, "Ceravolo Dataset\n(Noise = absence of fitting events)", ha="center", fontsize=12, fontweight="bold")

    handles, labels = all_axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.04), frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "accuracy_noise_combined_raw200.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overall(table_overall_noise: pd.DataFrame, algorithms: list):
    n_algs = len(algorithms)
    ncols = int(np.ceil(n_algs / 2))
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2), sharey=True)
    axes = axes.flatten()

    for idx, alg in enumerate(algorithms):
        ax = axes[idx]
        alg_data = table_overall_noise[table_overall_noise["Algorithm"] == alg].sort_values("Noise Level")

        if alg_data.empty:
            ax.set_title(alg, fontweight="bold", fontsize=11)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue

        nls = alg_data["Noise Level"].values
        precs = alg_data["Precision"].values
        recs = alg_data["Recall"].values
        f1s = alg_data["F1"].values

        for vals, style in [(precs, metric_styles["Precision"]), (recs, metric_styles["Recall"]), (f1s, metric_styles["F1"])]:
            vals_plot = np.where(np.isnan(vals), 0, vals)
            ax.plot(
                nls,
                vals_plot,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=5,
                label=style["label"],
                zorder=3,
            )
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                ax.scatter(nls[nan_mask], vals_plot[nan_mask], marker="x", color="red", s=40, zorder=4)

        ax.set_title(alg, fontweight="bold", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(sorted(table_overall_noise["Noise Level"].unique()))
        ax.grid(True, alpha=0.3)

        if idx % ncols == 0:
            ax.set_ylabel("Performance", fontsize=11)
        if idx >= ncols:
            ax.set_xlabel("Noise Level", fontsize=11)

    for idx in range(n_algs, nrows * ncols):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.04), frameon=True)

    fig.suptitle("Overall Accuracy (F1) per Algorithm — All Datasets by Noise Level", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "accuracy_overall_by_noise_raw200.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    table_dataset_noise, table_overall_noise = compute_metric_tables()
    algorithms = sorted(table_dataset_noise["Algorithm"].unique())

    print("Plotting Ostovar ...")
    plot_dataset(table_dataset_noise, algorithms, "Ostovar", [0, 2, 5], (int(np.ceil(len(algorithms) / 2)) * 3.2, 2 * 3), "accuracy_noise_ostovar_raw200.png")

    print("Plotting Ceravolo ...")
    plot_dataset(table_dataset_noise, algorithms, "Ceravolo", [0, 5, 10, 15, 20], (int(np.ceil(len(algorithms) / 2)) * 3.2, 2 * 3), "accuracy_noise_ceravolo_raw200.png")

    print("Plotting combined ...")
    plot_combined(table_dataset_noise, algorithms)

    print("Plotting overall ...")
    plot_overall(table_overall_noise, algorithms)

    print(f"Done. Raw-based plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

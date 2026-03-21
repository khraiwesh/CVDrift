import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cdrift-evaluation"))

import matplotlib.pyplot as plt
import pandas as pd
from cdrift import evaluation
from cdrift.utils.helpers import readCSV_Lists, convertToTimedelta, importLog
import numpy as np
from datetime import datetime
from statistics import mean, harmonic_mean, stdev
from scipy.stats import iqr
from typing import List, Tuple
import seaborn as sns
import re
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from tqdm.auto import tqdm


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

mapping_ostovar_to_shortnames = {
    "ConditionalMove": 'cm',
    "ConditionalRemoval": 'cre',
    "ConditionalToSequence": 'cf',
    "Frequency": 'fr',
    "Loop": 'lp',
    "ParallelMove": 'pm',
    "ParallelRemoval": 'pre',
    "ParallelToSequence": 'pl',
    "SerialMove": 'sm',
    "SerialRemoval": 'sre',
    "Skip": 'cb',
    "Substitute": 'rp',
    "Swap": 'sw',
}

def split_by_name(df):
    return [
        (alg, df[df["Algorithm"] == alg].copy())
        for alg in df["Algorithm"].unique()
    ]

def map_row_to_cp(row):
    logname = row["Log"]
    if row["Log Source"] == "Ceravolo":
        return logname.split("_")[-1]
    elif row["Log Source"] == "Ostovar":
        change_pattern = logname.split("_")[-1]
        if change_pattern.isnumeric():
            change_pattern = logname.split("_")[-2]
        return mapping_ostovar_to_shortnames[change_pattern]
    elif row["Log Source"] == "Bose":
        return None
    else:
        raise ValueError(f"Unknown Log Source for Versatility: {row['Log Source']}")

def map_row_to_noise_level(row):
    logname = row["Log"]
    if row["Log Source"] == "Ceravolo":
        return re.search('noise([0-9]*)_', logname).group(1)
    elif row["Log Source"] == "Ostovar":
        last_member = logname.split('_')[-1]
        if last_member.isnumeric():
            return last_member
        else:
            return '0'
    elif row["Log Source"] == "Bose":
        return None
    else:
        raise ValueError(f"Unknown Log Source for Noise Level: {row['Log Source']}")


def preprocess(_df):
    df = _df.copy()
    # Keep only CVDrift
    df = df[df["Algorithm"] == "CVDrift"]

    ## Only use Atomic Logs
    atomic_logs = {
        log
        for _, (source, log) in df[["Log Source", "Log"]].iterrows()
        if (source.lower() == "ostovar" and log.lower().startswith("atomic"))
        or (log.lower().startswith("bose"))
        or (source.lower() == "ceravolo" and set(log.lower().split("_")[-1]) != {'i', 'o', 'r'})
    }
    df = df[df["Log"].isin(atomic_logs)]

    df["Change Pattern"] = df.apply(lambda x: map_row_to_cp(x), axis=1)
    df["Noise Level"] = df.apply(lambda x: map_row_to_noise_level(x), axis=1)

    return df

def calcAccuracy_no_params(df: pd.DataFrame, lag_window: int):
    """Calculate accuracy for CVDrift (no parameter grouping)."""
    tps = 0
    fps = 0
    positives = 0
    detected = 0

    for index, row in df.iterrows():
        actual_cp = row["Actual Changepoints for Log"]
        detected_cp = row["Detected Changepoints"]
        tp, fp = evaluation.getTP_FP(detected_cp, actual_cp, lag_window)
        tps += tp
        fps += fp
        positives += len(actual_cp)
        detected += len(detected_cp)

    try:
        precision = tps / detected
    except ZeroDivisionError:
        precision = np.nan

    try:
        recall = tps / positives
    except ZeroDivisionError:
        recall = np.nan

    f1 = harmonic_mean([precision, recall])
    return precision, recall, f1

def calculate_accuracy_cvdrift(dataframe, lag_window):
    precision, recall, f1 = calcAccuracy_no_params(dataframe, lag_window)
    return f1, precision, recall

def plot_accuracy_cvdrift(f1, precision, recall, out_path, lag_window, color):
    accuracy_plot_df = pd.DataFrame([
        {"Algorithm": "CVDrift", "Metric": "F1-Score", "Value": f1},
        {"Algorithm": "CVDrift", "Metric": "Precision", "Value": precision},
        {"Algorithm": "CVDrift", "Metric": "Recall", "Value": recall},
    ])

    palette = sns.color_palette([color])
    plt.figure()
    plt.grid(zorder=0)
    ax = sns.barplot(x="Metric", y="Value", data=accuracy_plot_df, hue="Algorithm", palette=palette, zorder=5)
    ax.figure.set_size_inches(8, 4)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(ax.get_xlabel(), size=20)
    ax.set_ylabel("", size=20)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, prop={'size': 14})

    if not os.path.exists(f"{out_path}/Accuracy{lag_window}"):
        os.makedirs(f"{out_path}/Accuracy{lag_window}")
    plt.savefig(f"{out_path}/Accuracy{lag_window}/accuracy_c.pdf", bbox_inches="tight", format="pdf")
    plt.close('all')

def calcLatencies_no_params(df, lag_window):
    lags = []
    for index, row in df.iterrows():
        actual_cp = row["Actual Changepoints for Log"]
        detected_cp = row["Detected Changepoints"]
        assignments = evaluation.assign_changepoints(detected_cp, actual_cp, lag_window)
        for d, a in assignments:
            lags.append(abs(d - a))
    return lags

def calculate_latency_cvdrift(dataframe, lag_window, min_support=1):
    lags = calcLatencies_no_params(dataframe, lag_window)
    if len(lags) >= min_support:
        scaled = 1 - (mean(lags) / lag_window)
    else:
        scaled = np.nan
    return scaled, lags

def plot_latency_cvdrift(lags, out_path, lag_window, color):
    latency_plot_df = pd.DataFrame([
        {"Algorithm": "CVDrift", "Unscaled Latency": latency}
        for latency in lags
    ])

    plt.figure(figsize=(10, 4))
    palette = sns.color_palette([color])
    ax = sns.barplot(x="Algorithm", y="Unscaled Latency", data=latency_plot_df, palette=palette)
    plt.ylabel("Latency")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), size=18)
    ax.set_ylabel(ax.get_ylabel(), size=18)
    ax.figure.set_size_inches(8, 4)

    if not os.path.exists(f"{out_path}/Latency{lag_window}"):
        os.makedirs(f"{out_path}/Latency{lag_window}")
    plt.savefig(f"{out_path}/Latency{lag_window}/latency.pdf", bbox_inches="tight", format="pdf")
    plt.close('all')

def calc_versatility_cvdrift(dataframe, lag_window):
    recalls = dict()
    for change_pattern, cp_group in dataframe.groupby(by="Change Pattern"):
        TPS = 0
        POSITIVES = 0
        for index, row in cp_group.iterrows():
            detected_changepoints = row["Detected Changepoints"]
            actual_changepoints = row["Actual Changepoints for Log"]
            tp, _ = evaluation.getTP_FP(detected_changepoints, actual_changepoints, lag_window)
            TPS += tp
            POSITIVES += len(actual_changepoints)
        recall = TPS / POSITIVES if POSITIVES != 0 else np.nan
        recalls[change_pattern] = recall
    versatility = np.nanmean(list(recalls.values()))
    return versatility, recalls

def plot_versatility_cvdrift(recalls, out_path, lag_window, color):
    df_vers_plot = pd.DataFrame([
        {"Algorithm": "CVDrift", "Change Pattern": cp, "Versatility": rec}
        for cp, rec in recalls.items()
    ])

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.barplot(x="Change Pattern", y="Versatility", data=df_vers_plot, hue="Algorithm", ax=ax)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    if not os.path.exists(f"{out_path}/Versatility{lag_window}"):
        os.makedirs(f"{out_path}/Versatility{lag_window}")
    plt.savefig(f"{out_path}/Versatility{lag_window}/versatility.pdf", bbox_inches="tight", format="pdf")
    plt.close('all')

    # Radar chart
    fill_colors = {"CVDrift": color}
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        fig = px.line_polar(
            df_vers_plot,
            r='Versatility',
            theta='Change Pattern',
            line_close=True,
            color='Algorithm',
            color_discrete_map=fill_colors,
            title=None
        )
        fig.update_layout(showlegend=False, title_x=0.5, font=dict(size=18))
        fig.update_traces(fill='toself')

        if not os.path.exists(f"{out_path}/Versatility{lag_window}/Radar_Charts"):
            os.makedirs(f"{out_path}/Versatility{lag_window}/Radar_Charts")
        fig.write_image(f"{out_path}/Versatility{lag_window}/Radar_Charts/radar_chart_CVDrift.pdf", format="pdf")

def calculate_scalability_cvdrift(dataframe):
    result = dataframe["Duration (Seconds)"].mean()
    result_str = datetime.strftime(datetime.utcfromtimestamp(result), '%H:%M:%S')
    return {"avg_seconds": result, "str": result_str}

def plot_scalability_cvdrift(dataframe, out_path, lag_window, color):
    scalability_plot_df = pd.DataFrame([
        {"Algorithm": "CVDrift", "Duration": duration / 60}
        for duration in dataframe["Duration (Seconds)"].tolist()
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette([color])
    ax = sns.boxplot(
        data=scalability_plot_df,
        x="Duration",
        y="Algorithm",
        palette=palette,
        width=0.5,
        ax=ax,
        fliersize=0
    )
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel("Duration [Minutes]", size=18)
    ax.set_ylabel(ax.get_ylabel(), size=18)

    if not os.path.exists(f"{out_path}/Scalability{lag_window}"):
        os.makedirs(f"{out_path}/Scalability{lag_window}")
    plt.savefig(f"{out_path}/Scalability{lag_window}/scalability.pdf", bbox_inches="tight", format="pdf")
    plt.close('all')

    # Barplot
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x="Algorithm", y="Duration", data=scalability_plot_df, palette=palette)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), size=18)
    ax.set_ylabel("Duration [Minutes]", size=18)
    plt.savefig(f"{out_path}/Scalability{lag_window}/scalability_barplot.pdf", bbox_inches="tight", format="pdf")
    plt.close('all')

EVAL_LOGS_BASE = Path(os.path.dirname(os.path.abspath(__file__))) / "cdrift-evaluation" / "EvaluationLogs"

def calculate_rel_scalabilities_cvdrift(df, loglengths, loglengths_events):
    df_seconds = df.copy()
    df_seconds["Log Length (Cases)"] = df_seconds[["Log Source", "Log"]].apply(
        axis=1, func=lambda x: loglengths[EVAL_LOGS_BASE / x[0] / (x[1] + ".xes.gz")])
    df_seconds["Log Length (Events)"] = df_seconds[["Log Source", "Log"]].apply(
        axis=1, func=lambda x: loglengths_events[EVAL_LOGS_BASE / x[0] / (x[1] + ".xes.gz")])

    total_seconds = df_seconds["Duration (Seconds)"].sum()
    total_cases = df_seconds["Log Length (Cases)"].sum()
    total_events = df_seconds["Log Length (Events)"].sum()

    seconds_per_case = total_seconds / total_cases
    seconds_per_event = total_seconds / total_events

    df_seconds["Milliseconds per Event"] = df_seconds.apply(
        lambda x: (x["Duration (Seconds)"] * 1000) / x["Log Length (Events)"], axis=1)
    return df_seconds, seconds_per_case, seconds_per_event

def plot_rel_scalability_cvdrift(df_seconds, out_path, lag_window, color):
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette([color])
    ax = sns.boxplot(
        data=df_seconds,
        x="Milliseconds per Event",
        y="Algorithm",
        palette=palette,
        width=0.5,
        ax=ax,
        fliersize=0
    )
    ax.set_xscale("log")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel("Milliseconds per Event", size=18)
    ax.set_ylabel(ax.get_ylabel(), size=18)

    if not os.path.exists(f"{out_path}/Scalability{lag_window}"):
        os.makedirs(f"{out_path}/Scalability{lag_window}")
    plt.savefig(f"{out_path}/Scalability{lag_window}/scalability_mseconds_per_event.pdf", bbox_inches="tight", format="pdf")
    plt.close('all')

    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x="Algorithm", y="Milliseconds per Event", data=df_seconds, palette=palette)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), size=18)
    ax.set_ylabel("Milliseconds per Event", size=18)
    ax.set_yscale("log")
    plt.savefig(f"{out_path}/Scalability{lag_window}/scalability_mseconds_per_event_barplot.pdf", bbox_inches="tight", format="pdf")
    plt.close('all')

def calc_harm_means_cvdrift(dataframe, min_support, lag_window):
    robustnesses = dict()
    for noise_level, noise_df in dataframe.groupby("Noise Level"):
        f1, precision, recall = calculate_accuracy_cvdrift(noise_df, lag_window)
        scaled_latency, _ = calculate_latency_cvdrift(noise_df, lag_window, min_support=min_support)

        # Versatility: only if Ceravolo/Ostovar logs
        noise_df_v = noise_df[noise_df["Log Source"].isin(["Ceravolo", "Ostovar"])]
        if len(noise_df_v) > 0:
            versatility, _ = calc_versatility_cvdrift(noise_df_v, lag_window)
        else:
            versatility = np.nan

        robustnesses[noise_level] = {
            "CVDrift": harmonic_mean([f1, scaled_latency, versatility])
        }
    return robustnesses

def convert_harm_mean_to_auc(means, div_zero_default=0):
    def _convert_nan_to_zero(x):
        if np.isnan(x):
            x = 0
        return x

    means_reformatted = {
        approach: [
            (int(noise_level), _convert_nan_to_zero(means[noise_level][approach]))
            for noise_level in means.keys()
        ]
        for approach in means[list(means.keys())[0]].keys()
    }

    robustnesses = dict()
    for approach in means_reformatted.keys():
        points = sorted(means_reformatted[approach], key=lambda x: x[0])
        initial_robustness = points[0][1]
        ideal_auc = initial_robustness * (points[-1][0] - points[0][0])
        prev_noise, prev_robust = points[0]
        auc = 0
        for noise, robust in points[1:]:
            delta_noise = noise - prev_noise
            sum_robust = prev_robust + robust
            area = (delta_noise * sum_robust) / 2
            auc += area
            prev_noise = noise
            prev_robust = robust

        if ideal_auc != 0:
            robustnesses[approach] = auc / ideal_auc
        else:
            robustnesses[approach] = div_zero_default
    return robustnesses

def _plot_robustness_one_cvdrift(robustness_df, logset, color):
    fig, ax = plt.subplots(figsize=(8, 4))
    relevant_df = robustness_df[(robustness_df["Approach"] == "CVDrift") & (robustness_df["Log Set"] == logset)]
    sns.pointplot(x="Noise Level", y="Robustness", data=relevant_df, errorbar=None, color=color, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("CVDrift", size=16)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel("Noise Level", size=18)
    ax.set_ylabel("Performance", size=18)
    robustness_of_zero_noise = min(zip(relevant_df["Noise Level"], relevant_df["Robustness"]), key=lambda x: x[0])[1]
    if np.isnan(robustness_of_zero_noise):
        robustness_of_zero_noise = 0
    ax.hlines(robustness_of_zero_noise, color=color, linestyle='dashed', xmin=0,
              xmax=len(relevant_df["Noise Level"].unique()) - 1)
    fig.tight_layout()
    fig.set_facecolor('0.9')
    return fig, ax

def plot_robustness_cvdrift(means_ostovar, means_ceravolo, out_path, lag_window, color):
    plot_df_robust = pd.DataFrame(
        [
            {"Log Set": "Ostovar", "Approach": name, "Noise Level": int(noise_level), "Robustness": robustness}
            for noise_level, robust_dict in means_ostovar.items()
            for name, robustness in robust_dict.items()
        ] + [
            {"Log Set": "Ceravolo", "Approach": name, "Noise Level": int(noise_level), "Robustness": robustness}
            for noise_level, robust_dict in means_ceravolo.items()
            for name, robustness in robust_dict.items()
        ]
    )

    if not os.path.exists(f"{out_path}/Robustness{lag_window}"):
        os.makedirs(f"{out_path}/Robustness{lag_window}")

    fig_ost, _ = _plot_robustness_one_cvdrift(plot_df_robust, "Ostovar", color)
    fig_ost.savefig(f"{out_path}/Robustness{lag_window}/robustness_ostovar_logs.pdf", bbox_inches="tight", format="pdf")

    fig_cer, _ = _plot_robustness_one_cvdrift(plot_df_robust, "Ceravolo", color)
    fig_cer.savefig(f"{out_path}/Robustness{lag_window}/robustness_ceravolo_logs.pdf", bbox_inches="tight", format="pdf")

    plt.close('all')

def analyze_change_pattern_distribution(df, out_path):
    cp_count_results = {cp: dict() for cp in df["Change Pattern"].unique() if cp is not None}

    for name, group in df[["Log Source", "Log", "Change Pattern"]].drop_duplicates(inplace=False).groupby("Log Source"):
        series = group["Change Pattern"].value_counts().to_dict()
        for cp, count in series.items():
            cp_count_results[cp][name] = count

    for key in cp_count_results.keys():
        cp_count_results[key]["Bose"] = 0
    cp_count_results["Mixed (Bose)"] = {"Ostovar": 0, "Ceravolo": 0, "Bose": 1}

    cp_counts = pd.DataFrame(cp_count_results).fillna(0).astype(int)
    cp_counts_tr = cp_counts.transpose()
    total_logs = cp_counts.to_numpy().sum()
    cp_counts_tr["Relative Frequency"] = cp_counts_tr.apply(lambda row: row.sum() / total_logs * 100, axis=1)
    cp_counts_tr.to_csv(f"{out_path}/change_pattern_distribution.csv")


def evaluate(csv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithm_results_with_CVDrift.csv"),
             out_path="CVDrift_Evaluation_Results",
             lag_window: int = 200,
             verbose: bool = False):
    """Evaluate accuracy for CVDrift only."""

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    color = "#006BA2"

    # Preprocess
    df = readCSV_Lists(csv_path)
    df = preprocess(df)

    if len(df) == 0:
        raise ValueError("No CVDrift rows found in the CSV after preprocessing!")

    if verbose:
        print(f"CVDrift rows after preprocessing: {len(df)}")

    ## Split into Noisy/Noiseless Logs
    logs = zip(df["Log Source"], df["Log"])
    noiseless_logs = {
        log for source, log in logs if
        (source == "Ostovar" and not (log.endswith("_2") or log.endswith("_5"))) or
        (source == "Ceravolo" and log.split("_")[2] == "noise0") or
        source == "Bose"
    }

    df_noiseless = df[df["Log"].isin(noiseless_logs)]

    # Accuracy
    f1, precision, recall = calculate_accuracy_cvdrift(df_noiseless, lag_window)
    print(f"Accuracy (F1): {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    plot_accuracy_cvdrift(f1, precision, recall, out_path, lag_window, color)

    # Save results
    results_df = pd.DataFrame([{
        "Algorithm": "CVDrift",
        "Accuracy (F1)": f1,
        "Precision": precision,
        "Recall": recall,
    }])

    results_df.to_csv(f"{out_path}/accuracy_results{lag_window}.csv", index=False)
    print(f"\nResults saved to {out_path}/accuracy_results{lag_window}.csv")
    return results_df


if __name__ == "__main__":
    evaluate(verbose=True)

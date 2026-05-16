import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats as scipy_stats

from eda_mcp.utils import sample


def generate_plots(
    series: pl.Series,
    column_name: str,
    classification: str,
    output_dir: str,
) -> str | None:
    if classification == "high_cardinality":
        return None

    clean = series.drop_nulls()
    if len(clean) == 0:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in column_name)
    out_path = str(Path(output_dir) / f"{safe_name}_diagnostics.png")

    sns.set_style("whitegrid")

    try:
        if classification == "continuous":
            _plot_continuous(series, column_name, out_path)
        elif classification == "discrete":
            _plot_discrete(series, column_name, out_path)
        elif classification == "categorical":
            _plot_categorical(series, column_name, out_path)
        elif classification == "binary":
            _plot_binary(series, column_name, out_path)
        elif classification == "temporal":
            _plot_temporal(series, column_name, out_path)
    finally:
        plt.close("all")

    return out_path


def _plot_continuous(series: pl.Series, column_name: str, out_path: str) -> None:
    clean = series.drop_nulls()
    arr = clean.to_numpy()
    arr_sampled = sample(clean).to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{column_name} [continuous]", fontsize=14)

    sns.histplot(arr, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Histogram with KDE")
    axes[0, 0].set_xlabel(column_name)

    sns.boxplot(x=arr, ax=axes[0, 1])
    axes[0, 1].set_title("Boxplot")
    axes[0, 1].set_xlabel(column_name)

    scipy_stats.probplot(arr_sampled, plot=axes[1, 0])
    axes[1, 0].set_title("QQ Plot")

    sorted_arr = np.sort(arr)
    ecdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    axes[1, 1].plot(sorted_arr, ecdf)
    axes[1, 1].set_title("ECDF")
    axes[1, 1].set_xlabel(column_name)
    axes[1, 1].set_ylabel("Cumulative probability")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")


def _plot_discrete(series: pl.Series, column_name: str, out_path: str) -> None:
    arr = series.drop_nulls().to_numpy()
    vc = series.value_counts(sort=False).sort(series.name)
    val_col = vc.columns[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{column_name} [discrete]", fontsize=14)

    axes[0].bar([str(v) for v in vc[val_col].to_list()], vc["count"].to_list())
    axes[0].set_title("Value Counts")
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel("Count")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    sns.boxplot(x=arr, ax=axes[1])
    axes[1].set_title("Boxplot")
    axes[1].set_xlabel(column_name)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")


def _plot_categorical(series: pl.Series, column_name: str, out_path: str) -> None:
    n = len(series)
    vc = series.value_counts(sort=True).head(20)
    val_col = vc.columns[0]

    values = [str(v) for v in vc[val_col].to_list()][::-1]
    counts = vc["count"].to_list()[::-1]
    pcts = [c / n * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(8, max(4, len(values) * 0.4)))
    fig.suptitle(f"{column_name} [categorical]", fontsize=14)

    bars = ax.barh(values, counts)
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="left", va="center", fontsize=9,
        )
    ax.set_title("Top 20 Value Counts")
    ax.set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")


def _plot_binary(series: pl.Series, column_name: str, out_path: str) -> None:
    n = len(series)
    vc = series.value_counts(sort=True)
    val_col = vc.columns[0]

    values = [str(v) for v in vc[val_col].to_list()]
    counts = vc["count"].to_list()
    pcts = [c / n * 100 for c in counts]

    ratio = counts[0] / counts[1] if len(counts) >= 2 and counts[1] > 0 else float("inf")
    color = "tomato" if ratio > 3 else "steelblue"

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f"{column_name} [binary]", fontsize=14)

    bars = ax.bar(values, counts, color=color)
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{pct:.1f}%",
            ha="center", va="bottom",
        )
    ax.set_title("Class Balance" + (" (imbalanced)" if ratio > 3 else ""))
    ax.set_xlabel(column_name)
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")


def _plot_temporal(series: pl.Series, column_name: str, out_path: str) -> None:
    clean = series.drop_nulls().sort()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{column_name} [temporal]", fontsize=14)

    df = pl.DataFrame({column_name: clean})
    daily = df.group_by(column_name).agg(pl.len().alias("count")).sort(column_name)
    axes[0].plot(daily[column_name].to_list(), daily["count"].to_list())
    axes[0].set_title("Value Counts Over Time")
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel("Count")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    try:
        months = series.dt.month().drop_nulls()
        by_month = (
            pl.DataFrame({"month": months})
            .group_by("month")
            .agg(pl.len().alias("count"))
            .sort("month")
        )
        axes[1].bar(by_month["month"].to_list(), by_month["count"].to_list())
        axes[1].set_title("Counts by Month")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Count")
        axes[1].set_xticks(range(1, 13))
    except Exception:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

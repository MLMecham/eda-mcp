import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats as scipy_stats

from eda_mcp.stats import classify_column
from eda_mcp.utils import warn

_NUMERIC = {"continuous", "discrete"}
_MAX_SCATTER = 10


def numeric_columns(df: pl.DataFrame) -> list[str]:
    return [col for col in df.columns if classify_column(df[col]) in _NUMERIC and df[col].null_count() < len(df)]


def compute_correlations(df: pl.DataFrame, cols: list[str]) -> tuple[dict, dict]:
    pearson: dict = {c: {} for c in cols}
    spearman: dict = {c: {} for c in cols}

    for i, col_a in enumerate(cols):
        for j, col_b in enumerate(cols):
            if i == j:
                pearson[col_a][col_b] = 1.0
                spearman[col_a][col_b] = 1.0
                continue

            pair = df.select([col_a, col_b]).drop_nulls()
            a = pair[col_a].to_numpy()
            b = pair[col_b].to_numpy()

            if len(a) < 3:
                warn(f"Too few observations to correlate '{col_a}' and '{col_b}', skipping.")
                pearson[col_a][col_b] = None
                spearman[col_a][col_b] = None
                continue

            try:
                r_p, _ = scipy_stats.pearsonr(a, b)
                pearson[col_a][col_b] = round(float(r_p), 4)
            except Exception as e:
                warn(f"Pearson correlation failed for '{col_a}' vs '{col_b}': {e}")
                pearson[col_a][col_b] = None

            try:
                r_s, _ = scipy_stats.spearmanr(a, b)
                spearman[col_a][col_b] = round(float(r_s), 4)
            except Exception as e:
                warn(f"Spearman correlation failed for '{col_a}' vs '{col_b}': {e}")
                spearman[col_a][col_b] = None

    return pearson, spearman


def strong_pairs(
    pearson: dict,
    spearman: dict,
    cols: list[str],
    threshold: float,
) -> list[dict]:
    pairs = []
    for i, col_a in enumerate(cols):
        for j, col_b in enumerate(cols):
            if j <= i:
                continue
            r_s = spearman[col_a].get(col_b)
            if r_s is None or abs(r_s) < threshold:
                continue
            r_p = pearson[col_a].get(col_b)
            pairs.append({
                "column_a": col_a,
                "column_b": col_b,
                "spearman": r_s,
                "pearson": r_p,
                "flag": "highly correlated (|ρ| ≥ 0.9)" if abs(r_s) >= 0.9 else None,
            })

    pairs.sort(key=lambda x: abs(x["spearman"]), reverse=True)
    return pairs[:_MAX_SCATTER]


def plot_heatmap(spearman: dict, cols: list[str], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    matrix = np.array([[spearman[c1].get(c2, 0.0) or 0.0 for c2 in cols] for c1 in cols])

    size = max(6, len(cols))
    fig, ax = plt.subplots(figsize=(size, size - 1))
    sns.set_style("whitegrid")
    sns.heatmap(
        matrix,
        xticklabels=cols,
        yticklabels=cols,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Spearman Correlation Heatmap")
    plt.tight_layout()

    out_path = str(Path(output_dir) / "correlation_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out_path


def plot_scatter(
    df: pl.DataFrame,
    col_a: str,
    col_b: str,
    spearman: float,
    output_dir: str,
) -> str:
    pair = df.select([col_a, col_b]).drop_nulls()
    a = pair[col_a].to_numpy()
    b = pair[col_b].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_style("whitegrid")
    sns.scatterplot(x=a, y=b, ax=ax, alpha=0.6)
    ax.set_xlabel(col_a)
    ax.set_ylabel(col_b)
    ax.set_title(f"{col_a} vs {col_b}  (ρ={spearman:+.2f})")

    plt.tight_layout()

    def safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)

    out_path = str(Path(output_dir) / f"scatter_{safe(col_a)}_vs_{safe(col_b)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out_path

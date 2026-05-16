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
from eda_mcp.utils import sample, warn

_NUMERIC = {"continuous", "discrete"}
_CATEGORICAL = {"categorical", "binary"}


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def numeric_columns(df: pl.DataFrame) -> list[str]:
    return [col for col in df.columns if classify_column(df[col]) in _NUMERIC and df[col].null_count() < len(df)]


def categorical_columns(df: pl.DataFrame) -> list[str]:
    return [col for col in df.columns if classify_column(df[col]) in _CATEGORICAL and df[col].null_count() < len(df)]


# --- Numeric: Pearson + Spearman ---

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
    top_n: int = 10,
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
    return pairs[:top_n]


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

    out_path = str(Path(output_dir) / "numeric_heatmap.png")
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
    pair = sample(df.select([col_a, col_b]).drop_nulls())
    a = pair[col_a].to_numpy()
    b = pair[col_b].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_style("whitegrid")
    sns.scatterplot(x=a, y=b, ax=ax, alpha=0.6)
    ax.set_xlabel(col_a)
    ax.set_ylabel(col_b)
    ax.set_title(f"{col_a} vs {col_b}  (ρ={spearman:+.2f})")

    plt.tight_layout()

    out_path = str(Path(output_dir) / f"scatter_{_safe(col_a)}_vs_{_safe(col_b)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out_path


# --- Categorical: Cramér's V ---

def _cramers_v(a: pl.Series, b: pl.Series) -> float | None:
    joint = pl.DataFrame({"a": a, "b": b}).drop_nulls()
    if len(joint) < 3:
        return None
    try:
        a_vals = joint["a"].cast(pl.String).to_numpy()
        b_vals = joint["b"].cast(pl.String).to_numpy()
        unique_a = np.unique(a_vals)
        unique_b = np.unique(b_vals)
        contingency = np.zeros((len(unique_a), len(unique_b)), dtype=int)
        a_idx = {v: i for i, v in enumerate(unique_a)}
        b_idx = {v: i for i, v in enumerate(unique_b)}
        for av, bv in zip(a_vals, b_vals):
            contingency[a_idx[av], b_idx[bv]] += 1
        chi2, _, _, _ = scipy_stats.chi2_contingency(contingency)
        n = len(joint)
        k = min(contingency.shape) - 1
        return round(float(np.sqrt(chi2 / (n * k))), 4) if k > 0 else 0.0
    except Exception as e:
        warn(f"Cramér's V failed for '{a.name}' vs '{b.name}': {e}")
        return None


def compute_cramers_v(df: pl.DataFrame, cols: list[str]) -> dict:
    matrix: dict = {c: {} for c in cols}
    for i, col_a in enumerate(cols):
        for j, col_b in enumerate(cols):
            if i == j:
                matrix[col_a][col_b] = 1.0
            elif j < i:
                matrix[col_a][col_b] = matrix[col_b][col_a]
            else:
                matrix[col_a][col_b] = _cramers_v(df[col_a], df[col_b])
    return matrix


def strong_categorical_pairs(
    cramers_v: dict,
    cols: list[str],
    threshold: float,
    top_n: int = 3,
) -> list[dict]:
    pairs = []
    for i, col_a in enumerate(cols):
        for j, col_b in enumerate(cols):
            if j <= i:
                continue
            v = cramers_v[col_a].get(col_b)
            if v is None or v < threshold:
                continue
            pairs.append({
                "column_a": col_a,
                "column_b": col_b,
                "cramers_v": v,
                "flag": "strongly associated (V ≥ 0.7)" if v >= 0.7 else None,
            })
    pairs.sort(key=lambda x: x["cramers_v"], reverse=True)
    return pairs[:top_n]


def plot_categorical_heatmap(cramers_v: dict, cols: list[str], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    matrix = np.array([[cramers_v[c1].get(c2, 0.0) or 0.0 for c2 in cols] for c1 in cols])

    size = max(6, len(cols))
    fig, ax = plt.subplots(figsize=(size, size - 1))
    sns.set_style("whitegrid")
    sns.heatmap(
        matrix,
        xticklabels=cols,
        yticklabels=cols,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Cramér's V Association Heatmap")
    plt.tight_layout()

    out_path = str(Path(output_dir) / "categorical_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out_path


# --- Mixed: Eta-squared (categorical vs numeric) ---

def _eta_squared(numeric: pl.Series, categorical: pl.Series) -> float | None:
    joint = pl.DataFrame({"num": numeric, "cat": categorical}).drop_nulls()
    if len(joint) < 3:
        return None
    try:
        grand_mean = joint["num"].mean()
        ss_total = float(((joint["num"] - grand_mean) ** 2).sum())
        if ss_total == 0:
            return 0.0
        ss_between = sum(
            len(group_df) * (float(group_df["num"].mean()) - grand_mean) ** 2
            for _, group_df in joint.group_by("cat")
        )
        return round(float(ss_between / ss_total), 4)
    except Exception as e:
        warn(f"Eta-squared failed for '{numeric.name}' vs '{categorical.name}': {e}")
        return None


def compute_eta_squared(df: pl.DataFrame, num_cols: list[str], cat_cols: list[str]) -> dict:
    matrix: dict = {cat: {} for cat in cat_cols}
    for cat in cat_cols:
        for num in num_cols:
            matrix[cat][num] = _eta_squared(df[num], df[cat])
    return matrix


def strong_mixed_pairs(
    eta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    threshold: float,
    top_n: int = 3,
) -> list[dict]:
    pairs = []
    for cat in cat_cols:
        for num in num_cols:
            v = eta[cat].get(num)
            if v is None or v < threshold:
                continue
            pairs.append({
                "categorical": cat,
                "numeric": num,
                "eta_squared": v,
                "flag": "strong association (η² ≥ 0.14)" if v >= 0.14 else None,
            })
    pairs.sort(key=lambda x: x["eta_squared"], reverse=True)
    return pairs[:top_n]


def plot_mixed_heatmap(eta: dict, num_cols: list[str], cat_cols: list[str], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    matrix = np.array([[eta[cat].get(num, 0.0) or 0.0 for num in num_cols] for cat in cat_cols])

    fig, ax = plt.subplots(figsize=(max(6, len(num_cols)), max(4, len(cat_cols))))
    sns.set_style("whitegrid")
    sns.heatmap(
        matrix,
        xticklabels=num_cols,
        yticklabels=cat_cols,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Eta-squared: Categorical vs Numeric")
    plt.tight_layout()

    out_path = str(Path(output_dir) / "mixed_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out_path


def plot_boxplot(
    df: pl.DataFrame,
    numeric_col: str,
    categorical_col: str,
    eta: float,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    joint = sample(df.select([numeric_col, categorical_col]).drop_nulls())

    group_medians = (
        joint.group_by(categorical_col)
        .agg(pl.col(numeric_col).median().alias("median"))
        .sort("median")
    )
    order = group_medians[categorical_col].to_list()

    cat_vals = joint[categorical_col].cast(pl.String).to_list()
    num_vals = joint[numeric_col].to_list()

    fig, ax = plt.subplots(figsize=(max(6, len(order)), 5))
    sns.set_style("whitegrid")
    sns.boxplot(x=cat_vals, y=num_vals, order=[str(o) for o in order], ax=ax)
    ax.set_title(f"{numeric_col} by {categorical_col}  (η²={eta:.2f})")
    ax.set_xlabel(categorical_col)
    ax.set_ylabel(numeric_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = str(Path(output_dir) / f"boxplot_{_safe(categorical_col)}_vs_{_safe(numeric_col)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out_path


def plot_grouped_bar(
    df: pl.DataFrame,
    col_a: str,
    col_b: str,
    cramers_v: float,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    joint = df.select([col_a, col_b]).drop_nulls()

    totals = joint.group_by(col_a).agg(pl.len().alias("total"))
    counts = joint.group_by([col_a, col_b]).agg(pl.len().alias("count"))
    counts = counts.join(totals, on=col_a).with_columns(
        (pl.col("count") / pl.col("total")).alias("proportion")
    )

    a_vals = sorted(counts[col_a].cast(pl.String).unique().to_list())
    b_vals = sorted(counts[col_b].cast(pl.String).unique().to_list())

    x = np.arange(len(a_vals))
    width = 0.8 / len(b_vals)

    fig, ax = plt.subplots(figsize=(max(8, len(a_vals) * len(b_vals) * 0.4), 5))
    sns.set_style("whitegrid")

    for i, b_val in enumerate(b_vals):
        subset = counts.filter(pl.col(col_b).cast(pl.String) == b_val)
        props = []
        for a_val in a_vals:
            row = subset.filter(pl.col(col_a).cast(pl.String) == a_val)
            props.append(float(row["proportion"][0]) if len(row) > 0 else 0.0)
        ax.bar(x + i * width, props, width, label=str(b_val))

    ax.set_xticks(x + width * (len(b_vals) - 1) / 2)
    ax.set_xticklabels(a_vals, rotation=45, ha="right")
    ax.set_xlabel(col_a)
    ax.set_ylabel("Proportion")
    ax.set_title(f"{col_a} vs {col_b}  (V={cramers_v:.2f})")
    ax.legend(title=col_b, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = str(Path(output_dir) / f"grouped_bar_{_safe(col_a)}_vs_{_safe(col_b)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out_path

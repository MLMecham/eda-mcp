from pathlib import Path

import polars as pl
from mcp.server.fastmcp import FastMCP

from eda_mcp.correlations import (
    categorical_columns,
    compute_correlations,
    compute_cramers_v,
    compute_eta_squared,
    numeric_columns,
    plot_boxplot,
    plot_categorical_heatmap,
    plot_grouped_bar,
    plot_heatmap,
    plot_mixed_heatmap,
    plot_scatter,
    strong_categorical_pairs,
    strong_mixed_pairs,
    strong_pairs,
)
from eda_mcp.plots import generate_plots
from eda_mcp.reader import load_file, load_query
from eda_mcp.report import generate_markdown_report
from eda_mcp.stats import classify_column, get_summary
from eda_mcp.utils import handle_errors

mcp = FastMCP(
    "eda-mcp",
    instructions=(
        "This server runs locally on the user's machine as a stdio process. "
        "All file paths are local absolute paths on the user's filesystem — "
        "you have direct access to local files, no upload or URL is needed. "
        "Always use absolute paths when calling tools. "
        "Before calling generate_report, ask the user where they want the report saved (output_dir). "
        "The default saves to an output/ folder next to the source file — confirm this is acceptable first."
    ),
)

_COMPACT_KEYS = {
    "row_count",
    "missing_pct",
    "mean",
    "median",
    "std",
    "min",
    "Q1",
    "Q3",
    "max",
    "skewness",
    "outlier_pct",
}
_NUMERIC_DIFF_KEYS = [
    "mean",
    "median",
    "std",
    "min",
    "max",
    "outlier_pct",
    "missing_pct",
]


def _dataset_overview(df: pl.DataFrame) -> dict:
    n = df.shape[0]
    duplicate_rows = int(df.is_duplicated().sum())
    extra_rows = n - len(df.unique())
    return {
        "rows": n,
        "columns": df.shape[1],
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / n * 100, 2) if n > 0 else 0.0,
        "extra_rows": extra_rows,
        "extra_rows_pct": round(extra_rows / n * 100, 2) if n > 0 else 0.0,
        "column_names": df.columns,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "classifications": {col: classify_column(df[col]) for col in df.columns},
        "missing_counts": {col: df[col].null_count() for col in df.columns},
        "missing_pct": {
            col: round(df[col].null_count() / n * 100, 2) if n > 0 else 0.0
            for col in df.columns
        },
    }


def _resolve_output_dir(file_path: str, output_dir: str | None) -> str:
    if output_dir:
        return output_dir
    return str(Path(file_path.strip()).parent / "output")


@mcp.tool()
@handle_errors
def load_dataset(file_path: str, table: str | None = None) -> dict:
    """
    Load a local file and return a structural overview — shape, column types,
    classifications, missing values, and duplicate counts. Call this first
    when the data is already on disk and you want to load it all.

    Use query_dataset instead when you need SQL filtering, joins, remote
    sources (S3, HTTP), or a DuckDB database.

    Supports CSV, Parquet, Excel, JSON, NDJSON, Avro, SQLite (.db/.sqlite),
    DuckDB (.duckdb). For multi-table databases pass table; auto-selected if
    only one table exists.

    Returns: column names, dtypes, classifications (continuous, discrete,
    categorical, binary, temporal, high_cardinality), missing counts,
    duplicate_rows (all occurrences), extra_rows (removable by dedup).
    """
    df = load_file(file_path, table)
    return {"file_path": file_path, **_dataset_overview(df)}


@mcp.tool()
@handle_errors
def query_dataset(
    query: str,
    db_path: str | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Run a DuckDB SQL query and return the same structural overview as
    load_dataset. Use this instead of load_dataset when you need filtering,
    joins, remote sources, or a DuckDB database.

    query accepts a SQL statement or a bare file path (auto-wrapped):
      "SELECT * FROM 'sales.parquet' WHERE year=2024"
      "SELECT * FROM 's3://bucket/data.parquet'"
      "SELECT a.*, b.col FROM 'file1.csv' a JOIN 'file2.parquet' b ON a.id=b.id"
      "C:/data/sales.parquet"

    db_path: path to a .duckdb file when querying named tables inside it.
    output_path: where to save result as Parquet (default: system temp dir).

    Pass result_path to any other tool exactly like file_path.
    """
    import os
    import tempfile

    df = load_query(query, db_path)

    if df.shape[0] == 0:
        return {
            "error": "Query returned 0 rows — check join conditions, filters, or key compatibility."
        }

    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "eda_query_result.parquet")
    df.write_parquet(output_path)
    return {"result_path": output_path, "query": query, **_dataset_overview(df)}


@mcp.tool()
@handle_errors
def get_column_summary(
    file_path: str,
    column: str,
    table: str | None = None,
    classification: str | None = None,
    full_summary: bool = True,
) -> dict:
    """
    Return summary statistics for a single column. Stats depend on type:
    continuous/discrete: five-number summary, mean, std, skewness, kurtosis,
    outlier count (IQR), normality test. categorical/binary: mode, value counts,
    class balance. temporal: date range, gap count. high_cardinality: sample only.

    classification: override auto-detection — e.g. "categorical" on a discrete
    column for value counts. Valid: continuous, discrete, categorical, binary,
    temporal, high_cardinality. Invalid overrides fall back with a warning.
    full_summary: False returns compact subset matching get_column_summary_by_group.
    """
    df = load_file(file_path, table)
    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found. Available columns: {df.columns}"
        }
    summary = get_summary(df[column], classification)
    return (
        summary
        if full_summary
        else {k: v for k, v in summary.items() if k in _COMPACT_KEYS}
    )


@mcp.tool()
@handle_errors
def get_all_summaries(file_path: str, table: str | None = None) -> dict:
    """
    Return summary statistics for every column at once, keyed by column name.
    Same output as get_column_summary per column. For large datasets with many
    columns prefer get_column_summary on specific columns of interest.
    """
    df = load_file(file_path, table)
    results = {}
    for col in df.columns:
        try:
            results[col] = get_summary(df[col])
        except Exception as e:
            results[col] = {"error": str(e)}
    return results


@mcp.tool()
@handle_errors
def get_diagnostic_plot(
    file_path: str,
    column: str,
    output_dir: str,
    table: str | None = None,
    classification: str | None = None,
) -> dict:
    """
    Generate and save a diagnostic PNG for a single column. Plot type is
    auto-selected: continuous → histogram/KDE/boxplot/QQ/ECDF, discrete →
    value counts + boxplot, categorical → horizontal bar, binary → class
    balance bar, temporal → time series + monthly bar, high_cardinality →
    no plot. Saves to output_dir/{column}_diagnostics.png.

    classification: override auto-detection, same rules as get_column_summary.
    """
    df = load_file(file_path, table)
    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found. Available columns: {df.columns}"
        }
    if classification is not None:
        from eda_mcp.stats import _validate_classification_override

        classification = _validate_classification_override(df[column], classification)
    if classification is None:
        classification = classify_column(df[column])
    if classification == "high_cardinality":
        return {
            "error": f"No plot generated for '{column}': high cardinality column (likely ID or free text)."
        }
    path = generate_plots(df[column], column, classification, output_dir)
    return {"path": path} if path else {"error": f"No plot generated for '{column}'."}


@mcp.tool()
@handle_errors
def get_correlations(
    file_path: str,
    output_dir: str | None = None,
    numeric_threshold: float = 0.5,
    categorical_threshold: float = 0.3,
    mixed_threshold: float = 0.1,
    top_n: int = 3,
    table: str | None = None,
    plots: bool = False,
    numeric: bool = True,
    categorical: bool = True,
    mixed: bool = True,
) -> dict:
    """
    Compute pairwise associations between columns. Three types, all on by default:

    numeric: Pearson + Spearman between continuous/discrete columns.
    categorical: Cramér's V between categorical/binary columns (0–1).
    mixed: Eta-squared (η²) between categorical and numeric columns (0–1,
    measures variance explained).

    numeric_threshold: min |ρ| for strong_pairs (default 0.5).
    categorical_threshold: min V for strong_pairs (default 0.3).
    mixed_threshold: min η² for strong_pairs (default 0.1).
    top_n: max pairs returned per type (default 3).
    plots: True generates heatmaps + scatter/bar/boxplots in {stem}_correlations/.
    """
    df = load_file(file_path, table)
    stem = Path(file_path.strip()).stem
    out = str(Path(_resolve_output_dir(file_path, output_dir)) / f"{stem}_correlations")
    result = {}
    num_cols = numeric_columns(df) if (numeric or mixed) else []
    cat_cols = categorical_columns(df) if (categorical or mixed) else []

    if numeric:
        if len(num_cols) >= 2:
            pearson, spearman = compute_correlations(df, num_cols)
            pairs = strong_pairs(pearson, spearman, num_cols, numeric_threshold, top_n)
            result["numeric"] = {
                "columns": num_cols,
                "pearson": pearson,
                "spearman": spearman,
                "strong_pairs": pairs,
            }
            if plots:
                heatmap = plot_heatmap(spearman, num_cols, out)
                scatter_paths = [
                    plot_scatter(df, p["column_a"], p["column_b"], p["spearman"], out)
                    for p in pairs
                ]
                result["numeric"]["heatmap"] = heatmap
                result["numeric"]["scatter_plots"] = scatter_paths
        else:
            result["numeric"] = {"error": "Need at least 2 numeric columns."}

    if categorical:
        if len(cat_cols) >= 2:
            cramers = compute_cramers_v(df, cat_cols)
            pairs = strong_categorical_pairs(
                cramers, cat_cols, categorical_threshold, top_n
            )
            result["categorical"] = {
                "columns": cat_cols,
                "cramers_v": cramers,
                "strong_pairs": pairs,
            }
            if plots:
                result["categorical"]["heatmap"] = plot_categorical_heatmap(
                    cramers, cat_cols, out
                )
                result["categorical"]["grouped_bars"] = [
                    plot_grouped_bar(
                        df, p["column_a"], p["column_b"], p["cramers_v"], out
                    )
                    for p in pairs
                ]
        else:
            result["categorical"] = {"error": "Need at least 2 categorical columns."}

    if mixed:
        if num_cols and cat_cols:
            eta = compute_eta_squared(df, num_cols, cat_cols)
            pairs = strong_mixed_pairs(eta, num_cols, cat_cols, mixed_threshold, top_n)
            result["mixed"] = {
                "numeric_columns": num_cols,
                "categorical_columns": cat_cols,
                "eta_squared": eta,
                "strong_pairs": pairs,
            }
            if plots:
                result["mixed"]["heatmap"] = plot_mixed_heatmap(
                    eta, num_cols, cat_cols, out
                )
                result["mixed"]["boxplots"] = [
                    plot_boxplot(
                        df, p["numeric"], p["categorical"], p["eta_squared"], out
                    )
                    for p in pairs
                ]
        else:
            result["mixed"] = {
                "error": "Need at least 1 numeric and 1 categorical column."
            }

    return result


@mcp.tool()
@handle_errors
def generate_report(
    file_path: str,
    output_dir: str | None = None,
    table: str | None = None,
    numeric: bool = True,
    categorical: bool = True,
    mixed: bool = True,
) -> dict:
    """
    Generate a full EDA markdown report — dataset overview, data quality flags,
    per-column stats with diagnostic plots, and association analysis (Pearson,
    Cramér's V, eta-squared). Saves to {filename}_eda_report.md.

    Always ask the user for output_dir before calling. Default saves next to
    the source file in output/, which may not be what they want.

    Returns path, flags, column_classifications, and shape. After generating,
    present the path and summarize the flags to the user — ask which columns
    or issues they want to investigate next. Do not automatically call further
    tools without the user's direction.

    Toggle association sections with numeric/categorical/mixed bools.
    """
    df = load_file(file_path, table)
    path, flags, classifications = generate_markdown_report(
        df,
        file_path,
        _resolve_output_dir(file_path, output_dir),
        numeric=numeric,
        categorical=categorical,
        mixed=mixed,
    )
    return {
        "path": path,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "flags": flags,
        "column_classifications": classifications,
    }


@mcp.tool()
@handle_errors
def compare_distributions(
    source_a: str,
    source_b: str,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict:
    """
    Compare the distributions of two data slices column by column and return
    a statistical diff. Each source should resolve to one flat slice of data —
    use WHERE to define the group, not GROUP BY.

    Both source_a and source_b accept a file path or a SQL query, same as
    query_dataset:

      # Two file paths

            compare_distributions("sales_2023.parquet", "sales_2024.parquet")

      # Two slices from the same file — one condition per source
      compare_distributions(
          "SELECT * FROM 'diamonds.parquet' WHERE cut='Ideal'",
          "SELECT * FROM 'diamonds.parquet' WHERE cut='Fair'",
          label_a="Ideal", label_b="Fair"
      )

      # Subgroups — AND as many conditions as needed, still one flat slice per source
      compare_distributions(
          "SELECT * FROM 'diamonds.parquet' WHERE cut='Ideal' AND color='D'",
          "SELECT * FROM 'diamonds.parquet' WHERE cut='Fair' AND color='J'",
          label_a="Ideal-D", label_b="Fair-J"
      )

      # Two result paths from query_dataset
      compare_distributions(result_path_a, result_path_b, label_a="2023", label_b="2024")

    Returns:
    - shape: row and column count for each source
    - only_in_{label_a} / only_in_{label_b}: columns that appear in one source but not the other
    - per_column_diffs: for each shared column, the delta between the two sources —
      numeric deltas (mean_delta, median_delta, std_delta, outlier_pct_delta,
      missing_pct_delta), classification changes, and mode changes for categoricals

    label_a / label_b: optional human-readable names used in the response
    (e.g. "Ideal", "Fair", "2023", "2024"). Defaults to "a" and "b".
    """
    label_a = label_a or "a"
    label_b = label_b or "b"

    df_a = load_query(source_a)
    df_b = load_query(source_b)

    summaries_a = {col: get_summary(df_a[col]) for col in df_a.columns}
    summaries_b = {col: get_summary(df_b[col]) for col in df_b.columns}

    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    shared = cols_a & cols_b

    per_column_diffs = {}
    for col in shared:
        s_a = summaries_a[col]
        s_b = summaries_b[col]
        diff = {
            f"classification_{label_a}": s_a.get("classification"),
            f"classification_{label_b}": s_b.get("classification"),
            "classification_changed": s_a.get("classification")
            != s_b.get("classification"),
        }
        for key in _NUMERIC_DIFF_KEYS:
            v_a = s_a.get(key)
            v_b = s_b.get(key)
            if v_a is not None and v_b is not None:
                try:
                    diff[f"{key}_{label_a}"] = v_a
                    diff[f"{key}_{label_b}"] = v_b
                    diff[f"{key}_delta"] = round(float(v_b) - float(v_a), 4)
                except (TypeError, ValueError):
                    pass
        if (
            s_a.get("classification") == "categorical"
            or s_b.get("classification") == "categorical"
        ):
            diff[f"mode_{label_a}"] = s_a.get("mode")
            diff[f"mode_{label_b}"] = s_b.get("mode")
            diff["mode_changed"] = s_a.get("mode") != s_b.get("mode")
            diff[f"unique_count_{label_a}"] = s_a.get("unique_count")
            diff[f"unique_count_{label_b}"] = s_b.get("unique_count")

        per_column_diffs[col] = diff

    return {
        "shape": {
            label_a: {"rows": df_a.shape[0], "columns": df_a.shape[1]},
            label_b: {"rows": df_b.shape[0], "columns": df_b.shape[1]},
        },
        f"only_in_{label_a}": sorted(cols_a - cols_b),
        f"only_in_{label_b}": sorted(cols_b - cols_a),
        "per_column_diffs": per_column_diffs,
    }


@mcp.tool()
@handle_errors
def get_column_summary_by_group(
    file_path: str,
    column: str,
    group_by: str | list[str],
    table: str | None = None,
    full_summary: bool = False,
) -> dict:
    """
    Return summary statistics for a column broken down by each unique combination
    of one or more group columns. Equivalent to calling get_column_summary once
    per group, but in a single call.

    Use this after load_dataset to investigate how a column's distribution
    varies across categories — e.g. how BST varies by bond_tier, or how
    revenue varies by region and year together.

    Pass a single column name or a list of column names to group_by. Multiple
    group columns produce a cartesian breakdown — use load_dataset first to
    understand cardinality before grouping by high-cardinality columns, as this
    can produce a very large number of groups.

    For comparing exactly two groups with quantified deltas, use
    compare_distributions instead — it explicitly shows by how much each statistic
    changed. get_column_summary_by_group is better when you have many groups
    and want to see all of them at once.

    Returns a dict keyed by group value (or tuple of group values for multiple
    group columns), each containing the full summary stats for that group —
    same structure as get_column_summary. Also includes a top-level "groups"
    key listing all group combinations found, and "group_counts" showing the
    row count per group.

    column: the column to summarize within each group.
    group_by: column name or list of column names to group by.
    full_summary: if False (default), returns a compact summary per group —
    count, missing_pct, mean, median, std, min, Q1, Q3, max, skewness,
    outlier_pct. If True, returns the complete get_summary output including
    kurtosis, normality test, zero count, etc. Only use full_summary=True
    when you have very few groups (3 or fewer) and need the detailed stats.
    """
    df = load_file(file_path, table)

    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found. Available columns: {df.columns}"
        }

    group_cols = [group_by] if isinstance(group_by, str) else group_by
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        return {
            "error": f"Group column(s) not found: {missing}. Available columns: {df.columns}"
        }

    results = {}
    group_counts = {}

    for group_values, group_df in df.group_by(group_cols, maintain_order=True):
        key = group_values[0] if len(group_cols) == 1 else tuple(group_values)
        key = str(key)
        group_counts[key] = len(group_df)
        try:
            summary = get_summary(group_df[column])
            results[key] = (
                summary
                if full_summary
                else {k: v for k, v in summary.items() if k in _COMPACT_KEYS}
            )
        except Exception as e:
            results[key] = {"error": str(e)}

    return {
        "column": column,
        "group_by": group_by,
        "groups": list(results.keys()),
        "group_counts": group_counts,
        "summaries": results,
    }


def main():
    mcp.run(transport="stdio")

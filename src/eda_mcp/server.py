from pathlib import Path

import polars as pl
from mcp.server.fastmcp import FastMCP

from eda_mcp.correlations import (
    compute_correlations,
    numeric_columns,
    plot_heatmap,
    plot_scatter,
    strong_pairs,
)
from eda_mcp.plots import generate_plots
from eda_mcp.reader import load_file, load_query
from eda_mcp.report import generate_markdown_report
from eda_mcp.stats import classify_column, get_summary
from eda_mcp.utils import handle_errors

mcp = FastMCP("eda-mcp")

_COMPACT_KEYS = {"row_count", "missing_pct", "mean", "median", "std", "min", "Q1", "Q3", "max", "skewness", "outlier_pct"}
_NUMERIC_DIFF_KEYS = ["mean", "median", "std", "min", "max", "outlier_pct", "missing_pct"]


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
    Load a dataset and return a structural overview. Call this first when
    exploring an unfamiliar dataset — it gives you the shape, column types,
    classifications, and missing value counts you need to decide what to
    investigate next.

    Returns: column names, dtypes, row count, per-column classifications
    (continuous, discrete, categorical, binary, temporal, high_cardinality),
    missing value counts and percentages per column. Also returns duplicate_rows
    (total rows that have at least one duplicate) and extra_rows (rows that would
    be removed by deduplication — the actionable count).

    Supports CSV, Parquet, Excel (.xlsx/.xls), JSON, NDJSON, Avro, SQLite
    (.db/.sqlite), and DuckDB (.duckdb). For multi-table databases, pass the
    table name via `table`. If omitted and the database has exactly one table,
    it is loaded automatically.
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
    Run a DuckDB SQL query and return a structural overview of the result,
    identical to load_dataset. Use this instead of load_dataset when your
    data comes from a SQL query, a remote source, or a DuckDB database.

    query accepts either a SQL statement or a bare file path:
      "SELECT * FROM 's3://bucket/sales.parquet' WHERE year=2024"
      "SELECT * FROM read_parquet('s3://bucket/data/', hive_partitioning=true)"
      "C:/data/sales.parquet"  -- bare path, wrapped automatically

    Cross-file joins are supported — DuckDB can mix any readable sources in one
    query:
      "SELECT c.*, o.amount FROM 'customers.csv' c JOIN sqlite_scan('sales.db', 'orders') o ON c.id = o.customer_id"
      "SELECT * FROM 'users.parquet' u JOIN 'events.csv' e ON u.id = e.user_id"

    Remote sources (S3, GCS, HTTP) are supported via DuckDB's httpfs extension.
    Credentials are read from your environment (AWS_ACCESS_KEY_ID, etc.).

    db_path: optional path to a local .duckdb database file to query against.
    output_path: where to save the result as Parquet (default: system temp dir).

    Pass result_path from the response to any other tool — get_column_summary,
    get_all_summaries, get_correlations, generate_report — exactly like file_path.
    """
    import os
    import tempfile

    df = load_query(query, db_path)

    if df.shape[0] == 0:
        return {"error": "Query returned 0 rows — check join conditions, filters, or key compatibility."}

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
    Return full summary statistics for a single column. The column type is
    auto-detected and the appropriate statistics are computed:

    - continuous/discrete: five-number summary, mean, std, skewness with plain
      english label, kurtosis with label, outlier count (IQR method), zero count,
      infinite count, normality test (scipy normaltest p-value and result)
    - categorical: mode, top 10 value counts with percentages
    - binary: mode, top value counts, class balance ratio with imbalance flag
      (flagged if majority:minority ratio exceeds 3:1)
    - temporal: min/max date, date range in days, gap count, most common year
      and month
    - high_cardinality: flagged as likely ID or free text with sample values only

    Use this to investigate a specific column in depth after calling load_dataset
    to identify columns of interest.

    classification: optional override for the auto-detected column type. Use when
    the classifier gets it wrong — e.g. pass "categorical" to get value counts on
    a column detected as discrete, or "continuous" to get distribution stats on a
    column detected as high_cardinality. Valid values: continuous, discrete,
    categorical, binary, temporal, high_cardinality. Invalid overrides, dtype
    mismatches (e.g. "continuous" on a string column), and "binary" on a column
    with more than 2 unique values are rejected with a warning and fall back to
    auto-detection.
    full_summary: if True (default), returns all statistics. If False, returns
    the compact subset used by get_column_summary_by_group — useful for
    consistency when comparing a single column against grouped results.
    """
    df = load_file(file_path, table)
    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found. Available columns: {df.columns}"
        }
    summary = get_summary(df[column], classification)
    return summary if full_summary else {k: v for k, v in summary.items() if k in _COMPACT_KEYS}


@mcp.tool()
@handle_errors
def get_all_summaries(file_path: str, table: str | None = None) -> dict:
    """
    Return summary statistics for every column in the dataset in a single call,
    keyed by column name. Each value contains all statistics appropriate for the
    column's detected type — equivalent to calling get_column_summary once per
    column.

    Use this for a complete statistical overview of the entire dataset at once.
    For large datasets with many columns, prefer get_column_summary to inspect
    individual columns of interest rather than loading everything at once.
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
    file_path: str, column: str, output_dir: str, table: str | None = None, classification: str | None = None
) -> dict:
    """
    Generate and save a diagnostic plot for a single column as a PNG file.
    The plot type is automatically selected based on the column's classification:

    - continuous: 2x2 panel — histogram with KDE overlay, boxplot with outliers,
      QQ plot with reference line, ECDF
    - discrete: bar chart of value counts and boxplot side by side
    - categorical: horizontal bar chart of top 20 values with percentage labels
    - binary: bar chart of class balance with proportion labels; bars are red
      if the majority:minority ratio exceeds 3:1
    - temporal: line plot of counts over time and bar chart of counts by month
    - high_cardinality: no plot is generated; a message is returned instead

    Saves the PNG to output_dir/{column}_diagnostics.png and returns the file
    path. Use output_dir to control where plots land — the same folder as the
    dataset or a dedicated output directory both work well.

    classification: optional override for the auto-detected column type. Same
    validation rules as get_column_summary — invalid overrides fall back to
    auto-detection with a warning.
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
    threshold: float = 0.5,
    table: str | None = None,
    plots: bool = False,
) -> dict:
    """
    Compute pairwise correlations between all numeric columns in the dataset.

    Returns both Pearson and Spearman correlation matrices, the strongest pairs
    above the threshold (sorted by absolute Spearman correlation, max 10),
    and highly correlated flags for pairs with |ρ| >= 0.9.

    Only continuous and discrete columns are included — categorical, binary,
    temporal, and high_cardinality columns are excluded automatically.

    threshold controls which pairs are flagged as strong (default 0.5). Set
    higher e.g. 0.7 for only strong correlations, lower e.g. 0.3 to cast a
    wider net.

    plots: set to True to generate a Spearman heatmap and scatter plots for
    strong pairs. Plots are saved to output_dir/correlations/. Defaults to
    False — omit when you only need the numbers.
    """
    df = load_file(file_path, table)
    cols = numeric_columns(df)

    if len(cols) < 2:
        return {"error": "Need at least 2 numeric columns to compute correlations."}

    pearson, spearman = compute_correlations(df, cols)
    pairs = strong_pairs(pearson, spearman, cols, threshold)

    if not plots:
        return {
            "numeric_columns": cols,
            "pearson": pearson,
            "spearman": spearman,
            "strong_pairs": pairs,
        }

    stem = Path(file_path.strip()).stem
    out = str(Path(_resolve_output_dir(file_path, output_dir)) / f"{stem}_correlations")
    heatmap_path = plot_heatmap(spearman, cols, out)

    scatter_paths = []
    for pair in pairs:
        path = plot_scatter(
            df, pair["column_a"], pair["column_b"], pair["spearman"], out
        )
        scatter_paths.append(path)

    return {
        "numeric_columns": cols,
        "pearson": pearson,
        "spearman": spearman,
        "strong_pairs": pairs,
        "heatmap": heatmap_path,
        "scatter_plots": scatter_paths,
    }


@mcp.tool()
@handle_errors
def generate_report(
    file_path: str, output_dir: str | None = None, table: str | None = None
) -> dict:
    """
    Generate a complete EDA markdown report for the entire dataset. This is the
    main tool to call for a thorough, end-to-end analysis. The report includes:

    - Dataset overview: row count, column count, memory usage, total missing values
    - Data quality flags: columns with >20% missing values, imbalanced binary
      columns, high cardinality columns, columns with infinite values, columns
      with >10% outliers by IQR method
    - Per-column variable summaries: statistics table, diagnostic plot image,
      and a 2-3 sentence plain english interpretation of the distribution shape,
      outliers, and data quality for each column

    Saves the report as {filename}_eda_report.md in output_dir alongside the
    diagnostic plot PNGs. Returns the path to the saved report file.

    For quick inspection of a single column use get_column_summary or
    get_diagnostic_plot instead of running the full report.
    """
    df = load_file(file_path, table)
    path = generate_markdown_report(
        df, file_path, _resolve_output_dir(file_path, output_dir)
    )
    return {"path": path}


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
    - only_in_a / only_in_b: columns that appear in one source but not the other
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
            "classification_changed": s_a.get("classification") != s_b.get("classification"),
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
        if s_a.get("classification") == "categorical" or s_b.get("classification") == "categorical":
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
        return {"error": f"Column '{column}' not found. Available columns: {df.columns}"}

    group_cols = [group_by] if isinstance(group_by, str) else group_by
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        return {"error": f"Group column(s) not found: {missing}. Available columns: {df.columns}"}

    results = {}
    group_counts = {}

    for group_values, group_df in df.group_by(group_cols, maintain_order=True):
        key = group_values[0] if len(group_cols) == 1 else tuple(group_values)
        key = str(key)
        group_counts[key] = len(group_df)
        try:
            summary = get_summary(group_df[column])
            results[key] = summary if full_summary else {k: v for k, v in summary.items() if k in _COMPACT_KEYS}
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

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from eda_mcp.correlations import (
    compute_correlations,
    numeric_columns,
    plot_heatmap,
    plot_scatter,
    strong_pairs,
)
from eda_mcp.plots import generate_plots
from eda_mcp.reader import load_file
from eda_mcp.report import generate_markdown_report
from eda_mcp.stats import classify_column, get_summary
from eda_mcp.utils import handle_errors

mcp = FastMCP("eda-mcp")


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
    missing value counts and percentages per column.

    Supports CSV, Parquet, Excel (.xlsx/.xls), JSON, NDJSON, Avro, and SQLite
    (.db/.sqlite). For SQLite files with multiple tables, pass the table name
    via `table`. If omitted and the database has exactly one table, it is loaded
    automatically.
    """
    df = load_file(file_path, table)
    n = df.shape[0]
    duplicate_rows = int(df.is_duplicated().sum())
    return {
        "file_path": file_path,
        "rows": n,
        "columns": df.shape[1],
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / n * 100, 2) if n > 0 else 0.0,
        "column_names": df.columns,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "classifications": {col: classify_column(df[col]) for col in df.columns},
        "missing_counts": {col: df[col].null_count() for col in df.columns},
        "missing_pct": {
            col: round(df[col].null_count() / n * 100, 2) if n > 0 else 0.0
            for col in df.columns
        },
    }


@mcp.tool()
@handle_errors
def get_column_summary(file_path: str, column: str, table: str | None = None) -> dict:
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
    """
    df = load_file(file_path, table)
    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found. Available columns: {df.columns}"
        }
    return get_summary(df[column])


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
    file_path: str, column: str, output_dir: str, table: str | None = None
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
    """
    df = load_file(file_path, table)
    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found. Available columns: {df.columns}"
        }
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
) -> dict:
    """
    Compute pairwise correlations between all numeric columns in the dataset
    and generate a Spearman correlation heatmap. Scatter plots are generated
    for column pairs with a Spearman correlation above the threshold.

    Returns both Pearson and Spearman correlation matrices, the strongest pairs
    above the threshold (sorted by absolute Spearman correlation, max 10),
    highly correlated flags for pairs with |ρ| >= 0.9, and file paths for all
    generated plots.

    Only continuous and discrete columns are included — categorical, binary,
    temporal, and high_cardinality columns are excluded automatically.

    threshold controls which pairs get scatter plots (default 0.5). Set higher
    e.g. 0.7 for only strong correlations, lower e.g. 0.3 to cast a wider net.
    Scatter plots are capped at 10 pairs regardless of threshold.
    """
    df = load_file(file_path, table)
    cols = numeric_columns(df)

    if len(cols) < 2:
        return {"error": "Need at least 2 numeric columns to compute correlations."}

    out = _resolve_output_dir(file_path, output_dir)
    pearson, spearman = compute_correlations(df, cols)
    pairs = strong_pairs(pearson, spearman, cols, threshold)

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
def generate_report(file_path: str, output_dir: str | None = None, table: str | None = None) -> dict:
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
    path = generate_markdown_report(df, file_path, _resolve_output_dir(file_path, output_dir))
    return {"path": path}


def main():
    mcp.run(transport="stdio")

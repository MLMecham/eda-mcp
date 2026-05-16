# eda-mcp

An MCP server for exploratory data analysis. Point it at a dataset and let your AI assistant do the analysis — summary statistics, diagnostic plots, correlation analysis, and full markdown reports, all from a single conversation.

Built by [MLMecham](https://github.com/MLMecham).

---

## Quickstart

Run instantly with no install step:

```bash
uvx eda-mcp
```

Or install permanently:

```bash
pip install eda-mcp
```

---

## Connecting to Claude Desktop

Add this to your `claude_desktop_config.json`:

**Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "eda-mcp": {
      "command": "uvx",
      "args": ["eda-mcp"]
    }
  }
}
```

Restart Claude Desktop. The tools will appear automatically.

> **Tip:** Add `--refresh` to always pull the latest version from PyPI on startup:
> ```json
> "args": ["--refresh", "eda-mcp"]
> ```

---

## Troubleshooting

**Tools not appearing after install or update**

uvx caches the installed version and won't update automatically. Force a refresh:

```bash
uvx --refresh eda-mcp --help
```

Then fully quit and reopen Claude Desktop (not just close the window).

**Check server logs**

If the tools still don't appear, check the MCP server logs:

- **Windows:** `%APPDATA%\Claude\logs\mcp-server-eda-mcp.log`
- **Mac:** `~/Library/Logs/Claude/mcp-server-eda-mcp.log`

---

## Tools

| Tool | Description |
|---|---|
| `load_dataset` | Load a file and return a structural overview — column names, types, classifications, missing value counts, and duplicate stats. Start here. |
| `query_dataset` | Run a DuckDB SQL query and return the same overview as `load_dataset`. Supports local files, remote sources (S3, GCS, HTTP), SQLite, and cross-file joins. Result saved to Parquet for use with other tools. |
| `get_column_summary` | Full statistics for a single column. Accepts optional `classification` override and `full_summary=False` for a compact output. |
| `get_all_summaries` | Summary statistics for every column at once, keyed by column name. |
| `get_column_summary_by_group` | Summary statistics for a column broken down by one or more group columns. Compact by default, `full_summary=True` for detailed output. |
| `get_diagnostic_plot` | Generate a diagnostic plot for a single column. Plot type is auto-selected by classification. Accepts optional `classification` override. |
| `get_correlations` | Compute all three association types: Pearson + Spearman (numeric), Cramér's V (categorical), and eta-squared (mixed). Each type toggleable independently with separate thresholds. |
| `compare_distributions` | Compare the distributions of two data slices column by column. Accepts file paths or SQL queries. Returns labeled deltas for all numeric and categorical columns. |
| `generate_report` | Full EDA report — dataset overview, data quality flags, per-column summaries with plots, and full association analysis. Saved as markdown. |

---

## Supported File Formats

| Format | Extension |
|---|---|
| CSV | `.csv` |
| Parquet | `.parquet` |
| Excel | `.xlsx`, `.xls` |
| JSON | `.json` |
| Newline-delimited JSON | `.ndjson` |
| Avro | `.avro` |
| SQLite | `.db`, `.sqlite` |
| DuckDB | `.duckdb` |

String columns are automatically coerced to better types on load (integers, floats, dates) where unambiguous.

For SQLite and DuckDB files with multiple tables, pass the `table` parameter to specify which one. If the database has exactly one table it is loaded automatically.

### Querying with SQL

Use `query_dataset` for SQL-based loading, remote sources, or cross-file joins:

```sql
-- Filter before analysis
SELECT * FROM 's3://bucket/sales.parquet' WHERE year = 2024

-- Cross-file join — mix any DuckDB-readable sources
SELECT t.*, p.bst FROM 'trainers.csv' t JOIN 'pokemon.parquet' p ON t.pokemon = p.name

-- Query a local DuckDB database (pass db_path separately)
SELECT * FROM my_table

-- Hive-partitioned S3
SELECT * FROM read_parquet('s3://bucket/data/', hive_partitioning=true)
```

Pass the `result_path` from `query_dataset` to any other tool exactly like a regular `file_path`.

---

## Association Analysis

`get_correlations` computes three types of associations in one call:

| Type | Measure | Columns | Default threshold |
|---|---|---|---|
| `numeric` | Pearson + Spearman | continuous, discrete | 0.5 |
| `categorical` | Cramér's V | categorical, binary | 0.3 |
| `mixed` | Eta-squared (η²) | categorical vs numeric | 0.1 |

Toggle each type with `numeric=True/False`, `categorical=True/False`, `mixed=True/False`. Set `plots=True` to generate heatmaps and pair-level charts.

### Comparing distributions

`compare_distributions` diffs two slices column by column:

```python
# Compare two cut grades
compare_distributions(
    "SELECT * FROM 'diamonds.parquet' WHERE cut='Ideal'",
    "SELECT * FROM 'diamonds.parquet' WHERE cut='Fair'",
    label_a="Ideal", label_b="Fair"
)

# Compare two time periods
compare_distributions("sales_2023.parquet", "sales_2024.parquet", label_a="2023", label_b="2024")
```

Returns mean, median, std, outlier, and missing value deltas per column — Claude can immediately say how much each statistic changed.

---

## Column Classifications

Every column is automatically classified before analysis:

| Classification | Description |
|---|---|
| `continuous` | Floats, or integers with more than 20 unique values |
| `discrete` | Integers with 20 or fewer unique values |
| `categorical` | Strings with low cardinality (< 5% unique ratio or ≤ 10 unique values) |
| `binary` | Booleans, or any column with exactly 2 unique non-null values |
| `temporal` | Date, Datetime, or Duration columns |
| `high_cardinality` | Likely identifiers, UUIDs, or free text — statistical summary skipped |

Pass `classification="categorical"` to any summary or plot tool to override the auto-detected type.

---

## Using as a Python Library

The core functions are also importable directly:

```python
from eda_mcp import (
    load_file, load_query,
    classify_column, get_summary,
    numeric_columns, categorical_columns,
    compute_correlations, compute_cramers_v, compute_eta_squared,
    generate_markdown_report,
)

df = load_file("data/sales.parquet")
summary = get_summary(df["revenue"])
generate_markdown_report(df, "data/sales.parquet", "output/")

# Query and analyze
df = load_query("SELECT * FROM 'data.parquet' WHERE region='West'")
```

---

## Example Prompts

Once connected to Claude:

```
Analyze this dataset: /path/to/data.csv
```
```
Join the Batting and People tables in my Lahman SQLite database and generate a full EDA report
```
```
Compare the price distribution between Ideal and Fair cut diamonds
```
```
How does revenue vary across regions and product categories?
```
```
What columns in sales.parquet have missing values?
```
```
Generate a full EDA report for customers.xlsx
```

---

## Requirements

- Python 3.11+
- Dependencies are installed automatically via `uvx` or `pip`

---

## License

MIT

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
| `load_dataset` | Load a file and get column names, types, classifications, and missing value counts. Start here. |
| `query_dataset` | Run a DuckDB SQL query and return the same overview as `load_dataset`. Supports local files, remote sources (S3, GCS, HTTP), SQLite, and cross-file joins. Result is saved to Parquet for use with other tools. |
| `get_column_summary` | Full statistics for a single column — five-number summary, skewness, kurtosis, outlier count, normality test. Accepts an optional `classification` override. |
| `get_all_summaries` | Summary statistics for every column at once, keyed by column name. |
| `get_diagnostic_plot` | Generate a diagnostic plot for a single column. Plot type is auto-selected by classification. |
| `get_correlations` | Pearson and Spearman correlation matrices, a heatmap, and scatter plots for strongly correlated pairs. |
| `generate_report` | Full EDA report — dataset overview, data quality flags, per-column summaries with plots, and correlation analysis. Saved as markdown. |

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

-- Cross-file join
SELECT t.*, p.bst FROM 'trainers.csv' t JOIN 'pokemon.parquet' p ON t.pokemon = p.name

-- Query a DuckDB database
SELECT * FROM my_table  -- with db_path pointing to your .duckdb file
```

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

---

## Using as a Python Library

The core functions are also importable directly:

```python
from eda_mcp import load_file, classify_column, get_summary, generate_markdown_report

df = load_file("data/sales.parquet")
summary = get_summary(df["revenue"])
generate_markdown_report(df, "data/sales.parquet", "output/")
```

---

## Example Prompts

Once connected to Claude:

```
Analyze this dataset: /path/to/data.csv
```
```
What columns in sales.parquet have missing values?
```
```
Is age correlated with income in this file?
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

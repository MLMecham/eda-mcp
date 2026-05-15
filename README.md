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

---

## Tools

| Tool | Description |
|---|---|
| `load_dataset` | Load a file and get column names, types, classifications, and missing value counts. Start here. |
| `get_column_summary` | Full statistics for a single column — five-number summary, skewness, kurtosis, outlier count, normality test. |
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

String columns are automatically coerced to better types on load (integers, floats, dates) where unambiguous.

For SQLite files with multiple tables, pass the `table` parameter to specify which one.

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

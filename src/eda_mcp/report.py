from datetime import datetime
from pathlib import Path

import polars as pl
from jinja2 import Template

from eda_mcp.correlations import (
    compute_correlations,
    numeric_columns,
    plot_heatmap,
    plot_scatter,
    strong_pairs,
)
from eda_mcp.plots import generate_plots
from eda_mcp.stats import classify_column, get_summary
from eda_mcp.utils import warn

_TEMPLATE = Template("""\
# EDA Report: {{ filename }}
Generated: {{ timestamp }}

## Dataset Overview

| | |
|---|---|
| Source | `{{ file_path }}` |
| Rows | {{ rows }} |
| Columns | {{ columns }} |
| Total missing values | {{ total_missing }} ({{ total_missing_pct }}%) |
| Duplicate rows | {{ duplicate_rows }} ({{ duplicate_pct }}%) |
| Memory usage | {{ memory_mb }} MB |

## Data Quality Flags

{% if flags %}
{% for flag in flags %}
- {{ flag }}
{% endfor %}
{% else %}
No data quality issues detected.
{% endif %}

---

## Variable Summaries

{% for col in column_reports %}
### {{ col.name }} [{{ col.classification }}]

{% if col.stats_table %}
{{ col.stats_table }}

{% endif %}
{% if col.top_counts_table %}
**Top value counts:**

{{ col.top_counts_table }}

{% endif %}
{% if col.plot_path %}
![{{ col.name }} diagnostics]({{ col.plot_path }})

{% endif %}
*{{ col.interpretation }}*

---
{% endfor %}

{% if corr %}
## Correlation Analysis

Numeric columns: {{ corr.columns | join(', ') }}

{% if corr.heatmap %}
![Spearman Correlation Heatmap]({{ corr.heatmap }})

{% endif %}
{% if corr.strong_pairs %}
### Strong Correlations (|ρ| ≥ {{ corr.threshold }})

| Column A | Column B | Spearman ρ | Pearson r | Flag |
|---|---|---|---|---|
{% for pair in corr.strong_pairs %}| {{ pair.column_a }} | {{ pair.column_b }} | {{ pair.spearman }} | {{ pair.pearson }} | {{ pair.flag or '—' }} |
{% endfor %}

{% for pair in corr.strong_pairs %}
{% if pair.scatter_path %}
![{{ pair.column_a }} vs {{ pair.column_b }}]({{ pair.scatter_path }})

{% endif %}
{% endfor %}
{% else %}
No correlations found above threshold {{ corr.threshold }}.
{% endif %}
{% endif %}
""")


def generate_markdown_report(df: pl.DataFrame, file_path: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = Path(file_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows, columns = df.shape

    total_missing = sum(df[c].null_count() for c in df.columns)
    total_cells = rows * columns
    total_missing_pct = round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0
    duplicate_rows = int(df.is_duplicated().sum())
    duplicate_pct = round(duplicate_rows / rows * 100, 2) if rows > 0 else 0.0

    try:
        memory_mb = round(df.estimated_size("mb"), 2)
    except Exception:
        memory_mb = "unknown"

    summaries = {}
    for col in df.columns:
        try:
            summaries[col] = get_summary(df[col])
        except Exception as e:
            summaries[col] = {"classification": "unknown", "error": str(e)}

    flags = _collect_flags(summaries)
    images_dir = Path(output_dir) / f"{filename}_images"

    column_reports = []
    for col in df.columns:
        summary = summaries[col]
        classification = summary.get("classification", "unknown")

        if df[col].null_count() == rows:
            column_reports.append({
                "name": col,
                "classification": classification,
                "stats_table": None,
                "top_counts_table": None,
                "plot_path": None,
                "interpretation": f"{col} contains no data — all {rows} values are missing.",
            })
            continue

        try:
            plot_path = generate_plots(df[col], col, classification, str(images_dir))
            if plot_path:
                plot_path = f"{filename}_images/{Path(plot_path).name}"
        except Exception:
            plot_path = None

        column_reports.append({
            "name": col,
            "classification": classification,
            "stats_table": _stats_table(summary),
            "top_counts_table": _top_counts_table(summary.get("top_10_value_counts", [])),
            "plot_path": plot_path,
            "interpretation": _interpret(col, summary),
        })

    corr = _build_correlation_section(df, filename, images_dir)

    out_path = str(Path(output_dir) / f"{filename}_eda_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_TEMPLATE.render(
            filename=filename,
            timestamp=timestamp,
            file_path=file_path,
            rows=rows,
            columns=columns,
            total_missing=total_missing,
            total_missing_pct=total_missing_pct,
            duplicate_rows=duplicate_rows,
            duplicate_pct=duplicate_pct,
            memory_mb=memory_mb,
            flags=flags,
            column_reports=column_reports,
            corr=corr,
        ))

    return out_path


def _build_correlation_section(
    df: pl.DataFrame, filename: str, images_dir: Path
) -> dict | None:
    try:
        cols = numeric_columns(df)
        if len(cols) < 2:
            return None

        threshold = 0.5
        pearson, spearman = compute_correlations(df, cols)
        pairs = strong_pairs(pearson, spearman, cols, threshold)

        heatmap_abs = plot_heatmap(spearman, cols, str(images_dir))
        heatmap_rel = f"{filename}_images/{Path(heatmap_abs).name}"

        for pair in pairs:
            try:
                scatter_abs = plot_scatter(
                    df, pair["column_a"], pair["column_b"],
                    pair["spearman"], str(images_dir)
                )
                pair["scatter_path"] = f"{filename}_images/{Path(scatter_abs).name}"
            except Exception as e:
                warn(f"Scatter plot failed for {pair['column_a']} vs {pair['column_b']}: {e}")
                pair["scatter_path"] = None

        return {
            "columns": cols,
            "strong_pairs": pairs,
            "heatmap": heatmap_rel,
            "threshold": threshold,
        }
    except Exception as e:
        warn(f"Correlation analysis failed: {e}")
        return None


def _collect_flags(summaries: dict) -> list[str]:
    flags = []
    for col, s in summaries.items():
        if s.get("missing_pct", 0) > 20:
            flags.append(f"**{col}**: {s['missing_pct']}% missing values")
        if s.get("classification") == "high_cardinality":
            flags.append(f"**{col}**: high cardinality — possible ID or free text column")
        if s.get("infinite_count", 0) > 0:
            flags.append(f"**{col}**: {s['infinite_count']} infinite values")
        if s.get("outlier_pct", 0) > 10:
            flags.append(f"**{col}**: {s['outlier_pct']}% outliers by IQR method")
        cb = s.get("class_balance", {})
        if cb.get("imbalanced"):
            flags.append(f"**{col}**: imbalanced binary column ({cb['ratio']}:1 ratio)")
    return flags


def _stats_table(summary: dict) -> str:
    skip = {"classification", "top_10_value_counts", "flag", "sample_values"}
    rows = ["| Metric | Value |", "|---|---|"]
    for key, value in summary.items():
        if key in skip:
            continue
        label = key.replace("_", " ").title()
        if isinstance(value, dict):
            for subkey, subval in value.items():
                rows.append(f"| {label} — {subkey.replace('_', ' ')} | {subval} |")
        else:
            rows.append(f"| {label} | {value} |")
    return "\n".join(rows)


def _top_counts_table(top_10: list) -> str:
    if not top_10:
        return ""
    rows = ["| Value | Count | % |", "|---|---|---|"]
    for row in top_10:
        rows.append(f"| {row['value']} | {row['count']} | {row['pct']}% |")
    return "\n".join(rows)


def _interpret(column_name: str, summary: dict) -> str:
    classification = summary.get("classification", "unknown")
    parts = []

    if classification in ("continuous", "discrete"):
        norm = summary.get("normality_test", {})
        skew_label = summary.get("skewness_label", "")
        p = norm.get("p_value")
        if p is not None:
            parts.append(
                f"{column_name} is {norm.get('result', '')} with a {skew_label} distribution (p={p})."
            )
        elif skew_label:
            parts.append(f"{column_name} shows a {skew_label} distribution.")

        outlier_pct = summary.get("outlier_pct", 0)
        if outlier_pct > 0:
            parts.append(f"{outlier_pct}% of values are outliers by IQR method.")

        missing_pct = summary.get("missing_pct", 0)
        parts.append("No missing values detected." if missing_pct == 0 else f"{missing_pct}% of values are missing.")

    elif classification == "categorical":
        unique_count = summary.get("unique_count", 0)
        mode = summary.get("mode")
        parts.append(f"{column_name} has {unique_count} unique values.")
        if mode is not None:
            parts.append(f"The most frequent value is '{mode}'.")
        missing_pct = summary.get("missing_pct", 0)
        parts.append("No missing values detected." if missing_pct == 0 else f"{missing_pct}% of values are missing.")

    elif classification == "binary":
        cb = summary.get("class_balance", {})
        if cb:
            ratio = cb.get("ratio", 1)
            if cb.get("imbalanced"):
                parts.append(f"{column_name} is imbalanced with a {ratio}:1 majority-to-minority ratio.")
            else:
                parts.append(f"{column_name} has a balanced class distribution ({ratio}:1 ratio).")
        missing_pct = summary.get("missing_pct", 0)
        parts.append("No missing values detected." if missing_pct == 0 else f"{missing_pct}% of values are missing.")

    elif classification == "temporal":
        min_date = summary.get("min_date")
        max_date = summary.get("max_date")
        date_range = summary.get("date_range_days")
        if min_date and max_date:
            parts.append(f"{column_name} spans from {min_date} to {max_date} ({date_range} days).")
        gap_count = summary.get("gap_count", 0)
        if gap_count:
            parts.append(f"{gap_count} gaps detected in the time series.")
        missing_pct = summary.get("missing_pct", 0)
        if missing_pct > 0:
            parts.append(f"{missing_pct}% of values are missing.")

    elif classification == "high_cardinality":
        parts.append(
            f"{column_name} appears to be a high-cardinality identifier or free text column — statistical summary is not meaningful."
        )
        sample = summary.get("sample_values", [])
        if sample:
            parts.append(f"Sample values: {', '.join(str(v) for v in sample)}.")

    elif classification == "unknown":
        parts.append(f"Could not generate summary for {column_name}: {summary.get('error', 'unknown error')}.")

    return " ".join(parts)

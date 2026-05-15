from datetime import datetime
from pathlib import Path

import polars as pl
from jinja2 import Template

from .plots import generate_plots
from .stats import classify_column, get_summary

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

{{ col.stats_table }}

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
""")


def generate_markdown_report(df: pl.DataFrame, file_path: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = Path(file_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows, columns = df.shape

    total_missing = sum(df[c].null_count() for c in df.columns)
    total_cells = rows * columns
    total_missing_pct = round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0

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
            memory_mb=memory_mb,
            flags=flags,
            column_reports=column_reports,
        ))

    return out_path


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

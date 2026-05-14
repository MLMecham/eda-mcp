from scipy import stats as scipy_stats
import polars as pl

_INTEGER_TYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)
_FLOAT_TYPES = (pl.Float32, pl.Float64)
_STRING_TYPES = (pl.String, pl.Categorical)


def classify_column(series: pl.Series) -> str:
    dtype = series.dtype
    n_total = len(series)
    n_unique = series.n_unique()
    cardinality_ratio = n_unique / n_total if n_total > 0 else 0.0

    if dtype == pl.Boolean:
        return "binary"
    if dtype in (pl.Date, pl.Duration) or str(dtype).startswith("Datetime"):
        return "temporal"
    if n_unique == 2:
        return "binary"
    if dtype in _FLOAT_TYPES:
        return "continuous"
    if dtype in _INTEGER_TYPES:
        return "continuous" if n_unique > 20 else "discrete"
    if dtype in _STRING_TYPES or str(dtype) == "Utf8":
        return "categorical" if cardinality_ratio < 0.05 else "high_cardinality"
    return "high_cardinality"


def _skewness_label(value: float) -> str:
    if value < -1:
        return "strongly left skewed"
    if value < -0.5:
        return "moderately left skewed"
    if value <= 0.5:
        return "approximately normal"
    if value <= 1:
        return "moderately right skewed"
    return "strongly right skewed"


def _kurtosis_label(value: float) -> str:
    if value < -1:
        return "platykurtic/light tailed"
    if value <= 1:
        return "approximately normal"
    return "leptokurtic/heavy tailed"


def _numeric_stats(series: pl.Series) -> dict:
    n = len(series)
    clean = series.drop_nulls()

    q1 = series.quantile(0.25, interpolation="midpoint") or 0.0
    q3 = series.quantile(0.75, interpolation="midpoint") or 0.0
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = int(clean.filter((clean < lower) | (clean > upper)).len())

    skew = clean.skew()
    kurt = clean.kurtosis()
    is_float = series.dtype in _FLOAT_TYPES
    infinite_count = int(series.is_infinite().sum()) if is_float else 0
    zero_count = int(clean.filter(clean == 0).len())

    arr = clean.to_numpy()
    if len(arr) >= 8:
        _, p = scipy_stats.normaltest(arr)
        p = float(p)
        normality = {
            "p_value": round(p, 4),
            "result": "likely normal (p>0.05)" if p > 0.05 else "likely not normal (p<=0.05)",
        }
    else:
        normality = {"p_value": None, "result": "insufficient data for normality test"}

    return {
        "min": series.min(),
        "Q1": q1,
        "median": series.quantile(0.5, interpolation="midpoint"),
        "Q3": q3,
        "max": series.max(),
        "mean": round(float(series.mean()), 4) if series.mean() is not None else None,
        "std": round(float(series.std()), 4) if series.std() is not None else None,
        "skewness": round(float(skew), 4) if skew is not None else None,
        "skewness_label": _skewness_label(skew) if skew is not None else None,
        "kurtosis": round(float(kurt), 4) if kurt is not None else None,
        "kurtosis_label": _kurtosis_label(kurt) if kurt is not None else None,
        "outlier_count": outlier_count,
        "outlier_pct": round(outlier_count / n * 100, 2) if n > 0 else 0.0,
        "zero_count": zero_count,
        "zero_pct": round(zero_count / n * 100, 2) if n > 0 else 0.0,
        "infinite_count": infinite_count,
        "normality_test": normality,
    }


def _categorical_stats(series: pl.Series, include_class_balance: bool = False) -> dict:
    n = len(series)
    clean = series.drop_nulls()
    mode = clean.mode()[0] if len(clean) > 0 else None

    vc_df = series.value_counts(sort=True).head(10)
    top_10 = [
        {"value": row[0], "count": row[1], "pct": round(row[1] / n * 100, 2)}
        for row in vc_df.iter_rows()
    ]

    result: dict = {"mode": mode, "top_10_value_counts": top_10}

    if include_class_balance:
        counts = [row[1] for row in series.value_counts(sort=True).iter_rows()]
        if len(counts) >= 2:
            ratio = counts[0] / counts[1]
            result["class_balance"] = {
                "majority_count": counts[0],
                "minority_count": counts[1],
                "ratio": round(ratio, 2),
                "imbalanced": ratio > 3,
            }

    return result


def _temporal_stats(series: pl.Series) -> dict:
    clean = series.drop_nulls().sort()
    if len(clean) == 0:
        return {}

    min_date = clean[0]
    max_date = clean[-1]

    try:
        delta = max_date - min_date
        date_range_days = delta.days if hasattr(delta, "days") else None
    except Exception:
        date_range_days = None

    try:
        diffs = clean.diff().drop_nulls()
        gap_count = int(diffs.dt.total_days().filter(
            diffs.dt.total_days() > 1
        ).len())
    except Exception:
        gap_count = None

    try:
        most_common_year = int(series.dt.year().drop_nulls().mode()[0])
    except Exception:
        most_common_year = None

    try:
        most_common_month = int(series.dt.month().drop_nulls().mode()[0])
    except Exception:
        most_common_month = None

    return {
        "min_date": str(min_date),
        "max_date": str(max_date),
        "date_range_days": date_range_days,
        "gap_count": gap_count,
        "most_common_year": most_common_year,
        "most_common_month": most_common_month,
    }


def get_summary(series: pl.Series) -> dict:
    n = len(series)
    classification = classify_column(series)
    missing_count = series.null_count()

    summary: dict = {
        "dtype": str(series.dtype),
        "classification": classification,
        "row_count": n,
        "missing_count": missing_count,
        "missing_pct": round(missing_count / n * 100, 2) if n > 0 else 0.0,
        "unique_count": series.n_unique(),
        "cardinality_ratio": round(series.n_unique() / n, 4) if n > 0 else 0.0,
    }

    if classification in ("continuous", "discrete"):
        summary.update(_numeric_stats(series))
    elif classification == "categorical":
        summary.update(_categorical_stats(series))
    elif classification == "binary":
        summary.update(_categorical_stats(series, include_class_balance=True))
    elif classification == "temporal":
        summary.update(_temporal_stats(series))
    elif classification == "high_cardinality":
        summary["flag"] = "possible ID or free text column, statistical summary not meaningful"
        summary["sample_values"] = series.drop_nulls().head(5).to_list()

    return summary

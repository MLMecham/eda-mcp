import sqlite3
from pathlib import Path

import polars as pl

_SUPPORTED = {".csv", ".parquet", ".xlsx", ".xls", ".json", ".ndjson", ".avro", ".db", ".sqlite"}
_STRING_DTYPES = (pl.String, pl.Categorical, pl.Utf8)


def load_file(file_path: str, table: str | None = None) -> pl.DataFrame:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pl.read_csv(file_path, try_parse_dates=True)
    elif ext == ".parquet":
        df = pl.read_parquet(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pl.read_excel(file_path)
    elif ext == ".json":
        df = pl.read_json(file_path)
    elif ext == ".ndjson":
        df = pl.read_ndjson(file_path)
    elif ext == ".avro":
        df = pl.read_avro(file_path)
    elif ext in (".db", ".sqlite"):
        df = _load_sqlite(file_path, table)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {', '.join(sorted(_SUPPORTED))}"
        )

    return _coerce_types(df)


def _coerce_types(df: pl.DataFrame) -> pl.DataFrame:
    try:
        better = {}
        for col in df.columns:
            if df[col].dtype not in _STRING_DTYPES:
                continue
            series = df[col]
            original_nulls = series.null_count()
            for target in (pl.Date, pl.Datetime, pl.Int64, pl.Float64):
                try:
                    cast = series.cast(target, strict=False)
                    if cast.null_count() == original_nulls:
                        better[col] = cast
                        break
                except Exception:
                    continue
        if better:
            return df.with_columns([v.alias(k) for k, v in better.items()])
        return df
    except Exception:
        return df


def _load_sqlite(file_path: str, table: str | None) -> pl.DataFrame:
    with sqlite3.connect(file_path) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]

        if not tables:
            raise ValueError(f"No tables found in '{file_path}'.")

        if table is None:
            if len(tables) == 1:
                table = tables[0]
            else:
                raise ValueError(
                    f"'{file_path}' has multiple tables: {tables}. "
                    f"Specify one with the 'table' parameter."
                )
        elif table not in tables:
            raise ValueError(
                f"Table '{table}' not found in '{file_path}'. "
                f"Available tables: {tables}"
            )

        return pl.read_database(f'SELECT * FROM "{table}"', conn)

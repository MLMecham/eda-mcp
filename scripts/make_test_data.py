"""
Generate test datasets in all supported formats.

Reads raw CSVs from data/raw/ and writes every format to data/generated/.
Also generates a synthetic dataset that exercises every column classification.

Usage:
    python scripts/make_test_data.py
"""

import datetime
import random
import sqlite3
import sys
import uuid
from pathlib import Path

import duckdb
import polars as pl

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "generated"


def write_formats(df: pl.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT_DIR / f"{name}.csv")
    df.write_parquet(OUT_DIR / f"{name}.parquet")
    df.write_excel(OUT_DIR / f"{name}.xlsx")
    df.write_json(OUT_DIR / f"{name}.json")
    df.write_ndjson(OUT_DIR / f"{name}.ndjson")
    _write_sqlite(df, name, OUT_DIR / f"{name}.db")
    _write_duckdb(df, name, OUT_DIR / f"{name}.duckdb")
    print(f"  wrote {name} in 7 formats")


def _write_sqlite(df: pl.DataFrame, table: str, path: Path) -> None:
    _type_map = {
        pl.Int8: "INTEGER", pl.Int16: "INTEGER", pl.Int32: "INTEGER", pl.Int64: "INTEGER",
        pl.UInt8: "INTEGER", pl.UInt16: "INTEGER", pl.UInt32: "INTEGER", pl.UInt64: "INTEGER",
        pl.Float32: "REAL", pl.Float64: "REAL",
        pl.Boolean: "INTEGER",
    }

    def sql_type(dtype: pl.DataType) -> str:
        return _type_map.get(dtype, "TEXT")

    col_defs = ", ".join(f'"{c}" {sql_type(t)}' for c, t in zip(df.columns, df.dtypes))
    placeholders = ", ".join("?" * len(df.columns))

    def coerce(row: tuple) -> tuple:
        return tuple(str(v) if isinstance(v, (datetime.date, datetime.datetime)) else v for v in row)

    with sqlite3.connect(str(path)) as conn:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute(f'CREATE TABLE "{table}" ({col_defs})')
        conn.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', (coerce(r) for r in df.rows()))


def _write_duckdb(df: pl.DataFrame, table: str, path: Path) -> None:
    path.unlink(missing_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM df')
    conn.close()


def make_multi_table_duckdb(n: int = 300) -> None:
    """Two-table DuckDB for testing query_dataset joins."""
    random.seed(99)
    player_ids = [f"P{i:04d}" for i in range(n)]

    players = pl.DataFrame({
        "player_id": player_ids,
        "name": [f"Player {i}" for i in range(n)],
        "position": [random.choice(["OF", "IF", "P", "C"]) for _ in range(n)],
        "birth_year": [random.randint(1970, 2000) for _ in range(n)],
    })

    stats = pl.DataFrame({
        "player_id": [random.choice(player_ids) for _ in range(n * 3)],
        "season": [random.randint(2010, 2023) for _ in range(n * 3)],
        "games": [random.randint(1, 162) for _ in range(n * 3)],
        "hits": [random.randint(0, 200) for _ in range(n * 3)],
        "home_runs": [random.randint(0, 50) for _ in range(n * 3)],
        "avg": [round(random.uniform(0.150, 0.350), 3) for _ in range(n * 3)],
    })

    path = OUT_DIR / "multi_table.duckdb"
    path.unlink(missing_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE players AS SELECT * FROM players")
    conn.execute("CREATE TABLE stats AS SELECT * FROM stats")
    conn.close()
    print("  wrote multi_table.duckdb (players + stats tables)")


def make_synthetic(n: int = 300) -> pl.DataFrame:
    random.seed(42)

    categories = ["alpha", "beta", "gamma", "delta", "epsilon"]
    note_words = ["good", "bad", "average", "excellent", "poor", "needs review", "flagged", "ok"]
    base_date = datetime.date(2020, 1, 1)

    ids = [str(uuid.UUID(int=random.getrandbits(128))) for _ in range(n)]
    ages = [random.randint(18, 80) for _ in range(n)]
    scores = [random.randint(1, 10) for _ in range(n)]
    incomes = [round(abs(random.gauss(55000, 30000)) + random.expovariate(1 / 20000), 2) for _ in range(n)]
    cats = [random.choice(categories) for _ in range(n)]
    is_active = [random.random() < 0.82 for _ in range(n)]
    dates = [base_date + datetime.timedelta(days=random.randint(0, 365 * 3)) for _ in range(n)]
    notes = [f"entry-{uuid.uuid4().hex[:8]}: {random.choice(note_words)}" for _ in range(n)]
    measurements = [round(random.gauss(10.0, 2.5), 3) if random.random() > 0.25 else None for _ in range(n)]

    return pl.DataFrame({
        "id": pl.Series(ids),
        "age": pl.Series(ages, dtype=pl.Int32),
        "score": pl.Series(scores, dtype=pl.Int32),
        "income": pl.Series(incomes, dtype=pl.Float64),
        "category": pl.Series(cats),
        "is_active": pl.Series(is_active),
        "signup_date": pl.Series(dates),
        "measurement": pl.Series(measurements, dtype=pl.Float64),
        "notes": pl.Series(notes),
    })


def main() -> None:
    print("Generating test datasets into data/generated/\n")

    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        name = csv_path.stem
        print(f"[{name}]")
        try:
            df = pl.read_csv(csv_path, try_parse_dates=True)
            write_formats(df, name)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print("\n[synthetic]")
    write_formats(make_synthetic(), "synthetic")

    print("\n[multi_table]")
    make_multi_table_duckdb()

    print("\nDone.")


if __name__ == "__main__":
    main()

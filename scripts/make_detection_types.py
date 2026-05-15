"""
Generate data/raw/detection_types.csv — a dataset designed to exercise every
classification path in classify_column() and every coercion path in _coerce_types().

Column              Expected classification     Why
-----------         -----------------------     ---
row_id              high_cardinality            all-unique integers + "id" in name
customer_uuid       high_cardinality            UUID pattern detected in values
purchase_year       discrete                    integers in 1900-2100 range (year-like)
rating              discrete                    integers, <=20 unique values
age                 continuous                  integers, >20 unique values
income              continuous                  floats
region              categorical                 4 unique strings / 100 rows = 4% < 5%
is_churned          binary                      boolean column
outcome             binary                      exactly 2 unique string values
signup_date         temporal                    string with dashes -> coerces to Date
zip_code            high_cardinality            all-unique integers (no dashes, stays Int64)
notes               high_cardinality            unique free-text strings
score               continuous                  floats with decimals
int_as_string       continuous                  "42"-style strings -> coerce to Int64
"""

import csv
import datetime
from pathlib import Path

OUT = Path(__file__).parent.parent / "data" / "raw" / "detection_types.csv"
N = 100
REGIONS = ["North", "South", "East", "West"]
BASE_DATE = datetime.date(2022, 1, 1)


def make_uuid(i: int) -> str:
    hex_i = f"{i:012x}"
    return f"a1b2c3d4-e5f6-7890-abcd-{hex_i}"


def make_row(i: int) -> dict:
    return {
        "row_id":       i,
        "customer_uuid": make_uuid(i),
        "purchase_year": 2018 + (i % 5),
        "rating":       (i % 5) + 1,
        "age":          18 + (i % 57),
        "income":       round(30000 + i * 512.75, 2),
        "region":       REGIONS[i % 4],
        "is_churned":   i % 5 == 0,
        "outcome":      "win" if i % 3 != 0 else "loss",
        "signup_date":  str(BASE_DATE + datetime.timedelta(days=i - 1)),
        "zip_code":     10000 + i,
        "notes":        f"ref-{i:04d}: customer note entry number {i}",
        "score":        round(1.0 + (i / N) * 9.0, 2),
        "int_as_string": str(100 + i),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [make_row(i) for i in range(1, N + 1)]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {N} rows to {OUT}")


if __name__ == "__main__":
    main()

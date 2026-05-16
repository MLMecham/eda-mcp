from contextvars import ContextVar
from functools import wraps
from typing import Literal

import polars as pl

_warnings_var: ContextVar[list[str]] = ContextVar("warnings")


def sample(
    df: pl.DataFrame,
    n: int = 10_000,
    strategy: Literal["random", "stratified", "temporal", "outlier_preserving"] = "random",
    seed: int = 42,
) -> pl.DataFrame:
    """
    Sample a DataFrame down to at most n rows. Returns the full DataFrame unchanged
    if it already has n or fewer rows.

    strategy:
      "random" (default): uniform random sample. Fast and sufficient for most plots.
      "stratified": not yet implemented — falls back to random.
      "temporal": not yet implemented — falls back to random.
      "outlier_preserving": not yet implemented — falls back to random.
    """
    if len(df) <= n:
        return df
    if strategy != "random":
        warn(f"Sampling strategy '{strategy}' is not yet implemented — falling back to random.")
    return df.sample(n, seed=seed)


def warn(msg: str) -> None:
    try:
        _warnings_var.get().append(msg)
    except LookupError:
        pass


def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        token = _warnings_var.set([])
        try:
            result = func(*args, **kwargs)
            warnings = _warnings_var.get()
            if not warnings:
                return result
            if isinstance(result, dict):
                result["warnings"] = warnings
                return result
            return {"result": result, "warnings": warnings}
        except Exception as e:
            warnings = _warnings_var.get()
            err: dict = {"error": f"{func.__name__} failed: {e}"}
            if warnings:
                err["warnings"] = warnings
            return err
        finally:
            _warnings_var.reset(token)
    return wrapper

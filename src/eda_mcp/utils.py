from contextvars import ContextVar
from functools import wraps

_warnings_var: ContextVar[list[str]] = ContextVar("warnings")


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

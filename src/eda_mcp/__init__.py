"""eda-mcp: MCP server for exploratory data analysis."""

from importlib.metadata import version

__version__ = version("eda-mcp")

from eda_mcp.reader import load_file
from eda_mcp.stats import classify_column, get_summary
from eda_mcp.plots import generate_plots
from eda_mcp.report import generate_markdown_report
from eda_mcp.correlations import compute_correlations, numeric_columns

__all__ = [
    # Version
    "__version__",
    # Data loading
    "load_file",
    # Statistics
    "classify_column",
    "get_summary",
    # Plots
    "generate_plots",
    # Reports
    "generate_markdown_report",
    # Correlations
    "compute_correlations",
    "numeric_columns",
]

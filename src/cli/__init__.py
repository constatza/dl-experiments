"""CLI-facing helper functions for graph-cg scripts.

This package exposes pure-Python entry points used by the thin command-line
wrappers that live at the repository root. By keeping the functional logic here
we can import the same routines from both orchestration code and executable
scripts without duplicating behaviour.
"""

from .data import (  # noqa: F401
    process_data_from_config,
    load_data_config,
)
from .prediction import run_inference  # noqa: F401
from .training import (  # noqa: F401
    train_model,
    train_pca_preconditioner,
)
from .comparison import compare_preconditioners  # noqa: F401
from .analysis import analyze_dataset  # noqa: F401
from .raw_data import standardize_raw_filenames  # noqa: F401

__all__ = [
    "process_data_from_config",
    "load_data_config",
    "run_inference",
    "train_model",
    "train_pca_preconditioner",
    "compare_preconditioners",
    "analyze_dataset",
    "standardize_raw_filenames",
]

"""Diagnostics helpers for spectra and preconditioning metrics."""

from .spectra import (
    compute_condition_numbers,
    format_scientific,
    precondition_matrix,
    plot_condition_numbers,
)
from .plots import plot_condition_boxes, plot_condition_bars

__all__ = [
    "compute_condition_numbers",
    "format_scientific",
    "precondition_matrix",
    "plot_condition_numbers",
    "plot_condition_boxes",
    "plot_condition_bars",
]

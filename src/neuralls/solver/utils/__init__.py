"""Utility functions for solver implementation.

This package contains utility functions organized by purpose:
- numerics: Numerical stability utilities (stable_dot_product, compute_curvature)
- operators: Linear operator wrappers and conversions
- validation: Input validation utilities

Design Principles:
    - Pure functions where possible
    - Single Responsibility
    - No side effects (except logging)
"""

from .numerics import check_breakdown, compute_curvature, stable_dot_product

__all__ = [
    "stable_dot_product",
    "compute_curvature",
    "check_breakdown",
]

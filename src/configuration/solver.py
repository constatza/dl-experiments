"""Solver parameters - delegated to Pydantic models.

This module previously contained dataclass definitions for solver parameters.
These have been removed in favor of using Pydantic models directly:
- Use GeneralSolverConfig from solver_models.py
- Use SolverSpecConfig from solver_models.py

The Pydantic models provide validation and are used directly throughout the codebase
without intermediate dataclass conversion.
"""

from __future__ import annotations

# Re-export Pydantic models for backward compatibility with imports
from .solver_models import GeneralSolverConfig, SolverSpecConfig

__all__ = ["GeneralSolverConfig", "SolverSpecConfig"]

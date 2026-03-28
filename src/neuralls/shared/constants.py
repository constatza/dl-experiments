"""Constants and configuration values for graph-cg project.

This module centralizes magic values used across the codebase to improve
maintainability and avoid duplication.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from .math_utils import MatrixNormType

# =============================================================================
# Exit Codes
# =============================================================================
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_KEYBOARD_INTERRUPT = 130

# =============================================================================
# Default Paths
# =============================================================================
# Dynamically determine project root relative to this file (src/neuralls/constants.py)
# src/neuralls/constants.py -> src/neuralls -> src -> project_root
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _path_from_env(var_name: str, fallback: Path) -> Path:
    """Resolve default path from env override when present."""
    value = os.getenv(var_name)
    if value:
        return Path(value).expanduser().resolve()
    return fallback


DEFAULT_PROCESSED_DATA_DIR = _path_from_env(
    "GRAPH_CG_PROCESSED_DIR", DEFAULT_PROJECT_ROOT / "data" / "processed"
)
DEFAULT_OUTPUT_DIR = _path_from_env("GRAPH_CG_OUTPUT_DIR", DEFAULT_PROJECT_ROOT / "data" / "output")
DEFAULT_FIGURES_DIR = _path_from_env(
    "GRAPH_CG_FIGURES_DIR", DEFAULT_PROJECT_ROOT / "data" / "figures"
)
DEFAULT_CHECKPOINTS_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"
DEFAULT_MLRUNS_DIR = _path_from_env("GRAPH_CG_MLRUNS_DIR", DEFAULT_PROJECT_ROOT / "data" / "mlruns")
DEFAULT_MLARTIFACTS_DIR = _path_from_env(
    "GRAPH_CG_MLARTIFACTS_DIR", DEFAULT_PROJECT_ROOT / "data" / "mlartifacts"
)

# Legacy compatibility - these match old common.py constants
DEFAULT_PROCESSED_DIR = str(DEFAULT_PROCESSED_DATA_DIR)
DEFAULT_RESULTS_DIR = str(DEFAULT_OUTPUT_DIR)
DEFAULT_FIGURES_DIR_STR = str(DEFAULT_FIGURES_DIR)

# Default configuration file paths (relative to graph-cg root)
DEFAULT_MODEL_CONFIG = "configs/experiments/default/linear.toml"
DEFAULT_DATA_CONFIG = "configs/datasets/collect-504-solutions.toml"
DEFAULT_EXPERIMENTS_CONFIG = "configs/experiments.toml"

# Default Experiment Config Filenames
EXP_MODEL_CONFIG_NAME = "linear.toml"
EXP_DATA_CONFIG_NAME = "data.toml"
EXP_SOLVER_CONFIG_NAME = "solver.toml"

# =============================================================================
# CG Solver Defaults
# =============================================================================
DEFAULT_RTOL = 1e-6  # Relative tolerance for CG solves
DEFAULT_ATOL = 1e-14  # Absolute tolerance for convergence (in addition to relative tolerance)
DEFAULT_CG_MAX_ITERATIONS = 1000
DEFAULT_CG_STOPPING_CRITERION = "tolerance"

# Breakdown tolerance for CG algorithms (used in comparison and solver internals)
DEFAULT_BREAKDOWN_TOL = 1e-12  # Threshold for detecting numerical breakdown in denominators

# Prediction diagnostic thresholds
PREDICTION_NORM_EPSILON = 1e-12  # Epsilon for safe division in prediction norm ratio

# Reorthogonalization numerical tolerances
REORTHOG_ZERO_NORM_TOL = 1e-14  # Threshold for considering a norm as zero
REORTHOG_MAX_COEFF = 1e10  # Maximum safe reorthogonalization coefficient
REORTHOG_STRICT_THRESHOLD = 0.01  # Strict selective reorthogonalization threshold
REORTHOG_DEFAULT_WINDOW = 10  # Default window size for partial reorthogonalization

# CG Algorithm Safety Parameters
# These parameters control restart/breakdown behavior for flexible PCG with non-SPD preconditioners
DEFAULT_CURVATURE_EPSILON = 1e-14  # eps_curv: Restart if curvature d_k < eps_curv * ||p_k||^2
DEFAULT_BETA_MAX = 1e10  # Maximum allowed beta before restart (prevents runaway in non-SPD cases)
DEFAULT_RESIDUAL_REPLACEMENT_FREQ = 50  # m_replacement: Recompute true residual every N iterations
DEFAULT_DIVERGENCE_FACTOR = 1e10  # gamma_div: Declare divergence if ||r|| > gamma_div * ||b||

# FCG (Flexible Conjugate Gradient) Algorithm Parameters
# These control the truncated orthogonalization history for FCG variants
DEFAULT_M_MAX = 20  # Maximum history length for truncated orthogonalization in FCG
DEFAULT_FCG_HISTORY_LIMIT = 200  # Max search directions to retain for orthog/reorthog stability
FCG_ORTHOG_EPSILON = 1e-14  # Threshold for near-zero inner products in Gram-Schmidt

# Numerical Stability Thresholds (following SciPy conventions)
# These use np.finfo to get dtype-specific values, no magic numbers
# SciPy reference: scipy/sparse/linalg/_isolve/iterative.py (BICG, BICGSTAB, CGS)


def get_breakdown_tol(dtype: type[np.floating[Any]] = np.float64) -> float:
    """Get breakdown tolerance following SciPy BICG/BICGSTAB convention.

    SciPy uses eps**2 for vanishing denominators in iterative solvers like
    BICG, BICGSTAB, and CGS. For float64, this is approximately 4.93e-32.

    Why eps**2 (not eps):
        Machine epsilon (eps ≈ 2.22e-16 for float64) is the smallest value
        such that 1.0 + eps ≠ 1.0 in floating point. When dealing with:
        - Squared norms: ||r||^2 = sum(r_i^2)
        - Products of small quantities
        - Results of divisions

        The relevant precision is eps^2 because errors in squared quantities
        accumulate quadratically. A value below eps^2 has lost all significance
        in the mantissa when squared operations are involved.

    Physical Interpretation (float64):
        - eps ≈ 2.22e-16: Relative precision of single operations
        - eps^2 ≈ 4.93e-32: Relative precision of squared operations
        - For 3-element vector: each element must be > 1.28e-16 to avoid
          ||r||^2 < eps^2 breakdown

    Usage Context:
        Used to detect when denominators in CG iterations have lost numerical
        significance and division would amplify errors beyond machine precision:
        - Curvature d_k = p^T A p (quadratic form, involves products)
        - Residual norms ||r||^2 (already squared)
        - Step length denominators (ratios of these quantities)

    Note: Python's float is always 64-bit (double precision), so conversion
    from numpy.float64 to float is safe and doesn't lose precision.

    References:
        scipy/sparse/linalg/_isolve/iterative.py:
        rhotol = np.finfo(x.dtype.char).eps**2

        Notay (2000): "Flexible Conjugate Gradients"
        Section on numerical breakdown detection
    """
    # Cast to float for type checker; Python float is always 64-bit
    return float(np.finfo(dtype).eps ** 2)


def get_overflow_threshold(dtype: type[np.floating[Any]] = np.float64) -> float:
    """Get overflow threshold as 0.1 * dtype max.

    Uses a fraction of the maximum representable value to leave safety margin
    for intermediate calculations. For float64, this is approximately 1.8e307.

    Why 0.1 * max (not max):
        Maximum representable float64 ≈ 1.8e308, but using this as threshold
        would be too aggressive because:
        1. Intermediate calculations need headroom (multiplications, squares)
        2. Allows 10x safety margin for numerical operations
        3. Prevents false triggers on legitimate large-scale problems

    Physical Interpretation (float64):
        - finfo.max ≈ 1.8e308: Absolute maximum before overflow to inf
        - 0.1 * max ≈ 1.8e307: Practical threshold with safety margin
        - sqrt(0.1 * max) ≈ 1.3e154: Threshold for _stable_dot_product scaling
          (if max(|a|, |b|) > 1.3e154, products could overflow)

    Usage Context:
        Used in _stable_dot_product to decide when to apply preventive scaling.
        The actual threshold used is sqrt(overflow_threshold) ≈ 1.3e154, which
        ensures that individual products a_i * b_i won't overflow before we
        can apply balanced scaling.

    Why Fractional Max (SciPy Convention):
        SciPy and LAPACK libraries commonly use fractional thresholds (0.1 or 0.5)
        of maximum values to leave numerical headroom. This prevents edge cases
        where:
        - Overflow in (x * y) even though x, y < max individually
        - Loss of precision near maximum exponent
        - Underflow in rescaling operations

    Note: Python's float is always 64-bit (double precision), so conversion
    from numpy.float64 to float is safe and doesn't lose precision.

    References:
        LAPACK Working Note 176: "Design of Overflow and Underflow Detection"
        Discusses use of fractional thresholds for numerical stability
    """
    # Cast to float for type checker; Python float is always 64-bit
    return float(0.1 * np.finfo(dtype).max)


# =============================================================================
# Data Generation Defaults
# =============================================================================
DEFAULT_NUM_SAMPLES = 6000

# Dataset artifacts (split format)
DATASET_MANIFEST_FILENAME = "manifest.json"
RHS_ARRAY_FILENAME = "rhs.npy"
SOLUTIONS_ARRAY_FILENAME = "solutions.npy"
MATRIX_COO_DIRNAME = "matrix_coo"

# Strategy-specific iteration parameters
# These are now configured at the strategy level (not generation level):
# - DEFAULT_KRYLOV_ITERATIONS: Used by krylov strategy for Krylov subspace dimension
# - DEFAULT_RESIDUAL_TRACE_ITERS: Used by residual_traces, residuals, and gaussian_residuals strategies
DEFAULT_KRYLOV_ITERATIONS = 15
DEFAULT_RESIDUAL_TRACE_ITERS = 8

DEFAULT_RANDOM_SEED = 42
DEFAULT_NORMALIZE = "spectral"  # "none", "matrix", "rhs", "spectral", or "diagonal"
DEFAULT_SHUFFLE = True

# =============================================================================
# Comparison/Evaluation Defaults
# =============================================================================
DEFAULT_TEST_SAMPLE_INDEX = 0  # Which sample to extract for single-sample comparison tasks


# =============================================================================
# Eigenvector Selection
# =============================================================================
EIGENVECTOR_SELECT_SMALLEST = "smallest"
EIGENVECTOR_SELECT_LARGEST = "largest"
EIGENVECTOR_SELECT_RANDOM = "random"
EigenvectorSelectionMode = Literal["smallest", "largest", "random"]


# =============================================================================
# Matrix Norm Types (for dataset metadata)
# =============================================================================
# Convenience aliases for backward compatibility
MATRIX_NORM_SPECTRAL = MatrixNormType.SPECTRAL.value
MATRIX_NORM_FROBENIUS = MatrixNormType.FROBENIUS.value
MATRIX_NORM_NUCLEAR = MatrixNormType.NUCLEAR.value
MATRIX_NORM_ONE = MatrixNormType.ONE.value
MATRIX_NORM_INF = MatrixNormType.INF.value

# Default matrix norm for dataset metadata
DEFAULT_MATRIX_NORM_TYPE = MatrixNormType.SPECTRAL


# =============================================================================
# Config Section Names (for TOML parsing)
# =============================================================================
class ConfigSections:
    """TOML config section names."""

    SOURCE = "source"
    GENERATION = "generation"
    OUTPUT = "output"
    TRAINING = "TRAINING"
    MODEL = "MODEL"
    SESSION = "SESSION"
    DATASET = "DATASET"
    DATAMODULE = "DATAMODULE"
    MLFLOW = "MLFLOW"
    OPTUNA = "OPTUNA"
    EXTRAS = "EXTRAS"
    PATHS = "PATHS"


class ConfigKeys:
    """TOML config key names."""

    # Source section
    TYPE = "type"
    CASE_PATH = "case_path"
    MATRIX_FILE = "matrix_file"
    MATRIX_PATH = "matrix_path"
    RHS_PATH = "rhs_path"
    RHS_PATTERN = "rhs_pattern"
    SOLUTIONS_PATH = "solutions_path"

    # Generation section
    NUM_SAMPLES = "num_samples"
    NORMALIZE = "normalize"
    MIX = "mix"
    KRYLOV_ITERS = "krylov_iters"
    RESIDUAL_ITERS = "residual_iters"
    RHS_ARCHIVE_GLOB = "rhs_archive_glob"
    STRATEGY = "strategy"
    PERCENTAGE = "percentage"  # Deprecated: use SAMPLES instead
    SAMPLES = "samples"  # Number of samples to generate (0=skip, -1=all, >0=exact count)
    NAME = "name"
    RHS_GLOB = "rhs_glob"
    SOLUTIONS_GLOB = "solutions_glob"
    SEED = "seed"
    SHUFFLE = "shuffle"
    PROVIDE_RHS = "provide_rhs"

    # Output section
    PROCESSED_DIR = "processed_dir"

    # Expected types
    TYPE_RHS_ARCHIVE = "rhs_archive"
    TYPE_GENERATED = "generated"
    TYPE_SOLUTION_ARCHIVE = "solution_archive"

    # Normalization methods (for normalize parameter)
    NORM_NONE = "none"
    NORM_MATRIX = "matrix"
    NORM_RHS = "rhs"
    NORM_DIAGONAL = "diagonal"
    NORM_SPECTRAL = "spectral"


# =============================================================================
# File I/O
# =============================================================================
FILE_MODE_READ_BINARY = "rb"
FILE_MODE_WRITE_BINARY = "wb"
FILE_MODE_READ_TEXT = "r"
FILE_MODE_WRITE_TEXT = "w"
FILE_ENCODING_UTF8 = "utf-8"

# =============================================================================
# UI/Output Symbols
# =============================================================================
SYMBOL_SUCCESS = "✓"
SYMBOL_ERROR = "✗"
SYMBOL_ROCKET = "🚀"
SYMBOL_CHART = "📊"
SYMBOL_CHECKMARK = "✅"
SYMBOL_WARNING = "⚠️"

# =============================================================================
# Noise Analysis
# =============================================================================
NOISE_STRATEGY_NONE = "none"
DEFAULT_NOISE_LEVEL = 0.05
DEFAULT_NOISE_SEED = None

# =============================================================================
# Validation Thresholds
# =============================================================================
MIN_MATRIX_SIZE = 1
MIN_TOLERANCE = 1e-15
MAX_ITERATIONS_UPPER_LIMIT = 1_000_000

# =============================================================================
# Plot Settings
# =============================================================================
DEFAULT_PLOT_DPI = 150
DEFAULT_PLOT_FIGSIZE = (10, 6)
DEFAULT_PLOT_STYLE = "seaborn-v0_8-darkgrid"

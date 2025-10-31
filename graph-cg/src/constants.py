"""Constants and configuration values for graph-cg project.

This module centralizes magic values used across the codebase to improve
maintainability and avoid duplication.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Exit Codes
# =============================================================================
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_KEYBOARD_INTERRUPT = 130

# =============================================================================
# Default Paths
# =============================================================================
DEFAULT_PROJECT_ROOT = Path("/data/projects/graph-cg")
DEFAULT_PROCESSED_DATA_DIR = DEFAULT_PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "data" / "output"
DEFAULT_FIGURES_DIR = DEFAULT_PROJECT_ROOT / "data" / "figures"
DEFAULT_CHECKPOINTS_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"

# =============================================================================
# CG Solver Defaults
# =============================================================================
DEFAULT_CG_TOLERANCE = 1e-10
DEFAULT_CG_MAX_ITERATIONS = 1000
DEFAULT_CG_STOPPING_CRITERION = "tolerance"

# =============================================================================
# Data Generation Defaults
# =============================================================================
DEFAULT_NUM_SAMPLES = 6000
DEFAULT_KRYLOV_ITERATIONS = 15
DEFAULT_RESIDUAL_TRACE_ITERS = 8
DEFAULT_RANDOM_SEED = 42
DEFAULT_NORMALIZE = "spectral"  # "none", "matrix", "rhs", or "spectral"
DEFAULT_SHUFFLE = True

# =============================================================================
# Comparison/Evaluation Defaults
# =============================================================================
DEFAULT_TEST_SAMPLE_INDEX = 0  # Which sample to extract for single-sample comparison tasks

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
    PERCENTAGE = "percentage"
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

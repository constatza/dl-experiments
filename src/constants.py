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
DEFAULT_MLRUNS_DIR = DEFAULT_PROJECT_ROOT / "data" / "mlruns"
DEFAULT_MLARTIFACTS_DIR = DEFAULT_PROJECT_ROOT / "data" / "mlartifacts"

# Legacy compatibility - these match old common.py constants
DEFAULT_PROCESSED_DIR = str(DEFAULT_PROCESSED_DATA_DIR)
DEFAULT_RESULTS_DIR = str(DEFAULT_OUTPUT_DIR)
DEFAULT_FIGURES_DIR_STR = str(DEFAULT_FIGURES_DIR)

# Default configuration file paths (relative to graph-cg root)
DEFAULT_MODEL_CONFIG = "configs/ffnn.toml"
DEFAULT_DATA_CONFIG = "data-configs/collect-504-solutions.toml"
DEFAULT_EXPERIMENTS_CONFIG = "configs/experiments.toml"

# =============================================================================
# CG Solver Defaults
# =============================================================================
DEFAULT_ATOL = 1e-10  # Absolute tolerance for CG solves
DEFAULT_RTOL = 1e-6  # Relative tolerance for CG solves
DEFAULT_CG_MAX_ITERATIONS = 1000
DEFAULT_CG_STOPPING_CRITERION = "tolerance"

# Reorthogonalization numerical tolerances
REORTHOG_ZERO_NORM_TOL = 1e-14  # Threshold for considering a norm as zero
REORTHOG_MAX_COEFF = 1e10  # Maximum safe reorthogonalization coefficient
REORTHOG_STRICT_THRESHOLD = 0.01  # Strict selective reorthogonalization threshold
REORTHOG_DEFAULT_WINDOW = 10  # Default window size for partial reorthogonalization

# CG Algorithm Safety Parameters
# These parameters control restart/breakdown behavior for flexible PCG with non-SPD preconditioners
DEFAULT_CURVATURE_EPSILON = (
    1e-14  # eps_curv: Restart if curvature d_k < eps_curv * ||p_k||^2
)
DEFAULT_BETA_MAX = (
    1e10  # Maximum allowed beta before restart (prevents runaway in non-SPD cases)
)
DEFAULT_RESIDUAL_REPLACEMENT_FREQ = (
    50  # m_replacement: Recompute true residual every N iterations
)
DEFAULT_DIVERGENCE_FACTOR = (
    1e10  # gamma_div: Declare divergence if ||r|| > gamma_div * ||b||
)
DEFAULT_ATOL = (
    1e-14  # Absolute tolerance for convergence (in addition to relative tolerance)
)

# FCG (Flexible Conjugate Gradient) Algorithm Parameters
# These control the truncated orthogonalization history for FCG variants
DEFAULT_M_MAX = 10  # Maximum history length for truncated orthogonalization in FCG
DEFAULT_FCG_HISTORY_LIMIT = (
    200  # Max search directions to retain for orthog/reorthog stability
)
FCG_ORTHOG_EPSILON = 1e-14  # Threshold for near-zero inner products in Gram-Schmidt

# =============================================================================
# Data Generation Defaults
# =============================================================================
DEFAULT_NUM_SAMPLES = 6000
DEFAULT_KRYLOV_ITERATIONS = 15
DEFAULT_RESIDUAL_TRACE_ITERS = 8
DEFAULT_RANDOM_SEED = 42
DEFAULT_NORMALIZE = "spectral"  # "none", "matrix", "rhs", "spectral", or "diagonal"
DEFAULT_SHUFFLE = True

# =============================================================================
# Comparison/Evaluation Defaults
# =============================================================================
DEFAULT_TEST_SAMPLE_INDEX = (
    0  # Which sample to extract for single-sample comparison tasks
)


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
    SAMPLES = (
        "samples"  # Number of samples to generate (0=skip, -1=all, >0=exact count)
    )
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

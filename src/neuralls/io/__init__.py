"""I/O utilities package for neuralls.

This package provides a clean, organized API for all file and data I/O operations.
All modules are internal but commonly-used functions are re-exported here.

Public API:
-----------

Filesystem Operations (from io.filesystem):
    - ensure_dir: Create directory if it doesn't exist
    - sanitize_identifier: Convert string to filesystem-safe slug
    - derive_model_identifier: Extract model identifier from config
    - parse_data_dir_name: Parse data directory name to extract parameters
    - copy_file: Copy file from source to destination
    - get_file_size: Get file size in bytes
    - file_exists: Check if file exists
    - get_relative_path: Get relative path from base
    - format_file_size: Format size in human-readable format
    - save_text_file: Save text to file
    - load_text_file: Load text from file

Array/System Loading (from io.arrays):
    - load_system_data: Load matrix and RHS data
    - save_training_data: Save training arrays
    - load_case_data: Load complete dataset from directory
    - list_available_cases: List all available data cases
    - derive_checkpoint_path: Derive checkpoint path from config

Dataset Operations (from io.datasets):
    - load_dataset: Load dataset from .npz file
    - load_numpy_array: Load numpy array from file
    - save_numpy_array: Save numpy array to file
    - has_comparison_split: Check if dataset has comparison split
    - list_available_variants: List available dataset variants
    - DatasetFileTracker: Track file operations
    - file_tracker: Global file tracker instance
    - track_files: Get file tracker

Checkpoint Operations (from io.checkpoints):
    - get_latest_checkpoint: Find most recent checkpoint
    - clean_directory: Remove old files from directory

Base I/O (from io.base):
    - load_npz_entry: Load single entry from NPZ file
    - load_matrix: Load matrix from file

Comparison/Solver I/O (from io.comparison):
    - resolve_system_paths: Resolve matrix/RHS paths
    - load_system_arrays: Load system arrays from files
    - load_npz_entry: Load NPZ entry with warnings

Configuration Loading (from io.toml_loader):
    - load_model_config: Load and validate model config
    - load_data_config: Load and validate data config
    - load_solver_config: Load and validate solver config
    - load_raw_toml: Load TOML without validation
"""

# Filesystem operations
from neuralls.io.filesystem import (
    ensure_dir,
    sanitize_identifier,
    derive_model_identifier,
    parse_data_dir_name,
    copy_file,
    get_file_size,
    file_exists,
    get_relative_path,
    format_file_size,
    save_text_file,
    load_text_file,
)

# Array/system loading
from neuralls.io.arrays import (
    load_system_data,
    save_training_data,
    load_case_data,
    list_available_cases,
    derive_checkpoint_path,
)

# Dataset operations
from neuralls.io.datasets import (
    load_dataset,
    load_numpy_array,
    save_numpy_array,
    has_comparison_split,
    list_available_variants,
    DatasetFileTracker,
    file_tracker,
    track_files,
)

# Checkpoint operations
from neuralls.io.checkpoints import (
    get_latest_checkpoint,
    clean_directory,
)

# Base I/O (for low-level operations)
from neuralls.io.base import (
    load_npz_entry,
    load_matrix,
)

# Comparison/solver I/O
from neuralls.io.comparison import (
    resolve_system_paths,
    load_system_arrays,
)

# Configuration loading
from neuralls.io.toml_loader import (
    load_model_config,
    load_data_config,
    load_solver_config,
    load_raw_toml,
)

__all__ = [
    # Filesystem
    "ensure_dir",
    "sanitize_identifier",
    "derive_model_identifier",
    "parse_data_dir_name",
    "copy_file",
    "get_file_size",
    "file_exists",
    "get_relative_path",
    "format_file_size",
    "save_text_file",
    "load_text_file",
    # Arrays
    "load_system_data",
    "save_training_data",
    "load_case_data",
    "list_available_cases",
    "derive_checkpoint_path",
    # Datasets
    "load_dataset",
    "load_numpy_array",
    "save_numpy_array",
    "has_comparison_split",
    "list_available_variants",
    "DatasetFileTracker",
    "file_tracker",
    "track_files",
    # Checkpoints
    "get_latest_checkpoint",
    "clean_directory",
    # Base
    "load_npz_entry",
    "load_matrix",
    # Comparison
    "resolve_system_paths",
    "load_system_arrays",
    # Config loading
    "load_model_config",
    "load_data_config",
    "load_solver_config",
    "load_raw_toml",
]

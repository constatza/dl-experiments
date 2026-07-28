# Solver Export Utilities

Modular package for exporting solver results and configurations with NPZ support for efficient storage.

## Features

- **Three-level API**: Convenience, full control, and granular functions
- **Multiple formats**: NPZ (binary, fast), TXT (human-readable), or BOTH
- **NPZ benefits**: 3-10x smaller files, faster I/O, cleaner directory structure
- **Backward compatible**: Existing code works unchanged
- **Single responsibility**: Each module has one clear purpose

## Quick Start

### Level 1: Convenience API (Recommended)

```python
from pathlib import Path
import numpy as np
from neuralls.domain.solver.utils.export import quick_export, ArrayFormat

# Export everything in one call
quick_export(
    result=solver_result,
    config=solver_config,
    A=A,
    b=b,
    x_sol=x_sol,
    output_dir=Path("/tmp/output"),
    name_stem="solver_run",
    format=ArrayFormat.NPZ,  # Fast, small (default for new code)
)
```

### Level 2: Full Control API (Backward Compatible)

```python
from neuralls.domain.solver.utils.export import save_solver_data, ArrayFormat

save_solver_data(
    output_dir=Path("/tmp/output"),
    name_stem="solver_run",
    A=A,
    b=b,
    x0=x0,
    x_sol=x_sol,
    result=result,
    solver_config=config,
    data_config={"matrix_type": "tridiagonal", "size": 100},
    experiment_config={"project": "benchmarks"},
    format=ArrayFormat.NPZ,  # Explicitly choose format
)
```

### Level 3: Granular API (Advanced)

```python
from neuralls.domain.solver.utils.export import (
    save_system_arrays,
    save_iteration_history,
    save_full_config,
    save_solver_result,
    ArrayFormat,
)

# Export specific components
save_system_arrays(output_dir, A, b, x_sol, x0=x0, format=ArrayFormat.NPZ)
save_iteration_history(output_dir, history, format=ArrayFormat.NPZ)
save_full_config(output_dir / "config.toml", config)
save_solver_result(output_dir / "results.toml", result)
```

## Format Comparison

### NPZ Format (Recommended)
- **Size**: 3-10x smaller than TXT
- **Speed**: Faster load/save
- **Structure**: Clean (2 binary files + 2 TOML files)
- **Use case**: Production, large datasets, repeated I/O

```
solver_output/
├── system.npz         # A, b, x0, x_sol
├── history.npz        # residual_norms, residuals, solutions
├── config.toml        # Solver configuration
└── results.toml       # Outcomes
```

### TXT Format (Backward Compatible)
- **Size**: Larger (human-readable overhead)
- **Speed**: Slower load/save
- **Structure**: Many files (11+ text files)
- **Use case**: Debugging, external tools, human inspection

```
solver_output/
├── A.txt
├── b.txt
├── x0.txt
├── x_sol.txt
├── residual_norms.txt
├── residuals.txt       # If trace_mode=FULL
├── solutions.txt       # If trace_mode=FULL
├── directions.txt      # If trace_mode=FULL
├── config.toml
└── results.toml
```

### BOTH Format (Debugging)
- Exports both NPZ and TXT for maximum compatibility
- Useful during migration or when debugging NPZ issues

## Real-World Performance

Test case: 100x100 tridiagonal system with full trace (22 iterations)

| Format | Size  | Files | Load Time |
|--------|-------|-------|-----------|
| TXT    | 388K  | 11    | ~10ms     |
| NPZ    | 128K  | 4     | ~3ms      |
| Ratio  | 3.0x  | 2.75x | 3.3x      |

## Migration Guide

### Existing Code (TXT Default)
```python
# No changes needed - backward compatible
save_solver_data(
    output_dir=results_dir,
    name_stem="test_run",
    A=A,
    b=b,
    x0=x0,
    x_sol=x_sol,
    result=result,
    solver_config=config,
)
# Still exports TXT format by default
```

### New Code (NPZ Recommended)
```python
# Add format parameter
save_solver_data(
    output_dir=results_dir,
    name_stem="test_run",
    A=A,
    b=b,
    x0=x0,
    x_sol=x_sol,
    result=result,
    solver_config=config,
    format=ArrayFormat.NPZ,  # 3-10x smaller
)
```

## Loading NPZ Data

```python
import numpy as np

# Load system arrays
with np.load(output_dir / "system.npz") as data:
    A = data["A"]
    b = data["b"]
    x0 = data["x0"]
    x_sol = data["x_sol"]

# Load history
with np.load(output_dir / "history.npz") as data:
    residual_norms = data["residual_norms"]
    if "residuals" in data:  # FULL mode only
        residuals = data["residuals"]
        solutions = data["solutions"]
```

## Architecture

### Package Structure
```
export/
├── __init__.py         # Public API (3 levels)
├── arrays.py           # Numerical data export
├── configs.py          # TOML configuration export
├── results.py          # TOML results export
├── serialization.py    # Type conversion for TOML
├── formats.py          # ArrayFormat enum
└── README.md           # This file
```

### Design Principles
- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Add formats without touching existing code
- **DRY**: Reusable functions, no duplication
- **SOLID**: Follows project architecture standards

## Testing

```bash
# Comprehensive unit tests
uv run pytest tests/solver/test_export_package.py -v

# Integration test (backward compatibility)
uv run pytest tests/benchmarks/exactness/test_trace_export.py -v

# All solver tests
uv run pytest tests/solver/ -x
```

## Benefits Summary

1. **Efficiency**: 3-10x smaller files, faster I/O
2. **Organization**: Cleaner directory structure (4 files vs 11+)
3. **Modularity**: Single-responsibility modules, easy to extend
4. **Compatibility**: Backward compatible, smooth migration
5. **Standards**: Follows scientific Python conventions (NPZ is standard)
6. **Convenience**: `quick_export()` for common case (one-liner)
7. **Flexibility**: Three API levels for different needs

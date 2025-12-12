# Plan: Migrate to Pydantic Strategy Configuration Dataclasses

## Problem Summary

Currently, strategy configuration uses untyped `dict[str, Any]` with several issues:

1. **No type safety** - Parameters extracted with `.get()` and manual type conversions
2. **Silent failures** - Invalid parameters ignored, wrong types cause runtime errors
3. **Hidden dependencies** - Hard to discover what parameters each strategy accepts
4. **No validation** - Can't validate configs before passing to strategies
5. **Code duplication** - Each strategy reimplements the same `.get()` pattern
6. **Generation-level pollution** - `krylov_iters` and `residual_iters` propagated to ALL strategies even though only specific strategies use them

## User Decisions

✅ **Hard cutover migration** - Update all configs + code in one atomic commit
✅ **Apply to both parameters** - Migrate `krylov_iters` and `residual_iters` to strategy-level
✅ **Use Pydantic dataclasses** - Define typed config classes for each strategy with Literals for accepted kwargs
✅ **Add validation** - Automatic validation via Pydantic, fail fast on invalid configs

## Investigation Findings

### Current Configuration Pattern

**`residual_iters` configs:**
All 5 active data configs define `residual_iters` at generation level:
- `data-configs/collect-504-solutions.toml` - `residual_iters = 50`
- `data-configs/collect-2040-solutions.toml` - `residual_iters = 10`
- `data-configs/test-solutions.toml` - `residual_iters = 1`
- `data-configs/test-eigenvector-rhs.toml` - `residual_iters = 1`
- `data-configs/test-eigenvector-solution.toml` - `residual_iters = 1`

**`krylov_iters` configs:**
- **ZERO data configs** currently use `krylov_iters` (relies on hardcoded default of 15)

### Parameter-to-Strategy Mapping

| Parameter | Used By | Default | Config File |
|-----------|---------|---------|-------------|
| `residual_iters` | `cg_residual`, `cg_residual_error` | 8 | `constants.py:82` |
| `krylov_iters` | `krylov` | 15 | `constants.py:83` |

### Strategies That Ignore Both Parameters
6 strategies don't use either parameter:
- `random_normal`, `solution_archive`, `rhs_archive`
- `eigenvector_forward`, `eigenvector_inverse`
- (Note: `krylov` ignores `residual_iters`, residual strategies ignore `krylov_iters`)

### Code Propagation Flow
1. **Config reading** (`src/generation/config_processing.py:313-315`) - Reads from `[generation]` level
2. **Strategy override injection** (`config_processing.py:384-387`) - Applies to all strategies as default
3. **Orchestration** (`src/generation/orchestration.py:147-148`) - Passes via `cfg.setdefault()`
4. **Strategy execution** - Each strategy reads from `cfg` dict with its own fallback

## Proposed Architecture

### Strategy Configuration Dataclasses

Each strategy will have its own **Pydantic dataclass** defining accepted parameters:

```python
from pydantic.dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True, config={"extra": "forbid"})
class ResidualErrorConfig:
    samples: int
    residual_iters: int
    seed: int = 42
    archive_solutions: np.ndarray | None = None
    archive_rhs: np.ndarray | None = None

@dataclass(frozen=True, config={"extra": "forbid"})
class KrylovConfig:
    samples: int
    krylov_iters: int = 15
    seed: int = 42

@dataclass(frozen=True, config={"extra": "forbid"})
class EigenvectorForwardConfig:
    samples: int
    which: Literal["smallest", "largest", "both"] = "smallest"
    num_eigenvectors: int | None = None
    include_eigenvectors: bool = False
    seed: int = 42
```

Benefits:
- **`extra="forbid"`** → Validation error if unknown parameters passed
- **Type annotations** → IDE autocomplete and mypy checks
- **Defaults** → Centralized, documented in type definition
- **Frozen** → Immutable config objects

### Updated Strategy Interface

```python
class IDataGenerationStrategy(Protocol):
    name: str
    ConfigType: type  # New: Each strategy declares its config type

    def requires_rhs(self) -> bool: ...

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: Any,  # Will be strategy's specific ConfigType at runtime
    ) -> GeneratedSamples: ...
```

## Implementation Plan

### Phase 0: Add Pydantic Dependency

**File: `pyproject.toml`**

Add pydantic to dependencies:
```toml
dependencies = [
    "pydantic>=2.0",
    # ... existing dependencies
]
```

### Phase 1: Create Strategy Config Module

**File: `src/generation/strategy_configs.py` (NEW)**

Define all strategy configuration dataclasses in one place with all 8 strategy configs.

### Phase 2: Update Configuration Files (5 files)

Migrate `residual_iters` from generation-level to strategy-level in all data configs.

### Phase 3: Update Config Processing Logic

Remove generation-level parameter extraction and propagation.

### Phase 4: Update Orchestration Layer

Remove parameter injection, rely on strategy configs.

### Phase 5: Update Strategy Implementations (8 files)

Add `ConfigType` class variable and convert to typed config in `generate()`.

### Phase 6: Update Metadata Extraction

Read from strategy options only, remove generation-level check.

### Phase 7: Update Tests (2 files)

Use `strategy_overrides`, test Pydantic validation.

### Phase 8: Update Constants

Add clarifying comments for strategy-level parameters.

## Benefits After Migration

1. **Type safety** - Pydantic validates types automatically, no manual conversions
2. **Fail fast** - Invalid configs caught at validation time, not during execution
3. **No silent failures** - `extra="forbid"` rejects unknown parameters immediately
4. **Self-documenting** - Config dataclasses serve as API documentation
5. **IDE support** - Autocomplete and type checking in strategy implementations
6. **Centralized defaults** - All defaults in config dataclass, not scattered
7. **Simpler code** - No `.get()` boilerplate, just clean attribute access
8. **Better errors** - Pydantic provides detailed validation error messages

## Critical Files to Modify

### New Files (1)
1. `src/generation/strategy_configs.py` - All Pydantic config dataclasses

### Config Files (5)
1. `data-configs/collect-504-solutions.toml`
2. `data-configs/collect-2040-solutions.toml`
3. `data-configs/test-solutions.toml`
4. `data-configs/test-eigenvector-rhs.toml`
5. `data-configs/test-eigenvector-solution.toml`

### Core Logic (3)
1. `src/generation/config_processing.py`
2. `src/generation/orchestration.py`
3. `src/metadata_repository.py`

### Strategy Implementations (8 files)
1. `src/generation/strategies/residual_error.py`
2. `src/generation/strategies/residual_traces.py`
3. `src/generation/strategies/krylov.py`
4. `src/generation/strategies/eigenvector.py` (2 classes)
5. `src/generation/strategies/random_normal.py`
6. `src/generation/strategies/rhs_archive.py`
7. `src/generation/strategies/solution_archive.py`

### Tests (2)
1. `tests/generation/test_data_generation.py`
2. `tests/generation/test_generation_plan.py`

### Dependencies (1)
1. `pyproject.toml`

### Documentation (1)
1. `src/constants.py`

## Implementation Order

1. Add dependency - `pydantic>=2.0` to `pyproject.toml`
2. Create config module - `src/generation/strategy_configs.py` with all dataclasses
3. Update strategies - Add `ConfigType`, convert to typed config in `generate()`
4. Update orchestration - Remove param injection, add Pydantic error handling
5. Update config processing - Remove generation-level param extraction
6. Update metadata extraction - Read from strategy options only
7. Update config files - Move `residual_iters` to strategy level
8. Update tests - Use `strategy_overrides`, test Pydantic validation
9. Update constants - Add clarifying comments
10. Verify - Run full test suite, check mypy passes

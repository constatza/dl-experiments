# Generation Module

The generation package turns matrices and optional archives into processed
training datasets. Most users should interact with it through `process-data`
first and only then drop into the package internals.

## User Path

### Basic

Build one dataset from one config:

```bash
uv run process-data /path/to/dataset.toml \
  --case-config /path/to/case.toml
```

### Intermediate

Build every dataset declared in one case config:

```bash
uv run generate-all /path/to/case.toml
```

### Advanced

Import generation internals when you need to extend strategy behavior:

```python
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation import generate_mixture, run_generation
```

## Strategy Progression

Start with the simplest family that matches the model target.

| Level | Strategy | Output |
| --- | --- | --- |
| Basic | `normal` | `(b, x)` |
| Basic | `krylov` | `(b, x)` |
| Intermediate | `rhs_archive` | `(b, x)` |
| Intermediate | `solution_archive` | `(b, x)` |
| Advanced | `residual_traces` | `(r_k, x_k)` |
| Advanced | `residuals` | `(r_k, x_true - x_k)` |
| Advanced | `gaussian_residuals` | `(r_k, x_true - x_k)` |
| Advanced | `search_directions` | trace pairs for direction learning |

## Residual Families

The repo now uses explicit residual strategy names:

- `residual_traces` for residual-to-iterate pairs
- `residuals` for residual-to-error pairs
- `gaussian_residuals` for residual-to-error pairs without archive solutions

These names are the supported user-facing identifiers in dataset configs and
tests.

## Package Map

- `orchestration.py`: mixed-strategy payload assembly
- `payloads.py`: pure sparse dataset payload types and accumulation helpers
- `runner.py`: strategy registry and dispatch
- `providers.py`: archive or synthetic sample providers
- `transforms.py`: pure transforms such as `A @ x`
- `trace_utils.py`: trace trimming, offsets, and indexing helpers
- `strategies/`: concrete generation implementations

Config-driven generation entrypoints now live in
`neuralls.composition.generation.processing`, which wires the generation domain
to default tracing solvers through `neuralls.domain.generation.ports`.

## Extension Rules

When adding a strategy:

1. add a config model in `strategy_configs.py`
2. implement the strategy under `strategies/`
3. register it through `@register_strategy`
4. document the public strategy name and its required fields in user-facing docs
5. add generation and config tests

## Where It Connects

Generation stays inside the domain layer and depends only on:

- solver tracing for CG-derived strategies
- normalization trace containers
- shared constants and math helpers

Persistent dataset writing happens in
`neuralls.composition.generation.dataset_builder` via the platform storage
adapter, so the generation domain no longer writes files directly.

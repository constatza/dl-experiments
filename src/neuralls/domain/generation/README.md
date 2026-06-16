# Generation Module

The generation package turns matrices and optional archives into processed
training datasets. Most users should interact with it through `process-data`
first and only then drop into the package internals.

## Handling Arbitrarily-Named Matrix Files

When a colleague provides matrices with parameter-encoded filenames (no sequential
integer ID), set `enumerate_by` in the `[source]` block of the dataset TOML:

```toml
[source]
matrix_path = "/data/matrices/E1_*_E2_*.txt"
enumerate_by = "name"   # lexicographic — deterministic across runs
```

`enumerate_by` sorts the glob results by the chosen criterion and assigns sequential
IDs 0, 1, 2, …  No renaming or modification of the source files is required.

| Value | Sort criterion | When to use |
| --- | --- | --- |
| `"name"` | Lexicographic filename | **Default choice** — fully reproducible |
| `"ctime"` | File creation timestamp | Files arrived in a known order |
| `"mtime"` | Last-modified timestamp | Files were last touched in a known order |

`enumerate_by` and `sample_id_regex` are mutually exclusive; specifying both raises
a validation error.

Config-driven generation and the public composition entrypoint
`neuralls.composition.generation.dataset_builder.build_dataset(...)` both honor
`enumerate_by` and pass it through to the glob source streams.

For multi-matrix sources, dataset-level `counts` and `mix/total` budgets are
global across the matrix family rather than applied once per matrix. Set
`replacement = true` under `[generation]` only when you want supported random
strategies to reuse matrix bindings explicitly during that global allocation.
Archive-backed and deterministic strategies remain strict and do not honor
matrix replacement.

**Python API:**

```python
from neuralls.domain.generation.source_streams import GlobMatrixStream, EnumerateBy

stream = GlobMatrixStream("data/E1_*_E2_*.txt", enumerate_by=EnumerateBy.NAME)
# stream.sample_ids → (0, 1, 2, …)
```

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

- `orchestration.py`: mixed-strategy payload assembly; `build_dataset_payload()` requires an
  injected `ZarrAccumulatorPort` — the domain never creates storage objects directly
- `payloads.py`: pure DTO — `GeneratedDatasetPayload` only; no accumulation helpers
- `ports.py`: `ZarrAccumulatorPort`, `DatasetWriterPort`, and `TracingSolverPort` protocol
  definitions consumed by the composition layer
- `runner.py`: strategy registry and dispatch
- `providers.py`: archive or synthetic sample providers
- `transforms.py`: pure transforms such as `A @ x`
- `trace_utils.py`: trace trimming, offsets, and indexing helpers
- `strategies/`: concrete generation implementations

Config-driven generation entrypoints now live in
`neuralls.composition.generation.processing`, which wires the generation domain
to default tracing solvers through `neuralls.domain.generation.ports`.

## Dataset Storage

Zarr is the default on-disk format for new datasets (`zarr_coo`). Matrix samples
stream directly to disk via `ZarrSparseAccumulator` during generation — no
in-memory COO buffer is accumulated, so arbitrarily large datasets can be built
without OOM risk.

| Component | Location | Role |
| --- | --- | --- |
| `ZarrAccumulatorPort` | `domain/generation/ports.py` | Protocol — append samples, `finalize() -> Path` |
| `ZarrSparseAccumulator` | `platform/storage/datasets.py` | Concrete implementation; streams via `ZarrPackWriter` |
| `ZarrDatasetWriter` | `platform/storage/datasets.py` | Writes `rhs.npy`, `solutions.npy`, and the zarr pack |
| `SparseDatasetWriter` | `platform/storage/datasets.py` | Legacy writer — re-materialises zarr pack as `npy_coo` |

The `[output] dataset_format` field in dataset TOML selects the storage format
(`"zarr_coo"` default, or `"npy_coo"` for backward compatibility).
`open_sparse_pack()` from dlkit auto-detects the format at load time, so
existing `npy_coo` datasets continue to work without conversion.

## Normalization Metadata

Generation writes normalized matrix samples, RHS vectors, and solutions as one
consistent system. The dataset manifest stores one dataset-level normalization
block only.

For single-matrix datasets, that manifest block may include reversible scale
metadata such as `spectral_radius_bound` and `dimension_scale`.

For multi-matrix datasets, each matrix is still normalized independently before
storage. If those bindings do not share one exact scale payload, the manifest
intentionally leaves `normalization.scale` empty instead of pretending there is
one dataset-wide reversible scale.

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

`build_dataset()` in `neuralls.composition.generation.dataset_builder` creates a
`ZarrSparseAccumulator` and injects it into `build_dataset_payload()`. The
generation domain receives the accumulator via `ZarrAccumulatorPort` and never
imports or instantiates storage objects directly. All file I/O is confined to the
composition and platform layers.

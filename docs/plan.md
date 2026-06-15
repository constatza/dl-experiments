# Remaining Work Plan

## 1. Per-Matrix Solution Binding via `solution_ids`

### Problem

`solution_archive`, `scaled_solutions`, `validated_archive`, and `residual_traces` all load
solutions via a single `solutions_glob` config key — a **pool** shared across every binding.
For multi-matrix datasets this is incorrect: `matrix_i` may receive solutions computed for
`matrix_j`'s system. `rhs_ids` in `bind_sources()` already solves this correctly for RHS
vectors, but no equivalent exists for solutions.

`solution_sample_id` was removed as dead code, but the dead code existed because the feature
was never wired end-to-end — not because the requirement is absent.

### Design

Mirror the `rhs_path` / `rhs_ids` pattern exactly:

```
SourceConfig.solution_path  (str | None)
    ↓
_open_streams(…, solution_path)
    ↓ opens VectorSampleStream, passes solution_ids to bind_sources()
bind_sources(…, solution_ids)  → SystemBinding.solution_sample_id populated
    ↓
_process_binding(…, solution_stream)
    ↓ reads binding.solution_sample_id, loads vector
generate_mixture(…, single_solution=loaded_vec)
    ↓
strategy receives pre-loaded solution; skips its own glob loading
```

### Files to modify

| File | Change |
|---|---|
| `src/neuralls/domain/generation/source_streams.py` | Restore `SystemBinding.solution_sample_id: int \| None = None`; restore `solution_ids` param in `bind_sources()` and the solution-set matching logic |
| `src/neuralls/domain/generation/orchestration.py` | Add `solution_path: str \| None = None` to `build_dataset_payload()`; pass to `_open_streams()`; unpack 5-tuple; add `solution_stream` arg to `_process_binding()`; load solution vector from `binding.solution_sample_id` and pass as `single_solution` to `generate_mixture()` |
| `src/neuralls/domain/generation/orchestration.py` | `_process_binding` signature: add `solution_stream: VectorSampleStream \| None = None` |
| `src/neuralls/platform/config/models/data_models.py` | Add `solution_path: str \| None = None` to `SourceConfig`; add glob-expansion validator like `_expand_parameters_paths` |
| `src/neuralls/composition/generation/processing.py` | Thread `solution_path` from config through `_execute_plan`, `_execute_solution_archive`, `_execute_synthetic_generation` to `build_dataset()` |
| `src/neuralls/domain/generation/interfaces.py` | Add `single_solution: np.ndarray \| None = None` to `SingleRhsGenerationStrategy.generate()` and `MatrixGenerationStrategy.generate()` |
| `src/neuralls/domain/generation/strategies/solution_archive.py` | Accept `single_solution` kwarg; if provided, skip glob loading and use it directly |
| `src/neuralls/domain/generation/strategies/scaled_solutions.py` | Same |
| `src/neuralls/domain/generation/strategies/residual_traces.py` | Same (optional — these strategies derive solutions by solving, so `single_solution` would be a hint or overridden starting point; skip if not needed) |

### Key contracts

```python
# bind_sources
def bind_sources(
    matrix_ids: tuple[int, ...],
    rhs_ids: tuple[int, ...] | None = None,
    solution_ids: tuple[int, ...] | None = None,                  # RESTORED
    parameters_ids_list: tuple[tuple[int, ...], ...] = (),
) -> list[SystemBinding]: ...

# _process_binding receives the stream, loads vector at binding time
solution_stream: VectorSampleStream | None
if solution_stream and binding.solution_sample_id is not None:
    single_solution = solution_stream.load_sample(binding.solution_sample_id).vector
else:
    single_solution = None

# generate_mixture passes it through to the active strategy
generate_mixture(matrix_norm, …, single_solution=single_solution)
```

### Tests to add

- `tests/generation/test_source_streams.py`: `test_bind_sources_pairs_solution_ids_per_matrix`, `test_bind_sources_solution_mismatch_raises`
- `tests/generation/test_orchestration_solution_binding.py`: end-to-end with a mock strategy that asserts `single_solution` equals the expected loaded vector

---

## 2. DeepONet Query Provider Implementation

Implement `DeepONetQueryProvider.provide()` in
`src/neuralls/composition/experiments/feature_providers.py`:

```python
def provide(self, arrays: TrainingArrays) -> DataEntry:
    if not arrays.parameters_zarr:
        raise ValueError("DeepONet 'query' requires parameters_zarr[0].")
    params = np.asarray(zarr.open_array(str(arrays.parameters_zarr[0])))  # (N, param_dim)
    query = params[:, np.newaxis, :]                                        # (N, 1, param_dim)
    return ValueEntry(
        name="query",
        value=query,
        field_role=FieldRole.TARGET_COORDINATES,
        geometry_kind=GeometryKind.TABULAR,
    )
```

Open questions to resolve before implementing:
- Supervised target reshape: DLKit `BranchTrunkSpec` expects `(N, 1, n)` — confirm whether
  `solutions.zarr` needs reshaping or if a transform handles it
- `primary_input_name = "u"` (branch input) vs `"x"`: requires a separate
  `RuntimeDatasetContract` variant or a convention in the model TOML

---

## 3. SOLID Quick Wins

### 3a. OCP — Strategy property registry (`orchestration.py`)

Replace the two hardcoded `if strategy_name in {...}` blocks in
`_strategy_uses_finite_source()` and `_validate_replacement_support()` with a
module-level registry dict.

```python
@dataclass(frozen=True)
class _StrategyProperties:
    uses_finite_source: bool
    supports_replacement: bool

_STRATEGY_PROPERTIES: dict[str, _StrategyProperties] = {
    "solution_archive":  _StrategyProperties(uses_finite_source=True,  supports_replacement=False),
    "rhs_archive":       _StrategyProperties(uses_finite_source=True,  supports_replacement=False),
    "scaled_solutions":  _StrategyProperties(uses_finite_source=True,  supports_replacement=False),
    "validated_archive": _StrategyProperties(uses_finite_source=True,  supports_replacement=False),
    "residual_traces":   _StrategyProperties(uses_finite_source=True,  supports_replacement=False),
    "residuals":         _StrategyProperties(uses_finite_source=True,  supports_replacement=False),
    "gaussian_residuals":_StrategyProperties(uses_finite_source=True,  supports_replacement=False),
    "search_directions": _StrategyProperties(uses_finite_source=True,  supports_replacement=True),
}
```

`_strategy_uses_finite_source` and `strategy_supports_matrix_replacement` both do a single
`_STRATEGY_PROPERTIES.get(name)` lookup; fall back to existing heuristic for unknown names.
New strategies register here; no other function needs touching.

### 3b. SRP — Split `build_dataset_payload()` (`orchestration.py`)

Extract three private helpers, each with a single reason to change:

| Helper | Responsibility |
|---|---|
| `_prepare_generation_context(…) -> tuple[MatrixSampleStream, VectorSampleStream \| None, VectorSampleStream \| None, list[VectorSampleStream], list[SystemBinding], list[dict[str, int]]]` | Open streams, bind sources, resolve per-binding strategy counts |
| `_accumulate_bindings(…) -> tuple[list[np.ndarray], list[np.ndarray], list[list[np.ndarray]], list[float], list[float], list[ScaleMetadata \| None], int]` | Main loop: call `_process_binding`, feed accumulator, collect param blocks |
| `_finalize_payload(…) -> GeneratedDatasetPayload` | Stack arrays, resolve scale, call `_build_dataset_payload` |

`build_dataset_payload()` becomes an 8-line orchestrator calling these three in sequence.

---

## 4. Docs Update

After completing items 1–3, update the module-level `.md` files for every affected package:

| File | What to update |
|---|---|
| `src/neuralls/domain/generation/README.md` (or equivalent) | Document `solution_path` / `solution_ids` binding alongside `rhs_path`; update `bind_sources` contract table |
| `src/neuralls/composition/experiments/README.md` | Add `DeepONetQueryProvider` implementation notes and open questions resolved |
| `CLAUDE.md` module map (root) | No change needed — architecture unchanged |
| `docs/coupling_graph.dot` | Add edge if any new cross-package dependency introduced by solution binding |

---

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
uv run ty check src/
```

End-to-end FiLM training smoke test:
```bash
uv run neuralls train-model \
  --config configs/models/film/film-ffnn.toml \
  --data configs/datasets/train/45x15randomE/gaussian-cg100.toml
```

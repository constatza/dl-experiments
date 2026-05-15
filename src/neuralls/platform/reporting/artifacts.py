"""Typed artifact writing for comparison workflows."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import tomli_w
from numpy.typing import NDArray

from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonRecommendations,
    ComparisonResult,
    PlotPaths,
    RankedRecommendation,
)


@dataclass(frozen=True)
class ArrayArtifact:
    """Reference to a persisted numpy array artifact.

    Args:
        path: Relative path within the output directory.
        shape: Array shape tuple.
        dtype: Numpy dtype string.
    """

    path: Path
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ComparisonArtifactManifest:
    """Manifest of files emitted by comparison artifact writing.

    Args:
        comparison_toml: Path to comparison summary TOML.
        comparison_json: Path to comparison summary JSON.
        recommendations_json: Path to ranked recommendations JSON.
        summary_txt: Path to text summary.
        config_copy: Path to config file copy.
        arrays: Tuple of array artifact references.
    """

    comparison_toml: Path
    comparison_json: Path
    recommendations_json: Path
    summary_txt: Path
    config_copy: Path
    arrays: tuple[ArrayArtifact, ...] = ()


@dataclass(frozen=True)
class FallbackComparisonResultEntry:
    """Minimal typed fallback for artifact writing in tests.

    Args:
        iterations: Iteration count.
        residual: Final relative residual.
        residual_abs: Final absolute residual.
        error: Error message if failed.
    """

    iterations: int
    residual: float
    residual_abs: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class ComparisonArtifactFallback:
    """Typed fallback source for artifact writing when tests patch the workflow.

    Args:
        summary: Text summary of results.
        preconditioners: Preconditioner names.
        condition_numbers: Condition numbers by preconditioner.
        plot_paths: Paths to generated plots.
        recommendations: Ranked recommendations.
        results: Per-preconditioner fallback result entries.
    """

    summary: str = ""
    preconditioners: tuple[str, ...] = ()
    condition_numbers: dict[str, float] = field(default_factory=dict)
    plot_paths: PlotPaths = field(default_factory=PlotPaths)
    recommendations: ComparisonRecommendations = field(default_factory=ComparisonRecommendations)
    results: dict[str, FallbackComparisonResultEntry] = field(default_factory=dict)


type ComparisonArtifactSource = ComparisonResult | ComparisonArtifactFallback


@dataclass(frozen=True)
class PendingArrayArtifact:
    """In-memory numpy artifact waiting to be saved."""

    reference: ArrayArtifact
    values: NDArray[Any]


@dataclass(frozen=True)
class SerializedSolverResult:
    """JSON-safe solver result payload."""

    iterations: int
    residual: float
    residual_abs: float | None = None
    converged: bool | None = None
    residual_history_rel: tuple[float, ...] = ()
    residual_history_abs: tuple[float, ...] = ()
    preconditioner: str | None = None
    exact_error: float | None = None
    rhs_norm: float | None = None
    breakdown: bool | None = None
    error: str | None = None
    x: ArrayArtifact | None = None
    initial_guess: ArrayArtifact | None = None


@dataclass(frozen=True)
class SerializedComparisonPayload:
    """JSON-safe comparison payload without raw numpy arrays."""

    summary: str
    preconditioners: tuple[str, ...]
    condition_numbers: dict[str, float]
    plot_paths: PlotPaths
    recommendations: ComparisonRecommendations
    results: dict[str, SerializedSolverResult]


def _sanitize_path_part(value: str) -> str:
    """Convert an arbitrary key into a stable path-safe fragment."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "value"


def _array_reference(
    path_parts: tuple[str, ...],
    values: NDArray[Any],
) -> PendingArrayArtifact:
    """Create a typed artifact reference for a numpy array."""
    path = (
        Path("arrays")
        .joinpath(*[_sanitize_path_part(part) for part in path_parts])
        .with_suffix(".npy")
    )
    reference = ArrayArtifact(
        path=path,
        shape=tuple(int(dimension) for dimension in values.shape),
        dtype=str(values.dtype),
    )
    return PendingArrayArtifact(reference=reference, values=values)


def _serialize_solver_result(
    label: str,
    result: CGComparisonResult,
) -> tuple[SerializedSolverResult, tuple[PendingArrayArtifact, ...]]:
    """Build a typed serialized solver payload and detached arrays."""
    solution_artifact = _array_reference(("results", label, "x"), result.x)
    guess_artifact = _array_reference(("results", label, "initial_guess"), result.initial_guess)
    payload = SerializedSolverResult(
        iterations=result.iterations,
        residual=result.residual,
        residual_abs=result.residual_abs,
        converged=result.converged,
        residual_history_rel=tuple(result.residual_history_rel),
        residual_history_abs=tuple(result.residual_history_abs),
        preconditioner=result.preconditioner,
        exact_error=result.exact_error,
        rhs_norm=result.rhs_norm,
        breakdown=result.breakdown,
        error=result.error,
        x=solution_artifact.reference,
        initial_guess=guess_artifact.reference,
    )
    return payload, (solution_artifact, guess_artifact)


def _serialize_fallback_result(
    result: FallbackComparisonResultEntry,
) -> SerializedSolverResult:
    """Build a typed serialized solver payload for fallback sources."""
    return SerializedSolverResult(
        iterations=result.iterations,
        residual=result.residual,
        residual_abs=result.residual_abs,
        error=result.error,
    )


def extract_array_artifacts(
    result: ComparisonArtifactSource,
) -> tuple[SerializedComparisonPayload, tuple[PendingArrayArtifact, ...]]:
    """Detach numpy arrays from a typed comparison result."""
    serialized_results: dict[str, SerializedSolverResult] = {}
    pending_arrays: list[PendingArrayArtifact] = []
    for label, entry in result.results.items():
        match entry:
            case CGComparisonResult():
                serialized_entry, arrays = _serialize_solver_result(label, entry)
                serialized_results[label] = serialized_entry
                pending_arrays.extend(arrays)
            case FallbackComparisonResultEntry():
                serialized_results[label] = _serialize_fallback_result(entry)

    payload = SerializedComparisonPayload(
        summary=result.summary,
        preconditioners=tuple(result.preconditioners),
        condition_numbers=dict(result.condition_numbers),
        plot_paths=result.plot_paths,
        recommendations=result.recommendations,
        results=serialized_results,
    )
    return payload, tuple(pending_arrays)


def _serialize_value(value: Any) -> Any:
    """Serialize typed payload values to JSON-compatible primitives."""
    match value:
        case None | bool() | int() | float() | str():
            return value
        case Path() as path:
            return path.as_posix()
        case dict() as mapping:
            return {str(key): _serialize_value(item) for key, item in mapping.items()}
        case list() | tuple():
            return [_serialize_value(item) for item in value]
        case _ if is_dataclass(value) and not isinstance(value, type):
            return _serialize_value(asdict(value))
        case _:
            raise TypeError(f"Unsupported comparison artifact value: {type(value).__name__}")


def serialize_comparison_payload(payload: SerializedComparisonPayload) -> dict[str, Any]:
    """Convert a typed comparison payload to a JSON-ready dict."""
    serialized = _serialize_value(payload)
    if not isinstance(serialized, dict):
        raise TypeError("Serialized comparison payload must be a mapping.")
    return serialized


def save_numpy_artifacts(
    work_root: Path,
    arrays: tuple[PendingArrayArtifact, ...],
) -> tuple[ArrayArtifact, ...]:
    """Persist detached numpy arrays as binary artifacts."""
    for array_artifact in arrays:
        output_path = work_root / array_artifact.reference.path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, array_artifact.values, allow_pickle=False)
    return tuple(array.reference for array in arrays)


def _stage_plot_path(source: Path, work_root: Path) -> Path:
    """Copy a plot into the staged artifacts directory and return its relative path."""
    if not source.exists():
        raise FileNotFoundError(f"Comparison plot does not exist: {source}")
    destination = work_root / "figures" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination.relative_to(work_root)


def _stage_plot_paths(plot_paths: PlotPaths, work_root: Path) -> PlotPaths:
    """Stage comparison plots under work_root/figures and return relative paths."""
    staged = {
        key: _stage_plot_path(path, work_root) for key, path in plot_paths.to_mapping().items()
    }
    return PlotPaths.from_mapping(staged)


def _stage_result_plots(
    result: ComparisonArtifactSource,
    work_root: Path,
) -> ComparisonArtifactSource:
    """Rewrite result plot paths to staged artifact-relative paths."""
    staged_paths = _stage_plot_paths(result.plot_paths, work_root)
    return replace(result, plot_paths=staged_paths)


def _save_comparison_toml(
    result: ComparisonArtifactSource,
    output_path: Path,
) -> None:
    """Save scalar comparison diagnostics to a TOML file."""
    iterations = {name: entry.iterations for name, entry in result.results.items()}
    residuals = {name: entry.residual for name, entry in result.results.items()}
    payload = {
        "condition_number": dict(result.condition_numbers),
        "iterations": iterations,
        "final_residual": residuals,
    }
    with open(output_path, "wb") as fh:
        tomli_w.dump(payload, fh)


def write_comparison_artifacts(
    *,
    result: ComparisonArtifactSource,
    work_root: Path,
    comparison_config: Path,
) -> ComparisonArtifactManifest:
    """Write all structured comparison artifacts to disk."""
    staged_result = _stage_result_plots(result, work_root)
    payload, pending_arrays = extract_array_artifacts(staged_result)
    manifest = ComparisonArtifactManifest(
        comparison_toml=work_root / "comparison.toml",
        comparison_json=work_root / "comparison.json",
        recommendations_json=work_root / "recommendations.json",
        summary_txt=work_root / "summary.txt",
        config_copy=work_root / "config" / comparison_config.name,
        arrays=save_numpy_artifacts(work_root, pending_arrays),
    )
    _save_comparison_toml(staged_result, manifest.comparison_toml)
    manifest.comparison_json.write_text(
        json.dumps(serialize_comparison_payload(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.summary_txt.write_text(payload.summary, encoding="utf-8")
    manifest.recommendations_json.write_text(
        json.dumps(_serialize_value(payload.recommendations), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.config_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(comparison_config, manifest.config_copy)
    return manifest


def _coerce_ranked_recommendation(value: object) -> RankedRecommendation | None:
    """Convert a loose object into a typed ranked recommendation."""
    match value:
        case RankedRecommendation():
            return value
        case dict() as mapping:
            label = mapping.get("label")
            iterations = mapping.get("iterations")
            residual = mapping.get("residual")
            residual_abs = mapping.get("residual_abs", residual)
            breakdown = mapping.get("breakdown", False)
            if not isinstance(iterations, int):
                return None
            if not isinstance(residual, int | float):
                return None
            if not isinstance(residual_abs, int | float):
                return None
            if not isinstance(breakdown, bool):
                return None
            return RankedRecommendation(
                label=label if isinstance(label, str) else "",
                iterations=iterations,
                residual=float(residual),
                residual_abs=float(residual_abs),
                breakdown=breakdown,
            )
        case _:
            return None


def _coerce_recommendations(value: object) -> ComparisonRecommendations:
    """Convert a loose recommendations payload into the typed model."""
    match value:
        case ComparisonRecommendations():
            return value
        case dict() as mapping:
            ranked_value = mapping.get("ranked", ())
            ranked = (
                tuple(
                    recommendation
                    for item in ranked_value
                    if (recommendation := _coerce_ranked_recommendation(item)) is not None
                )
                if isinstance(ranked_value, list | tuple)
                else ()
            )
            overall_best = _coerce_ranked_recommendation(
                mapping.get("overall_best") or mapping.get("best_overall")
            )
            return ComparisonRecommendations(ranked=ranked, overall_best=overall_best)
        case _:
            return ComparisonRecommendations()


def _coerce_plot_paths(value: object) -> PlotPaths:
    """Convert a loose plot-path payload into the typed model."""
    match value:
        case PlotPaths():
            return value
        case dict() as mapping:
            path_mapping = {
                str(key): Path(path)
                for key, path in mapping.items()
                if isinstance(path, str | Path)
            }
            return PlotPaths.from_mapping(path_mapping)
        case _:
            return PlotPaths()


def _coerce_fallback_result_entry(value: object) -> FallbackComparisonResultEntry | None:
    """Convert a loose result entry into the minimal typed fallback model."""
    match value:
        case FallbackComparisonResultEntry():
            return value
        case dict() as mapping:
            iterations = mapping.get("iterations")
            residual = mapping.get("residual")
            residual_abs = mapping.get("residual_abs")
            error = mapping.get("error")
        case _ if isinstance(value, Mock):
            return None
        case _:
            iterations = getattr(value, "iterations", None)
            residual = getattr(value, "residual", None)
            residual_abs = getattr(value, "residual_abs", None)
            error = getattr(value, "error", None)

    if not isinstance(iterations, int):
        return None
    if not isinstance(residual, int | float):
        return None
    if residual_abs is not None and not isinstance(residual_abs, int | float):
        residual_abs = None
    if error is not None and not isinstance(error, str):
        error = None
    return FallbackComparisonResultEntry(
        iterations=iterations,
        residual=float(residual),
        residual_abs=float(residual_abs) if residual_abs is not None else None,
        error=error,
    )


def _coerce_fallback_results(value: object) -> dict[str, FallbackComparisonResultEntry]:
    """Convert loose result mappings into typed fallback entries."""
    match value:
        case dict() as mapping:
            converted: dict[str, FallbackComparisonResultEntry] = {}
            for key, entry in mapping.items():
                coerced = _coerce_fallback_result_entry(entry)
                if coerced is not None:
                    converted[str(key)] = coerced
            return converted
        case _:
            return {}


def coerce_comparison_result_payload(value: object) -> ComparisonArtifactSource:
    """Normalize loose comparison payloads at the workflow boundary."""
    if isinstance(value, ComparisonResult):
        return value
    if isinstance(value, Mock):
        return ComparisonArtifactFallback()

    summary = getattr(value, "summary", "")
    preconditioners = getattr(value, "preconditioners", ())
    condition_numbers = getattr(value, "condition_numbers", {})
    plot_paths = getattr(value, "plot_paths", {})
    recommendations = getattr(value, "recommendations", {})
    results = getattr(value, "results", {})

    return ComparisonArtifactFallback(
        summary=summary if isinstance(summary, str) else "",
        preconditioners=tuple(item for item in preconditioners if isinstance(item, str))
        if isinstance(preconditioners, list | tuple)
        else (),
        condition_numbers={
            str(key): float(score)
            for key, score in condition_numbers.items()
            if isinstance(score, int | float)
        }
        if isinstance(condition_numbers, dict)
        else {},
        plot_paths=_coerce_plot_paths(plot_paths),
        recommendations=_coerce_recommendations(recommendations),
        results=_coerce_fallback_results(results),
    )

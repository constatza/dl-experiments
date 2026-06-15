"""Feature provider registry for runtime model-family–specific dataset entries.

Each model family declares which features it needs in its TOML's [[DATASET.features]]
section. This registry fulfills those declarations from storage artifacts, keeping
training orchestration agnostic to layout differences between model families.

To add a new model family: register a new FeatureProvider in _FEATURE_REGISTRY.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dlkit.common.geometry import FieldRole, GeometryKind
from dlkit.infrastructure.config.data_entries import DataEntry, ZarrEntry

from neuralls.platform.storage.training_artifacts import TrainingArrays


@runtime_checkable
class FeatureProvider(Protocol):
    """Protocol for mapping a declared feature name to a DLKit DataEntry."""

    def provide(self, arrays: TrainingArrays) -> DataEntry:
        """Build a DLKit DataEntry from storage artifacts.

        Args:
            arrays: Resolved training artifact paths (rhs, solutions, matrix, parameters).

        Returns:
            Configured DataEntry ready for injection into DLKit's feature list.
        """
        ...


class FiLMConditionProvider:
    """Provides the ``condition`` entry for FiLM / conditional FFNN models.

    Maps ``parameters_0.zarr`` (shape ``(N, param_dim)``) directly to a
    ``ZarrEntry`` named ``condition``. DLKit loads the array lazily from disk.

    The conditioning vector encodes the system parameters (e.g. ``[E₁, E₂, E₃, E₄]``)
    that characterise the matrix ``A`` and drive FiLM's modulation layers.
    """

    def provide(self, arrays: TrainingArrays) -> DataEntry:
        """Return a lazy ZarrEntry for the FiLM conditioning vector.

        Args:
            arrays: Training artifact paths; ``parameters_zarr[0]`` is used.

        Returns:
            ZarrEntry with ``model_input=True`` and ``FieldRole.FEATURE``.

        Raises:
            ValueError: If no ``parameters_*.zarr`` files are present in the dataset.
        """
        if not arrays.parameters_zarr:
            raise ValueError(
                "FiLM 'condition' requires parameters_0.zarr in the dataset. "
                "Set source.parameters_paths in the dataset config."
            )
        return ZarrEntry(
            name="condition",
            path=arrays.parameters_zarr[0],
            model_input=True,
            field_role=FieldRole.FEATURE,
            geometry_kind=GeometryKind.TABULAR,
        )


class DeepONetQueryProvider:
    """Provides the ``query`` (trunk input) entry for DeepONet branch-trunk models.

    Loads ``parameters_0.zarr`` (shape ``(N, param_dim)``), reshapes to
    ``(N, 1, param_dim)`` via ``np.newaxis``, and returns a ``ValueEntry``
    with ``FieldRole.TARGET_COORDINATES`` matching DLKit's branch-trunk contract.
    """

    def provide(self, arrays: TrainingArrays) -> DataEntry:
        """Build a ValueEntry carrying the query tensor for DeepONet.

        Args:
            arrays: Training artifact paths; ``parameters_zarr[0]`` is used.

        Returns:
            ValueEntry with shape ``(N, 1, param_dim)`` and
            ``field_role=FieldRole.TARGET_COORDINATES``.

        Raises:
            ValueError: If no ``parameters_*.zarr`` files are present.
        """
        import zarr
        import numpy as np
        from dlkit.infrastructure.config.data_entries import ValueEntry

        if not arrays.parameters_zarr:
            raise ValueError(
                "DeepONet 'query' requires parameters_zarr[0] in the dataset. "
                "Set source.parameters_paths in the dataset config."
            )
        params = np.asarray(
            zarr.open_array(str(arrays.parameters_zarr[0]), mode="r")[:]
        )  # (N, param_dim)
        query = params[:, np.newaxis, :]  # (N, 1, param_dim)
        return ValueEntry(
            name="query",
            value=query,
            field_role=FieldRole.TARGET_COORDINATES,
            geometry_kind=GeometryKind.TABULAR,
        )


_FEATURE_REGISTRY: dict[str, FeatureProvider] = {
    "condition": FiLMConditionProvider(),
    "query": DeepONetQueryProvider(),
}


def registered_extra_names() -> frozenset[str]:
    """Return the set of extra feature names the registry can fulfill.

    Returns:
        Frozenset of declared-feature names beyond the base ``x`` and ``matrix``.
    """
    return frozenset(_FEATURE_REGISTRY)


def resolve_extra_feature(name: str, arrays: TrainingArrays) -> DataEntry:
    """Fulfill a declared extra feature name from storage artifacts.

    Args:
        name: Feature name declared in ``[[DATASET.features]]`` in the model TOML.
        arrays: Resolved training artifact paths.

    Returns:
        Configured DataEntry for injection into DLKit's feature list.

    Raises:
        ValueError: If ``name`` has no registered provider.
    """
    match _FEATURE_REGISTRY.get(name):
        case None:
            raise ValueError(
                f"No feature provider registered for '{name}'. "
                f"Known extra features: {sorted(_FEATURE_REGISTRY)}"
            )
        case provider:
            return provider.provide(arrays)

"""Unit tests for preconditioner registry.

Tests registry pattern, type lookup, and Open/Closed extensibility.

Follows project principles:
- Use fixtures for all test data
- Type hints throughout
- Focus on registry logic, not builder internals
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuralls.configuration.preconditioner import StandardPreconditionerConfig
from neuralls.preconditioner.builders import (
    IdentityBuilder,
    ILUBuilder,
    JacobiBuilder,
    NeuralBuilder,
    PreconditionerBuilder,
)
from neuralls.preconditioner.registry import (
    PreconditionerRegistry,
    create_default_registry,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from neuralls.configuration.preconditioner import PreconditionerConfig
    from neuralls.preconditioner.builders import PreconditionerFn


# ==============================================================================
# Custom Builder for Extensibility Testing
# ==============================================================================


class CustomScalingBuilder:
    """Custom builder for testing Open/Closed principle.

    Scales residual by a custom factor to demonstrate registry extensibility.
    """

    def __init__(self, scale_factor: float = 2.0) -> None:
        """Initialize with scale factor."""
        self.scale_factor = scale_factor

    def build(
        self, matrix: NDArray, config: PreconditionerConfig
    ) -> "PreconditionerFn":
        """Build custom scaling preconditioner."""
        return lambda residual: residual * self.scale_factor

    def supported_types(self) -> set[str]:
        """Return supported type names."""
        return {"custom", "scaling"}


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def test_matrix() -> NDArray:
    """Test matrix for registry tests.

    Returns:
        4x4 diagonal matrix
    """
    return np.diag([4.0, 3.0, 2.0, 1.0])


@pytest.fixture
def residual_vector() -> NDArray:
    """Test residual vector.

    Returns:
        4D residual vector
    """
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def empty_registry() -> PreconditionerRegistry:
    """Create empty registry for testing.

    Returns:
        Empty PreconditionerRegistry instance
    """
    return PreconditionerRegistry()


@pytest.fixture
def populated_registry() -> PreconditionerRegistry:
    """Create registry with standard builders.

    Returns:
        Registry with identity, jacobi, and ilu builders
    """
    registry = PreconditionerRegistry()
    registry.register(IdentityBuilder())
    registry.register(JacobiBuilder())
    registry.register(ILUBuilder())
    return registry


# ==============================================================================
# Registry Core Functionality (5 tests)
# ==============================================================================


def test_registry_register_builder(
    empty_registry: PreconditionerRegistry,
) -> None:
    """Test registering a builder adds supported types."""
    # Initially empty
    assert len(empty_registry.available_types()) == 0

    # Register IdentityBuilder (supports "identity" and "none")
    empty_registry.register(IdentityBuilder())

    # Check types were added
    types = empty_registry.available_types()
    assert "identity" in types
    assert "none" in types
    assert len(types) == 2


def test_registry_register_duplicate_type_raises_error(
    empty_registry: PreconditionerRegistry,
) -> None:
    """Test that registering duplicate type raises ValueError."""
    # Register first builder
    empty_registry.register(IdentityBuilder())

    # Create second builder that also claims "identity" type
    class DuplicateIdentityBuilder:
        def build(self, matrix: NDArray, config: PreconditionerConfig) -> PreconditionerFn:  # type: ignore[type-arg]
            return lambda r: r

        def supported_types(self) -> set[str]:
            return {"identity"}  # Conflicts!

    # Attempting to register should raise ValueError
    with pytest.raises(ValueError, match="already registered"):
        empty_registry.register(DuplicateIdentityBuilder())  # type: ignore[arg-type]


def test_registry_create_with_valid_type(
    populated_registry: PreconditionerRegistry,
    test_matrix: NDArray,
    residual_vector: NDArray,
) -> None:
    """Test creating preconditioner with valid registered type."""
    # Create Jacobi preconditioner
    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
    precond_fn = populated_registry.create(test_matrix, config)

    # Should be callable
    assert callable(precond_fn)

    # Should apply correctly (z_i = r_i / A_ii)
    result = precond_fn(residual_vector)
    expected = residual_vector / np.diag(test_matrix)
    np.testing.assert_array_equal(result, expected)


def test_registry_create_with_invalid_type_raises_error(
    populated_registry: PreconditionerRegistry, test_matrix: NDArray
) -> None:
    """Test that creating with unregistered type raises ValueError."""
    # Create a mock config with unregistered type
    # We bypass Pydantic validation by creating a simple class
    class UnknownConfig:
        def __init__(self):
            self.name = "unknown"
            self.type = "unknown"

    config = UnknownConfig()  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unsupported preconditioner type"):
        populated_registry.create(test_matrix, config)


def test_registry_available_types_returns_all_registered(
    empty_registry: PreconditionerRegistry,
) -> None:
    """Test that available_types returns all registered type names."""
    # Register multiple builders
    empty_registry.register(IdentityBuilder())  # identity, none
    empty_registry.register(JacobiBuilder())  # jacobi
    empty_registry.register(ILUBuilder())  # ilu

    types = empty_registry.available_types()

    # Should have all 4 types
    assert types == {"identity", "none", "jacobi", "ilu"}


# ==============================================================================
# Default Registry (3 tests)
# ==============================================================================


def test_default_registry_includes_standard_types() -> None:
    """Test that create_default_registry includes all standard types."""
    registry = create_default_registry()

    types = registry.available_types()

    # Should include identity, none, jacobi, ilu, neural
    assert "identity" in types
    assert "none" in types
    assert "jacobi" in types
    assert "ilu" in types
    assert "neural" in types


def test_default_registry_handles_none_alias_for_identity(
    test_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test that 'none' type is supported by IdentityBuilder.

    The Pydantic validator normalizes 'none' to 'identity' before validation,
    so we test that IdentityBuilder declares support for 'none'.
    """
    from neuralls.preconditioner.builders import IdentityBuilder

    registry = create_default_registry()

    # Check that IdentityBuilder supports "none"
    identity_builder = IdentityBuilder()
    assert "none" in identity_builder.supported_types()

    # Use identity type (none is normalized to identity at config level)
    config = StandardPreconditionerConfig(name="identity", type="identity")
    precond_fn = registry.create(test_matrix, config)

    # Should behave like identity (returns copy)
    result = precond_fn(residual_vector)
    np.testing.assert_array_equal(result, residual_vector)


def test_default_registry_creates_each_type_successfully(
    test_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test that default registry can create all standard types.

    Note: Neural type requires valid checkpoint, so we skip it here.
    """
    registry = create_default_registry()

    # Test identity
    identity_config = StandardPreconditionerConfig(name="identity", type="identity")
    identity_fn = registry.create(test_matrix, identity_config)
    identity_result = identity_fn(residual_vector)
    np.testing.assert_array_equal(identity_result, residual_vector)

    # Test jacobi
    jacobi_config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
    jacobi_fn = registry.create(test_matrix, jacobi_config)
    jacobi_result = jacobi_fn(residual_vector)
    expected_jacobi = residual_vector / np.diag(test_matrix)
    np.testing.assert_array_equal(jacobi_result, expected_jacobi)

    # Test ilu
    ilu_config = StandardPreconditionerConfig(name="ilu", type="ilu")
    ilu_fn = registry.create(test_matrix, ilu_config)
    ilu_result = ilu_fn(residual_vector)
    assert ilu_result.shape == residual_vector.shape

    # Skip neural (requires checkpoint)


# ==============================================================================
# Extensibility Tests (2 tests - demonstrates Open/Closed)
# ==============================================================================


def test_registry_accepts_custom_builder(
    empty_registry: PreconditionerRegistry,
) -> None:
    """Test that registry accepts custom builder implementations.

    Demonstrates Open/Closed Principle: registry is open for extension
    (can add new builders) but closed for modification (no code changes needed).
    """
    # Register custom builder
    custom_builder = CustomScalingBuilder(scale_factor=3.0)
    empty_registry.register(custom_builder)  # type: ignore[arg-type]

    # Check types were registered
    types = empty_registry.available_types()
    assert "custom" in types
    assert "scaling" in types


def test_custom_builder_integrates_without_modification(
    empty_registry: PreconditionerRegistry,
    test_matrix: NDArray,
    residual_vector: NDArray,
) -> None:
    """Test that custom builder integrates seamlessly without registry modification.

    This validates the Open/Closed Principle: the registry code never changed,
    but we extended its functionality by adding a new builder.
    """
    # Register custom builder
    custom_builder = CustomScalingBuilder(scale_factor=5.0)
    empty_registry.register(custom_builder)  # type: ignore[arg-type]

    # Create mock config with custom type (bypasses Pydantic validation)
    class CustomConfig:
        def __init__(self):
            self.name = "custom"
            self.type = "custom"

    config = CustomConfig()  # type: ignore[assignment]
    precond_fn = empty_registry.create(test_matrix, config)

    # Apply preconditioner
    result = precond_fn(residual_vector)

    # Should scale by 5.0
    expected = residual_vector * 5.0
    np.testing.assert_array_equal(result, expected)

    # Registry code was NEVER modified - this is extension, not modification!

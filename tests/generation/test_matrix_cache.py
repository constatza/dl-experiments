"""Tests verifying @functools.cache correctness for the _get_matrix closure.

The nested ``_get_matrix`` closure inside ``build_dataset_payload`` uses
``@functools.cache`` to avoid redundant normalization computations.  These
tests verify the key logical invariants directly — without invoking the full
``build_dataset_payload`` pipeline — by replicating the caching pattern with
counting mocks.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from unittest.mock import MagicMock, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def counting_loader() -> tuple[MagicMock, list[int]]:
    """A MagicMock loader paired with a call-count list.

    Each invocation creates a **new** ``np.eye(3)`` array so that cache-hit
    vs. cache-miss can be distinguished by object identity (``is``).  The
    ``call_count`` list is mutated on every cache-miss so tests can assert
    the exact number of underlying load operations.

    Returns:
        Tuple of (mock_loader, call_count_list) where call_count_list[0]
        increments on every invocation of the mock.
    """
    call_count: list[int] = [0]
    loader = MagicMock()

    def _side_effect(sid: int) -> np.ndarray:
        call_count[0] += 1
        # Return a fresh array each time so identity checks are meaningful.
        return np.eye(3, dtype=np.float64) * (sid + 1)

    loader.side_effect = _side_effect
    return loader, call_count


# ---------------------------------------------------------------------------
# Helper: build a fresh cached getter (simulates one build_dataset_payload call)
# ---------------------------------------------------------------------------


def _make_cached_getter(loader: MagicMock) -> Callable[[int], np.ndarray]:
    """Return a freshly decorated ``@cache`` function wrapping *loader*.

    Each call to this factory produces a NEW closure with a NEW independent
    cache, mirroring the behaviour of ``build_dataset_payload`` which creates
    a fresh ``@cache``-decorated ``_get_matrix`` on every invocation.

    Args:
        loader: Callable ``(sample_id: int) -> np.ndarray`` used as the
            underlying data source.

    Returns:
        A ``@cache``-decorated callable ``(int) -> np.ndarray``.
    """

    @cache
    def _get(sid: int) -> np.ndarray:
        return loader(sid)

    return _get


# ---------------------------------------------------------------------------
# Test 1: same id → single loader call, identity-preserving cache
# ---------------------------------------------------------------------------


def test_cache_same_id_calls_loader_once(
    counting_loader: tuple[MagicMock, list[int]],
) -> None:
    """Same sample_id must hit the underlying loader exactly once.

    Requesting the same sample_id multiple times must return the **identical**
    Python object (``is`` check) and must NOT invoke the loader a second time.

    Args:
        counting_loader: Fixture providing a (mock, call_count) pair.
    """
    loader, call_count = counting_loader
    _get = _make_cached_getter(loader)

    result_a = _get(0)
    result_b = _get(0)
    result_c = _get(1)

    # Same id → same object from cache
    assert result_a is result_b, "Cache must return the identical object for the same sample_id"
    # Different id → different object
    assert result_a is not result_c, "Different sample_ids must produce independent cache entries"
    # Loader called exactly twice: once for id=0, once for id=1
    assert call_count[0] == 2, f"Expected 2 loader calls, got {call_count[0]}"
    loader.assert_has_calls([call(0), call(1)], any_order=False)


# ---------------------------------------------------------------------------
# Test 2: fresh cache per closure (two separate build_dataset_payload calls)
# ---------------------------------------------------------------------------


def test_fresh_cache_per_closure_instance() -> None:
    """Each build_dataset_payload invocation must get a fresh, empty cache.

    When ``build_dataset_payload`` is called twice, the second invocation
    must call the loader again for each sample_id — there must be NO
    cross-invocation cache sharing.
    """
    total_calls: list[int] = [0]
    loader = MagicMock()

    def _side_effect(sid: int) -> np.ndarray:
        total_calls[0] += 1
        return np.eye(3, dtype=np.float64)

    loader.side_effect = _side_effect

    # First "build" invocation — cache lives here
    _get_first = _make_cached_getter(loader)
    _get_first(0)
    _get_first(0)  # cached within first closure
    assert total_calls[0] == 1, "First closure: loader should be called once for id=0"

    # Second "build" invocation — brand new cache, no cross-contamination
    _get_second = _make_cached_getter(loader)
    _get_second(0)
    _get_second(0)  # cached within second closure
    assert total_calls[0] == 2, (
        "Second closure must call the loader again for id=0 — "
        "caches must not be shared across closure instances"
    )


# ---------------------------------------------------------------------------
# Test 3: mutation safety — cache returns same object; document the contract
# ---------------------------------------------------------------------------


def test_cache_mutation_returns_same_object(
    counting_loader: tuple[MagicMock, list[int]],
) -> None:
    """Mutating the returned array affects subsequent reads (shared reference).

    ``@functools.cache`` stores and returns the **same** object on every
    cache hit.  This test documents the known behaviour: if a caller mutates
    the cached array in-place, subsequent cache reads will observe the mutation.
    Callers that need isolated copies must call ``array.copy()`` themselves.

    Args:
        counting_loader: Fixture providing a (mock, call_count) pair.
    """
    loader, call_count = counting_loader
    _get = _make_cached_getter(loader)

    first = _get(0)

    # Mutate the returned array in-place
    first[0, 0] = 999.0

    # Cache hit: returns the *same* Python object — mutation IS visible
    second = _get(0)
    assert second is first, "Cache must return the identical object on a hit"
    assert second[0, 0] == 999.0, (
        "Mutation of the cached array IS visible on subsequent reads — "
        "this is expected @cache behaviour; callers needing isolation must .copy()"
    )

    # Loader was only called once despite two _get(0) calls
    assert call_count[0] == 1, "Loader must not be called again after a cache hit"

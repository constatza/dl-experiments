"""TOML serialization utilities for solver data.

This module provides type conversion for TOML export, handling:
- Enums → strings (via .value)
- Dataclasses → dicts (via asdict())
- Numpy types → Python types (int, float, list)
- None filtering (remove None values from dicts)

All functions are pure with well-defined type conversions.
"""

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np


def sanitize_for_toml(obj: Any) -> Any:
    """Convert Python objects to TOML-compatible types.

    Recursively converts nested structures (dicts, lists, tuples) and handles
    common Python types that are not TOML-compatible:
    - Enums → strings via .value
    - Dataclasses → dicts via asdict()
    - Numpy integers → int
    - Numpy floats → float
    - Numpy arrays → lists
    - None values filtered from dicts

    Args:
        obj: Object to convert (dict, list, dataclass, Enum, numpy type, etc.)

    Returns:
        TOML-compatible object (str, int, float, bool, list, dict).

    Example:
        >>> from enum import Enum
        >>> class Color(Enum):
        ...     RED = "red"
        >>> sanitize_for_toml(Color.RED)
        'red'

        >>> import numpy as np
        >>> sanitize_for_toml(np.int64(42))
        42

        >>> sanitize_for_toml({"a": 1, "b": None, "c": 3})
        {'a': 1, 'c': 3}
    """
    if obj is None:
        return None  # Will be filtered by dict comprehension

    if isinstance(obj, dict):
        return {k: sanitize_for_toml(v) for k, v in obj.items() if v is not None}

    if isinstance(obj, list | tuple):
        return [sanitize_for_toml(x) for x in obj if x is not None]

    if isinstance(obj, Enum):
        return obj.value

    if is_dataclass(obj) and not isinstance(obj, type):
        return sanitize_for_toml(asdict(obj))

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj

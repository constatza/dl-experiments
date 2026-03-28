"""Array export format options."""

from enum import StrEnum


class ArrayFormat(StrEnum):
    """Array export format options.

    Attributes:
        NPZ: Numpy binary format (fast, small, ~10x smaller than txt).
        TXT: Text format (human-readable, slower, larger files).
        BOTH: Export both NPZ and TXT formats.

    Example:
        >>> format = ArrayFormat.NPZ
        >>> format.value
        'npz'
    """

    NPZ = "npz"
    """Numpy binary format - fast, compact, supports multiple arrays in one file."""

    TXT = "txt"
    """Text format - human-readable, larger files, slower I/O."""

    BOTH = "both"
    """Export both formats for debugging and compatibility."""

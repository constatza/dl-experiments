"""Pure helpers for auto-id and display-name generation in case configs.

This module provides functions to infer missing 'id' and 'display_name' fields
in experiment and comparison entries, with sensible fallback logic and validation.
No Pydantic imports — only plain Python and regex.
"""

from __future__ import annotations

import re
from typing import cast

_ID_PATTERN = re.compile(r"^[\w\-]+$")


def _validate_id_chars(id_val: str, context: str = "id") -> None:
    r"""Raise ValueError if id_val contains characters outside ^[\w\-]+$.

    Args:
        id_val: The ID string to validate.
        context: Description of what is being validated (for error messages).

    Raises:
        ValueError: If id_val contains invalid characters.
    """
    if not _ID_PATTERN.fullmatch(id_val):
        raise ValueError(
            f"{context} contains invalid characters: {id_val!r} must match ^[\\w\\-]+$"
        )


def _slugify(display_name: str) -> str:
    """Lowercase, replace spaces with hyphens, validate result against _ID_PATTERN.

    Args:
        display_name: The display name to slugify.

    Returns:
        The slugified ID string.

    Raises:
        ValueError: If the result is invalid after slugification, or if input is empty.
    """
    if not display_name:
        raise ValueError("invalid display_name: cannot be empty")

    # Strip surrounding whitespace, lowercase, and replace spaces with hyphens
    slugified = display_name.strip().lower().replace(" ", "-")

    # Validate the result
    if not _ID_PATTERN.fullmatch(slugified):
        raise ValueError(f"invalid id after slugification: {slugified!r}")

    return slugified


def _build_display_lookup(entries: list[object]) -> dict[str, str]:
    """Return {id: effective_display_name} from a list of raw registry entry dicts.

    Non-dict entries are silently ignored. Falls back to the entry id when
    display_name is absent or blank.

    Args:
        entries: List of registry entries (mixed types; only dicts are processed).

    Returns:
        Mapping from id to display name for each valid entry.
    """
    result: dict[str, str] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        entry_dict = cast(dict[str, object], entry)
        entry_id = entry_dict.get("id")
        if entry_id is None:
            continue

        # Treat entry_id as string for lookup purposes
        id_str = str(entry_id)

        # Use display_name if present and non-blank, otherwise fall back to id
        display_name = entry_dict.get("display_name", "")
        if isinstance(display_name, str) and display_name.strip():
            result[id_str] = display_name.strip()
        else:
            result[id_str] = id_str

    return result


def _infer_experiment_id(
    dataset_id: str,
    model_id: str,
    user_id: str | None,
    user_dn: str | None,
) -> str:
    """Return the resolved experiment id.

    Priority:
      1. user_id (stripped, if non-blank)
      2. _slugify(user_dn) if user_dn non-blank
      3. f"{model_id}-{dataset_id}"  (model first)

    Validates result with _validate_id_chars before returning.

    Args:
        dataset_id: The dataset identifier.
        model_id: The model identifier.
        user_id: Explicit user-supplied id (may be None or blank).
        user_dn: Explicit user-supplied display_name (may be None or blank).

    Returns:
        The resolved experiment id string.

    Raises:
        ValueError: If the resolved id contains invalid characters.
    """
    if user_id and user_id.strip():
        final_id = user_id.strip()
    elif user_dn and user_dn.strip():
        final_id = _slugify(user_dn)
    else:
        final_id = f"{model_id}-{dataset_id}"

    _validate_id_chars(final_id, "id")
    return final_id


def _infer_experiment_display_name(
    dataset_id: str,
    model_id: str,
    user_dn: str | None,
    dataset_display: dict[str, str],
    model_display: dict[str, str],
) -> str:
    """Return the resolved experiment display name.

    Priority:
      1. user_dn (stripped, if non-blank)
      2. f"{model_label} | {dataset_label}"
         (labels from lookups, fallback to raw id when key absent)

    Args:
        dataset_id: The dataset identifier.
        model_id: The model identifier.
        user_dn: Explicit user-supplied display_name (may be None or blank).
        dataset_display: Lookup from dataset id to display label.
        model_display: Lookup from model id to display label.

    Returns:
        The resolved experiment display name string.
    """
    if user_dn and user_dn.strip():
        return user_dn.strip()

    dataset_label = dataset_display.get(dataset_id, dataset_id)
    model_label = model_display.get(model_id, model_id)
    return f"{model_label} | {dataset_label}"


def _infer_comparison_id(
    matrix_id: str,
    rhs_id: str,
    user_id: str | None,
    user_dn: str | None,
) -> str:
    """Return the resolved comparison id.

    Priority:
      1. user_id (stripped, if non-blank)
      2. _slugify(user_dn) if user_dn non-blank
      3. matrix_id           (when matrix_id == rhs_id)
         f"{matrix_id}-{rhs_id}"  otherwise (single dash)

    Validates result with _validate_id_chars before returning.

    Args:
        matrix_id: The matrix dataset identifier.
        rhs_id: The right-hand side dataset identifier.
        user_id: Explicit user-supplied id (may be None or blank).
        user_dn: Explicit user-supplied display_name (may be None or blank).

    Returns:
        The resolved comparison id string.

    Raises:
        ValueError: If the resolved id contains invalid characters.
    """
    if user_id and user_id.strip():
        final_id = user_id.strip()
    elif user_dn and user_dn.strip():
        final_id = _slugify(user_dn)
    elif matrix_id == rhs_id:
        final_id = matrix_id
    else:
        final_id = f"{matrix_id}-{rhs_id}"

    _validate_id_chars(final_id, "id")
    return final_id


def _infer_comparison_display_name(
    matrix_id: str,
    rhs_id: str,
    user_dn: str | None,
    dataset_display: dict[str, str],
) -> str:
    """Return the resolved comparison display name.

    Priority:
      1. user_dn (stripped, if non-blank)
      2. matrix_label           (when matrix_id == rhs_id)
         f"{matrix_label} | {rhs_label}"  otherwise

    Args:
        matrix_id: The matrix dataset identifier.
        rhs_id: The right-hand side dataset identifier.
        user_dn: Explicit user-supplied display_name (may be None or blank).
        dataset_display: Lookup from dataset id to display label.

    Returns:
        The resolved comparison display name string.
    """
    if user_dn and user_dn.strip():
        return user_dn.strip()

    matrix_label = dataset_display.get(matrix_id, matrix_id)
    rhs_label = dataset_display.get(rhs_id, rhs_id)

    if matrix_id == rhs_id:
        return matrix_label
    return f"{matrix_label} | {rhs_label}"

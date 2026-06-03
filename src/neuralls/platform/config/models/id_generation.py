"""Pure helpers for auto-id and display-name generation in case configs.

This module provides functions to auto-fill missing 'id' and 'display_name' fields
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
    if not _ID_PATTERN.match(id_val):
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
        raise ValueError("display_name cannot be empty")

    # Strip surrounding whitespace, lowercase, and replace spaces with hyphens
    slugified = display_name.strip().lower().replace(" ", "-")

    # Validate the result
    if not _ID_PATTERN.match(slugified):
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


def _fill_experiment_entry(
    exp: dict[str, object],
    dataset_display: dict[str, str],
    model_display: dict[str, str],
) -> None:
    """Fill missing id and display_name in-place for a raw experiment entry dict.

    Priority for id:
      1. Explicit user value (non-blank)
      2. _slugify(display_name) — when display_name provided
      3. f"{model_id}-{dataset_id}" — model FIRST, then dataset

    Priority for display_name:
      1. Explicit user value (non-blank)
      2. f"{dataset_label} | {model_label}"
         where labels come from dataset_display/model_display lookups
         (fall back to the raw id when not in the lookup)

    Also validate the final id with _validate_id_chars.

    Args:
        exp: Experiment entry dict to fill (modified in-place).
        dataset_display: Lookup from dataset id to display label.
        model_display: Lookup from model id to display label.

    Raises:
        ValueError: If the final id is invalid or if required keys are missing.
    """
    # Extract dataset_id and model_id (TOML alias: 'dataset' -> dataset_id, 'model' -> model_id)
    dataset_id = str(exp.get("dataset", ""))
    model_id = str(exp.get("model", ""))

    # 1. Determine the ID
    user_id = exp.get("id")
    if user_id and isinstance(user_id, str) and user_id.strip():
        # Explicit user value
        final_id = user_id.strip()
    elif "display_name" in exp:
        user_dn = exp.get("display_name")
        if isinstance(user_dn, str) and user_dn.strip():
            # Derive from display_name
            final_id = _slugify(user_dn)
        else:
            # display_name exists but is blank, use fallback
            final_id = f"{model_id}-{dataset_id}"
    else:
        # No display_name, use fallback
        final_id = f"{model_id}-{dataset_id}"

    # Validate the final id
    _validate_id_chars(final_id, "id")
    exp["id"] = final_id

    # 2. Determine the display_name
    user_dn = exp.get("display_name")
    if user_dn and isinstance(user_dn, str) and user_dn.strip():
        # Explicit user value, keep as-is
        pass
    else:
        # Auto-generate from lookups
        dataset_label = dataset_display.get(dataset_id, dataset_id)
        model_label = model_display.get(model_id, model_id)
        final_dn = f"{dataset_label} | {model_label}"
        exp["display_name"] = final_dn


def _fill_comparison_entry(
    comp: dict[str, object],
    dataset_display: dict[str, str],
) -> None:
    """Fill missing id and display_name in-place for a raw comparison entry dict.

    Priority for id:
      1. Explicit user value
      2. _slugify(display_name) — when display_name provided
      3. matrix_dataset_id — when matrix_dataset == rhs_dataset
         f"{matrix_dataset_id}-{rhs_dataset_id}" — otherwise (single dash)

    Priority for display_name:
      1. Explicit user value
      2. matrix_dataset label — when matrix_dataset == rhs_dataset
         f"{matrix_dn} | {rhs_dn}" — otherwise (pipe separator)

    Also validate the final id with _validate_id_chars.

    Args:
        comp: Comparison entry dict to fill (modified in-place).
        dataset_display: Lookup from dataset id to display label.

    Raises:
        ValueError: If the final id is invalid or if required keys are missing.
    """
    # Extract dataset IDs
    matrix_dataset_id = str(comp.get("matrix_dataset", ""))
    rhs_dataset_id = str(comp.get("rhs_dataset", ""))

    # 1. Determine the ID
    user_id = comp.get("id")
    if user_id and isinstance(user_id, str) and user_id.strip():
        # Explicit user value
        final_id = user_id.strip()
    elif "display_name" in comp:
        user_dn = comp.get("display_name")
        if isinstance(user_dn, str) and user_dn.strip():
            # Derive from display_name
            final_id = _slugify(user_dn)
        else:
            # display_name exists but is blank, use fallback
            if matrix_dataset_id == rhs_dataset_id:
                final_id = matrix_dataset_id
            else:
                final_id = f"{matrix_dataset_id}-{rhs_dataset_id}"
    else:
        # No display_name, use fallback
        if matrix_dataset_id == rhs_dataset_id:
            final_id = matrix_dataset_id
        else:
            final_id = f"{matrix_dataset_id}-{rhs_dataset_id}"

    # Validate the final id
    _validate_id_chars(final_id, "id")
    comp["id"] = final_id

    # 2. Determine the display_name
    user_dn = comp.get("display_name")
    if user_dn and isinstance(user_dn, str) and user_dn.strip():
        # Explicit user value, keep as-is
        pass
    else:
        # Auto-generate from lookups
        matrix_label = dataset_display.get(matrix_dataset_id, matrix_dataset_id)
        rhs_label = dataset_display.get(rhs_dataset_id, rhs_dataset_id)

        if matrix_dataset_id == rhs_dataset_id:
            final_dn = matrix_label
        else:
            final_dn = f"{matrix_label} | {rhs_label}"

        comp["display_name"] = final_dn

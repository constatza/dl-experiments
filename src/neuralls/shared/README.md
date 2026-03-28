# Shared Module

The shared package holds cross-layer primitives only.

## Package Map

- `constants.py`: package-wide constants and config keys
- `types.py`: shared numerical metadata types
- `normalization.py`: normalization helpers and trace containers
- `workspace.py`: experiment and workspace DTOs
- `inference.py`: inference DTOs
- `training.py`: training and batch result DTOs
- `comparison/`: comparison DTOs and result payload types
- `hashing.py`, `functional.py`, `math_utils.py`, `validation.py`: pure utility helpers

## Semantic Difference

Shared code exists only to prevent duplication across the other five top-level
packages. A module belongs here when it is pure, dependency-light, and used by
multiple layers. If a helper is specific to one owner package, it should live
with that owner instead of becoming generic by convenience.

## Boundary

Shared code must stay dependency-free with respect to the other architectural
layers. If a type is needed by more than one layer, move it here instead of
letting `application`, `platform`, or `composition` import one another.

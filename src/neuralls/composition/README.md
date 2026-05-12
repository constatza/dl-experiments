# Composition Module

The composition package owns wiring and config-driven assembly.

## Package Map

- `experiments/`: registry loading, experiment wiring, training, comparison, inference
- `comparison/`: single-run comparison assembly around application/domain logic
- `generation/`: config-driven dataset orchestration, dataset persistence wiring, and default tracing services
- `inference/`: inference data-loading composition helpers
- `preconditioners/`: config-to-preconditioner factory wiring
- `tracking/`: tracking tag and run-spec assembly

## Semantic Difference

Composition is the only layer allowed to know both the abstract workflow shape
and the concrete adapter set needed to run it. If a module mostly wires config
models, ports, and runtime collaborators together, it belongs here. If it
starts doing reusable numerical work, it belongs in `domain`. If it starts
owning concrete IO or MLflow mechanics, that code belongs in `platform`.

## Boundary

Composition is where config models, platform adapters, workflow DTOs, and
domain services are connected. Entry modules that still assemble concrete
collaborators belong here rather than under `application`.

CLI entry points now create `NeurallsSettings` from the active machine profile
plus explicit env overrides and pass them top-down into composition. Dataset
generation, training, comparison, and inference assembly all expect resolved
settings instead of reading environment variables or normalizing config paths
locally.

Training orchestration now keeps DLKit's unified `execute()` entrypoint at the
composition boundary. The loader is responsible for producing the correct
workflow-specific DLKit config type before invoking DLKit runtime execution.

Model-reference resolution also keeps MLflow URI/model lookup helpers on the
composition side because the installed DLKit package no longer exposes the
older convenience functions that previous code imported from `dlkit`.

Comparison orchestration also owns preflight validation of comparison input
paths before MLflow tracking starts. Missing benchmark datasets now fail the
individual comparison outcome early with a remediation hint instead of opening
partial runs. Neural comparison model downloads use stable internal ids for
their temporary artifact directories so case display names remain human-readable
without leaking Windows-invalid characters into filesystem paths.

Case-config assembly now assumes that platform loaders have already resolved
machine roots from the active profile, expanded environment-backed path
placeholders, and resolved registry-relative paths, so composition code works
with concrete `Path` values instead of Unix-specific string conventions.
When composition accepts explicit path overrides or env-provided case-config
paths, it routes them through `platform.config.resolution` rather than
re-implementing fallback logic locally. Composition still owns workflow-local
artifact layout decisions such as comparison output directories and workspace
selection, but shared normalization policy and MLflow URI derivation stay in
platform so composition remains wiring-only.

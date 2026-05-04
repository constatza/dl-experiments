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

Training orchestration now keeps DLKit's unified `execute()` entrypoint at the
composition boundary. The loader is responsible for producing the correct
workflow-specific DLKit config type before invoking DLKit runtime execution.

Model-reference resolution also keeps MLflow URI/model lookup helpers on the
composition side because the installed DLKit package no longer exposes the
older convenience functions that previous code imported from `dlkit`.

Master-config assembly now assumes that platform loaders have already expanded
environment-backed path placeholders and resolved registry-relative paths, so
composition code works with concrete `Path` values instead of Unix-specific
string conventions.

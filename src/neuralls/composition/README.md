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

Composition assumes that platform has already resolved configuration and
runtime concerns into concrete values and adapters. Its job is to select the
right collaborators, construct workflow-local DTOs, and hand execution to
application or platform entrypoints without re-implementing path, environment,
or client policy.

Training, inference, comparison, and generation assembly all follow the same
rule: composition may decide which collaborators participate in a workflow, but
it must not absorb low-level IO mechanics, config normalization policy, or
tensor-level runtime behavior.

For MLflow naming, composition only propagates the resolved case-config names.
Training uses `names.training` and comparison uses `names.comparison`, with the
defaults owned by the case-config Pydantic models rather than composition-layer
constants.

Comparison model resolution treats one resolved MLflow `run_id` as the hard
boundary for checkpoint discovery. When downloaded run artifacts contain
multiple `.ckpt` files, composition canonicalizes byte-identical duplicate
copies and raises on distinct candidates instead of silently picking the first
path.

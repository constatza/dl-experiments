# CLI Module

The CLI package contains command entry points only.

## Package Map

- `process_data.py`: build one dataset from one dataset config
- `generate_multiple.py`: build all datasets declared in a case config
- `train_model.py`: train one model on one dataset
- `train_multiple.py`: train a registry batch and emit aggregate reporting
- `predict.py`: run inference for one model/checkpoint pair
- `compare_preconditioners.py`: run comparison profiles for a case config
- `run_experiments.py`: end-to-end case execution

## Boundary

CLI modules parse user input, print progress, and delegate immediately to
`neuralls.composition`. They must not load configs directly, instantiate
platform adapters directly, or contain business logic.

Commands that load checked-in configs first resolve one explicit case config
plus optional `--env-file` overrides, then pass the resulting
`NeurallsSettings` into the composition layer. This keeps settings discovery at
the CLI boundary rather than inside workflow orchestration code.

`process-data`, `train-model`, and `predict` require explicit case selection
through `--case-config` or `NEURALLS_CASE_CONFIG`. Case-driven commands such as
`generate-all`, `train-all`, `compare-all`, and `run-experiments` use their
passed case config path directly.

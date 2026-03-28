# CLI Module

The CLI package contains command entry points only.

## Package Map

- `process_data.py`: build one dataset from one dataset config
- `generate_multiple.py`: build all datasets declared in an experiments registry
- `train_model.py`: train one model on one dataset
- `train_multiple.py`: train a registry batch and emit aggregate reporting
- `predict.py`: run inference for one model/checkpoint pair
- `compare_preconditioners.py`: run comparison profiles for a registry
- `run_experiments.py`: end-to-end registry execution

## Boundary

CLI modules parse user input, print progress, and delegate immediately to
`neuralls.composition`. They must not load configs directly, instantiate
platform adapters directly, or contain business logic.

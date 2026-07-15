# CLI Module

The CLI package defines one public executable: `neuralls`.

## Public Surface

- `neuralls config ...`: manage machine-specific profiles
- `neuralls generate <case.toml>`: build every dataset declared in one case
- `neuralls generate-single <dataset.toml> --case-config <case.toml>`: build one dataset config
- `neuralls train <case.toml>`: train every assignment declared in one case
- `neuralls eval <case.toml>`: evaluate completed assignment checkpoints on their logged test splits
- `neuralls run <case.toml>`: generate datasets and train the full case matrix
- `neuralls compare <case.toml>`: run every comparison profile declared in one case after their benchmark datasets exist

## Package Map

- `main.py`: root Typer assembler for the public command surface
- `config.py`: profile management subcommands
- `generate.py`: case-wide dataset generation
- `generate_single.py`: explicit single-dataset generation
- `train.py`: case-wide training and aggregate reporting
- `eval.py`: case-wide checkpoint evaluation and aggregate reporting
- `run.py`: end-to-end case execution
- `compare.py`: case-wide solver benchmarking
- `options.py`: shared option aliases for profile and env-file resolution

## Boundary

CLI modules parse user input, print progress, resolve runtime settings at the
boundary, and delegate immediately to `neuralls.composition`. They must not
instantiate platform adapters directly or contain workflow business logic.

CLI owns argument parsing, top-level option semantics, and user-facing error
messages. It may resolve the active settings/profile context, but it should not
contain workflow assembly, filesystem policy, or service-integration logic.
Generation commands therefore render failures at the CLI boundary, while
lower layers supply the detailed operation/path context needed for diagnosis.

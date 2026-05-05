# CLI Module

The CLI package defines one public executable: `neuralls`.

## Public Surface

- `neuralls config ...`: manage machine-specific profiles
- `neuralls generate <case.toml>`: build every dataset declared in one case
- `neuralls generate-single <dataset.toml> --case-config <case.toml>`: build one dataset config
- `neuralls train <case.toml>`: train every experiment declared in one case
- `neuralls run <case.toml>`: generate datasets and train the full case matrix
- `neuralls compare <case.toml>`: run every comparison profile declared in one case

## Package Map

- `main.py`: root Typer assembler for the public command surface
- `config.py`: profile management subcommands
- `generate.py`: case-wide dataset generation
- `generate_single.py`: explicit single-dataset generation
- `train.py`: case-wide training and aggregate reporting
- `run.py`: end-to-end case execution
- `compare.py`: case-wide solver benchmarking
- `options.py`: shared option aliases for profile and env-file resolution

## Boundary

CLI modules parse user input, print progress, resolve runtime settings at the
boundary, and delegate immediately to `neuralls.composition`. They must not
instantiate platform adapters directly or contain workflow business logic.

The batch workflow commands accept one explicit case config positional argument
plus optional `--env-file` and `--profile` overrides. `neuralls
generate-single` accepts a dataset config plus `--case-config` to resolve settings.
This keeps settings discovery at the CLI boundary rather than inside workflow
orchestration code.

## Machine Configuration

Machine-specific roots are managed outside the repo with `neuralls config`.
Profiles live under the user config directory and provide:

- `raw_dir`: raw matrix and archive inputs
- `processed_dir`: processed datasets
- `output_dir`: MLflow state, checkpoints, figures, and reports

Typical flow:

```bash
uv run neuralls config init
uv run neuralls config create default --raw-dir /data/raw --processed-dir /data/processed --output-dir /data/output
uv run neuralls config list
uv run neuralls config show
uv run neuralls config set output-dir /new/output
uv run neuralls config delete laptop
```

Every public workflow command also accepts `--profile` / `-p` to override the active
`NEURALLS_PROFILE` selection for one invocation.

`neuralls config init` writes a commented starter file. `neuralls config create` is
non-interactive and expects explicit `--raw-dir`, `--processed-dir`, and
`--output-dir` flags.

`neuralls config set` overwrites an existing field value for an existing
profile. `neuralls config delete` removes a named profile; the `default`
profile is preserved.

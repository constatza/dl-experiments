import os  # noqa: D100

import numpy as np
import polars as pl
import torch
from tomlkit import dumps

from dlkit.io.settings import load_validated_settings
from dlkit.transforms import PCA, MinMaxScaler
from dlkit.transforms.chain import Pipeline


def _only_ascending(time: np.array) -> np.array:
    diff = np.diff(time)
    # find first negative difference
    non_positive_idx: int = np.nonzero(diff <= 0)[0][0].item()
    if non_positive_idx < len(diff) // 2:
        raise ValueError("More than half of the timesteps are non-positive.")
    return time[: non_positive_idx + 1]


def _repeat_time_until(time: np.array, target_timesteps: int):
    """Repeat the last element of the input array until it has the desired length."""
    residual_length = target_timesteps - len(time)
    repeated_last_element = np.tile(time[-1], residual_length)
    if residual_length < 0:
        raise ValueError(
            "Input array is already longer than or equal to the desired length."
        )

    return np.concatenate([time, repeated_last_element], axis=0)


def _process_time(
    time: np.array, target_timesteps: int, num_reps: int = 1
) -> (np.ndarray, np.ndarray):
    """Process the time array to ensure it has the desired length and repeat it if necessary."""
    dt = np.diff(_only_ascending(time))
    dt = np.append(dt, dt[-1])
    dt = _repeat_time_until(dt, target_timesteps)
    time = _repeat_time_until(time, target_timesteps)
    # repeat time to match the number of samples
    return np.tile(time, num_reps), np.tile(dt, num_reps)


def _read_solutions(variables: tuple[str, ...], input_dir: dict[str, str]) -> np.array:
    data = [np.load(f"{input_dir}{os.sep}{var}.npy") for var in variables]
    # concat along axis 2
    data = np.concatenate(data, axis=1)
    data = np.transpose(data, (0, 2, 1)).astype(np.float32)
    return data


def main():
    """Read solutions from files and preprocess the data."""
    variables = ("u", "p", "cox", "tcell")
    settings_path = "./config.toml"
    settings = load_validated_settings(settings_path)
    paths = settings.PATHS
    reduced_dims = 3
    solutions = _read_solutions(variables, paths.input_dir)
    N, T, D = solutions.shape
    parameters = np.load(paths.parameters)
    # repeat each parameter T times for each timestep
    parameters = np.repeat(parameters, T, axis=0)

    time = np.loadtxt(paths.time)
    time, dt = _process_time(time, T, N)

    solutions_flat = solutions.reshape(-1, D)

    chain = Pipeline(
        [
            MinMaxScaler(dim=[0, 1]),
            PCA(n_components=reduced_dims),
        ]
    )
    parameters_chain = Pipeline(
        shape=parameters.shape[1:], feature_transforms=[MinMaxScaler(dim=0)]
    )

    features_pca = chain.fit_transform(torch.from_numpy(solutions_flat)).numpy()
    parameters = parameters_chain.fit_transform(torch.from_numpy(parameters)).numpy()

    time_df = pl.LazyFrame(
        data={"time": time, "dt": dt},
        schema={"time": pl.datatypes.Float32, "dt": pl.datatypes.Float32},
    ).with_columns(
        [
            pl.arange(0, N * T).alias("sample") // T,
            pl.arange(0, N * T).alias("step") % T,
        ]
    )
    features_df = pl.LazyFrame(
        data=features_pca,
        schema={f"pc_{i}": pl.datatypes.Float32 for i in range(reduced_dims)},
    )
    params_df = pl.LazyFrame(
        data=parameters,
        schema={f"param_{i}": pl.datatypes.Float32 for i in range(parameters.shape[1])},
    )

    df = pl.concat([features_df, params_df, time_df], how="horizontal")

    # check if there are NA values
    nans = df.select(
        pl.sum_horizontal(pl.all().is_infinite()).alias("inf_count"),
        pl.sum_horizontal(pl.all().is_nan()).alias("nan_count"),
        pl.sum_horizontal(pl.all().is_null()).alias("null_count"),
    ).sum()

    num_invalids = nans.select(pl.sum_horizontal(pl.all())).collect().item()

    if num_invalids > 0:
        raise ValueError("DataFrame contains NA values.")

    # write to parquet file

    df.collect().write_parquet(
        settings.PATHS.input_dir / "pca.parquet",
        compression="snappy",
    )

    schema = {key: str(value) for key, value in df.collect_schema().items()}
    # write to toml

    with open(settings.PATHS.input_dir / "pca-schema.toml", "w") as f:
        f.write(dumps(schema))

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()

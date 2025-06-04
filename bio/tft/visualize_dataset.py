"""This script visualizes the dataset."""

import seaborn as sns
import polars as pl
from matplotlib import pyplot as plt

from dlkit.io.settings import load_validated_settings


if __name__ == "__main__":
    sns.set_theme("paper")

    config = load_validated_settings("./config.toml")
    df = pl.read_parquet(config.PATHS.features)

    g = sns.PairGrid(
        df,
        x_vars=["time"],
        y_vars=["pc_0", "pc_1", "pc_2"],
        hue="sample",
    )
    g.map(sns.lineplot, errorbar=None, estimator=None)

    f = sns.PairGrid(
        df, x_vars=["time"], y_vars=["param_0", "param_1", "param_2"], hue="sample"
    )
    f.map(sns.lineplot, errorbar=None, estimator=None)

    plt.tight_layout()
    plt.show()

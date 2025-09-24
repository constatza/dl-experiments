"""This script visualizes the dataset."""

import seaborn as sns
import polars as pl
from matplotlib import pyplot as plt

from dlkit.settings.environment import env as dl_env


if __name__ == "__main__":
    sns.set_theme("paper")

    root = dl_env.get_root_path()
    df = pl.read_parquet(root / "input" / "pca.parquet")

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

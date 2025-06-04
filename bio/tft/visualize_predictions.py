import seaborn as sns
import polars as pl

import torch
from matplotlib import pyplot as plt

from dlkit.io.settings import load_validated_settings


if __name__ == "__main__":
    sns.set_theme("paper")

    config = load_validated_settings("./config.toml")
    df = pl.read_parquet(config.PATHS.features)

    predictions = (
        torch.load(config.PATHS.predictions_dir / "predictions_0.pt").cpu().numpy()
    )

    plt.scatter(df["time"].to_numpy(), predictions)
    plt.show()

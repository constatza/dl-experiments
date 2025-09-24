import seaborn as sns
import polars as pl

import torch
from matplotlib import pyplot as plt

from pathlib import Path

from dlkit.io.locations import predictions_dir
from dlkit.settings.environment import env as dl_env


if __name__ == "__main__":
    sns.set_theme("paper")

    root = dl_env.get_root_path()
    input_dir = root / "input"
    df = pl.read_parquet(input_dir / "pca.parquet")

    pred_path = predictions_dir() / "predictions_0.pt"
    predictions = torch.load(pred_path).cpu().numpy()

    plt.scatter(df["time"].to_numpy(), predictions)
    plt.show()

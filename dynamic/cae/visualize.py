import matplotlib.pyplot as plt  # noqa: D100
import numpy as np
import torch

from pathlib import Path

from dlkit.settings.environment import env as dl_env
from dlkit.metrics.temporal import mase
from dlkit.postprocessing import plot_pred_vs_true, plot_residuals
from dlkit.io.index import load_split_indices

num_plots = 3
dof_idx = 99
variable = "U"
root = dl_env.get_root_path()
output_dir = root / "outputs"
input_dir = root / "input"
figures_dir = output_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

features = np.load(input_dir / "train_solutions.npy", mmap_mode="r")
parameters = np.load(input_dir / "train_model_params.npy", mmap_mode="r")
predictions = np.load(output_dir / "predictions.npy", mmap_mode="r")
idx_split = load_split_indices(input_dir / "idx_split.json")

latent = np.load(output_dir / "latent.npy", mmap_mode="r")
# stack first two axes
predictions = predictions.reshape(-1, predictions.shape[-2], predictions.shape[-1])
timesteps = np.arange(0, predictions.shape[-1])

sample_idx = np.random.choice(idx_split.test, num_plots, replace=False)


selected_features = features[sample_idx, dof_idx : dof_idx + 1, :]
selected_predictions = predictions[sample_idx, dof_idx : dof_idx + 1, :]


error = mase(
    torch.from_numpy(selected_predictions).float(),
    torch.from_numpy(selected_features).float(),
)
title = f"Variable: {variable}, MASE: {error:.4f}"
# common x-axis
fig, ax = plt.subplots(num_plots, 1, figsize=(10, 10), sharex=True)
fig.suptitle(title)
for i, idx in enumerate(sample_idx):
    ax[i].plot(timesteps, selected_features[i, 0, :].squeeze().T, label="Original")
    ax[i].plot(timesteps, selected_predictions[i, 0, :].squeeze().T, label="Predicted")
    ax[i].set_title(f"Sample {idx}")
    ax[i].legend()
    ax[i].set_xlabel("Timestep")
    ax[i].set_ylabel(f"{variable}")
# common x-axis
fig.savefig(figures_dir / f"{variable}_predictions.png", dpi=600)

fig = plot_pred_vs_true(selected_predictions.ravel(), selected_features.ravel(), title)
fig.savefig(figures_dir / f"{variable}_pred_vs_true.png", dpi=600)

fig = plot_residuals(selected_predictions.ravel(), selected_features.ravel(), title)
fig.savefig(figures_dir / f"{variable}_residuals.png", dpi=600)

# visualize latent space
# add figure with three 3d subplots
fig = plt.figure(figsize=(15, 10))
for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    scatter = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 0], c=parameters[:, 0])
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")
    # add colorbar
    cbar = plt.colorbar(scatter, ax=ax, location="bottom")
    cbar.set_label(f"Problem Parameter {i + 1}")
fig.suptitle("Latent Space")
fig.savefig(figures_dir / f"{variable}_latent_space.png", dpi=600)

plt.show()

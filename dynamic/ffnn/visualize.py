
import matplotlib.pyplot as plt
import numpy as np
import torch

from dlkit.postprocessing import plot_pred_vs_true, plot_residuals
from dlkit.metrics.temporal import mase

from dlkit.io.settings import load_validated_settings

num_plots = 3
dof_idx = 99
variable = "U"
config_path = "config.toml"
config = load_validated_settings(config_path)
paths = config.PATHS
output_dir = paths.figures
output_dir.mkdir(exist_ok=True, parents=True)


features = np.load(paths.features, mmap_mode="r")
plt.plot(features[:, 299, :].T)
predictions_all = np.load(paths.predictions, mmap_mode="r")
predictions = predictions_all["predictions"]
latent = predictions_all["latent"]
# stack first two axes
predictions = predictions.reshape(-1, predictions.shape[-2], predictions.shape[-1])
timesteps = np.arange(0, predictions.shape[-1])


# random num_plots indices from axis 0
sample_idx = np.random.randint(0, features.shape[0], num_plots)
# random num_plots indices from axis 1


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
    ax[i].plot(timesteps, selected_features[:, 0, :].T, label="Original")
    ax[i].plot(timesteps, selected_predictions[:, 0, :].T, label="Predicted")
    ax[i].set_title(f"Sample {idx}")
    ax[i].legend()
    ax[i].set_xlabel("Timestep")
    ax[i].set_ylabel(f"{variable}")
# common x-axis
fig.savefig(output_dir / f"{variable}_predictions.png", dpi=600)


fig = plot_pred_vs_true(selected_predictions.ravel(), selected_features.ravel(), title)
fig.savefig(output_dir / f"{variable}_pred_vs_true.png", dpi=600)


fig = plot_residuals(selected_predictions.ravel(), selected_features.ravel(), title)
fig.savefig(output_dir / f"{variable}_residuals.png", dpi=600)
plt.show()

# visualize latent space
# add figure with three 3d subplots
fig = plt.figure(figsize=(15, 10))
for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    scatter = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2])
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")
    # add colorbar
    cbar = plt.colorbar(scatter, ax=ax, location="bottom")
    cbar.set_label(f"Problem Parameter {i + 1}")
fig.suptitle("Latent Space")
fig.savefig(output_dir / f"{variable}_latent_space.png", dpi=600)

plt.show()

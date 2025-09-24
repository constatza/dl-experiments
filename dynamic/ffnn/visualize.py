# noqa: D100
import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path

from dlkit.settings.environment import env as dl_env


variable = "latent"
root = dl_env.get_root_path()
input_dir = root / "input"
output_dir = root / "outputs"

features = np.load(input_dir / "train_model_params.npy")
targets = np.load(output_dir / "latent.npy")
predictions = np.load(output_dir / "predictions_latent.npy")

plt.scatter(features[:, 0], targets[:, 0], label="True")
plt.scatter(features[:, 0], predictions[:, 0], label="Predicted")
plt.legend()
plt.show()

# noqa: D100
import matplotlib.pyplot as plt
import numpy as np


from dlkit.io.settings import load_validated_settings
from dlkit.io.index import load_split_indices


dof_idx = 99
variable = "latent"
config_path = "config.toml"
config = load_validated_settings(config_path)
idx_split = load_split_indices(config.PATHS.idx_split)
paths = config.PATHS

features = np.load(paths.features)
targets = np.load(paths.targets)
predictions = np.load(paths.predictions)

plt.scatter(features[:, 0], targets[:, 0], label="Targets")
plt.scatter(features[:, 0], predictions[:, 0], label="Predicted")
plt.legend()
plt.show()

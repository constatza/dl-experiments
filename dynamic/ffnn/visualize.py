# noqa: D100
import matplotlib.pyplot as plt
import numpy as np


from dlkit.io.settings import load_validated_settings


dof_idx = 99
variable = "latent"
config_path = "config.toml"
config = load_validated_settings(config_path)
paths = config.PATHS

features = np.load(paths.features)
targets = np.load(paths.targets)
predictions = np.load(paths.predictions)

plt.scatter(features[:, 0], targets[:, 0], label="Targets")
plt.scatter(features[:, 1], predictions[:, 0], label="Predicted")
plt.legend()
plt.show()

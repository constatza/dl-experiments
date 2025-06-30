# noqa: D100
import matplotlib.pyplot as plt
import numpy as np


from dlkit.io.settings import load_validated_settings


variable = "latent"
config_path = "config.toml"
config = load_validated_settings(config_path)
paths = config.PATHS

features = np.load(paths.features)
targets = np.load(paths.targets)
predictions = np.load(paths.predictions)

plt.scatter(features[:, 0], targets[:, 0], label="True")
plt.scatter(features[:, 0], predictions[:, 0], label="Predicted")
plt.legend()
plt.show()

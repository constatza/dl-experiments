"""Training script for graph-cg using new dlkit API."""

from dlkit.api import train
from dlkit.settings import GeneralSettings

if __name__ == "__main__":
    config_path = "./config-ffnn.toml"

    # Load configuration using new dlkit API
    settings = GeneralSettings.from_file(config_path)

    # Run training with MLflow strategy
    result = train(settings, strategy="mlflow")

    print(f"Training completed: {result}")
    print(f"Final metrics: {result.metrics}")

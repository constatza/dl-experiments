from dlkit.io.settings import load_validated_settings
from dlkit.run.mlflow_training import train_mlflow

if __name__ == "__main__":
    config = load_validated_settings("./config.toml")
    train_mlflow(config)

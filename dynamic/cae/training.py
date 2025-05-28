from dlkit.io.settings import load_validated_settings  # noqa: D100
from dlkit.run.optuna_training import train_optuna

if __name__ == "__main__":
    config_path = "./config.toml"
    settings = load_validated_settings(config_path)
    train_optuna(settings)

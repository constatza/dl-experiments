from dlkit.io.settings import load_validated_settings  # noqa: D100
from dlkit.run.training import train

if __name__ == "__main__":
    config_path = "./config.toml"
    settings = load_validated_settings(config_path)
    train(settings)

from dlkit.io.settings import load_validated_settings
from dlkit.scripts.training import train

if __name__ == "__main__":
    config = load_validated_settings("./config.toml")
    train(config)

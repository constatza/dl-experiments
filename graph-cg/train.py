from dlkit.run import run_from_path
from dlkit.datasets import get_dataset

if __name__ == "__main__":
    config_path = "./config.toml"
    graph_ds = get_dataset("GraphDataset")
    run_from_path(config_path)

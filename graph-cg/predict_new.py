"""Prediction script for graph-cg using new dlkit API."""

import matplotlib.pyplot as plt
from dlkit.api import infer
from dlkit.settings import GeneralSettings


def plot_pred_vs_true(y_hat, y, sample):
    """Plot predictions vs true values."""
    plt.figure(figsize=(10, 5))
    plt.scatter(y, y_hat)
    # plot y = x
    plt.plot(
        [min(y), max(y)],
        [min(y), max(y)],
        linestyle="dashed",
        color="orange",
        label="Identity",
    )
    plt.legend()
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.grid()

    plt.figure(figsize=(10, 5))
    plt.plot(y_hat, label="Predicted")
    plt.plot(y, label="True")
    plt.legend()
    plt.xlabel("Dof")
    plt.ylabel("Value")
    plt.title(f"Sample {sample}")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    config_path = "./config-ffnn.toml"

    # Load configuration using new dlkit API
    settings = GeneralSettings.from_file(config_path)

    # Run inference using new dlkit API
    result = infer(settings, checkpoint_path="./outputs/ffnn.ckpt")

    # Extract predictions and targets from result
    predictions = result.predictions

    # Get a sample for visualization
    if predictions:
        # Assuming predictions is a list of tensors or similar structure
        sample_idx = 0
        if isinstance(predictions, dict) and "y" in predictions:
            y_hat = predictions["y"][sample_idx].cpu().numpy()
            # Get corresponding true values from test dataset
            # This may need adjustment based on the actual result structure
            y = (
                result.targets["y"][sample_idx].cpu().numpy()
                if hasattr(result, "targets")
                else None
            )

            if y is not None:
                plot_pred_vs_true(y_hat, y, sample_idx)
            else:
                print("No target values available for comparison")
        else:
            print(f"Prediction result structure: {type(predictions)}")
            print(
                f"Available keys: {predictions.keys() if hasattr(predictions, 'keys') else 'N/A'}"
            )

    print(f"Inference completed: {result}")

import torch  # noqa: D100
from dlkit.run.training import train_state_from_path
from dlkit.utils.torch_utils import dataloader_to_tensor
from pydantic import FilePath
from matplotlib import pyplot as plt


def predict(
    ffnn_path: FilePath = "../ffnn/inference.toml",
    cae_path: FilePath = "../cae/inference.toml",
):
    # Load model and input array
    device = torch.device("cuda")

    ffnn_state, ffnn_config = train_state_from_path(ffnn_path)
    cae_state, cae_config = train_state_from_path(cae_path)

    ffnn = ffnn_state.model.to(device)
    cae = cae_state.model.to(device)

    ffnn.eval()
    cae.eval()
    test_loader_ffnn = ffnn_state.datamodule.predict_dataloader()
    test_loader_cae = cae_state.datamodule.predict_dataloader()

    x, _ = dataloader_to_tensor(test_loader_ffnn)
    y, _ = dataloader_to_tensor(test_loader_cae)

    x = x.to(device)
    y = y.to(device)

    with torch.inference_mode():
        x = ffnn(x)
        y_unscaled = cae.model.decode(x)
        y_hat = cae.targets_chain.inverse_transform(y_unscaled)

    return y_hat.cpu().numpy(), y.cpu().numpy(), x.cpu().numpy()


if __name__ == "__main__":
    predictions, targets, x = predict()
    plt.plot(predictions[0, 0, :], label="Predicted")
    plt.plot(targets[0, 0, :], label="True")
    plt.xlabel("Timesteps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

import torch  # noqa: D100
from pydantic import FilePath
from matplotlib import pyplot as plt
from dlkit.utils.torch_utils import dataloader_to_tensor
from dlkit.run import run_from_path


def predict(
    ffnn_path: FilePath = "../ffnn/config.toml",
    cae_path: FilePath = "../cae/config.toml",
):
    # Load model and input array
    device = torch.device("cuda")

    ffnn_state = run_from_path(ffnn_path, mode="inference")
    cae_state = run_from_path(cae_path, mode="inference")

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
    dof = 100
    predictions, targets, x = predict()
    plt.plot(targets[0, dof, :], label="True")
    plt.plot(predictions[0, dof, :], label="Predicted")
    plt.xlabel("Timesteps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

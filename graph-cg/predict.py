import torch
from matplotlib import pyplot as plt
from dlkit.settings import Settings
from dlkit.run import run


if __name__ == "__main__":
    settings_path = "./config.toml"
    settings = Settings.from_file(settings_path)
    train_state = run(
        settings,
        mode="inference",
        checkpoint=settings.PATHS.output_dir / "graph.ckpt",
    )
    model = train_state.model.to("cuda")
    dataset = train_state.datamodule.dataset

    dataloader = train_state.datamodule.test_dataloader()
    model.eval()
    data = next(iter(dataloader)).to("cuda")
    sample = torch.randint(low=0, high=len(dataset) - 1, size=(1,))[0]
    # data = dataset[sample].to("cuda")
    with torch.inference_mode():
        y_hat = model(data).cpu().numpy()
        y = data.y.cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.scatter(y, y_hat)
    # plot y =x
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

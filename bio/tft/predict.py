from pytorch_forecasting import TemporalFusionTransformer  # noqa: D100

from dlkit.io.settings import load_validated_settings
from dlkit.run.vanilla_training import build_training_state


if __name__ == "__main__":
    settings = load_validated_settings("./config.toml")
    training_state = build_training_state(settings)

    module = training_state.datamodule

    module.setup("test")
    inference_loader = module.test_dataloader()
    best_tft: TemporalFusionTransformer = training_state.model

    raw_predictions = best_tft.predict(
        inference_loader,
        return_predictions=True,
        return_x=True,
        return_y=True,  # optionally get true values if you have them
        # output_dir=settings.PATHS.predictions_dir,
    )

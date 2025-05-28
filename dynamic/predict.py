import json
import sys

import numpy as np  # noqa: D100
import torch

from dlkit.networks.blocks.base import PipelineNetwork
from loguru import logger


def main(features_path, solution_path, decoder_path, ffnn_path):
    # Load model and input array
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cae = PipelineNetwork.load_from_checkpoint(decoder_path)
    ffnn = PipelineNetwork.load_from_checkpoint(ffnn_path)
    cae = cae.to(device)
    ffnn = ffnn.to(device)

    ffnn.eval()
    cae.eval()
    x = np.load(features_path)
    with torch.no_grad():
        x = torch.from_numpy(x).float().to(device)
        x = ffnn(x)
        y = cae.model.decoder(x).cpu().numpy()

    np.save(solution_path, np.squeeze(y))
    logger.info(f"Prediction saved to {solution_path}")


# click.command("main")
# click.argument("settings_path", type=str, help="Path to the settings file")
# click.argument("results_file", type=str, help="Path to the timings file")
# click.argument("log_file", type=str, help="Path to the error log file")
def main_cli(
    # settings_path: str = r"M:\shared\Serafeim_Atzarakis\results\CantileverDynamicLinear\settings_sample.json",
    settings_path,
    results_file,
    log_file,
):
    """Main function to run the prediction script.

    Args:
        settings_path (str): Path to the settings file.
    """
    with open(settings_path) as f:
        settings = json.load(f)
    main(
        features_path=settings["ModelParamsPath"],
        solution_path=settings["SolutionVectorPath"],
        decoder_path=settings["ModelDecoderPath"],
        ffnn_path=settings["ModelFfnnPath"],
    )


# region debug
if __name__ == "__main__":
    # Actual script
    # parent = Path(r"M:\shared\Serafeim_Atzarakis\results\CantileverDynamicLinear")
    # main(
    #     features_path=parent / "test_model_params.npy",
    #     solution_path=parent / "test_solutions.npy",
    #     decoder_path=parent / r"checkpoints\cae.ckpt",
    #     ffnn_path=parent / r"checkpoints\ffnn.ckpt",
    # )
    path_settings = sys.argv[1]
    results_file = sys.argv[2]
    log_file = sys.argv[3]
    main_cli(path_settings, results_file, log_file)

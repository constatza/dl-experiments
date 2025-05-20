import json
import shutil

import numpy as np
from pathlib import Path


def get_dofs(data_path: Path):
    return (
        np.loadtxt(data_path / "u-dofs.txt").astype(np.int64).flatten(),
        np.loadtxt(data_path / "p-dofs.txt").astype(np.int64).flatten(),
    )


def porous_from_u_and_p(dofs_path: Path, u, p):
    udofs, pdofs = get_dofs(dofs_path)
    total_dofs = len(udofs) + len(pdofs)
    up = np.zeros((u.shape[0], total_dofs, u.shape[-1]))
    up[:, udofs, :] = u
    up[:, pdofs, :] = p
    return up


def remove_u_and_p():
    del data_mmap["u"]
    del data_mmap["p"]


if __name__ == "__main__":
    num_samples = 10
    mode = "x4"
    input_path = Path(r"M:\constantinos\data\bio\5-equations\input")
    output_path = Path(r"M:\constantinos\data\bio\5-equations\output")
    dofs_path = input_path / "dofs"
    idx_split_path = input_path / "idx_split.json"
    predictions_dir = output_path / "predictions"
    msolve_indir = output_path / "msolve-input"
    msolve_outdir = output_path / "msolve-output" / mode

    with open(idx_split_path) as f:
        test_indices = json.load(f)["test"]
    samples = np.random.choice(test_indices, num_samples, replace=False)

    names = ["u", "p", "cox", "tcell", "porous"]
    data_paths = [predictions_dir / f"{p}-predictions.npy" for p in names]
    paths_dict = dict(zip(names, data_paths))

    paths_dict["parameters"] = input_path / "parameters.npy"

    data_mmap = {}
    for variable, path in paths_dict.items():
        if path.exists():
            data_mmap[variable] = np.load(path, mmap_mode="r")
        elif variable == "porous":
            data_mmap["porous"] = porous_from_u_and_p(
                dofs_path, data_mmap["u"], data_mmap["p"]
            )
            np.save(paths_dict["porous"], data_mmap["porous"])
    remove_u_and_p()

    output_filenames = {
        variable: f"{variable}Solutions_AnalysisNo_x.txt"
        for variable in data_mmap.keys()
    }
    output_filenames["parameters"] = "parameters_miTumor_kth_tumor_sv_AnalysisNo_x.txt"

    # delete old dir and create new dir
    for directory in [msolve_indir]:
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        sample_dir = msolve_indir / str(sample)
        out_dir = msolve_outdir / str(sample)
        # delete old dir and create new dir
        for directory in [sample_dir, out_dir]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)

        for variable, array in data_mmap.items():
            sample_data = np.atleast_2d(array[sample, ...])
            if variable != "parameters":
                sample_data = sample_data.T
            path = sample_dir / output_filenames[variable]
            np.savetxt(path, sample_data)

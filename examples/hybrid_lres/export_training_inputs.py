"""Author-only export of accepted tables/config/experimental source metadata.

    python export_training_inputs.py --workspace /author/workspace --output inputs

This reads existing accepted outputs; it does not fit or modify the source study.
Large table/covariance files are copied only if not already at the destination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(workspace, output):
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(workspace / "paper_local_checks_20261019"))
    import queue_models

    _, bank = queue_models.full_bank()
    exp = bank["exp"]
    manager = exp["correlation_manager"].to_dict()
    params = manager["_pending_correlation_params"]
    unknown = [tag for tag in manager["correlation_groups"]
               if str(params.get(tag, "")).partition(":")[0].lower() in ("jetscape", "ptgauss")]
    experiment = {key: exp[key] for key in ("y", "y_err_stat", "y_err_syst", "systematic_names",
                  "observable_ranges", "per_observable_external_stat_cov") if key in exp}
    experiment["correlation_manager"] = manager
    meta = dict(schema="hybrid-lres-training-inputs-v1", features=bank["features"],
                experiment=experiment, unknown_shape_tags=unknown,
                coordinate_convention="Original full-measurement normalized bin centers, before cuts")
    path = output / "experiment_sources.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(clean(meta), indent=2, allow_nan=False) + "\n")
    native = workspace / "paper_local_checks_20261019/models/all/configs/main.yaml"
    config = yaml.safe_load(native.read_text())
    config["output_dir"] = "OUTPUT"
    config["observable_table_dir"] = "INPUTS"
    config["observable_config_dir"] = "INPUTS"
    for settings in config["analyses"]["analysis_raa_unit"]["parameters"]["emulators"].values():
        settings["simulation_noise"]["covariance_file"] = "full_training.npz"
    (output / "native_template.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    root = workspace / ("atlas_rg_inference_20261015/recovery_20261016/precision_continuation/timeout_completion")
    source = root / "local_inference/inference/grid.npz"
    with np.load(source, allow_pickle=False) as z:
        np.savez_compressed(output / "grid_axes.npz", q=z["q"], kappa=z["kappa"])
    sources = {"accepted_observables.h5": root / "local_inference/results/main/analysis_raa_unit_Lres-E-loss/observables.h5",
               "full_training.npz": root / "export/full_simulation_covariance_rg.npz"}
    for name, src in sources.items():
        dest = output / name
        if not dest.exists():
            shutil.copyfile(src, dest)
        if digest(dest) != digest(src):
            raise ValueError("Wrong existing training input: " + name)
    # Reference results are separate from the inputs and never used for fitting.
    with np.load(workspace / "paper_local_checks_20261019/queue/covariances.npz", allow_pickle=False) as z:
        np.savez_compressed(output / "reference_covariances.npz", **{key: z[key] for key in z.files})
    receipt = dict(schema=meta["schema"], files={p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                   for p in output.iterdir() if p.name in ("accepted_observables.h5", "full_training.npz",
                   "experiment_sources.json", "native_template.yaml", "grid_axes.npz", "reference_covariances.npz")},
                   source_native_config_sha256=digest(native),
                   reference_files_not_training_inputs=["reference_covariances.npz"])
    (output / "TRAINING_INPUTS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.workspace.resolve(), args.output.resolve()), indent=2))

"""Verify and summarize saved Darcy Rhizome/FNO runs at widths 8 and 16.

Run only after the new run has completed checkpoint selection and test export.
This script performs no training and evaluates metrics from saved predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SHARED_CONFIG = (
    "steps", "batch_size", "modes", "updates", "step_size", "lr", "val_every",
    "n_val", "seed", "split_seed", "threads", "resolution", "boundary",
)


def metrics(prediction, truth):
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    truth = truth.astype(np.float64)
    norms = np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
    if np.any(norms == 0):
        raise ValueError("relative L2 requires nonzero target norms")
    relative = np.linalg.norm(error.reshape(len(error), -1), axis=1) / norms
    if not np.isfinite(relative).all():
        raise ValueError("non-finite prediction metrics")
    return relative, {
        "n_fields": len(truth), "relative_l2_mean": float(relative.mean()),
        "relative_l2_median": float(np.quantile(relative, 0.5)),
        "relative_l2_p95": float(np.quantile(relative, 0.95)),
        "rmse_stored_units": float(np.sqrt(np.mean(error ** 2))),
        "mae_stored_units": float(np.mean(np.abs(error))),
    }


def load_run(path):
    metadata = json.loads((path / "metadata.json").read_text())
    result = json.loads((path / "results.json").read_text())
    with np.load(path / "test_predictions.npz", allow_pickle=False) as arrays:
        saved = {k: arrays[k].copy() for k in ("coefficient", "truth", "prediction", "row")}
    relative, checked = metrics(saved["prediction"], saved["truth"])
    for key, value in checked.items():
        if not np.isclose(value, result["test"][key], rtol=1e-12, atol=1e-12):
            raise ValueError(f"saved metrics disagree for {path}: {key}")
    return metadata, result, saved, relative


def compare(new_path, rhizome16_path, fno8_path, output):
    paths = {"rhizome_w8": new_path, "rhizome_w16": rhizome16_path, "fno_w8": fno8_path}
    runs = {key: load_run(path) for key, path in paths.items()}
    current = runs["rhizome_w8"]
    if current[0]["config"]["width"] != 8 or current[0]["config"].get("architecture", "rhizome") != "rhizome":
        raise ValueError("new run must be a width-8 Rhizome")
    if runs["rhizome_w16"][0]["config"]["width"] != 16:
        raise ValueError("Rhizome reference must have width 16")
    fno_config = runs["fno_w8"][0]["config"]
    if fno_config["width"] != 8 or fno_config["architecture"] != "fno":
        raise ValueError("FNO reference must have width 8")
    for name in ("rhizome_w16", "fno_w8"):
        other = runs[name]
        for key in ("files_sha256", "train_indices", "val_indices", "stats"):
            if current[0][key] != other[0][key]:
                raise ValueError(f"uncontrolled {key}: {name}")
        for key in SHARED_CONFIG:
            if current[0]["config"][key] != other[0]["config"][key]:
                raise ValueError(f"uncontrolled config {key}: {name}")
        if current[1]["samples_seen"] != other[1]["samples_seen"]:
            raise ValueError(f"training exposure differs: {name}")
        for key in ("row", "truth", "coefficient"):
            if not np.array_equal(current[2][key], other[2][key]):
                raise ValueError(f"test data/ordering differs for {name}: {key}")

    summary = {"comparability_checks_passed": True, "shared_config": {
        key: current[0]["config"][key] for key in SHARED_CONFIG},
        "dataset_sha256": current[0]["files_sha256"],
        "n_train": current[0]["n_train"], "n_validation": current[0]["n_validation"],
        "samples_seen": current[1]["samples_seen"], "runs": {}, "paired_comparisons": {},
        "interpretation": "One training seed; equal width and optimizer steps do not imply equal parameters or compute. Timings come from separate runs and are observational.",
        "test_status": "Previously inspected public test set; exploratory capacity comparison, not a fresh confirmatory holdout."}
    for name, (meta, result, _, _) in runs.items():
        summary["runs"][name] = {
            "path": str(paths[name].resolve()), "width": meta["config"]["width"],
            "source_sha256": meta["source_sha256"],
            "parameters_real": result["parameters_real"], "best_step": result["best_step"],
            "best_validation_relative_l2": result["best_validation_relative_l2"],
            "train_seconds_including_validation": result["train_seconds_including_validation"],
            "test": result["test"]}
    current_error = current[3]
    for name in ("rhizome_w16", "fno_w8"):
        other_error = runs[name][3]
        summary["paired_comparisons"][f"rhizome_w8_vs_{name}"] = {
            "mean_relative_l2_reduction_fraction": float(1 - current_error.mean() / other_error.mean()),
            "mean_relative_l2_absolute_difference": float((current_error - other_error).mean()),
            "fields_with_lower_error": int(np.sum(current_error < other_error)),
            "fields_with_equal_error": int(np.sum(current_error == other_error)),
            "fields_with_higher_error": int(np.sum(current_error > other_error)),
            "parameter_ratio": current[1]["parameters_real"] / runs[name][1]["parameters_real"],
            "recorded_training_time_ratio": current[1]["train_seconds_including_validation"] / runs[name][1]["train_seconds_including_validation"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "per_field_relative_l2.npz", row=current[2]["row"],
                        **{name: values[3] for name, values in runs.items()})
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhizome8", type=Path, required=True)
    parser.add_argument("--rhizome16", type=Path, required=True)
    parser.add_argument("--fno8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.rhizome8, args.rhizome16, args.fno8, args.out)


if __name__ == "__main__":
    main()

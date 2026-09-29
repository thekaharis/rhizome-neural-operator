import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ebm21cm.data.cache import SliceDataset
from ebm21cm.model import RecurrentFNO2d, RhizomeOperator2d
from ebm21cm.train_recurrent import (
    ExperimentConfig,
    field_metrics,
    load_checkpoint,
    predict,
    run_experiment,
)


def test_field_metrics_separate_mixed_phases():
    truth = np.array([[[0, 1], [0, 1]], [[1, 1], [1, 1]]], dtype=float)
    result = field_metrics(truth, truth)
    assert result["all"]["n_slices"] == 2
    assert result["mixed"]["n_slices"] == 1
    assert result["all"]["rmse"] == 0
    assert result["all"]["ionized_iou"] == 1
    neutral = field_metrics(np.ones((1, 2, 2)), np.ones((1, 2, 2)))
    assert neutral["all"]["ionized_iou"] is None
    assert neutral["mixed"] == {"n_slices": 0}
    json.dumps(neutral, allow_nan=False)


@pytest.mark.parametrize("options", [
    {"steps": 0}, {"batch_size": 0}, {"modes": -1}, {"updates": 0},
    {"lr": float("nan")}, {"step_size": 0}, {"step_size": 1.1},
    {"threads": 0}, {"seed": -1},
])
def test_invalid_experiment_config(options):
    with pytest.raises(ValueError):
        ExperimentConfig(**options).validate()


@pytest.mark.parametrize("variants", [("recurrent", "untied"), ("rhizome", "rhizome_untied")])
def test_recurrent_training_checkpoint_and_evaluation(toy_cache, tmp_path, variants):
    run = tmp_path / "run"
    tied, untied = variants
    # Deliberately larger than the training split: never drop the only batch.
    config = ExperimentConfig(steps=4, batch_size=100, width=4, modes=2, updates=2,
                              val_every=2, threads=1)
    result = run_experiment(toy_cache, run, config, variants=variants)
    assert (run / "examples.png").exists()
    assert (run / tied / "rollout.png").exists()
    assert json.loads((run / "results.json").read_text()) == result
    metadata = json.loads((run / "metadata.json").read_text())
    assert set(metadata["splits"]["train"]["cones"]).isdisjoint(metadata["splits"]["test"]["cones"])
    assert result[untied]["parameters_real"] > result[tied]["parameters_real"]
    for variant in variants:
        report = result[variant]
        assert np.isfinite(report["best_validation_bce"])
        assert report["best_validation_bce"] < report["initial_validation_bce"]
        assert report["best_step"] > 0
        assert len(report["history"]) == 2
        assert 0 <= report["test"]["all"]["rmse"] <= 1
        model, checkpoint = load_checkpoint(run / variant / "best.pt")
        expected_class = RhizomeOperator2d if tied == "rhizome" else RecurrentFNO2d
        assert isinstance(model, expected_class)
        ds = SliceDataset(toy_cache, "test", checkpoint["stats"])
        fields, _ = predict(model, DataLoader(ds, batch_size=100), torch.device("cpu"))
        with np.load(run / variant / "test_predictions.npz") as stored:
            for key, value in fields.items():
                assert np.array_equal(value, stored[key])
    assert set(result[tied]["iteration_sweep"]) == {"1", "2", "4"}
    assert result[tied]["rollout"]["max_abs_state"] <= 1.000001
    assert "iteration_sweep" not in result[untied]
    with pytest.raises(ValueError, match="not empty"):
        run_experiment(toy_cache, run, config)


def test_unsupported_variants_and_device_fail_before_creating_run(toy_cache, tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="distinct variants"):
        run_experiment(toy_cache, run, variants=("recurrent", "recurrent"))
    with pytest.raises(ValueError, match="cpu or cuda"):
        run_experiment(toy_cache, run, device="mps")
    assert not run.exists()


def test_legacy_checkpoint_without_architecture_tag(tmp_path):
    config = {"in_channels": 1, "width": 4, "modes": 2, "n_steps": 2}
    model = RecurrentFNO2d(**config).eval()
    path = tmp_path / "legacy.pt"
    checkpoint = {"model": model.state_dict(), "model_config": config, "stats": {}}
    torch.save(checkpoint, path)
    loaded, _ = load_checkpoint(path)
    x = torch.randn(2, 1, 8, 8)
    assert isinstance(loaded, RecurrentFNO2d)
    assert torch.equal(model(x), loaded(x))
    checkpoint["architecture"] = "unrecognized"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="unknown checkpoint architecture"):
        load_checkpoint(path)

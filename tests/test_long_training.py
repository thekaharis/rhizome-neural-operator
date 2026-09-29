from dataclasses import replace
import json

import pytest
import torch

from ebm21cm.train_rhizome import LongRunConfig, PlateauState, run


def test_plateau_reduces_lr_then_waits_at_floor():
    config = LongRunConfig(min_steps=6, min_lr=0.0015, lr_patience=2,
                           stop_patience=3, relative_min_delta=0.01)
    state = PlateauState(best=1.0, reference=1.0)
    lr, stop, improved = state.observe(0.8, 1, config.lr, config)
    assert improved and not stop and state.bad_checks == 0
    lr, stop, _ = state.observe(0.8, 2, lr, config)
    assert lr == config.lr and not stop
    lr, stop, _ = state.observe(0.8, 3, lr, config)
    assert lr == config.min_lr and state.bad_checks == 0 and not stop
    for step in (4, 5):
        lr, stop, _ = state.observe(0.8, step, lr, config)
        assert not stop
    _, stop, _ = state.observe(0.8, 6, lr, config)
    assert stop and state.reductions == 1


def test_small_improvements_update_checkpoint_and_accumulate_against_reference():
    config = LongRunConfig(relative_min_delta=0.01, lr_patience=4)
    state = PlateauState(best=1.0, reference=1.0)
    _, _, improved = state.observe(0.995, 1, config.lr, config)
    assert improved and state.best == 0.995 and state.reference == 1.0 and state.bad_checks == 1
    _, _, improved = state.observe(0.989, 2, config.lr, config)
    assert improved and state.reference == 0.989 and state.bad_checks == 0
    with pytest.raises(FloatingPointError):
        state.observe(float("nan"), 3, config.lr, config)


@pytest.mark.parametrize("kwargs", [
    {"min_steps": 0}, {"min_steps": 21_000}, {"min_lr": 0}, {"min_lr": 0.01},
    {"lr_factor": 1}, {"stop_patience": 0}, {"relative_min_delta": 1},
    {"lr_patience": -1}, {"train_probe_rows": 0},
])
def test_invalid_long_run_config(kwargs):
    with pytest.raises(ValueError):
        LongRunConfig(**kwargs).validate()


def tiny_config(**kwargs):
    return replace(LongRunConfig(max_steps=8, min_steps=2, batch_size=5, width=4, modes=2,
                                 updates=2, val_every=3, lr_patience=20, stop_patience=20,
                                 train_probe_rows=4, threads=1), **kwargs)


def _checkpoint(path):
    return torch.load(path / "last.pt", weights_only=True, map_location="cpu")


def _assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_equal(x, y)
    else:
        assert a == b


def test_resume_between_validations_matches_uninterrupted_training(toy_cache, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    config = tiny_config()
    complete = run(toy_cache, full, config)
    part = run(toy_cache, resumed, replace(config, max_steps=4))
    assert not part["test_evaluated"] and part["stop_reason"] == "max_steps"
    assert not (resumed / "test_predictions.npz").exists()
    continued = run(toy_cache, resumed, config, resume=True)
    assert continued["best_validation_bce"] == complete["best_validation_bce"]
    a, b = _checkpoint(full), _checkpoint(resumed)
    for key in ("model", "optimizer", "best_model", "plateau", "stream", "history",
                "torch_rng", "step", "samples_seen", "window_loss", "window_pixels"):
        _assert_equal(a[key], b[key])
    assert not complete["test_evaluated"] and not continued["test_evaluated"]
    assert (resumed / "learning_curves.png").exists()
    with pytest.raises(ValueError, match="differs"):
        run(toy_cache, resumed, replace(config, lr=0.002), resume=True)
    with pytest.raises(ValueError, match="only increase"):
        run(toy_cache, resumed, replace(config, max_steps=7), resume=True)
    # Simulate interruption at step 8 of a previously requested 20-step run:
    # a new budget of 10 is above the saved step but below the prior budget.
    b["max_steps_requested"] = 20
    torch.save(b, resumed / "last.pt")
    with pytest.raises(ValueError, match="only increase"):
        run(toy_cache, resumed, replace(config, max_steps=10), resume=True)
    with pytest.raises(ValueError, match="not empty"):
        run(toy_cache, resumed, config)


def test_test_fields_are_only_evaluated_after_stopping_criterion(toy_cache, tmp_path, monkeypatch):
    from ebm21cm.data.cache import SliceDataset

    original = SliceDataset.raw
    test_reads = []

    def record_reads(self, row):
        if self.cone_id[row] == test_cone:
            test_reads.append(row)
        return original(self, row)

    test_cone = SliceDataset(toy_cache, "test").cone_id[0]
    monkeypatch.setattr(SliceDataset, "raw", record_reads)
    # Deliberately coarse threshold for a fast mechanics test, NOT the actual
    # training protocol. Rate drops at step 1, then a fresh check stops at 2.
    config = tiny_config(max_steps=4, val_every=1, lr_patience=1, stop_patience=1,
                         relative_min_delta=0.99, min_lr=0.0015)
    result = run(toy_cache, tmp_path / "stopped", config)
    assert result["criterion_met"] and result["test_evaluated"]
    assert result["stop_reason"] == "validation_plateau_at_min_lr"
    assert result["steps"] == 2 and result["final_lr"] == config.min_lr
    assert test_reads
    test_reads.clear()
    budget = tiny_config(max_steps=2)
    result = run(toy_cache, tmp_path / "budget", budget)
    assert not result["test_evaluated"] and test_reads == []
    saved = json.loads((tmp_path / "stopped" / "results.json").read_text())
    assert saved["test"]["all"]["n_slices"] > 0


def test_resume_requires_checkpoint(toy_cache, tmp_path):
    with pytest.raises(ValueError, match="existing last.pt"):
        run(toy_cache, tmp_path / "missing", tiny_config(), resume=True)


def test_recover_metadata_only_initialization(toy_cache, tmp_path):
    config = tiny_config(max_steps=2)
    original, recovered = tmp_path / "original", tmp_path / "recovered"
    run(toy_cache, original, config)
    recovered.mkdir()
    (recovered / "metadata.json").write_text((original / "metadata.json").read_text())
    (recovered / "last.pt.tmp").write_bytes(b"interrupted initial checkpoint")
    result = run(toy_cache, recovered, config, resume=True)
    assert result["steps"] == 2
    _assert_equal(_checkpoint(original)["model"], _checkpoint(recovered)["model"])

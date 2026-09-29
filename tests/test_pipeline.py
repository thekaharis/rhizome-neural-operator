"""End to end on toy data: train a few steps, resume, sample, evaluate."""

import json

import h5py
import numpy as np
import pytest

from ebm21cm import evaluate, sample, train


def _cfg(tmp_path, steps):
    cfg = {"model": {"ch": 16, "mults": [1, 2], "num_res": 1, "attn_levels": [1], "heads": 2},
           "train": {"batch_size": 4, "steps": steps, "warmup": 2, "workers": 0, "log_every": 1,
                     "val_every": 2, "val_rows": 4, "ckpt_every": 2, "sample_every": 4,
                     "sample_rows": 2, "sample_steps": 3}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_train_resume_sample_evaluate(toy_cache, tmp_path):
    run = tmp_path / "run"
    train.main(["--cache", str(toy_cache), "--run-dir", str(run), "--config", str(_cfg(tmp_path, 4)),
                "--device", "cpu"])
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    assert list((run / "samples").glob("*.png"))
    with pytest.raises(SystemExit):  # refuses to overwrite silently
        train.main(["--cache", str(toy_cache), "--run-dir", str(run), "--device", "cpu"])
    train.main(["--cache", str(toy_cache), "--run-dir", str(run), "--config", str(_cfg(tmp_path, 6)),
                "--device", "cpu", "--resume"])
    recs = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in recs if "loss" in r]
    assert steps == [1, 2, 3, 4, 5, 6]

    # Candidates: the truth itself should get exactly the truth's energy.
    with h5py.File(toy_cache, "r") as h:
        rows = np.flatnonzero(h["split"][:] == 2)[:2]
        cand = tmp_path / "cand.h5"
        with h5py.File(cand, "w") as c:
            c["row"] = rows
            c["xhi"] = h["xhi"][rows].astype(np.float32)
            c["tb"] = h["tb"][rows].astype(np.float32)
    out = tmp_path / "samples.h5"
    sample.main(["--run-dir", str(run), "--n-rows", "3", "--n-samples", "3", "--batch-size", "2",
                 "--steps", "3", "--mala-steps", "1", "--candidates", str(cand), "--device", "cpu",
                 "--out", str(out)])
    with h5py.File(out, "r") as h:
        assert h["xhi_samples"].shape == (3, 3, 16, 16)
        has = h["has_candidate"][:]
        assert has.any()
        assert np.allclose(h["E_candidate"][:][has], h["E_truth"][:][has], rtol=1e-4)
        assert (h["xhi_samples"][:] >= 0).all() and (h["xhi_samples"][:] <= 1).all()
    res, d = evaluate.evaluate(out, out_dir=tmp_path / "eval", n_rays=200)
    assert (d / "metrics.json").exists() and (d / "examples.png").exists() and (d / "energy.png").exists()
    assert res["overall"]["n_slices"] == 3 and res["overall"]["xhi_crps"] is not None

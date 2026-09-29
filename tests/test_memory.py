"""Reionization-window sampling and the in-memory, device-side data path."""

import json

import h5py
import numpy as np
import pytest
import torch

from ebm21cm.data import cache
from ebm21cm.data.memory import MemorySplit, MemoryStream, dihedral, parse_channels
from ebm21cm.train_recurrent import load_checkpoint
from ebm21cm.train_rhizome import LongRunConfig, run, train_cone_subset

from test_long_training import _assert_equal, _checkpoint, tiny_config


def test_window_picks_are_stratified_in_level():
    # Fine enough that nearest matches stay inside their stratum.
    z = np.linspace(5, 25, 20000)
    # A history that crawls towards 1: index strata would crowd the top levels.
    history = np.clip((z - 5) / 3, 0, 1) ** 0.2
    rng = np.random.default_rng(0)
    idx = cache.pick_window_indices(z, history, 4, 4, 8, 3, (0.1, 0.9), None, rng)
    inside = idx[(history[idx] >= 0.1) & (history[idx] <= 0.9)]
    assert len(inside) == 8 and len(idx) == 11
    assert np.all(np.diff(idx) > 0) and idx.min() >= 4 and idx.max() < len(z) - 4
    # One pick per equal-width stratum of the window's x_HI range.
    window = history[4:len(z) - 4]
    window = window[(window >= 0.1) & (window <= 0.9)]
    edges = np.linspace(window.min(), window.max(), 9)
    assert np.all(np.histogram(history[inside], edges)[0] == 1)
    never = np.ones_like(z)
    assert len(cache.pick_window_indices(z, never, 4, 4, 8, 3, (0.1, 0.9), None, rng)) == 3


def test_window_cache_records_options(toy_dir, tmp_path):
    files = cache.discover(toy_dir, "*.h5")
    kw = dict(slices_per_cone=3, bands="-2,0,2", xhi_window=(0.2, 0.8), background_slices=1,
              chunk_cache_bytes=2**22, log=lambda *a: None)
    single = cache.build(files, tmp_path / "single.h5", **kw)
    shards = [cache.build(files, tmp_path / f"s{i}.h5", shard=i, n_shards=2, **kw) for i in range(2)]
    merged = cache.merge(shards, tmp_path / "merged.h5", log=lambda *a: None)
    for path in (single, merged):
        with h5py.File(path, "r") as h:
            assert json.loads(h.attrs["xhi_window"]) == [0.2, 0.8] and h.attrs["background_slices"] == 1
            counts = np.unique(h["cone_id"][:], return_counts=True)[1]
            assert counts.max() <= 4


@pytest.mark.parametrize("spec,expected", [("all", [0, 1, 2, 3, 4]), ("center", [2]),
                                           ("singles", [1, 2, 3]), ("0,2", [0, 2])])
def test_parse_channels(spec, expected):
    assert parse_channels(spec, [(-8, -4), (-1, 0), (0, 1), (1, 2), (4, 8)]) == expected


def test_parse_channels_rejects_bad_indices():
    with pytest.raises(ValueError):
        parse_channels("0,0", [(0, 1), (1, 2)])
    with pytest.raises(ValueError):
        parse_channels("5", [(0, 1), (1, 2)])


def test_memory_split_matches_slice_dataset(toy_cache):
    stats = cache.read_stats(toy_cache)
    ds = cache.SliceDataset(toy_cache, "validation", stats)
    mem = MemorySplit(toy_cache, "validation", stats)
    assert len(mem) == len(ds) and mem.center_band == ds.center_band
    batch = mem.batch(torch.arange(len(mem)))
    for j in range(len(ds)):
        item = ds[j]
        assert torch.allclose(batch["cond"][j], item["cond"], atol=1e-4)
        assert torch.allclose(batch["target"][j, 0], item["target"][0], atol=1e-6)
        assert torch.allclose(batch["scalars"][j], item["scalars"], atol=1e-6)
        assert int(batch["row"][j]) == item["row"] and int(batch["cone_id"][j]) == item["cone_id"]
    sub = MemorySplit(toy_cache, "validation", stats, channels="center")
    assert sub.n_cond == 1 and torch.allclose(sub.batch([0])["cond"][0, 0], batch["cond"][0, ds.center_band])


def test_memory_augmentation_is_joint_per_sample(toy_cache):
    mem = MemorySplit(toy_cache, "train", cache.read_stats(toy_cache))
    positions = torch.arange(8)
    plain = mem.batch(positions)
    codes = torch.arange(8)
    augmented = mem.batch(positions, codes)
    for j, code in enumerate(codes.tolist()):
        for key in ("cond", "target"):
            assert torch.equal(augmented[key][j], dihedral(plain[key][j], code))
    assert len({dihedral(plain["cond"][0], c).numpy().tobytes() for c in range(8)}) == 8


def test_memory_stream_resumes_exactly(toy_cache):
    mem = MemorySplit(toy_cache, "train", cache.read_stats(toy_cache))
    a = MemoryStream(mem, 5, seed=3)
    for _ in range(4):
        a.next()
    state = a.state_dict()
    expected = [a.next() for _ in range(6)]
    b = MemoryStream(mem, 5, seed=99)
    b.load_state_dict(state)
    for batch in expected:
        other = b.next()
        assert torch.equal(batch["cond"], other["cond"]) and torch.equal(batch["row"], other["row"])


def test_train_cone_subset_is_fixed():
    assert train_cone_subset(range(10), 0) is None and train_cone_subset(range(10), 10) is None
    subset = train_cone_subset(range(10), 4)
    assert len(subset) == 4 and np.array_equal(subset, train_cone_subset(reversed(range(10)), 4))


def test_memory_config_validation():
    for kwargs in ({"data": "disk"}, {"channels": "center"}, {"train_cones": 3},
                   {"data": "memory", "train_cones": -1}, {"eval_batch_size": -1},
                   {"data": "memory", "storage": "disk"}):
        with pytest.raises(ValueError):
            LongRunConfig(**kwargs).validate()


def test_memory_training_resumes_and_records_channels(toy_cache, tmp_path):
    config = tiny_config(data="memory", channels="singles", eval_batch_size=7)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    complete = run(toy_cache, full, config)
    run(toy_cache, resumed, tiny_config(data="memory", channels="singles", eval_batch_size=7, max_steps=4))
    continued = run(toy_cache, resumed, config, resume=True)
    assert continued["best_validation_bce"] == complete["best_validation_bce"]
    a, b = _checkpoint(full), _checkpoint(resumed)
    for key in ("model", "optimizer", "best_model", "stream", "history", "step"):
        _assert_equal(a[key], b[key])
    model, checkpoint = load_checkpoint(full / "best.pt")
    assert checkpoint["channels"] == [1, 2, 3] and model.in_channels == 3
    record = a["history"][-1]
    assert record["validation_rmse"] > 0 and "validation_mixed_slice_rmse" in record
    metadata = json.loads((full / "metadata.json").read_text())
    assert metadata["channels"] == [1, 2, 3] and metadata["train_subset"] is None


def test_memory_train_subset(toy_cache, tmp_path):
    run(toy_cache, tmp_path / "subset", tiny_config(data="memory", train_cones=2, max_steps=3))
    metadata = json.loads((tmp_path / "subset" / "metadata.json").read_text())
    subset = metadata["train_subset"]
    assert len(subset["cones"]) == 2 and set(subset["cones"]) <= set(metadata["splits"]["train"]["cones"])
    with h5py.File(toy_cache, "r") as h:
        cone_of_row = h["cone_id"][:]
    assert set(cone_of_row[metadata["train_probe_cache_rows"]].tolist()) <= set(subset["cones"])


def test_tb_overflow_clip_keeps_cone_and_excludes_it_from_stats(toy_dir, tmp_path):
    import shutil
    src = sorted(cache.discover(toy_dir, "*.h5"))
    files = [shutil.copy(f, tmp_path / f.name) for f in src]
    with h5py.File(files[0], "r+") as f:
        f["lightcone/brightness_temp"][0, 0, :] = 7.0e4
    kw = dict(slices_per_cone=3, bands="-2,0,2", log=lambda *a: None)
    skipped = cache.build(files, tmp_path / "skip.h5", **kw)
    clipped = cache.build(files, tmp_path / "clip.h5", tb_overflow="clip", **kw)
    with h5py.File(skipped, "r") as a, h5py.File(clipped, "r") as b:
        assert len(json.loads(a.attrs["skipped"])) == 1 and json.loads(a.attrs["skipped"])[0][1].endswith("float16")
        cone = json.loads(b.attrs["tb_clipped"])
        assert len(cone) == 1 and cone[0] in b["cone_id"][:] and json.loads(b.attrs["skipped"]) == []
        assert b["tb"][:].max() == np.float16(6.0e4)
        # T_b stats ignore the clipped cone, so they match the cache that skipped it.
        if b["split"][:][b["cone_id"][:] == cone[0]][0] == 0:
            assert cache.read_stats(clipped)["tb_mean"] == pytest.approx(cache.read_stats(skipped)["tb_mean"])
    with pytest.raises(ValueError):
        cache.build(files, tmp_path / "bad.h5", tb_overflow="drop", **kw)

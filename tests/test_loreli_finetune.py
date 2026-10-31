import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from ebm21cm import loreli_metrics as lm
from ebm21cm.data.loreli import LoreliAdapter, LoreliCone, read_cube
from ebm21cm.data.loreli_cache import (
    SCHEMA,
    padding_mask,
    rows_from_cone,
    select_target_indices,
)
from ebm21cm.data.loreli_memory import LoReLiMemorySplit, LoReLiStream
from ebm21cm.data.loreli_splits import (
    filter_sims_by_split,
    load_split,
    make_split,
    make_v2_split,
    nested_train_blocks,
)
from ebm21cm.evaluate_loreli import fingerprint
from ebm21cm.loreli_inference import load_model
from ebm21cm.model.freeze import (
    category_of,
    freeze_model,
    optimizer_parameter_groups,
    parameter_groups,
)
from ebm21cm.model.remap import remapped_model
from ebm21cm.model.rhizome import build_model
from ebm21cm.train_loreli_finetune import _batch_loss
from tests.test_loreli import REAL_STATS, _fingerprint_args, write_cone


def _tiny_model(modes=4, n_steps=1):
    return build_model({
        "in_channels": 26, "out_channels": 2, "scalar_dim": 38,
        "width": 8, "modes": modes, "n_steps": n_steps, "step_size": 0.5,
    })


def _write_memory_cache(path, n=24, domain=32, n_out=20, seed=0, split_id=0, block=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as h:
        h.create_dataset("cond", data=rng.standard_normal((n, 26, domain, domain)).astype(np.float16))
        h.create_dataset("target", data=rng.random((n, 2, domain, domain)).astype(np.float16))
        h.create_dataset("phys", data=(rng.random((n, domain, domain)) + 1.0).astype(np.float32))
        h.create_dataset("scalars", data=rng.standard_normal((n, 38)).astype(np.float32))
        h.create_dataset("mask", data=padding_mask(domain, n_out))
        h.create_dataset("z", data=np.linspace(5, 15, n))
        h.create_dataset("cone_id", data=np.repeat(np.arange(n // 2) + 10001, 2).astype(np.int64))
        h.create_dataset("block", data=np.full(n, block, np.int32))
        h.create_dataset("split", data=np.full(n, split_id, np.int8))
        h.attrs.update({"schema": SCHEMA, "domain": domain, "n_out": n_out})


def test_disk_storage_matches_cpu_batches_loss_and_augmentation(tmp_path):
    path = tmp_path / "cache.h5"
    _write_memory_cache(path, n=24)
    cpu = LoReLiMemorySplit(path, "train", blocks=[0], storage="cpu")
    disk = LoReLiMemorySplit(path, "train", blocks=[0], storage="disk")
    assert np.array_equal(cpu.rows, disk.rows)
    assert disk.cond is None and disk.target is None and disk.phys is None
    assert disk.nbytes < cpu.nbytes
    assert disk.dataset_nbytes == cpu.dataset_nbytes == len(cpu) * cpu.row_bytes

    # Same seed, same exact global permutation, across the epoch boundary at 24 rows.
    train_cpu, train_disk = LoReLiStream(cpu, 4, 0), LoReLiStream(disk, 4, 0)
    model = _tiny_model().eval()
    for step in range(8):
        a, b = train_cpu.next(), train_disk.next()
        for key in ("cond", "target", "phys", "mask", "scalars", "cone_id", "z"):
            assert torch.equal(a[key], b[key]), f"{key} differs at step {step}"
        loss_cpu, *_ = _batch_loss(model, a, (0.0, 1.0), 1.0)
        loss_disk, *_ = _batch_loss(model, b, (0.0, 1.0), 1.0)
        assert torch.equal(loss_cpu, loss_disk), f"loss differs at step {step}"
    assert train_disk.position == train_cpu.position == 8
    assert disk.io_stats["reads"] > 0 and disk.io_stats["bytes_read"] > 0

    # Explicit dihedral codes: all eight elements, mixing 90-degree rotations and flips.
    for codes in ([0, 1, 2, 3], [4, 5, 6, 7], [7, 0, 3, 5]):
        a, b = cpu.batch([0, 5, 11, 23], codes), disk.batch([0, 5, 11, 23], codes)
        for key in ("cond", "target", "phys", "mask", "scalars", "cone_id", "z"):
            assert torch.equal(a[key], b[key]), f"{key} differs for codes {codes}"
        loss_cpu, *_ = _batch_loss(model, a, (0.0, 1.0), 1.0)
        loss_disk, *_ = _batch_loss(model, b, (0.0, 1.0), 1.0)
        assert torch.equal(loss_cpu, loss_disk)

    disk.close()


def test_disk_storage_rows_survive_a_block_subset(tmp_path):
    path = tmp_path / "cache.h5"
    _write_memory_cache(path, n=24)
    with h5py.File(path, "a") as h:
        h["block"][:] = np.array([0, 0, 0, 1, 1, 1] * 4, np.int32)
    cpu = LoReLiMemorySplit(path, "train", blocks=[1], storage="cpu")
    disk = LoReLiMemorySplit(path, "train", blocks=[1], storage="disk")
    assert np.array_equal(cpu.rows, disk.rows)
    assert cpu.rows.tolist() == [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23]
    for positions in ([0], [1, 4], [11, 0, 6]):
        a, b = cpu.batch(positions, [2] * len(positions)), disk.batch(positions, [2] * len(positions))
        assert torch.equal(a["cond"], b["cond"]) and torch.equal(a["mask"], b["mask"])
        assert torch.equal(a["phys"], b["phys"])
    disk.close()


def test_padding_mask_is_the_valid_interior():
    for l_perp, n_out in ((294.0, 206), (300.0, 210)):
        mask = padding_mask(280, n_out)
        assert mask.shape == (280, 280)
        assert int(mask.sum()) == n_out * n_out
        pad = (280 - n_out) // 2
        assert mask[pad:pad + n_out, pad:pad + n_out].all()
        assert not mask[:pad].any() and not mask[-pad:].any()
        assert not mask[:, :pad].any() and not mask[:, -pad:].any()


def test_select_target_indices_keeps_all_active_and_two_background_slices():
    mean = np.array([0.0, 0.01, 0.02, 0.1, 0.5, 0.97, 0.98, 0.99, 1.0])
    indices, background = select_target_indices(mean, (0.02, 0.98), background=2)
    assert np.array_equal(indices, np.array([0, 2, 3, 4, 5, 6, 8]))
    assert background.tolist() == [True, False, False, False, False, False, True]
    # All active slices are present regardless of the background choice.
    assert set(np.flatnonzero((mean >= 0.02) & (mean <= 0.98))) <= set(indices.tolist())


def test_block_split_is_nested_deterministic_and_loadable(tmp_path):
    sims = [10000 + i for i in range(228)]
    split = make_split(sims, seed=0)
    assert len(split["train_blocks"]) == 28
    assert len(split["val_blocks"]) == 5
    assert len(split["test_blocks"]) == 5
    assert len(set(split["train_blocks"]) | set(split["val_blocks"]) | set(split["test_blocks"])) == 38

    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    loaded = load_split(path)
    assert loaded["train_blocks"] == split["train_blocks"]
    assert loaded["train_sims"] == sorted(s for b in split["train_blocks"] for s in split["blocks"][str(b)]["sims"])
    four = nested_train_blocks(loaded, 4)
    eight = nested_train_blocks(loaded, 8)
    assert len(four) == 4 and four == eight[:4]
    assert len(nested_train_blocks(loaded, 28)) == 28
    assert filter_sims_by_split(loaded["train_sims"] + loaded["test_sims"], loaded, "test") == loaded["test_sims"]


def test_v2_split_stratifies_60_blocks_and_interleaves():
    partial_sizes = [1] * 23 + [2] * 2 + [3] + [4] * 2 + [5] * 2
    sims = []
    for block in range(30):
        sims += list(range(10000 + 6 * block, 10006 + 6 * block))
    for index, size in enumerate(partial_sizes, start=30):
        sims += list(range(10000 + 6 * index, 10000 + 6 * index + size))
    assert len(sims) == 228

    payload = make_v2_split(sims, seed=0)
    assert payload["n_blocks"] == 60
    assert payload["split_sizes"] == [44, 8, 8]
    path = Path(tempfile.mkdtemp()) / "v2.json"
    path.write_text(json.dumps(payload))
    loaded = load_split(path)
    assert (len(loaded["train_blocks"]), len(loaded["val_blocks"]), len(loaded["test_blocks"])) == (44, 8, 8)
    assert len(loaded["train_sims"]) + len(loaded["val_sims"]) + len(loaded["test_sims"]) == 228
    assert len(loaded["sim_to_split"]) == 228

    for name, expected in (("train", 44), ("val", 8), ("test", 8)):
        blocks = loaded[f"{name}_blocks"]
        assert len(blocks) == expected
        sizes = [len(loaded["block_to_sims"][b]) for b in blocks]
        assert sum(size == 6 for size in sizes) == expected // 2
        assert sum(size != 6 for size in sizes) == expected // 2

    previous = []
    for n in (4, 8, 16, 32, 44):
        prefix = nested_train_blocks(loaded, n)
        assert prefix[: len(previous)] == previous
        sizes = [len(loaded["block_to_sims"][b]) for b in prefix]
        assert any(size == 6 for size in sizes) and any(size != 6 for size in sizes)
        previous = prefix

    # A block cannot appear in two splits, nor can its cones.
    for block in loaded["block_to_sims"]:
        owners = [name for name in ("train", "val", "test") if block in loaded[f"{name}_blocks"]]
        assert len(owners) == 1


def test_block_split_accepts_shorthand_lists(tmp_path):
    blocks = list(range(38))
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"seed": 0, "train": blocks[:28], "val": blocks[28:33], "test": blocks[33:]}))
    loaded = load_split(path)
    assert loaded["train_blocks"] == blocks[:28]
    assert loaded["test_blocks"] == blocks[33:]
    assert loaded["test_sims"][0] == 10000 + 6 * blocks[33]


def test_block_split_accepts_binac_splits_schema(tmp_path):
    """The file Binac writes uses top-level `splits` with irregular block sizes."""
    blocks = {391: [12352], 392: [12353, 12354], 446: [12677], 452: [12713, 12714], 505: [13036],
              511: [13072], 571: [13432], 614: [13685]}
    assignment = {391: "train", 392: "train", 446: "train", 452: "train", 505: "train",
                  511: "val", 571: "val", 614: "test"}
    splits = {}
    for name in ("train", "val", "test"):
        chosen = {str(b): sims for b, sims in blocks.items() if assignment[b] == name}
        splits[name] = {"n_blocks": len(chosen), "blocks": chosen,
                        "sims": [sim for sims in chosen.values() for sim in sims]}
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"seed": 0, "n_blocks": 95, "splits": splits}))
    loaded = load_split(path)
    assert loaded["train_blocks"] == [391, 392, 446, 452, 505]
    assert loaded["val_blocks"] == [511, 571]
    assert loaded["test_blocks"] == [614]
    assert loaded["train_sims"] == [12352, 12353, 12354, 12677, 12713, 12714, 13036]
    two = nested_train_blocks(loaded, 2)
    six = nested_train_blocks(loaded, 6)
    assert len(two) == 2 and two == six[:2]
    assert set(two) <= set(loaded["train_blocks"])
    assert filter_sims_by_split([12352, 13036, 13685], loaded, "test") == [13685]


def test_ids_from_split_filters_and_fingerprint(tmp_path):
    args = _fingerprint_args(tmp_path)
    old = fingerprint(args, {"step": 1})
    assert "ids_from_split" not in old and "split_sha256" not in old

    test_blocks = {
        "100": [10001, 10002, 10003],
        "101": [10004, 10005],
        "102": [10006, 10007],
        "103": [10008, 10009],
        "104": [10010, 10011],
    }
    split = {
        "seed": 0, "splits": {
            "train": {"blocks": {"1": [10012], "2": [10013]}, "sims": [10012, 10013]},
            "val": {"blocks": {"3": [10014]}, "sims": [10014]},
            "test": {"blocks": test_blocks, "sims": [s for v in test_blocks.values() for s in v]},
        },
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    loaded = load_split(path)
    assert filter_sims_by_split(range(10001, 10015), loaded, "test") == loaded["test_sims"]
    assert len(loaded["test_sims"]) == 11

    args.ids_from_split = "test"
    args.split_file = path
    new = fingerprint(args, {"step": 1})
    assert new["ids_from_split"] == "test"
    assert new["split_sha256"] == loaded["sha256"]
    assert new["split_file"] == str(path.resolve())
    # No split flags: the old fingerprint is byte-identical for the same run.
    del args.ids_from_split
    del args.split_file
    assert fingerprint(args, {"step": 1}) == old


def test_cache_rows_equal_evaluation_inputs():
    """One cache row is bit-identical (to float16 storage) to adapter.inputs."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu10001.dat"
        write_cone(path, n_native=16, n_los=96, tail=0)
        cone = LoreliCone(path, l_perp=294, n_native=16, n_los=96, data=read_cube(path))
        adapter = LoreliAdapter(cone, REAL_STATS, velocity="linear", l_perp=294, device="cpu")
        index = len(adapter.target_distance) // 2
        arrays, _ = rows_from_cone(
            cone, adapter, (0.0, 1.0), indices=np.array([index]), background=0
        )
        inp = adapter.inputs(index, index + 1)
        assert arrays["cond"].shape == (1, 26, 280, 280)
        assert arrays["scalars"].shape == (1, 38)
        assert np.allclose(arrays["cond"][0].astype(np.float32), np.asarray(inp["cond"])[0],
                           atol=5e-3, rtol=1e-3)
        assert np.array_equal(arrays["phys"][0], np.asarray(inp["phys"])[0, 0])
        assert np.allclose(arrays["scalars"][0], inp["scalars"][0], atol=1e-6)
        assert int(arrays["los_index"][0]) == index
        mask = padding_mask(280, adapter.n_out).astype(bool)
        assert np.isfinite(arrays["target"][0]).all()
        assert float(arrays["target"][0, 0][mask].min()) >= 0.0
        assert float(arrays["target"][0, 0][mask].max()) <= 1.0
        # The interior target is the resampled truth, not a padded copy.
        from ebm21cm.data.loreli_cache import _resampled_truth

        x_target, tb_target = _resampled_truth(cone, adapter)
        n_out = adapter.n_out
        interior = arrays["target"][0, 0][mask].reshape(n_out, n_out)
        assert np.allclose(interior, np.clip(x_target[..., index], 0, 1), atol=2e-3)
        assert np.allclose(
            arrays["target"][0, 1][mask].reshape(n_out, n_out), tb_target[..., index], atol=3e-2
        )


def test_masked_loss_ignores_the_padding():
    model = _tiny_model().eval()
    batch = {
        "cond": torch.randn(1, 26, 32, 32),
        "scalars": torch.randn(1, 38),
        "target": torch.rand(1, 2, 32, 32),
        "phys": torch.rand(1, 1, 32, 32) + 1.0,
        "mask": torch.from_numpy(padding_mask(32, 20))[None, None].float(),
    }
    a, *_ = _batch_loss(model, batch, (0.0, 1.0), 1.0)
    changed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    changed["target"][:, :, :6, :6] = 100.0  # entirely in the mirrored border
    b, *_ = _batch_loss(model, changed, (0.0, 1.0), 1.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_freeze_groups_cover_exact_counts_and_block_gradients():
    model = _tiny_model()
    groups = parameter_groups(model)
    assert set(groups) == {"decoder", "ends", "pointwise", "spectral"}
    assert sum(len(v) for v in groups.values()) == len(list(model.parameters()))
    assert all(category_of(name) in groups for name in dict(model.named_parameters()))

    previous = -1
    for setting in ("decoder", "ends", "pointwise", "none"):
        report = freeze_model(model, setting)
        assert report["total"] == sum(p.numel() for p in model.parameters())
        assert report["trainable"] >= previous
        previous = report["trainable"]
        assert report["trainable_counts"]["spectral"] == 0 if setting != "none" else report["counts"]["spectral"]
    assert report["trainable"] == report["total"]

    # A backward pass must leave frozen parameters without a gradient.
    freeze_model(model, "pointwise")
    out = model(torch.randn(2, 26, 32, 32), torch.randn(2, 38))
    out.square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
        else:
            assert parameter.grad is None, name
    spectral = optimizer_parameter_groups(model, scratch=False)["pretrained"]
    assert spectral == []  # pointwise setting freezes every spectral kernel
    assert len(optimizer_parameter_groups(model, scratch=False)["new"]) > 0


def test_reference_freeze_counts():
    """Pin the production width-64 / 48-mode parameter counts."""
    model = build_model({
        "in_channels": 26, "out_channels": 2, "scalar_dim": 38,
        "width": 64, "modes": 48, "n_steps": 6, "step_size": 0.5,
    })
    expected = {
        "decoder": 4_290,
        "ends": 8_514,
        "pointwise": 45_570,
        "none": 18_530_818,
    }
    for setting, count in expected.items():
        report = freeze_model(model, setting)
        assert report["trainable"] == count
    assert report["counts"] == {
        "decoder": 4_290, "ends": 4_224, "pointwise": 37_056, "spectral": 18_485_248
    }
    groups = optimizer_parameter_groups(model, scratch=False)
    assert groups["pretrained"] and groups["new"]
    freeze_model(model, "pointwise")
    assert optimizer_parameter_groups(model, scratch=False)["pretrained"] == []


def test_init_from_reproduces_zero_shot_and_domain_marker_skips_double_remap(tmp_path):
    config = {
        "in_channels": 26, "out_channels": 2, "scalar_dim": 38,
        "width": 8, "modes": 4, "n_steps": 2, "step_size": 0.5,
    }
    torch.manual_seed(0)
    state = build_model(config).state_dict()
    run = tmp_path / "ref"
    run.mkdir()
    torch.save({
        "model": state, "model_config": config, "head": "structured",
        "tb_norm": (0.0, 1.0), "step": 1, "validation_score": 0.0,
    }, run / "best.pt")
    from ebm21cm.train_loreli_finetune import build_finetune_model

    model, model_config, _ = build_finetune_model(run)
    expected = remapped_model(config, state, factor=2).eval()
    model.eval()
    cond = torch.randn(1, 26, 32, 32)
    scalars = torch.randn(1, 38)
    with torch.no_grad():
        assert torch.allclose(model(cond, scalars), expected(cond, scalars), atol=1e-6)
    assert model_config["modes"] == 8

    # A fine-tuned checkpoint is already on the 280-cell domain.
    fine = tmp_path / "fine"
    fine.mkdir()
    fine_config = dict(model_config)
    torch.save({
        "model": model.state_dict(), "model_config": fine_config, "head": "structured",
        "tb_norm": (0.0, 1.0), "step": 4, "domain_factor": 2,
    }, fine / "best.pt")
    native, remapped, _ = load_model(fine)
    assert native is remapped
    assert native.modes == 8


def test_cache_build_writes_the_documented_schema(tmp_path, monkeypatch):
    """Exercise the HDF5 writer without a real 480 MB cone."""
    import ebm21cm.data.loreli_cache as cache_mod

    split = make_split([10000 + i for i in range(228)], seed=0)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    block = split["train_blocks"][0]
    sim = 10000 + 6 * block

    training = tmp_path / "training.h5"
    with h5py.File(training, "w") as h:
        h.attrs["bands"] = json.dumps([(-32, -16)] * 13)
        h.attrs["normalization"] = json.dumps({"density": {"offset": 0, "scale": 1}})
        h.attrs["param_names"] = json.dumps([f"p{i}" for i in range(11)])
        h.attrs["scalar_names"] = json.dumps(["1/(1+z)"] + [f"p{i}" for i in range(11)] + [f"h{i}" for i in range(26)])
    reference = tmp_path / "ref"
    reference.mkdir()
    torch.save({"model_config": {"in_channels": 26, "out_channels": 2, "scalar_dim": 38,
                                 "width": 8, "modes": 4, "n_steps": 1},
                "head": "structured", "tb_norm": (0.0, 1.0), "step": 1},
               reference / "best.pt")

    class FakeCone:
        def __init__(self, path, **kwargs):
            self.sim = sim

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            pass

    def fake_rows(cone, adapter, tb_norm, **kwargs):
        return {
            "cond": np.zeros((1, 26, 280, 280), np.float16),
            "target": np.zeros((1, 2, 280, 280), np.float16),
            "phys": np.ones((1, 280, 280), np.float32),
            "scalars": np.zeros((1, 38), np.float32),
            "z": np.array([8.0]),
            "los_index": np.array([0], np.int32),
            "xhi_mean": np.array([0.5], np.float32),
            "stage": np.array([1], np.int8),
            "background": np.array([False]),
        }, {"cone_id": sim, "n_rows": 1, "n_active": 1}

    monkeypatch.setattr(cache_mod, "LoreliCone", FakeCone)
    monkeypatch.setattr(cache_mod, "LoreliAdapter", FakeAdapter)
    monkeypatch.setattr(cache_mod, "rows_from_cone", fake_rows)
    monkeypatch.setattr(cache_mod, "loreli_files", lambda data: ([tmp_path / "sim.dat"], [sim]))
    monkeypatch.setattr(cache_mod, "read_cube", lambda path: None)

    out = tmp_path / "cache.h5"
    cache_mod.build(out, tmp_path, training, reference, split_path, device="cpu",
                    l_perp=294.0, domain=280, limit=1, log=lambda *a: None)
    with h5py.File(out, "r") as h:
        assert h["cond"].shape == (1, 26, 280, 280)
        assert h["target"].shape == (1, 2, 280, 280)
        assert h["phys"].shape == (1, 280, 280)
        assert int(h["split"][0]) == 0
        assert int(h["block"][0]) == block
        assert int(h["mask"][:].sum()) == 206 * 206
        assert h.attrs["n_rows"] == 1 and h.attrs["n_cones"] == 1
        assert h.attrs["split_sha256"] == load_split(split_path)["sha256"]
        assert h.attrs["schema"] == SCHEMA


def test_memory_split_reads_mask_and_batches(tmp_path):
    domain, n_out = 32, 20
    mask = padding_mask(domain, n_out).astype(np.uint8)
    path = tmp_path / "cache.h5"
    n = 3
    with h5py.File(path, "w") as h:
        h.create_dataset("cond", data=np.random.randn(n, 26, domain, domain).astype(np.float16))
        h.create_dataset("target", data=np.random.rand(n, 2, domain, domain).astype(np.float16))
        h.create_dataset("phys", data=np.ones((n, domain, domain), np.float16))
        h.create_dataset("scalars", data=np.random.randn(n, 38).astype(np.float32))
        h.create_dataset("mask", data=mask)
        h.create_dataset("z", data=np.linspace(5, 15, n))
        h.create_dataset("cone_id", data=np.full(n, 10001, np.int64))
        h.create_dataset("block", data=np.zeros(n, np.int32))
        h.create_dataset("split", data=np.zeros(n, np.int8))
        h.attrs.update({"schema": SCHEMA, "domain": domain, "n_out": n_out})
    split = LoReLiMemorySplit(path, "train", blocks=[0])
    assert len(split) == 3
    batch = split.batch([0, 2])
    assert batch["cond"].shape == (2, 26, domain, domain)
    assert batch["target"].shape == (2, 2, domain, domain)
    assert batch["mask"].shape == (2, 1, domain, domain)
    assert batch["mask"].sum() == 2 * n_out * n_out


def test_trainer_smoke_builds_a_loreli_checkpoint(tmp_path):
    """One optimizer step on a tiny cache; the saved best.pt is LoReLi-domain."""
    from ebm21cm.train_loreli_finetune import FinetuneConfig, run as train_run

    config = {
        "in_channels": 26, "out_channels": 2, "scalar_dim": 38,
        "width": 8, "modes": 4, "n_steps": 2, "step_size": 0.5,
    }
    reference = tmp_path / "ref"
    reference.mkdir()
    torch.manual_seed(0)
    torch.save({
        "model": build_model(config).state_dict(), "model_config": config,
        "head": "structured", "tb_norm": (0.0, 1.0), "step": 1,
    }, reference / "best.pt")

    split = make_split([10000 + i for i in range(228)], seed=0)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    split_info = load_split(split_path)
    train_block = split["train_blocks"][0]
    val_block = split["val_blocks"][0]

    domain, n_out = 32, 20
    cache = tmp_path / "cache.h5"
    n = 4
    with h5py.File(cache, "w") as h:
        h.create_dataset("cond", data=np.random.randn(n, 26, domain, domain).astype(np.float16))
        h.create_dataset("target", data=np.random.rand(n, 2, domain, domain).astype(np.float16))
        h.create_dataset("phys", data=np.ones((n, domain, domain), np.float16))
        h.create_dataset("scalars", data=np.random.randn(n, 38).astype(np.float32))
        h.create_dataset("mask", data=padding_mask(domain, n_out))
        h.create_dataset("z", data=np.linspace(5, 15, n))
        h.create_dataset("cone_id", data=np.array([10001, 10001, 10002, 10002], np.int64))
        h.create_dataset("block", data=np.array([train_block, train_block, val_block, val_block], np.int32))
        h.create_dataset("split", data=np.array([0, 0, 1, 1], np.int8))
        h.attrs.update({"schema": SCHEMA, "domain": domain, "n_out": n_out,
                        "split_sha256": split_info["sha256"]})

    run_dir = tmp_path / "run"
    result = train_run(
        cache, run_dir, reference, freeze="decoder", split_file=split_path,
        config=FinetuneConfig(
            max_steps=2, min_steps=1, batch_size=2, eval_batch_size=2,
            lr_new=1e-3, lr_pretrained=1e-3, min_lr=1e-5, val_every=1,
            seed=0, threads=1, storage="cpu",
        ),
        device="cpu", log=lambda *a: None,
    )
    assert result["steps"] == 2
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert best["domain_factor"] == 2
    assert best["model_config"]["modes"] == 8
    native, remapped, _ = load_model(run_dir)
    assert native is remapped and native.modes == 8

    # Resume must continue the same run; scratch/device changes must refuse.
    resumed = train_run(
        cache, run_dir, reference, freeze="decoder", split_file=split_path,
        config=FinetuneConfig(
            max_steps=3, min_steps=1, batch_size=2, eval_batch_size=2,
            lr_new=1e-3, lr_pretrained=1e-3, min_lr=1e-5, val_every=1,
            seed=0, threads=1, storage="cpu",
        ),
        device="cpu", resume=True, log=lambda *a: None,
    )
    assert resumed["steps"] == 3
    with pytest.raises(ValueError, match="resume"):
        train_run(
            cache, run_dir, reference, freeze="none", scratch=True, split_file=split_path,
            config=FinetuneConfig(
                max_steps=3, min_steps=1, batch_size=2, eval_batch_size=2,
                lr_new=1e-3, lr_pretrained=1e-3, min_lr=1e-5, val_every=1,
                seed=0, threads=1, storage="cpu",
            ),
            device="cpu", resume=True, log=lambda *a: None,
        )
    with pytest.raises(ValueError, match="resume"):
        train_run(
            cache, run_dir, reference, freeze="decoder", split_file=split_path,
            config=FinetuneConfig(
                max_steps=3, min_steps=1, batch_size=2, eval_batch_size=2,
                lr_new=1e-3, lr_pretrained=1e-3, min_lr=1e-5, val_every=1,
                seed=0, threads=1, storage="cpu",
            ),
            device="cpu:0", resume=True, log=lambda *a: None,
        )

    # Scratch: same architecture, random init, every parameter trainable.
    scratch_dir = tmp_path / "run_scratch"
    scratch = train_run(
        cache, scratch_dir, reference, freeze="none", scratch=True, split_file=split_path,
        config=FinetuneConfig(
            max_steps=1, min_steps=1, batch_size=2, eval_batch_size=2,
            lr_new=1e-3, lr_pretrained=1e-3, min_lr=1e-5, val_every=1,
            seed=0, threads=1, storage="cpu",
        ),
        device="cpu", log=lambda *a: None,
    )
    scratch_best = torch.load(scratch_dir / "best.pt", map_location="cpu", weights_only=False)
    assert scratch["steps"] == 1 and scratch_best["scratch"] is True
    assert scratch["trainable_parameters"] == sum(
        p.numel() for p in load_model(reference)[1].parameters()
    )

    # lr=0 with freeze=none must reproduce the zero-shot initialization exactly.
    zero_dir = tmp_path / "run_zero"
    result_zero = train_run(
        cache, zero_dir, reference, freeze="none", split_file=split_path,
        config=FinetuneConfig(
            max_steps=1, min_steps=1, batch_size=2, eval_batch_size=2,
            lr_new=0.0, lr_pretrained=0.0, min_lr=0.0, val_every=1,
            seed=0, threads=1, storage="cpu",
        ),
        device="cpu", log=lambda *a: None,
    )
    assert result_zero["steps"] == 1
    zero_best = torch.load(zero_dir / "best.pt", map_location="cpu", weights_only=False)
    initial = load_model(reference)[1].state_dict()
    for name, tensor in initial.items():
        assert torch.allclose(zero_best["model"][name], tensor, atol=1e-9), name


def test_trainer_disk_storage_is_bit_identical_and_resume_guarded(tmp_path):
    """A full cpu run and a disk run (same seed) agree in batches, loss and weights."""
    from ebm21cm.train_loreli_finetune import FinetuneConfig, run as train_run

    model_config = {
        "in_channels": 26, "out_channels": 2, "scalar_dim": 38,
        "width": 8, "modes": 4, "n_steps": 2, "step_size": 0.5,
    }
    reference = tmp_path / "ref"
    reference.mkdir()
    torch.manual_seed(0)
    torch.save({
        "model": build_model(model_config).state_dict(), "model_config": model_config,
        "head": "structured", "tb_norm": (0.0, 1.0), "step": 1,
    }, reference / "best.pt")

    split = make_split([10000 + i for i in range(228)], seed=0)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    info = load_split(split_path)
    train_block, val_block = split["train_blocks"][0], split["val_blocks"][0]

    domain, n_out = 32, 20
    cache = tmp_path / "cache.h5"
    n = 4
    with h5py.File(cache, "w") as h:
        h.create_dataset("cond", data=np.random.randn(n, 26, domain, domain).astype(np.float16))
        h.create_dataset("target", data=np.random.rand(n, 2, domain, domain).astype(np.float16))
        h.create_dataset("phys", data=np.ones((n, domain, domain), np.float16))
        h.create_dataset("scalars", data=np.random.randn(n, 38).astype(np.float32))
        h.create_dataset("mask", data=padding_mask(domain, n_out))
        h.create_dataset("z", data=np.linspace(5, 15, n))
        h.create_dataset("cone_id", data=np.array([10001, 10001, 10002, 10002], np.int64))
        h.create_dataset("block", data=np.array([train_block, train_block, val_block, val_block], np.int32))
        h.create_dataset("split", data=np.array([0, 0, 1, 1], np.int8))
        h.attrs.update({"schema": SCHEMA, "domain": domain, "n_out": n_out,
                        "split_sha256": info["sha256"]})

    def config(storage, max_steps=2):
        return FinetuneConfig(
            max_steps=max_steps, min_steps=1, batch_size=2, eval_batch_size=2,
            lr_new=1e-3, lr_pretrained=1e-3, min_lr=1e-5, val_every=1,
            seed=0, threads=1, storage=storage,
        )

    cpu_dir, disk_dir = tmp_path / "run_cpu", tmp_path / "run_disk"
    cpu = train_run(cache, cpu_dir, reference, freeze="decoder", split_file=split_path,
                    config=config("cpu"), device="cpu", log=lambda *a: None)
    disk = train_run(cache, disk_dir, reference, freeze="decoder", split_file=split_path,
                     config=config("disk"), device="cpu", log=lambda *a: None)
    assert cpu["storage"] == "cpu" and disk["storage"] == "disk"
    assert cpu["disk_reads"] == 0 and disk["train_disk_reads"] > 0
    assert disk["val_disk_reads"] == 0  # the validation split stays in RAM
    assert disk["resident_bytes"] < cpu["resident_bytes"]
    assert disk["dataset_bytes"] == cpu["dataset_bytes"] > 0
    assert disk["resident_bytes"] < disk["dataset_bytes"]
    for key in ("steps", "samples_seen", "best_step", "best_validation_score",
                "trainable_parameters", "train_blocks"):
        assert cpu[key] == disk[key], key
    a = torch.load(cpu_dir / "best.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(disk_dir / "best.pt", map_location="cpu", weights_only=False)["model"]
    assert set(a) == set(b)
    for name in a:
        assert torch.equal(a[name], b[name]), name

    resumed = train_run(cache, disk_dir, reference, freeze="decoder", split_file=split_path,
                        config=config("disk", max_steps=3), device="cpu", resume=True,
                        log=lambda *a: None)
    assert resumed["steps"] == 3

    with pytest.raises(ValueError, match="resume"):
        train_run(cache, disk_dir, reference, freeze="decoder", split_file=split_path,
                  config=config("cpu", max_steps=3), device="cpu", resume=True, log=lambda *a: None)


def test_cache_build_assigned_only_drops_unassigned_cones(tmp_path, monkeypatch):
    import ebm21cm.data.loreli_cache as cache_mod

    split = make_split([10000 + i for i in range(228)], seed=0)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    assigned = 10000 + 6 * split["train_blocks"][0]
    unassigned = 99999
    paths = {str(tmp_path / "a.dat"): assigned, str(tmp_path / "b.dat"): unassigned}

    training = tmp_path / "training.h5"
    with h5py.File(training, "w") as h:
        h.attrs["bands"] = json.dumps([(-32, -16)] * 13)
        h.attrs["normalization"] = json.dumps({"density": {"offset": 0, "scale": 1}})
        h.attrs["param_names"] = json.dumps([f"p{i}" for i in range(11)])
        h.attrs["scalar_names"] = json.dumps(
            ["1/(1+z)"] + [f"p{i}" for i in range(11)] + [f"h{i}" for i in range(26)])
    reference = tmp_path / "ref"
    reference.mkdir()
    torch.save({"model_config": {"in_channels": 26, "out_channels": 2, "scalar_dim": 38,
                                 "width": 8, "modes": 4, "n_steps": 1},
                "head": "structured", "tb_norm": (0.0, 1.0), "step": 1},
               reference / "best.pt")

    class FakeCone:
        def __init__(self, path, **kwargs):
            self.sim = paths[str(path)]

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            pass

    def fake_rows(cone, adapter, tb_norm, **kwargs):
        sim = cone.sim
        return {
            "cond": np.zeros((1, 26, 280, 280), np.float16),
            "target": np.zeros((1, 2, 280, 280), np.float16),
            "phys": np.ones((1, 280, 280), np.float32),
            "scalars": np.zeros((1, 38), np.float32),
            "z": np.array([8.0]),
            "los_index": np.array([0], np.int32),
            "xhi_mean": np.array([0.5], np.float32),
            "stage": np.array([1], np.int8),
            "background": np.array([False]),
        }, {"cone_id": sim, "n_rows": 1, "n_active": 1}

    monkeypatch.setattr(cache_mod, "LoreliCone", FakeCone)
    monkeypatch.setattr(cache_mod, "LoreliAdapter", FakeAdapter)
    monkeypatch.setattr(cache_mod, "rows_from_cone", fake_rows)
    monkeypatch.setattr(cache_mod, "loreli_files",
                        lambda data: ([Path(p) for p in paths], list(paths.values())))
    monkeypatch.setattr(cache_mod, "read_cube", lambda path: None)

    kept = cache_mod.build(tmp_path / "kept.h5", tmp_path, training, reference, split_path,
                           device="cpu", assigned_only=True, log=lambda *a: None)
    assert kept["n_rows"] == 1
    with h5py.File(tmp_path / "kept.h5", "r") as h:
        assert int(h.attrs["n_cones"]) == 1
        assert int(h.attrs["n_unassigned"]) == 0
        assert bool(h.attrs["assigned_only"])
        assert int(h["split"][:].min()) >= 0
        assert int(h.attrs["n_rows"]) == 1

    cache_mod.build(tmp_path / "all.h5", tmp_path, training, reference, split_path,
                    device="cpu", log=lambda *a: None)
    with h5py.File(tmp_path / "all.h5", "r") as h:
        assert int(h.attrs["n_rows"]) == 2
        assert int(h.attrs["n_unassigned"]) == 1
        assert int(h.attrs["n_cones"]) == 2
        assert int(h["split"][:].min()) == -1

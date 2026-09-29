import json

import h5py
import numpy as np
import pytest

from ebm21cm.cosmology import FlatLCDM
from ebm21cm.data import cache
from ebm21cm.data.lightcone import DEFAULT_PARAM_NAMES, Lightcone
from ebm21cm.data.toy import write_toy

REF_Z = [6.0, 11.8850857, 35.0621589]


def test_distance_roundtrip():
    c = FlatLCDM(0.31, 0.68)
    z = np.linspace(5, 35, 50)
    assert np.allclose(c.z_at_distance(c.comoving_distance(z)), z, atol=1e-5)


def test_matches_py21cmfast_cosmology():
    """Reference: astropy Planck15.clone(H0=67.66, Om0=0.30964, Ob0=0.04897, Neff=3.044),
    the construction py21cmfast 4.x uses. Omitting massive neutrinos is off by ~0.5 at z~34."""
    c = FlatLCDM(0.30964144154550644, 0.6766)
    z = c.z_at_distance([8425.28018056, 10000.0, 11775.28018056])
    assert np.allclose(z, REF_Z, atol=2e-6)


def test_schemas_agree(tmp_path):
    raw = write_toy(tmp_path / "raw.h5", seed=3, H=16, cell=4.0, z_lo=6, z_hi=9, schema="raw_v2")
    nat = write_toy(tmp_path / "nat.h5", seed=3, H=16, cell=4.0, z_lo=6, z_hi=9, schema="native")
    with Lightcone(raw) as a, Lightcone(nat) as b:
        assert (a.schema, b.schema) == ("raw_v2", "native")
        assert a.n_los == b.n_los and a.shape_hw == b.shape_hw == (16, 16)
        assert a.cell_size == pytest.approx(4.0) and b.cell_size == pytest.approx(4.0)
        # Native redshifts come from inverting distances; raw ones are stored.
        assert np.allclose(a.redshifts, b.redshifts, atol=1e-4)
        assert np.allclose(a.params(DEFAULT_PARAM_NAMES), b.params(DEFAULT_PARAM_NAMES))
        assert np.array_equal(a.read_range("density", 5, 9), b.read_range("density", 5, 9))


def test_reversed_los_is_reordered(tmp_path):
    p = write_toy(tmp_path / "rev.h5", seed=1, H=16, cell=4.0, z_lo=6, z_hi=9)
    with Lightcone(p) as lc:
        fwd = lc.read_range("density", 0, lc.n_los)
        z = lc.redshifts.copy()
    with h5py.File(p, "r+") as f:
        for k in ("density", "neutral_fraction", "brightness_temp", "los_velocity"):
            a = f["lightcone"][k][:]
            del f["lightcone"][k]
            f["lightcone"][k] = a[:, :, ::-1]
        for k in ("lightcone_distances", "lightcone_redshifts"):
            a = f["lightcone"][k][:]
            del f["lightcone"][k]
            f["lightcone"][k] = a[::-1]
    with Lightcone(p) as lc:
        assert np.allclose(lc.redshifts, z)
        assert np.array_equal(lc.read_range("density", 0, lc.n_los), fwd)
        assert np.array_equal(lc.read_range("density", 3, 7), fwd[3:7])


def test_bands():
    assert cache.parse_bands("-8:-4,0,4:8") == [(-8, -4), (0, 1), (4, 8)]
    assert cache.band_extent([(-8, -4), (0, 1), (4, 8)]) == (8, 7)
    block = np.arange(20, dtype=float)[:, None, None] * np.ones((1, 2, 2))
    m = cache.band_means(block, 10, [(-2, 0), (0, 1), (1, 3)])
    assert m[:, 0, 0].tolist() == [8.5, 10.0, 11.5]


def test_cache_contents_match_source(toy_dir, toy_cache):
    with h5py.File(toy_cache, "r") as h:
        bands = [tuple(b) for b in json.loads(h.attrs["bands"])]
        split_of = {int(k): v for k, v in json.loads(h.attrs["split_of"]).items()}
        cone, idx, split = h["cone_id"][:], h["los_index"][:], h["split"][:]
        delta, xhi, tb, z = h["delta"][:], h["xhi"][:], h["tb"][:], h["z"][:]
    # Splits are per simulation and cover all three sets.
    for c in np.unique(cone):
        assert set(split[cone == c].tolist()) == {split_of[int(c)]}
    assert set(split.tolist()) == {0, 1, 2}
    below, above = cache.band_extent(bands)
    r = 3
    with Lightcone(toy_dir / f"toy_sample{int(cone[r]):06d}.h5") as lc:
        i = int(idx[r])
        assert below <= i < lc.n_los - above
        block = lc.read_range("density", i - below, i + above + 1)
        assert np.allclose(delta[r], cache.band_means(block, below, bands), atol=2e-3, rtol=2e-3)
        assert np.allclose(xhi[r], lc.read_range("neutral_fraction", i, i + 1)[0], atol=1e-3)
        assert np.allclose(tb[r], lc.read_range("brightness_temp", i, i + 1)[0], atol=0.05, rtol=1e-3)
        assert z[r] == pytest.approx(lc.redshifts[i])


def test_stats_use_train_only(toy_cache):
    st = cache.read_stats(toy_cache)
    with h5py.File(toy_cache, "r") as h:
        tr = h["split"][:] == 0
        tb = h["tb"][:][tr].astype(np.float64)
    assert st["tb_mean"] == pytest.approx(tb.mean(), rel=1e-6)
    assert st["tb_std"] == pytest.approx(tb.std(), rel=1e-6)
    assert st["sigma_data"] > 0


def test_shard_merge_equals_single(toy_dir, tmp_path):
    files = cache.discover(toy_dir, "*.h5")
    kw = dict(slices_per_cone=3, bands="-2,0,2", log=lambda *a: None)
    single = cache.build(files, tmp_path / "single.h5", **kw)
    shards = [cache.build(files, tmp_path / f"s{i}.h5", shard=i, n_shards=2, **kw) for i in range(2)]
    merged = cache.merge(shards, tmp_path / "merged.h5", log=lambda *a: None)
    with h5py.File(single, "r") as a, h5py.File(merged, "r") as b:
        oa = np.lexsort((a["los_index"][:], a["cone_id"][:]))
        ob = np.lexsort((b["los_index"][:], b["cone_id"][:]))
        for k in ("delta", "xhi", "split", "z"):
            assert np.array_equal(a[k][:][oa], b[k][:][ob])
    assert cache.read_stats(single) == pytest.approx(cache.read_stats(merged))


def test_dataset_and_augmentation(toy_cache):
    ds = cache.SliceDataset(toy_cache, "train", augment_data=True, seed=0)
    item = ds[0]
    assert item["cond"].shape == (ds.n_cond, 16, 16)
    assert item["target"].shape == (2, 16, 16)
    assert item["scalars"].shape == (ds.n_scalars,)
    # x_HI channel stays in [-1, 1] under augmentation; physical round trip.
    assert item["target"][0].abs().max() <= 1.0 + 1e-6
    raw = ds.raw(0)
    plain = cache.SliceDataset(toy_cache, "train")[0]
    xhi, tb = ds.norm.to_physical(plain["target"].numpy()[None])
    assert np.allclose(xhi[0], raw["xhi"], atol=1e-3)
    assert np.allclose(tb[0], raw["tb"], atol=0.05, rtol=1e-3)


def test_augment_is_joint():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((16, 16))
    cond = np.stack([base, 2 * base])
    target = np.stack([3 * base, -base])
    for _ in range(10):
        c, t = cache.augment(cond, target, rng)
        assert np.allclose(c[1], 2 * c[0]) and np.allclose(t[0], 3 * c[0]) and np.allclose(t[1], -c[0])

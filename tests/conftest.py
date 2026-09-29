import pytest

from ebm21cm.data import cache
from ebm21cm.data.toy import write_toy


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    for i in range(6):
        write_toy(d / f"toy_sample{i:06d}.h5", seed=i, H=16, cell=4.0, z_lo=6.0, z_hi=10.0)
    return d


@pytest.fixture(scope="session")
def toy_cache(toy_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("cache") / "slices.h5"
    files = cache.discover(toy_dir, "*.h5")
    cache.build(files, out, slices_per_cone=4, bands="-8:-4,-1,0,1,4:8", log=lambda *a: None)
    return out


@pytest.fixture
def tiny_model_cfg():
    return {"ch": 16, "mults": [1, 2], "num_res": 1, "attn_levels": [1], "heads": 2}

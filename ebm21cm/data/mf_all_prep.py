"""fno-21cm preparation over ALL raw lightcones, keeping the multi-field split's
validation and test cones and its train-only normalization.

    python -m ebm21cm.data.mf_all_prep --out $WORK/data/compressed/preparation_mf_all_raw.json

The multi-field preparation (``preparation_multifield_2000_tbclean``) covers the
first 2,000 samples through a cleaned-T_b native mirror. This one points
fno-21cm's native-window reader at every raw ``21cmfast_11d_sample*.h5`` file:

* validation / test = exactly the multi-field split's validation / test samples
  (by sample id), so results stay comparable with every multi-field model;
* train = every other sample (the 1,594 multi-field training cones plus all
  samples beyond the 2,000 subset);
* normalization and parameter normalization are copied unchanged from the
  multi-field preparation, so inputs and targets keep the same scale.

The raw files hold the un-clipped T_b; ``mf_slices build --clean-tb`` applies
fno-21cm's ``tools_clean_brightness_temp.clean`` per cone (cones whose T_b stays
non-finite are skipped there).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from .mf_slices import FNO_ROOT, PREPARATION

RAW = "/pfs/10/work/hd_id260-fno_training/data/data/21cmfast_11d_sample*.h5"


def sample_id(path):
    return int(re.findall(r"\d+", Path(path).stem)[-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=PREPARATION, help="multi-field preparation to take val/test and statistics from")
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--fno-root", default=FNO_ROOT)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    from ..train_rhizome3d import fno_pipeline

    fm, FieldMapping, FieldRegistry, _, _ = fno_pipeline(a.fno_root)
    from dataset.los_windows import NativeLightconeDataset

    base = fm.read_json(a.base)
    base_files = [f["path"] for f in base["source"]["files"]]
    held = {name: {sample_id(base_files[r]) for r in base["split"][name]} for name in ("val", "test")}
    files = sorted(glob.glob(a.raw), key=sample_id)
    registry = FieldRegistry.from_dict(base["registry"])
    mapping = FieldMapping.create(["density", "los_velocity"], ["neutral_fraction", "brightness_temp"],
                                  base["conditioning"], registry)
    dataset = NativeLightconeDataset(mapping, files=files, registry=registry)
    ids = [sample_id(p) for p in dataset.file_paths]
    rows = {name: [int(c) for c, s in zip(dataset.cone_ids, ids) if s in held[name]] for name in ("val", "test")}
    taken = set(rows["val"]) | set(rows["test"])
    rows["train"] = [int(c) for c in dataset.cone_ids if int(c) not in taken]
    for name in ("val", "test"):
        if len(rows[name]) != len(held[name]):
            raise SystemExit(f"{name}: {len(held[name]) - len(rows[name])} held-out samples have no raw file")
    prep = {"schema_version": 1, "source": dataset.source_description(), "source_fingerprint": dataset.fingerprint(),
            "registry": base["registry"], "conditioning": base["conditioning"], "split_seed": base.get("split_seed"),
            "split": rows, "normalization": base["normalization"],
            "parameter_normalization": base["parameter_normalization"],
            "derived_from": {"preparation": str(a.base), "rule": "val/test = base val/test sample ids; train = all others; "
                                                                  "normalization copied from base"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(prep))
    print(f"wrote {a.out}: {len(files)} files, train/val/test = "
          f"{len(rows['train'])}/{len(rows['val'])}/{len(rows['test'])}")


if __name__ == "__main__":
    main()

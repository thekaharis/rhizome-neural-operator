"""21cmEVAL python adapter: predictions stored in ``eval_cubes`` (``export_slicewise_mf``, fno-21cm exports).

    --model rhizome=python:scripts/eval_cubes_adapter.py:predict_rhizome2d_mf_w64_all

``predict_<tag>`` returns {"xhi": pred, "tb": tb_pred} from $EVAL_CUBES/<tag>/<sample id>.npz, where the
sample id is the cube's file stem (``cone_000024``; manifests from ``scripts/eval_cubes_cm21eval.py``).
The prediction's redshift grid must equal the sample's. Inference cost is not measured here; do not
enable the ``efficiency`` metric.
"""
import os
from pathlib import Path

import numpy as np

EVAL_CUBES = Path(os.environ.get("EVAL_CUBES", "/pfs/10/work/hd_id260-fno_training/data/eval_cubes"))


def __getattr__(attr):
    if not attr.startswith("predict_"):
        raise AttributeError(attr)
    tag = attr[len("predict_"):]
    if not (EVAL_CUBES / tag).is_dir():
        # AttributeError, not FileNotFoundError: getattr(module, "predict_draws", None) probes must see "absent"
        raise AttributeError(f"no eval_cubes for {tag!r} under {EVAL_CUBES}")

    def predict(sample):
        with np.load(EVAL_CUBES / tag / f"{sample.id}.npz") as d:
            z = d["z_native"] if "z_native" in d.files else d.get("z_grid")
            if z is not None and sample.meta.z_grid is not None and not np.allclose(z, sample.meta.z_grid):
                raise ValueError(f"{tag}/{sample.id}: prediction z grid differs from the sample's")
            return {"xhi": d["pred"].astype(np.float32), "tb": d["tb_pred"].astype(np.float32)}
    predict.__name__ = attr
    return predict


def describe():
    return {"framework": "precomputed (eval_cubes)"}

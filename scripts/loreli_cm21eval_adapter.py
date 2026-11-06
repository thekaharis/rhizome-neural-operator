"""21cmEVAL python adapter: precomputed rhizome predictions on LoReLi test lightcones.

    --model b8_decoder=python:scripts/loreli_cm21eval_adapter.py:predict_b8_decoder

``predict_<model>`` returns {"xhi", "tb"} from $LORELI_EVAL_PREDS/<model>/<sample id>.npz, written by
``loreli_cm21eval_predict.py`` (which runs the network; see its provenance.json for run, histories,
forward cost and the exactness check against the cache).  Inference cost is therefore not measured
here; do not enable the ``efficiency`` metric.
"""
import json
import os
from pathlib import Path

import numpy as np

PREDS = Path(os.environ.get("LORELI_EVAL_PREDS", Path(__file__).resolve().parents[1] / "runs/cm21eval/preds"))
_names = []


def __getattr__(attr):
    if not attr.startswith("predict_"):
        raise AttributeError(attr)
    name = attr[len("predict_"):]
    if not (PREDS / name).is_dir():
        # AttributeError, not FileNotFoundError: eval21 probes optional hooks
        # (getattr(module, "predict_draws", None)) and must see "absent".
        raise AttributeError(f"no predictions for model {name!r} under {PREDS}")
    _names.append(name)

    def predict(sample):
        with np.load(PREDS / name / f"{sample.id}.npz") as d:
            z = d["z_grid"]
            if sample.meta.z_grid is not None and not np.allclose(z, sample.meta.z_grid):
                raise ValueError(f"{name}/{sample.id}: prediction z grid differs from the sample's")
            return {"xhi": d["xhi"].astype(np.float32), "tb": d["tb"].astype(np.float32)}
    predict.__name__ = attr
    return predict


def describe():
    info = {"framework": "torch (precomputed)", "model": "rhizome_mf2d, slice-wise, 26-ch conditioning"}
    for name in _names:
        prov = PREDS / name / "provenance.json"
        if prov.exists():
            p = json.loads(prov.read_text())
            info.update({"run_dir": p["run"], "best_step": p["step"], "histories": p["histories"],
                         "ms_per_plane_forward": p["ms_per_plane"]})
    return info

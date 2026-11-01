"""Run 21cmEVAL on LoReLi v3 lightcones, with stages taken from the true x_HI also for the T_b pass.

    python scripts/loreli_cm21eval_run.py run --data .../manifest_tb.json --target tb --set target=tb ...

21cmEVAL bins slices into reionization stages by the slice mean of the *target*.  For a T_b
target that would bin by T_b.  When the sample carries the true x_HI as an input
(``manifest_tb.json`` declares inputs [density, xhi]), every stage mask is computed from the
x_HI slice means instead.  For the x_HI pass nothing changes.  Other arguments go to cm21eval.

(``--target`` alone only reaches the data loader; ``Settings.target`` needs ``--set target=...``.)
"""
import importlib
import sys

import numpy as np
import cm21eval.physics as physics
import cm21eval.metrics.base as base
from cm21eval.cli import main

_current = {}
_stage_masks = physics.stage_masks
_target = base.Metric.target


def target(self, sample):
    xhi = sample.inputs.get("xhi")
    _current["xbar"] = None if xhi is None else np.asarray(xhi, np.float64).mean(axis=(0, 1))
    return _target(self, sample)


def stage_masks(xbar, stage_edges, active_range, min_slices=1):
    true = _current.get("xbar")
    if true is not None and np.shape(true) == np.shape(xbar):
        xbar = true
    return _stage_masks(xbar, stage_edges, active_range, min_slices)


base.Metric.target = target
for mod in ("physics", "metrics.base", "metrics.spectrum", "metrics.eft", "metrics.observable"):
    m = importlib.import_module(f"cm21eval.{mod}")
    if hasattr(m, "stage_masks"):
        m.stage_masks = stage_masks
sys.exit(main(sys.argv[1:]))

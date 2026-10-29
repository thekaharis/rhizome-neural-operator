"""Apply a 21cmFAST-trained slice-wise rhizome to LoReLi II without retraining.

The model's spectral kernel is indexed by mode number, so a checkpoint trained
on the 200 Mpc / 140-cell box acts at the wrong physical wavenumbers on a
400 Mpc / 280-cell field. :func:`load_model` therefore builds an inference copy
with the weights remapped by :mod:`ebm21cm.model.remap`; the native model is
returned too, for the 21cmFAST degradation test.

No node phase: LoReLi's lightcone is not built by interpolating between node
boxes, so a checkpoint with ``node_phase`` set is refused rather than fed a
fabricated ``w``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data.loreli import area_matrix
from .model.remap import remapped_model
from .model.rhizome import build_model
from .train_rhizome_mf2d import tb_normalized

TRAINING_CELL_MPC = 200.0 / 140
TRAINING_BOX_MPC = 200.0


def load_model(run, factor=2):
    """(native model, remapped model, checkpoint) from a slice-wise mf2d run directory."""
    best = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    if best.get("node_phase"):
        raise SystemExit(f"{run}: node-phase checkpoint; LoReLi has no node interpolation, use the plain reference")
    config = dict(best["model_config"])
    if config.get("in_channels") != 26:
        raise SystemExit(f"{run}: expected 26 conditioning channels, got {config.get('in_channels')}")
    keep = best.get("scalar_keep")
    expected = 38 - (0 if keep is None else len(keep))
    if config.get("scalar_dim") != expected:
        raise SystemExit(f"{run}: scalar_dim {config.get('scalar_dim')} != {expected} (38 minus dropped history)")
    native = build_model(config).eval()
    native.load_state_dict(best["model"])
    if int(best.get("domain_factor", 1)) > 1:
        # A fine-tuned checkpoint is already stored on the LoReLi 280-cell
        # domain.  Applying the 2x remap again would double the modes.
        return native, native, best
    return native, remapped_model(config, best["model"], factor).eval(), best


@torch.no_grad()
def predict(model, inputs, best, device):
    """(x_HI, T_b [mK]) for one adapter chunk; ``inputs`` as from :meth:`LoreliAdapter.inputs`."""
    cond = torch.as_tensor(inputs["cond"]).to(device)
    scalars = torch.as_tensor(inputs["scalars"]).to(device)
    keep = best.get("scalar_keep")
    if keep is not None:
        scalars = scalars[:, keep]
    out = model(cond, scalars)
    tb_offset, tb_scale = best["tb_norm"]
    tb = tb_normalized(out, torch.as_tensor(inputs["phys"]).to(device), best["head"], tb_offset, tb_scale)
    return (out[:, 0].sigmoid().float().cpu().numpy(),
            (tb[:, 0] * tb_scale + tb_offset).float().cpu().numpy())


def predict_cone(adapter, model, best, device="cuda", batch=16, progress=None):
    """Predict every target slice of one cone; (x_HI, T_b) as (n_target, n_out, n_out)."""
    n = len(adapter.target_distance)
    xhi = np.empty((n, adapter.domain, adapter.domain), np.float32)
    tb = np.empty_like(xhi)
    for a in range(0, n, batch):
        b = min(n, a + batch)
        x, t = predict(model, adapter.inputs(a, b), best, device)
        xhi[a:b], tb[a:b] = x, t
        if progress is not None:
            progress(b, n)
    return xhi, tb


def _concat(parts):
    return torch.cat(parts) if isinstance(parts[0], torch.Tensor) else np.concatenate(parts)


def predict_targets(adapter, model, best, device="cuda", batch=16, targets=None):
    """Predict a sorted subset of target indices; (x_HI, T_b, targets).

    Contiguous runs are built as one adapter batch and concatenated until the
    model batch is full, so a stage-stratified stride does not degenerate into
    batch-size-1 forwards.
    """
    targets = np.asarray(targets, dtype=np.int64)
    if targets.size == 0:
        raise ValueError("empty target selection")
    groups = np.split(targets, np.flatnonzero(np.diff(targets) > 1) + 1)
    xs, ts, pending = [], [], []

    def flush():
        nonlocal pending
        if not pending:
            return
        inputs = {key: _concat([p[key] for p in pending]) for key in ("cond", "scalars", "phys")}
        x, t = predict(model, inputs, best, device)
        xs.append(x)
        ts.append(t)
        pending = []

    for group in groups:
        inputs = adapter.inputs(int(group[0]), int(group[-1]) + 1)
        pending.append(inputs)
        if sum(len(p["z"]) for p in pending) >= batch:
            flush()
    flush()
    return np.concatenate(xs), np.concatenate(ts), targets


def downsample_transverse(planes, adapter, cone):
    """Crop the 280-cell domain to the valid n_out plane and area-average to the native grid."""
    start = (adapter.domain - adapter.n_out) // 2
    planes = planes[..., start:start + adapter.n_out, start:start + adapter.n_out]
    weights = area_matrix(cone.n_native, adapter.n_out)
    return np.einsum("ij,...jk,lk->...il", weights, planes, weights, optimize=True).astype(np.float32)


def border_mask(adapter, cone, border=0):
    """Transverse mask: native cell centres at least ``border`` resampled cells from the plane edge."""
    x = (np.arange(cone.n_native) + 0.5) * (cone.l_perp / cone.n_native)
    edge = np.minimum(x, cone.l_perp - x)
    return np.minimum(edge[:, None], edge[None, :]) >= border * (cone.l_perp / adapter.n_out)


def map_to_native(xhi, tb, adapter, cone, border=0):
    """Crop the 280-cell prediction, average to the 128-cell LoReLi grid, and interpolate to native z.

    ``border`` is in resampled cells: native cells whose centres lie within that
    physical distance of the plane edge are marked False in the returned mask.
    Returns (x_HI, T_b) as (n_native, 128, 128), NaN outside the valid run.
    """
    xhi = downsample_transverse(xhi, adapter, cone)
    tb = downsample_transverse(tb, adapter, cone)
    valid = np.flatnonzero(cone.valid)
    position = np.interp(cone.distances[valid], adapter.target_distance, np.arange(len(adapter.target_distance)))
    i0 = np.floor(position).astype(np.int64).clip(0, len(adapter.target_distance) - 1)
    i1 = (i0 + 1).clip(0, len(adapter.target_distance) - 1)
    t = (position - i0).astype(np.float32)[:, None, None]
    shape = (len(cone.z), cone.n_native, cone.n_native)
    out_x = np.full(shape, np.nan, np.float32)
    out_t = np.full(shape, np.nan, np.float32)
    out_x[valid] = xhi[i0] * (1 - t) + xhi[i1] * t
    out_t[valid] = tb[i0] * (1 - t) + tb[i1] * t
    return out_x, out_t, border_mask(adapter, cone, border)

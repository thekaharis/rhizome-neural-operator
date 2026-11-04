"""Run a trained 2-D spectral model on a larger periodic domain.

The spectral convolution in :mod:`ebm21cm.model.recurrent` retains integer
modes ``-(modes-1) .. modes-1`` of whatever grid it is given. A model trained
on ``L`` therefore acts at twice the physical wavenumber on a grid of side
``2L``: new mode ``n`` has ``k = 2*pi*n/(2L) = 2*pi*(n/2)/L``, i.e. it is the
trained mode ``m = n/2``.

:func:`remap_spectral_conv` re-indexes those weights onto the larger domain.
Even modes are exact copies of the trained mode; odd modes are linearly
interpolated between the two neighbouring trained modes (complex-linear, so
real and imaginary parts separately), clamped at the top of the trained band.
The DC block is copied and the pointwise maps, gates and scalar embedding are
untouched. A field that is itself periodic on the small domain (a ``2 x 2``
tiling) has power only at even modes, so the remapped model reproduces the
native prediction exactly up to floating point: the ``2 x 2`` exactness test
in ``tests/test_loreli.py`` and ``scripts/check_loreli_domain.py`` pins this.

The one-based kx axis of ``positive``, and the ky axis of ``axis`` and
``negative``, start at mode 1 rather than 0; the ``offset`` argument below keeps
that bookkeeping explicit. Getting it wrong shifts the whole band by one mode,
which the tiling test catches.
"""

from __future__ import annotations

import torch

from .rhizome import build_model


def _interp(w, n, dim, factor, offset=0):
    """Resample a spectral weight along ``dim`` from ``m`` to ``factor*m`` modes.

    Result index ``j`` is physical mode ``j + offset``; it maps to trained index
    ``(j + offset)/factor - offset``. Neighbours are averaged linearly and the
    ends are clamped to the trained band.
    """
    m = w.shape[dim]
    pos = (torch.arange(n, dtype=torch.float64) + offset) / factor - offset
    i0 = pos.floor().clamp(0, m - 1).long()
    i1 = (i0 + 1).clamp(0, m - 1)
    t = (pos - i0).clamp(0, 1).to(w.real.dtype)
    shape = [1] * w.ndim
    shape[dim] = n
    t = t.view(shape)
    return w.index_select(dim, i0) * (1 - t) + w.index_select(dim, i1) * t


def remap_spectral_conv(dc, axis, positive, negative, modes, factor=2):
    """New ``(dc, axis, positive, negative)`` tensors for ``factor`` times the domain.

    Shapes follow ``SpectralConv2d``: ``dc`` (in, out), ``axis`` (in, out, m-1)
    for ky = 1..m-1 at kx = 0, ``positive`` (in, out, m, m-1) for ky = 0..m-1 and
    kx = 1..m-1, ``negative`` (in, out, m-1, m-1) for ky = -(m-1)..-1 and the same
    kx. The returned tensors have ``new = factor*m`` retained modes.
    """
    if not factor >= 1:
        raise ValueError("factor must be >= 1")
    new = int(factor * modes)
    if int(factor) != factor:
        raise ValueError("the retained band must scale by an integer factor")
    dc = dc.detach().clone()
    if axis.numel() == 0:  # modes = 1: no off-DC modes to remap
        axis_out = axis.new_zeros(axis.shape[:2] + (new - 1,))
        positive_out = positive.new_zeros(positive.shape[:2] + (new, new - 1))
        negative_out = negative.new_zeros(negative.shape[:2] + (new - 1, new - 1))
        return dc, axis_out, positive_out, negative_out
    axis_out = _interp(axis, new - 1, 2, factor, offset=1)
    positive_out = _interp(_interp(positive, new, 2, factor, offset=0), new - 1, 3, factor, offset=1)
    negative_out = _interp(_interp(negative, new - 1, 2, factor, offset=1), new - 1, 3, factor, offset=1)
    return dc, axis_out, positive_out, negative_out


def remap_state_dict(state, factor=2):
    """A model state dict for a domain ``factor`` times wider, spectral weights remapped."""
    out, modes = {}, {}
    for key, value in state.items():
        suffix = key.rsplit(".", 1)[-1]
        if suffix in ("axis", "positive", "negative"):
            m = value.shape[2] + (0 if suffix == "positive" else 1)
            if modes.setdefault(key.rsplit(".", 1)[0], m) != m:
                raise ValueError(f"{key}: retained modes disagree with its block")
        if suffix == "dc":
            out[key] = value.detach().clone()
        elif suffix in ("axis", "positive", "negative"):
            out[key] = None
        else:
            out[key] = value.detach().clone()
    for block, m in modes.items():
        dc, axis, positive, negative = remap_spectral_conv(
            state[f"{block}.dc"], state[f"{block}.axis"], state[f"{block}.positive"],
            state[f"{block}.negative"], m, factor)
        out[f"{block}.dc"], out[f"{block}.axis"] = dc, axis
        out[f"{block}.positive"], out[f"{block}.negative"] = positive, negative
    return out


def remapped_model(config, state, factor=2):
    """A new model with the same weights on a ``factor`` times wider periodic domain."""
    config = dict(config)
    for key in ("modes", "coarse_modes"):
        if key in config:
            config[key] = int(config[key]) * int(factor)
    model = build_model(config)
    model.load_state_dict(remap_state_dict(state, factor))
    return model

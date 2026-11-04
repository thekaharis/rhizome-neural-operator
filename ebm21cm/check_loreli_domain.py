"""Domain-transfer checks for the LoReLi adapter: 2 x 2 exactness and the mirror-seam edge curve.

    python -m ebm21cm.check_loreli_domain --run runs/rhizome_mf2d_w64_structured_all \
        --cache /pfs/.../rhizome_mf_slices_all.h5 --out runs/eval3d/loreli/domain_check

A **2 x 2 tiling** of a 21cmFAST slice is periodic on the 400 Mpc domain, so
the remapped model must reproduce the native prediction tiled, to floating
point. The **edge test** takes a centred crop of a real slice, mirror-pads it
to 280 cells (as the LoReLi adapter does) and compares the prediction with the
native model on the full periodic slice; the error is binned by distance to the
seam. Two crops are used: 100 cells (the conservative test) and 206 cells (the
actual LoReLi valid-plane size, padded by 37 cells per side).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .data.loreli import pad_to
from .loreli_inference import load_model
from .train_rhizome_mf2d import tb_normalized


def edge_curve(err2, side):
    """RMS error per distance (cells) from the nearest edge of a side x side crop."""
    d = np.minimum(np.arange(side), np.arange(side)[::-1])
    distance = np.minimum(d[:, None], d[None, :])
    sums = np.bincount(distance.ravel(), weights=err2.ravel(), minlength=side // 2 + 1)
    counts = np.bincount(distance.ravel(), minlength=side // 2 + 1)
    return np.sqrt(np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0))


def pad_batch(cond, side, size=280):
    """Mirror-pad every channel of one (C, H, W) crop to (C, size, size)."""
    return torch.stack([torch.as_tensor(pad_to(cond[c].numpy(), size)) for c in range(len(cond))])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=6, help="cache test rows to average")
    ap.add_argument("--sides", type=int, nargs="+", default=[100, 206])
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    native, model, best = load_model(args.run)
    with h5py.File(args.cache, "r") as h:
        rows = np.flatnonzero(h["split"][:] == 2)[:args.rows]
        cond = torch.as_tensor(np.asarray(h["cond"][rows], np.float32))
        scalars = torch.as_tensor(np.asarray(h["scalars"][rows], np.float32))
        phys = torch.as_tensor(np.asarray(h["phys"][rows], np.float32))
        z = h["z"][rows]
    print(f"{len(rows)} test rows: z {z.min():.2f}..{z.max():.2f}", flush=True)

    with torch.no_grad():
        native_out = native(cond, scalars)
    tb_offset, tb_scale = best["tb_norm"]
    with torch.no_grad():
        native_x = native_out[:, 0].sigmoid()
        native_tb = tb_normalized(native_out, phys[:, None], best["head"], tb_offset, tb_scale)[:, 0]

    result = {"rows": rows.tolist(), "z": z.tolist(), "tiling": {}, "edge": {}}
    for side in args.sides:
        curves_x, curves_tb = [], []
        for i in range(len(rows)):
            if side <= 140:
                start = (140 - side) // 2
                crop = cond[i, :, start:start + side, start:start + side]
                crop_phys = phys[i, start:start + side, start:start + side]
                target_x = native_x[i, start:start + side, start:start + side]
                target_tb = native_tb[i, start:start + side, start:start + side]
            else:  # a 2 x 2 tiled slice, the LoReLi-size valid plane
                tiled = torch.cat([torch.cat([cond[i], cond[i]], -1)] * 2, -2)
                tiled_phys = torch.cat([torch.cat([phys[i], phys[i]], -1)] * 2, -2)
                with torch.no_grad():
                    full = model(tiled[None], scalars[i:i + 1])
                full_x = full[0, 0].sigmoid()
                full_tb = tb_normalized(full, tiled_phys[None, None], best["head"], tb_offset, tb_scale)[0, 0]
                start = (280 - side) // 2
                crop = tiled[:, start:start + side, start:start + side]
                crop_phys = tiled_phys[start:start + side, start:start + side]
                target_x = full_x[start:start + side, start:start + side]
                target_tb = full_tb[start:start + side, start:start + side]
            with torch.no_grad():
                out = model(pad_batch(crop, side)[None], scalars[i:i + 1])
                x = out[0, 0].sigmoid()
                padded_phys = torch.as_tensor(pad_to(crop_phys.numpy(), 280))[None, None]
                tb = tb_normalized(out, padded_phys, best["head"], tb_offset, tb_scale)[0, 0]
            pad = (280 - side) // 2
            x = x[pad:pad + side, pad:pad + side]
            tb = tb[pad:pad + side, pad:pad + side]
            curves_x.append(edge_curve((x - target_x).numpy() ** 2, side))
            curves_tb.append(edge_curve((tb - target_tb).numpy() ** 2, side))
        distance = np.arange(len(curves_x[0]))
        result["edge"][str(side)] = {
            "distance_cells": distance.tolist(),
            "xhi_rms": np.nanmean(curves_x, 0).tolist(),
            "tb_rms_mk": np.nanmean(curves_tb, 0).tolist(),
            "xhi_rms_worst_row": np.nanmax(curves_x, 0).tolist(),
            "crop_xhi_mean": [float(t.mean()) for t in curves_x]}

    # The 2 x 2 tiling exactness is side-independent; report it explicitly for the full 140 slice.
    with torch.no_grad():
        tiled = torch.cat([torch.cat([cond, cond], -1)] * 2, -2)
        remapped = model(tiled, scalars)
    err = (remapped[:, :, :140, :140] - native_out).abs()
    result["tiling"] = {"max_abs_output_error": float(err.max()),
                        "max_abs_xhi_error": float((remapped[:, 0, :140, :140].sigmoid()
                                                    - native_x).abs().max()),
                        "output_scale": float(native_out.abs().max())}
    (args.out / "domain_check.json").write_text(json.dumps(result, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for side in args.sides:
            entry = result["edge"][str(side)]
            d = np.asarray(entry["distance_cells"])
            axes[0].plot(d, entry["xhi_rms"], label=f"{side}-cell crop")
            axes[1].plot(d, entry["tb_rms_mk"], label=f"{side}-cell crop")
        axes[0].set(xlabel="distance from mirror seam [cells]", ylabel="x_HI RMS", yscale="log")
        axes[1].set(xlabel="distance from mirror seam [cells]", ylabel="T_b RMS [mK]", yscale="log")
        for ax in axes:
            ax.axvline(8, color="0.6", ls="--")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("LoReLi adapter: error from the mirror seam")
        fig.savefig(args.out / "edge_curve.png", dpi=160)
    except ImportError:
        pass
    summary = {k: result[k] for k in ("tiling", "z")}
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()

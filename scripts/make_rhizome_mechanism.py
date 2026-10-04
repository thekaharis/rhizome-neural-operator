"""Paper figure: the rhizome interaction of a trained checkpoint on one slice.

    python scripts/make_rhizome_mechanism.py \\
        --checkpoint runs/rhizome_long_seed1/fit/best.pt \\
        --cache runs/rhizome_long_seed1/slices.h5 --out docs/figures/rhizome_mechanism

For one receiver cell i on an ionization front, it plots the learned kernel
norm ||kappa(r_i - r_j)||_F, the channel-RMS receiver and source gates at the
last update, and, for every update t, how the gates re-weight the incoming
connections of cell i relative to the ungated kernel:

    M_t(j) = ||diag(A_i) kappa(r_i - r_j) diag(B_j)||_F / ||kappa(r_i - r_j)||_F

shown as M_t / mean_j M_t (spatial re-weighting) with the mean gain printed.
Kernel entries are impulse responses of the trained SpectralConv2d, so the
maps use the model's own weights. They are gains on the transmitted value V_j,
not a Jacobian of the output, and not calibrated physical couplings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ebm21cm.data.cache import SliceDataset  # noqa: E402
from ebm21cm.train_recurrent import load_checkpoint  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


def kernel_impulse(kernel, width, h, w):
    """K[o, i, dy, dx]: output channel o at displacement (dy, dx) from a unit source in channel i."""
    x = torch.zeros(width, width, h, w, dtype=torch.float64)
    x[torch.arange(width), torch.arange(width), 0, 0] = 1.0
    with torch.no_grad():
        return kernel.double()(x).permute(1, 0, 2, 3).numpy()


def pick_slice(ds):
    """Validation row whose true neutral fraction is closest to one half."""
    means = np.array([float((ds[k]["target"][0] + 1).mean() / 2) for k in range(len(ds))])
    return int(np.argmin(np.abs(means - 0.5))), means


def pick_receiver(truth):
    """Front cell (neutral next to ionized) closest to the box centre."""
    xh = truth > 0.5
    front = xh & (~np.roll(xh, 1, 0) | ~np.roll(xh, -1, 0) | ~np.roll(xh, 1, 1) | ~np.roll(xh, -1, 1))
    ys, xs = np.nonzero(front)
    h, w = truth.shape
    k = np.argmin((ys - h / 2) ** 2 + (xs - w / 2) ** 2)
    return int(ys[k]), int(xs[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--row", type=int, default=None, help="dataset row; default: mean x_HI closest to 0.5")
    ap.add_argument("--band", type=int, default=6, help="density band shown in panel (a)")
    ap.add_argument("--out", default="docs/figures/rhizome_mechanism")
    args = ap.parse_args()

    model, ck = load_checkpoint(args.checkpoint)
    model = model.double().eval()
    ds = SliceDataset(args.cache, args.split, ck["stats"])
    row, means = (args.row, None) if args.row is not None else pick_slice(ds)
    item = ds[row]
    cond = item["cond"][None].double()
    scalars = item["scalars"][None].double()
    truth = ((item["target"][0] + 1) / 2).numpy()

    with torch.no_grad():
        logits, states = model(cond, scalars, return_states=True)
        forcing = model.lift(cond) + model.scalar_embed(scalars)[:, :, None, None]
        inter = model.cells[0].interaction
        factors = [inter.factors(s, forcing) for s in states[:-1]]
    pred = logits.sigmoid()[0, 0].numpy()
    width, (h, w) = model.width, truth.shape

    K = kernel_impulse(inter.kernel, width, h, w)
    iy, ix = pick_receiver(truth)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    Kd = K[:, :, (iy - yy) % h, (ix - xx) % w]           # (C_out, C_in, H, W): kappa(r_i - r_j)
    kernel_norm = np.sqrt((Kd ** 2).sum((0, 1)))
    eff, mod, rms_a, rms_b = [], [], [], []
    for source, receiver, _ in factors:
        B, Afield = source[0].numpy(), receiver[0].numpy()
        A = Afield[:, iy, ix]
        e = np.sqrt(np.einsum("o,oiyx,iyx->yx", A ** 2, Kd ** 2, B ** 2))
        eff.append(e)
        mod.append(e / kernel_norm)
        rms_a.append(np.sqrt((Afield ** 2).mean(0)))
        rms_b.append(np.sqrt((B ** 2).mean(0)))
    gains = [float(m.mean()) for m in mod]
    rel = [m / m.mean() for m in mod]
    span = max(abs(r - 1).max() for r in rel)

    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.size": 8,
        "axes.titlesize": 8, "axes.linewidth": 0.5,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    })
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.35), constrained_layout=True)

    def show(ax, img, cmap, title, norm=None, cbar=True, label=None, ring="#009E73"):
        im = ax.imshow(img, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
        ax.contour(truth, levels=[0.5], colors="w" if cmap == "gray" else "k", linewidths=0.45, alpha=0.75)
        ax.plot(ix, iy, marker="o", ms=5, mfc="none", mec=ring, mew=1.3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, pad=3)
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
            if label: cb.set_label(label, fontsize=6.5)
        return im

    bands = json.loads(ds.attrs["bands"])
    T = len(factors)
    show(axes[0, 0], cond[0, args.band].numpy(), "gray",
         r"(a) density $\delta$, band " + f"{bands[args.band]}".replace(" ", ""), label="normalized")
    show(axes[0, 1], kernel_norm / kernel_norm.max(), "Blues",
         r"(b) kernel $\|\kappa_\theta(\mathbf r_i-\mathbf r_j)\|_F$", label="rel.\\ to max")
    show(axes[0, 2], rms_a[-1], "Greens", rf"(c) receiver gate $\|\mathbf A\|_{{\mathrm{{rms}}}}$, $t{{=}}{T}$",
         label=r"$\in(0,2)$", ring="k")
    show(axes[0, 3], rms_b[-1], "Oranges", rf"(d) source gate $\|\mathbf B\|_{{\mathrm{{rms}}}}$, $t{{=}}{T}$",
         label=r"$\in(0,2)$")
    rnorm = TwoSlopeNorm(1.0, 1 - span, 1 + span)
    for t, ax in enumerate(axes[1]):
        show(ax, rel[t], "PuOr_r", rf"({'efgh'[t]}) $M_t$, update $t{{=}}{t + 1}$", norm=rnorm,
             cbar=t == T - 1, label=r"$M_t(j)\,/\,\langle M_t\rangle_j$")
        ax.text(0.03, 0.04, rf"gain $\langle M_t\rangle={gains[t]:.2f}$", transform=ax.transAxes, fontsize=6.5,
                bbox=dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.8))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    meta = {
        "checkpoint": str(args.checkpoint), "checkpoint_step": ck.get("step"), "split": args.split,
        "row": row, "cone_id": int(item["cone_id"]), "redshift": float(item["z"]),
        "true_mean_xhi": float(truth.mean()), "receiver_yx": [iy, ix],
        "mean_gate_gain_per_update": gains,
        "reweighting_range_per_update": [[float(r.min()), float(r.max())] for r in rel],
        "effective_shape_change_first_to_last": float(np.linalg.norm(
            eff[-1] / np.linalg.norm(eff[-1]) - eff[0] / np.linalg.norm(eff[0]))),
        "prediction_rmse": float(np.sqrt(((pred - truth) ** 2).mean())),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

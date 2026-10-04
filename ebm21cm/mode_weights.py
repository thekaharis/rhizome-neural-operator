"""Spectral mode-weight diagnostic for 2-D rhizome checkpoints.

    python -m ebm21cm.mode_weights --out runs/eval3d/mode_weights \\
        --run "x_HI w64=runs/rhizome_xhi_w64_fnosplit" --run "sweep m48=runs/sweep1/m48"

The 2-D counterpart of the SirenFNO note's diagnostic: the RMS over the
(in, out) channel matrix of the learned Fourier kernel at every retained mode
(ky, kx), relative to the initialization scale (in*out)^-1/2, so 1 = untouched.
Per run it reports the radial profile over |k| (mode units and Mpc^-1 on the
200 Mpc / 140-cell box), the low-25% fraction (share of the summed per-mode RMS
held by modes with |k| <= m/4, against the share those modes have under a flat
profile), and the cutoff ratio (mean RMS in the outer quarter of the band
3m/4 < |k| <= m-1 over the inner quarter |k| <= m/4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BOX_MPC = 200.0


def mode_map(state, prefix):
    """(ky, kx, rms) for every retained mode of one SpectralConv2d, rms relative to init."""
    dc, axis = state[prefix + "dc"].float(), state[prefix + "axis"]
    pos, neg = state[prefix + "positive"], state[prefix + "negative"]
    n_in, n_out = dc.shape
    scale = (n_in * n_out) ** -0.5
    rms = lambda w: (w.abs().square().mean((0, 1)).sqrt() / scale).numpy()
    m = pos.shape[2]
    out = [(0, 0, float(dc.square().mean().sqrt() / scale))]
    out += [(ky, 0, v) for ky, v in zip(range(1, m), rms(axis))]
    p, n = rms(pos), rms(neg)
    out += [(ky, kx, p[ky, kx - 1]) for ky in range(m) for kx in range(1, m)]
    out += [(ky, kx, n[ky + m - 1, kx - 1]) for ky in range(-(m - 1), 0) for kx in range(1, m)]
    return np.array(out, dtype=np.float64), m


def summarize(modes, m):
    k = np.hypot(modes[:, 0], modes[:, 1])
    w = modes[:, 2]
    inside = k <= m - 1
    inner, outer = inside & (k <= (m - 1) / 4), inside & (k > 3 * (m - 1) / 4)
    flat_share = inner.sum() / inside.sum()
    return {"modes": m, "mean_rms": float(w[inside].mean()),
            "low25_fraction": float(w[inner].sum() / w[inside].sum()), "low25_flat": float(flat_share),
            "cutoff_ratio": float(w[outer].mean() / w[inner].mean()),
            "k_max_mpc": float(2 * np.pi * (m - 1) / BOX_MPC)}


def radial(modes, m, bins=None):
    k = np.hypot(modes[:, 0], modes[:, 1])
    edges = np.arange(0, m) if bins is None else bins
    centers, values = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (k >= a) & (k < b)
        if sel.any():
            centers.append(0.5 * (a + b))
            values.append(modes[sel, 2].mean())
    return np.array(centers), np.array(values)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=run_dir (best.pt)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    table, profiles, maps = {}, {}, {}
    for spec in a.run:
        label, path = spec.split("=", 1)
        state = torch.load(Path(path) / "best.pt", map_location="cpu", weights_only=False)["model"]
        prefixes = sorted({k[: -len("dc")] for k in state if k.endswith("kernel.dc")})
        per_cell = [mode_map(state, p) for p in prefixes]
        modes = per_cell[0][0].copy()
        modes[:, 2] = np.mean([pc[0][:, 2] for pc in per_cell], axis=0)
        m = per_cell[0][1]
        table[label] = {**summarize(modes, m), "kernels": len(prefixes),
                        "width": int(state[prefixes[0] + "dc"].shape[0])}
        profiles[label] = radial(modes, m)
        maps[label] = (modes, m)
        print(label, json.dumps(table[label]))
    (a.out / "mode_weights.json").write_text(json.dumps(table, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, (c, v) in profiles.items():
        axes[0].plot(2 * np.pi * c / BOX_MPC, v, marker="o", ms=3, label=label)
    axes[0].axhline(1, color="grey", ls="--", lw=1, label="initialization")
    axes[0].set_xlabel(r"$|k_\perp|$  [Mpc$^{-1}$]")
    axes[0].set_ylabel("per-mode kernel RMS / init scale")
    axes[0].set_title("Learned Fourier-kernel weight vs |k| (radial mean)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    labels = list(table)
    x = np.arange(len(labels))
    axes[1].bar(x - 0.2, [table[l]["cutoff_ratio"] for l in labels], 0.4, label="cutoff ratio (outer/inner quarter)")
    axes[1].bar(x + 0.2, [table[l]["low25_fraction"] / table[l]["low25_flat"] for l in labels], 0.4,
                label="low-25% share / flat share")
    axes[1].axhline(1, color="grey", ls="--", lw=1)
    axes[1].set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_title("Spectral bias summary (1 = flat)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "mode_weight_profiles.png", dpi=140)
    plt.close(fig)

    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    for ax, (label, (modes, m)) in zip(axes[0], maps.items()):
        grid = np.full((2 * m - 1, m), np.nan)
        for ky, kx, v in modes:
            grid[int(ky) + m - 1, int(kx)] = v
        im = ax.imshow(grid, origin="lower", aspect="auto", extent=(-0.5, m - 0.5, -(m - 0.5), m - 0.5), cmap="viridis")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("$k_x$ (mode)")
        ax.set_ylabel("$k_y$ (mode)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(a.out / "mode_weight_maps.png", dpi=140)
    plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

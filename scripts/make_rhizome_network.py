"""Paper figure: how cells exchange information, tree versus rhizome.

    python scripts/make_rhizome_network.py \\
        --checkpoint runs/rhizome_long_seed1/fit/best.pt \\
        --cache runs/rhizome_long_seed1/slices.h5 --out docs/figures/rhizome_network

(a) A U-Net-style hierarchy: cells meet only at common coarse ancestors.
(b) An ungated global kernel, as in a Fourier layer: the trained rhizome's own
    kappa with A = B = 1, so the same pattern surrounds every cell.
(c) The trained rhizome at its last update: edge (i, j) carries
    ||diag(A_i) kappa(r_i - r_j) diag(B_j)||_F, so every cell sends and listens
    with its own state-dependent weights.
Panels (b) and (c) draw the incoming connections ("fans") of the same few
receiver cells on an 8x8 lattice, chosen to include the strongest and weakest
listener, with the same width/opacity mapping relative to each panel's maximum.
Edges follow the minimal-image (periodic) displacement; edges that wrap across
the boundary are omitted. Network statistics over all lattice pairs are
recorded in the JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ebm21cm.data.cache import SliceDataset  # noqa: E402
from ebm21cm.train_recurrent import load_checkpoint  # noqa: E402
from make_rhizome_mechanism import kernel_impulse, pick_slice  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#4D4D4D"


def lattice(n_cells, n_nodes):
    step = n_cells // n_nodes
    c = np.arange(n_nodes) * step + step // 2
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.stack([yy.ravel(), xx.ravel()], 1), step


def fans(ax, nodes, weights, receivers, n, color, scale, thresh=0.1):
    """Incoming edges w[r, j] / scale into each receiver r, lightest drawn first."""
    order = []
    for r in receivers:
        for j in range(len(nodes)):
            if j == r:
                continue
            d = (nodes[r] - nodes[j] + n // 2) % n - n // 2
            if np.any(nodes[j] + d != nodes[r]):
                continue                                       # wraps around the torus
            wt = weights[r, j] / scale
            if wt >= thresh:
                order.append((wt, j, r))
    order.sort()
    rgb = matplotlib.colors.to_rgb(color)
    paths = [MPath([nodes[j][::-1].astype(float), nodes[r][::-1].astype(float)])
             for _, j, r in order]
    ax.add_collection(PathCollection(
        paths, facecolors="none", capstyle="round", zorder=2,
        edgecolors=[(*rgb, 0.10 + 0.85 * wt ** 1.2) for wt, _, _ in order],
        linewidths=[0.25 + 2.4 * wt ** 1.2 for wt, _, _ in order]))
    return len(order)


def pick_receivers(nodes, a_rms, n_axis, k=4):
    """Strongest and weakest listener away from the border, then spread out the rest."""
    idx = np.arange(len(nodes))
    iy, ix = idx // n_axis, idx % n_axis
    inner = idx[(iy > 0) & (iy < n_axis - 1) & (ix > 0) & (ix < n_axis - 1)]
    def dist(a, b):
        d = np.abs(np.array([iy[a] - iy[b], ix[a] - ix[b]]))
        return np.hypot(*np.minimum(d, n_axis - d))
    chosen = [int(inner[np.argmax(a_rms[inner])])]
    far = [i for i in inner if dist(i, chosen[0]) >= 3]
    chosen.append(int(far[np.argmin(a_rms[far])]))
    while len(chosen) < k:
        chosen.append(int(max((i for i in inner if i not in chosen), key=lambda i: min(dist(i, c) for c in chosen))))
    return chosen


def background(ax, truth, n):
    cmap = matplotlib.colors.ListedColormap(["white", "#E6ECF3"])
    ax.imshow(truth > 0.5, cmap=cmap, origin="lower", interpolation="nearest", zorder=0)
    ax.contour(truth, levels=[0.5], colors="#9AA5B4", linewidths=0.45, zorder=1)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#666666")


def tree_panel(ax, n, levels, a_leaf, b_leaf):
    """Quadtree: leaves (finest) link to block parents up to one root."""
    nodes_by_level = []
    for lev in range(levels + 1):
        k = 2 ** (levels - lev)                                # nodes per axis at this level
        s = n / k
        c = (np.arange(k) + 0.5) * s
        nodes_by_level.append([(x, y) for y in c for x in c])
    path_nodes = set()
    for leaf in (a_leaf, b_leaf):
        x, y = leaf
        for lev in range(levels + 1):
            k = 2 ** (levels - lev); s = n / k
            path_nodes.add((lev, int(x // s), int(y // s)))
    for lev in range(levels):
        k = 2 ** (levels - lev); s = n / k
        for (x, y) in nodes_by_level[lev]:
            ix, iy = int(x // s), int(y // s)
            px, py = ((ix // 2) + 0.5) * 2 * s, ((iy // 2) + 0.5) * 2 * s
            on = (lev, ix, iy) in path_nodes
            ax.plot([x, px], [y, py], color=GREEN if on else "#8C8C8C", lw=(1.6 if on else 0.35 + 0.3 * lev),
                    alpha=0.95 if on else 0.7, zorder=3 if on else 2, solid_capstyle="round")
    for lev, pts in enumerate(nodes_by_level):
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=3 + 9 * lev ** 1.6, c="white", edgecolors="#6E6E6E", linewidths=0.55, zorder=4)
    for leaf in (a_leaf, b_leaf):
        ax.scatter(*leaf, s=22, c=GREEN, edgecolors="k", linewidths=0.4, zorder=5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--nodes", type=int, default=8, help="lattice nodes per axis")
    ap.add_argument("--update", type=int, default=0, help="update shown in (c), 1-based; 0 = last")
    ap.add_argument("--out", default="docs/figures/rhizome_network")
    args = ap.parse_args()

    model, ck = load_checkpoint(args.checkpoint)
    model = model.double().eval()
    ds = SliceDataset(args.cache, args.split, ck["stats"])
    row = args.row if args.row is not None else pick_slice(ds)[0]
    item = ds[row]
    cond, scalars = item["cond"][None].double(), item["scalars"][None].double()
    truth = ((item["target"][0] + 1) / 2).numpy()
    n = truth.shape[0]

    with torch.no_grad():
        _, states = model(cond, scalars, return_states=True)
        forcing = model.lift(cond) + model.scalar_embed(scalars)[:, :, None, None]
        inter = model.cells[0].interaction
        t = (args.update or len(states) - 1) - 1
        B, A, _ = (f[0].numpy() for f in inter.factors(states[t], forcing))
    K2 = kernel_impulse(inter.kernel, model.width, n, n) ** 2  # (C_out, C_in, H, W)

    nodes, step = lattice(n, args.nodes)
    m = len(nodes)
    kern, eff = np.zeros((m, m)), np.zeros((m, m))             # [receiver, source]
    eff_a, eff_b = np.zeros((m, m)), np.zeros((m, m))          # receiver gate only / sender gate only
    for a, (ya, xa) in enumerate(nodes):
        Ai2 = A[:, ya, xa] ** 2
        for b, (yb, xb) in enumerate(nodes):
            if a == b:
                continue
            Kab = K2[:, :, (ya - yb) % n, (xa - xb) % n]
            kern[a, b] = np.sqrt(Kab.sum())
            Bj2 = B[:, yb, xb] ** 2
            eff[a, b] = np.sqrt(Ai2 @ Kab @ Bj2)
            eff_a[a, b] = np.sqrt(Ai2 @ Kab.sum(1))
            eff_b[a, b] = np.sqrt(Kab.sum(0) @ Bj2)
    rx = int(np.argmin(((nodes - np.array([n // 2, n // 2])) ** 2).sum(1)))

    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.85), constrained_layout=True)

    background(axes[0], truth, n)
    tree_panel(axes[0], n, int(np.log2(args.nodes)), (float(nodes[rx][1]), float(nodes[rx][0])),
               (float(nodes[rx][1] + step), float(nodes[rx][0] + step)))
    axes[0].set_title(r"(a) tree: U-Net hierarchy", pad=3)
    axes[0].set_xlabel("neighbouring cells across a block\nboundary meet only at the root", fontsize=7, labelpad=3)

    a_rms = np.sqrt((A ** 2).mean(0))[nodes[:, 0], nodes[:, 1]]
    b_rms = np.sqrt((B ** 2).mean(0))[nodes[:, 0], nodes[:, 1]]
    recv = pick_receivers(nodes, a_rms, args.nodes)

    def lattice_dots(ax):
        ax.scatter(nodes[:, 1], nodes[:, 0], s=5, c="#A7A7A7", linewidths=0, zorder=3)

    background(axes[1], truth, n)
    ek = fans(axes[1], nodes, kern, recv, n, BLUE, kern[recv].max())
    lattice_dots(axes[1])
    axes[1].scatter(nodes[recv, 1], nodes[recv, 0], s=26, c="white", edgecolors="k", linewidths=0.8, zorder=5)
    axes[1].set_title(r"(b) global, fixed: $\kappa_\theta(\mathbf r_i-\mathbf r_j)$", pad=3)
    axes[1].set_xlabel("every cell hears every other through\nthe same pattern (Fourier layer)", fontsize=7, labelpad=3)

    background(axes[2], truth, n)
    ee = fans(axes[2], nodes, eff, recv, n, BLUE, eff[recv].max())
    bn = (b_rms - b_rms.min()) / (np.ptp(b_rms) or 1)
    axes[2].scatter(nodes[:, 1], nodes[:, 0], s=9, c=plt.get_cmap("Oranges")(0.2 + 0.8 * bn),
                    edgecolors="#9A5A2E", linewidths=0.3, zorder=3)
    axes[2].scatter(nodes[recv, 1], nodes[recv, 0], s=10 + 140 * (a_rms[recv] / a_rms.max()) ** 2, c="none",
                    edgecolors=GREEN, linewidths=1.3, zorder=5)
    axes[2].scatter(nodes[recv, 1], nodes[recv, 0], s=12, c="k", linewidths=0, zorder=6)
    for r in recv:
        axes[2].annotate(rf"$A_i^{{\mathrm{{rms}}}}={a_rms[r]:.2f}$", (nodes[r][1], nodes[r][0]), xytext=(5, 6),
                         textcoords="offset points", fontsize=6, color="#0B6B4F",
                         bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=7)
    axes[2].set_title(rf"(c) rhizome: $\mathbf A_i\,\kappa_\theta\,\mathbf B_j$, update $t{{=}}{t + 1}$", pad=3)
    axes[2].set_xlabel("edge $j\\to i$ scaled by the receiver gate\n$\\mathbf A_i$ (ring) and the sender gate $\\mathbf B_j$ (shade)", fontsize=7, labelpad=3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    off = ~np.eye(m, dtype=bool)
    asym = np.abs(eff - eff.T)[off] / (eff + eff.T)[off]
    kn, en = kern / kern.max(), eff / eff.max()
    meta = {
        "checkpoint": str(args.checkpoint), "checkpoint_step": ck.get("step"), "split": args.split, "row": row,
        "cone_id": int(item["cone_id"]), "redshift": float(item["z"]), "update_shown": t + 1,
        "lattice_nodes": int(m), "lattice_step_cells": int(step),
        "receivers_yx": nodes[recv].tolist(), "receiver_gate_rms": [float(a_rms[r]) for r in recv],
        "edges_drawn_kernel": ek, "edges_drawn_rhizome": ee, "edge_threshold_rel": 0.1, "wrapping_edges_omitted": True,
        "in_strength_cv_kernel": float(kn.sum(1).std() / kn.sum(1).mean()),
        "in_strength_cv_rhizome": float(en.sum(1).std() / en.sum(1).mean()),
        "out_strength_cv_rhizome": float(en.sum(0).std() / en.sum(0).mean()),
        # log of the gate factor eff/kern, split into receiver-only (B=1) and sender-only (A=1) parts
        "gate_factor_log_variance": {"total": float(np.var(np.log(eff[off] / kern[off]))),
                                      "receiver_only": float(np.var(np.log(eff_a[off] / kern[off]))),
                                      "sender_only": float(np.var(np.log(eff_b[off] / kern[off])))},
        "fan_receiver_factor": [float((eff_a[r] / kern[r])[np.arange(m) != r].mean()) for r in recv],
        "fan_sender_factor_range": [[float(v.min()), float(v.max())] for v in
                                    ((eff[r] / eff_a[r])[np.arange(m) != r] for r in recv)],
        "corr_in_strength_vs_receiver_gate": float(np.corrcoef(eff.sum(1), a_rms)[0, 1]),
        "median_directed_asymmetry": float(np.median(asym)),
        "receiver_gate_rms_range": [float(a_rms.min()), float(a_rms.max())],
        "source_gate_rms_range": [float(b_rms.min()), float(b_rms.max())],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

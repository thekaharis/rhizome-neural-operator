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
Panels (b) and (c) use the same 8x8 lattice of cells, the same edge threshold
and minimal-image (periodic) displacements. Edges are drawn undirected with the
larger of the two directed weights; the asymmetry is recorded in the JSON.
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


def curved(p, q, bend):
    """Quadratic Bezier from p to q (x, y), bowed sideways by ``bend`` of its length."""
    m = (p + q) / 2
    d = q - p
    ctrl = m + bend * np.array([-d[1], d[0]])
    return MPath([p, ctrl, q], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])


def edge_layer(ax, nodes, weights, n, color, scale, highlight=None, hcolor=GREEN, thresh=0.05):
    """Edges between lattice nodes along the minimal-image displacement.

    Each undirected edge shows sqrt(w_ij w_ji) / scale, so a cell that listens
    weakly thins all of its edges. The highlighted node's edges show its own
    incoming weights w_{h j} / scale. Edges that wrap across the periodic
    boundary are omitted for legibility.
    """
    w = np.sqrt(weights * weights.T) / scale
    m = len(nodes)
    order, drawn = [], 0
    for a in range(m):
        for b in range(a + 1, m):
            d = (nodes[b] - nodes[a] + n // 2) % n - n // 2
            if np.any(nodes[a] + d != nodes[b]):
                continue                                       # wraps around the torus
            hi = highlight is not None and highlight in (a, b)
            wt = weights[highlight, b if a == highlight else a] / scale if hi else w[a, b]
            if wt >= thresh:
                order.append((hi, wt, a, b))
    order.sort()
    paths, lws, cols = [], [], []
    for hi, wt, a, b in order:
        pa, pb = nodes[a][::-1].astype(float), nodes[b][::-1].astype(float)
        bend = 0.16 * (1 if (a * 31 + b * 17) % 2 else -1)
        paths.append(curved(pa, pb, bend))
        rgb = matplotlib.colors.to_rgb(hcolor if hi else color)
        lws.append((0.3 + 2.0 * wt) if hi else (0.12 + 1.9 * wt ** 1.3))
        cols.append((*rgb, 0.95 if hi else min(1.0, 0.04 + 0.8 * wt ** 1.4)))
        drawn += 1
    ax.add_collection(PathCollection(paths, facecolors="none", edgecolors=cols, linewidths=lws,
                                     capstyle="round", zorder=2))
    return drawn


def background(ax, truth, n):
    ax.imshow(truth, cmap="Greys", vmin=-0.4, vmax=2.6, origin="lower", interpolation="nearest", zorder=0)
    ax.contour(truth, levels=[0.5], colors=GREY, linewidths=0.4, alpha=0.6, zorder=1)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)


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
            ax.plot([x, px], [y, py], color=GREEN if on else GREY, lw=(1.6 if on else 0.35 + 0.35 * lev),
                    alpha=0.95 if on else 0.55, zorder=3 if on else 2, solid_capstyle="round")
    for lev, pts in enumerate(nodes_by_level):
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=4 + 9 * lev ** 1.6, c="white", edgecolors=GREY, linewidths=0.6, zorder=4)
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
    kern = np.zeros((m, m)); eff = np.zeros((m, m))            # [receiver, source]
    for a, (ya, xa) in enumerate(nodes):
        Ai2 = A[:, ya, xa] ** 2
        for b, (yb, xb) in enumerate(nodes):
            if a == b:
                continue
            Kab = K2[:, :, (ya - yb) % n, (xa - xb) % n]
            kern[a, b] = np.sqrt(Kab.sum())
            eff[a, b] = np.sqrt(Ai2 @ Kab @ (B[:, yb, xb] ** 2))
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

    background(axes[1], truth, n)
    ek = edge_layer(axes[1], nodes, kern, n, BLUE, kern.max(), highlight=rx)
    axes[1].scatter(nodes[:, 1], nodes[:, 0], s=9, c="white", edgecolors=GREY, linewidths=0.6, zorder=4)
    axes[1].set_title(r"(b) global, fixed: $\kappa_\theta(\mathbf r_i-\mathbf r_j)$", pad=3)
    axes[1].set_xlabel("one input-independent pattern\naround every cell (Fourier layer)", fontsize=7, labelpad=3)

    background(axes[2], truth, n)
    ee = edge_layer(axes[2], nodes, eff, n, BLUE, eff.max(), highlight=rx)
    a_rms = np.sqrt((A ** 2).mean(0))[nodes[:, 0], nodes[:, 1]]
    b_rms = np.sqrt((B ** 2).mean(0))[nodes[:, 0], nodes[:, 1]]
    bn = (b_rms - b_rms.min()) / (np.ptp(b_rms) or 1)
    axes[2].scatter(nodes[:, 1], nodes[:, 0], s=4 + 60 * (a_rms / a_rms.max()) ** 2, c="none",
                    edgecolors=GREEN, linewidths=0.8, zorder=4)
    axes[2].scatter(nodes[:, 1], nodes[:, 0], s=9, c=plt.get_cmap("Oranges")(0.25 + 0.75 * bn),
                    edgecolors=ORANGE, linewidths=0.5, zorder=5)
    axes[2].set_title(rf"(c) rhizome: $\mathbf A_i\,\kappa_\theta\,\mathbf B_j$, update $t{{=}}{t + 1}$", pad=3)
    axes[2].set_xlabel("every cell sends ($\\mathbf B$, fill) and listens\n($\\mathbf A$, ring) with its own weights",
                       fontsize=7, labelpad=3)

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
        "lattice_nodes": int(m), "lattice_step_cells": int(step), "highlighted_receiver_yx": nodes[rx].tolist(),
        "edges_drawn_kernel": ek, "edges_drawn_rhizome": ee, "edge_threshold_rel": 0.05, "wrapping_edges_omitted": True,
        "in_strength_cv_kernel": float(kn.sum(1).std() / kn.sum(1).mean()),
        "in_strength_cv_rhizome": float(en.sum(1).std() / en.sum(1).mean()),
        "out_strength_cv_rhizome": float(en.sum(0).std() / en.sum(0).mean()),
        "median_directed_asymmetry": float(np.median(asym)),
        "receiver_gate_rms_range": [float(a_rms.min()), float(a_rms.max())],
        "source_gate_rms_range": [float(b_rms.min()), float(b_rms.max())],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

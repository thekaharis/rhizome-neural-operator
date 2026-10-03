"""Per-slice signed errors (prediction - truth) of several models, side by side.

    python -m ebm21cm.delta_slices --cone 1063 1066 32 652 \\
        --model "2-D rhizome slice-wise=$CUBES/rhizome2d_w128_slicewise/manifest_test.json" \\
        --model "3-D fno cnn_whno_es=$CUBES/fno_cnn_whno_es/manifest.json" \\
        --out runs/eval3d/slicewise2d_vs_3d_test21/deltas

Models are ``eval_cubes`` manifests on the same redshift grid (truth is checked
to agree). Per cone, one page with a row per ionization stage (the slice whose
mean x_HI is closest to each ``--stages`` level): truth, then per model its
prediction - truth (red: too neutral, blue: too ionized) with the slice RMSE.
An overview page shows the middle stage of every cone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(manifest: str, cone: int):
    spec = json.loads(Path(manifest).read_text())
    (entries,) = spec["models"].values()
    entry = next((e for e in entries if int(e["cone_id"]) == cone), None)
    if entry is None:
        return None
    with np.load(entry["npz"]) as d:
        return d["truth"], d["pred"], d["z_native"]


def pick(truth, stages):
    means = truth.mean((0, 1))
    return [int(np.argmin(np.abs(means - s))) for s in stages]


def draw(axes, truth, deltas, k, z, label_row):
    axes[0].imshow(truth[:, :, k], origin="lower", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_ylabel(f"{label_row}\nz={z[k]:.2f}  <x_HI>={truth[:, :, k].mean():.2f}", fontsize=8)
    im = None
    for ax, (name, d) in zip(axes[1:], deltas.items()):
        im = ax.imshow(d[:, :, k], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        rmse = float(np.sqrt(np.mean(d[:, :, k] ** 2)))
        ax.text(0.02, 0.97, f"RMSE {rmse:.3f}", transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return im


def page(rows, names, out, title):
    fig, axes = plt.subplots(len(rows), 1 + len(names), figsize=(3.0 * (1 + len(names)) + 0.6, 3.0 * len(rows) + 0.6),
                             squeeze=False)
    im = None
    for i, (truth, deltas, k, z, label) in enumerate(rows):
        im = draw(axes[i], truth, deltas, k, z, label)
    for j, name in enumerate(["truth x_HI"] + [f"{n}\nprediction − truth" for n in names]):
        axes[0, j].set_title(name, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.86, 0.98))
    cax = fig.add_axes((0.88, 0.25, 0.015, 0.5))
    fig.colorbar(im, cax=cax).set_label("signed error (red: too neutral)", fontsize=8)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cone", type=int, nargs="+", required=True)
    ap.add_argument("--model", action="append", required=True, help="LABEL=manifest.json")
    ap.add_argument("--stages", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [s.split("=", 1) for s in args.model]
    names = [n for n, _ in sources]
    overview = []
    for cone in args.cone:
        loaded = {n: load(src, cone) for n, src in sources}
        if any(v is None for v in loaded.values()):
            print(f"cone {cone}: skipped (missing in a manifest)")
            continue
        truth, _, z = next(iter(loaded.values()))
        for n, (t, _, zz) in loaded.items():
            if not (np.allclose(t, truth, atol=1e-6) and np.allclose(zz, z)):
                raise ValueError(f"cone {cone}: truth or redshifts of {n} differ")
        deltas = {n: p.astype(np.float32) - truth for n, (_, p, _) in loaded.items()}
        ks = pick(truth, args.stages)
        page([(truth, deltas, k, z, f"x̄≈{s:.1f}") for s, k in zip(args.stages, ks)], names,
             args.out / f"cone_{cone}_deltas.png", f"Test cone {cone}: prediction − truth per slice")
        mid = ks[len(ks) // 2 - (1 if len(ks) % 2 == 0 else 0)]
        overview.append((truth, deltas, mid, z, f"cone {cone}"))
    if overview:
        page(overview, names, args.out / "overview_deltas.png",
             f"Prediction − truth at x̄_HI ≈ {args.stages[len(args.stages) // 2 - (1 if len(args.stages) % 2 == 0 else 0)]:.1f}")


if __name__ == "__main__":
    main()

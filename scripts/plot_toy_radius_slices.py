"""Plot test x_HI slices predicted by the toy radius-sweep runs side by side.

Rows are partly ionized test slices from different cones, spanning the
neutral fraction range; columns are the centre density band, the truth, and
each kernel's prediction (per-slice RMSE in the panel title).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ebm21cm.data.cache import SliceDataset, read_stats

DEFAULT_RUNS = ("rhizome", "rhizome_r1", "rhizome_r2", "rhizome_r4", "rhizome_r8")


def label(run):
    return "spectral (4 modes)" if run == "rhizome" else f"R = {run.removeprefix('rhizome_r')}"


def pick_rows(truth, cones, n):
    """Partly ionized slices nearest evenly spaced neutral fractions, one per cone."""
    means = truth.mean((1, 2))
    rows, used = [], set()
    for target in np.linspace(0.2, 0.8, n):
        for j in np.argsort(np.abs(means - target)):
            if cones[j] not in used and 0.05 < means[j] < 0.95:
                rows.append(int(j))
                used.add(cones[j])
                break
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fields = {}
    for run in args.runs:
        with np.load(args.sweep / run / run / "test_predictions.npz") as arrays:
            fields[run] = {k: arrays[k].copy() for k in ("prediction", "truth", "cone_id", "z")}
    base = fields[args.runs[0]]
    for run in args.runs[1:]:
        if not np.array_equal(fields[run]["truth"], base["truth"]):
            raise ValueError(f"test ordering differs: {run}")
    cache = json.loads((args.sweep / args.runs[0] / "metadata.json").read_text())["cache"]
    test = SliceDataset(cache, "test", read_stats(cache))
    rows = pick_rows(base["truth"], base["cone_id"], args.n)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    xhi_map = LinearSegmentedColormap.from_list("xhi", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"])
    ncol = 2 + len(args.runs)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.05 * ncol, 2.2 * len(rows) + 0.6), squeeze=False)
    for i, j in enumerate(rows):
        truth = base["truth"][j]
        density = test.raw(j)["delta"][test.center_band]
        panels = [("density, centre band", density, "Greys", None),
                  (f"truth  z={base['z'][j]:.2f}\nx_HI={truth.mean():.2f}", truth, xhi_map, (0, 1))]
        for run in args.runs:
            pred = fields[run]["prediction"][j]
            rmse = np.sqrt(np.mean((pred - truth) ** 2))
            panels.append((f"{label(run)}\nRMSE {rmse:.3f}", pred, xhi_map, (0, 1)))
        for k, (title, image, cmap, limits) in enumerate(panels):
            ax = axes[i, k]
            kw = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
            im = ax.imshow(image, origin="lower", cmap=cmap, interpolation="nearest", **kw)
            ax.set_title(title, fontsize=8.5, color="#0b0b0b")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#c3c2b7")
        axes[i, 0].set_ylabel(f"cone {base['cone_id'][j]}", fontsize=8.5, color="#52514e")
    fig.colorbar(im, ax=axes[:, 1:], fraction=0.015, pad=0.01, label="x_HI (truth binary; predictions sigmoid)")
    fig.suptitle("Toy x_HI test slices: compact radius-R kernels vs spectral (4 updates, 4 Mpc cells)",
                 fontsize=10, color="#0b0b0b", x=0.02, ha="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"rows {rows} -> {args.out}")


if __name__ == "__main__":
    main()

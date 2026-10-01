"""Per-cone comparison of 3-D x_HI predictions from several models, on identical planes.

    python -m ebm21cm.compare_cones3d --cone 24 \\
        --model "Rhizome w48=$EXPORT/rhizome3d_w48/native" \\
        --model "fno cnn_whno_es=$EXPORT/fno_cnn_whno_es/native" \\
        --model "fno multi-field=$WORK/data/eval_cubes/mf_cnn_whno_es_histtb_tbclean_ep27/manifest.json" \\
        --out runs/eval3d/cones

A model source is a directory of native-resolution ``cone_<id>.h5`` files
(``target``/``prediction`` groups, ``target_z``; ``export_cubes3d`` or
fno-21cm's prediction export) or an ``eval_cubes`` ``manifest.json``
(``pred``/``truth`` on native planes nearest a common redshift grid). All
models are compared on the same native planes: those of the first manifest
source if any, otherwise the native planes nearest a 512-point grid. Truth
must agree across sources on those planes (checked).

One page per cone, in fno-21cm's colours: lightcone strips (x vs z at the
middle y row) over the active reionization band for truth and each model,
signed-error strips, transverse slices at four ionization stages, and global
x_HI(z). RMSEs are printed per model over all planes and over the active band.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(source, cone):
    """(truth, prediction, z) with the LOS last, in increasing redshift."""
    source = Path(source)
    if source.suffix == ".json":
        spec = json.loads(source.read_text())
        (entries,) = spec["models"].values()
        entry = next((e for e in entries if int(e["cone_id"]) == cone), None)
        if entry is None:
            return None
        with np.load(entry["npz"]) as data:
            return data["truth"].astype(np.float32), data["pred"].astype(np.float32), data["z_native"]
    path = source / f"cone_{cone}.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        return (f["target/neutral_fraction"][:].astype(np.float32),
                f["prediction/neutral_fraction"][:].astype(np.float32), f["target_z"][:])


def align(fields, z_common):
    """Planes of a (truth, pred, z) triple at the native redshifts ``z_common``."""
    truth, pred, z = fields
    idx = np.abs(z[None, :] - z_common[:, None]).argmin(axis=1)
    if np.abs(z[idx] - z_common).max() > 1e-6:
        raise ValueError("a source lacks some of the common native planes")
    return truth[..., idx], pred[..., idx]


def page(cone, truth, preds, z, out, y_row=None):
    names = list(preds)
    means = truth.mean((0, 1))
    active = np.flatnonzero((means > 0.02) & (means < 0.98))
    lo, hi = (active.min(), active.max()) if len(active) else (0, len(z) - 1)
    pad = max(4, (hi - lo) // 10)
    band = slice(max(0, lo - pad), min(len(z), hi + pad + 1))
    y = truth.shape[1] // 2 if y_row is None else y_row
    levels = [lv for lv in (0.2, 0.4, 0.6, 0.8) if means.min() < lv < means.max()] or [float(np.median(means))]
    planes = [int(np.abs(means - lv).argmin()) for lv in levels]

    rows = len(names) + 1
    fig = plt.figure(figsize=(18, 1.7 * rows + 3.0 * len(planes) + 0.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.7 * rows, 3.0 * len(planes)])
    strips = grid[0].subgridspec(rows, 2, width_ratios=[1, 1])
    zb = z[band]
    extent = (zb[0], zb[-1], 0, truth.shape[0])
    image = err_image = None
    limit = max(0.05, float(np.quantile(np.abs(np.concatenate(
        [(p - truth)[:, y, band].ravel() for p in preds.values()])), 0.99)))
    for r, (name, field) in enumerate([("Truth", truth)] + list(preds.items())):
        ax = fig.add_subplot(strips[r, 0])
        image = ax.imshow(field[:, y, band], origin="lower", aspect="auto", extent=extent,
                          cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        rmse = "" if name == "Truth" else (f"  RMSE all {np.sqrt(np.mean((field - truth) ** 2)):.4f}, "
                                           f"band {np.sqrt(np.mean((field - truth)[..., band] ** 2)):.4f}")
        ax.set_ylabel(name, fontsize=9)
        ax.set_title(f"{name}{rmse}", fontsize=9, loc="left")
        if r < rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("redshift")
        ax_e = fig.add_subplot(strips[r, 1])
        if name == "Truth":
            ax_e.plot(z, means, color="black", label="truth")
            for (n, p), c in zip(preds.items(), ("#264653", "#e07a5f", "#168aad", "#7b2cbf")):
                ax_e.plot(z, p.mean((0, 1)), color=c, label=n, lw=1.2)
            ax_e.axvspan(zb[0], zb[-1], color="0.9", zorder=0)
            ax_e.set_ylabel(r"global $x_{HI}$")
            ax_e.legend(fontsize=7)
            ax_e.set_xlim(z[0], z[-1])
            ax_e.set_title("global history (grey: strip band)", fontsize=9, loc="left")
        else:
            err_image = ax_e.imshow((field - truth)[:, y, band], origin="lower", aspect="auto", extent=extent,
                                    cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            ax_e.set_title(f"{name}: prediction - truth", fontsize=9, loc="left")
            if r < rows - 1:
                ax_e.set_xticklabels([])
            else:
                ax_e.set_xlabel("redshift")
    fig.colorbar(image, ax=fig.axes[0], shrink=0.8, label="$x_{HI}$")
    fig.colorbar(err_image, ax=fig.axes[-1], shrink=0.8, label="signed error")

    slices = grid[1].subgridspec(len(planes), rows)
    for i, k in enumerate(planes):
        for j, (name, field) in enumerate([("Truth", truth)] + list(preds.items())):
            ax = fig.add_subplot(slices[i, j])
            ax.imshow(field[:, :, k], origin="lower", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"z={z[k]:.2f}\nmean {means[k]:.2f}", fontsize=9)
            else:
                rmse = np.sqrt(np.mean((field[:, :, k] - truth[:, :, k]) ** 2))
                ax.text(0.02, 0.98, f"RMSE={rmse:.3f}", transform=ax.transAxes, va="top", ha="left",
                        fontsize=8, color="white", bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
    fig.suptitle(f"Test cone {cone}: lightcone strips at y={y}, transverse slices at four stages", fontsize=13)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cone", type=int, nargs="+", required=True)
    ap.add_argument("--model", action="append", required=True, help="LABEL=native_dir or manifest.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-z", type=int, default=512)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [spec.split("=", 1) for spec in args.model]
    table = []
    for cone in args.cone:
        loaded = {name: load(src, cone) for name, src in sources}
        missing = [n for n, v in loaded.items() if v is None]
        if missing:
            print(f"cone {cone}: skipped, missing in {missing}")
            continue
        manifests = [n for n, src in sources if src.endswith(".json")]
        if manifests:
            z_common = loaded[manifests[0]][2]
        else:
            z = next(iter(loaded.values()))[2]
            grid = np.linspace(max(z[0], 5.001), min(z[-1], 24.97), args.n_z)
            z_common = z[np.unique(np.abs(z[None, :] - grid[:, None]).argmin(axis=1))]
        aligned = {n: align(v, z_common) for n, v in loaded.items()}
        truth = next(iter(aligned.values()))[0]
        for n, (t, _) in aligned.items():
            if not np.allclose(t, truth, atol=1e-6):
                raise ValueError(f"cone {cone}: truth of {n} differs from the other sources")
        preds = {n: p for n, (_, p) in aligned.items()}
        out = args.out / f"cone_{cone}.png"
        page(cone, truth, preds, z_common, out)
        row = {"cone": cone, "planes": len(z_common),
               **{n: float(np.sqrt(np.mean((p - truth) ** 2))) for n, p in preds.items()}}
        table.append(row)
        print(f"cone {cone}: " + "  ".join(f"{n} {v:.4f}" for n, v in row.items() if n not in ("cone", "planes"))
              + f"  -> {out}")
    (args.out / "rmse.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()

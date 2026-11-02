"""Per-cone comparison of 3-D x_HI (and T_b) predictions from several models, on identical planes.

    python -m ebm21cm.compare_cones3d --cone 24 \\
        --model "Rhizome w48=$EXPORT/rhizome3d_w48/native" \\
        --model "fno cnn_whno_es=$EXPORT/fno_cnn_whno_es/native" \\
        --model "fno multi-field=$WORK/data/eval_cubes/mf_cnn_whno_es_histtb_tbclean_ep27/manifest.json" \\
        --out runs/eval3d/cones

A model source is a directory of native-resolution ``cone_<id>.h5`` files
(``target``/``prediction`` groups, ``target_z``; ``export_cubes3d`` or
fno-21cm's prediction export) or an ``eval_cubes`` ``manifest.json``
(``pred``/``truth`` and optionally ``tb_pred``/``tb_truth`` on native planes
nearest a common redshift grid). All models are compared on the same native
planes: those of the first manifest source if any, otherwise the native planes
nearest a 512-point grid. Truth must agree across sources on those planes
(checked).

One gallery page per cone (``cone_<id>.png`` + ``.pdf``): a row per ionization
stage (slice-mean x_HI 0.8 / 0.6 / 0.4 / 0.2, as far as the cone covers them),
columns x_HI truth, each model, each model - truth, then the same for T_b when
every source provides it. x_HI in viridis on [0, 1], T_b in magma on the
truth's 1-99.5 % range per row, residuals in RdBu_r (x_HI +-1, T_b symmetric
99 % of the largest model residual per row); the slice RMSE of each model panel
is printed in its corner. RMSEs over all planes and over the active band go to
``rmse.json``.
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

FIELDS = ("xhi", "tb")


def load(source, cone):
    """``{"xhi": (truth, pred), "tb": (truth, pred) or absent, "z": z}``, LOS last, increasing redshift."""
    source = Path(source)
    if source.suffix == ".json":
        spec = json.loads(source.read_text())
        (entries,) = spec["models"].values()
        entry = next((e for e in entries if int(e["cone_id"]) == cone), None)
        if entry is None:
            return None
        with np.load(entry["npz"]) as data:
            out = {"xhi": (data["truth"].astype(np.float32), data["pred"].astype(np.float32)),
                   "z": data["z_native"]}
            if "tb_pred" in data.files and "tb_truth" in data.files:
                out["tb"] = (data["tb_truth"].astype(np.float32), data["tb_pred"].astype(np.float32))
            return out
    path = source / f"cone_{cone}.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        out = {"xhi": (f["target/neutral_fraction"][:].astype(np.float32),
                       f["prediction/neutral_fraction"][:].astype(np.float32)), "z": f["target_z"][:]}
        if "brightness_temp" in f["target"] and "brightness_temp" in f["prediction"]:
            out["tb"] = (f["target/brightness_temp"][:].astype(np.float32),
                         f["prediction/brightness_temp"][:].astype(np.float32))
        return out


def align(fields, z_common):
    """The loaded fields restricted to the planes at the native redshifts ``z_common``."""
    z = fields["z"]
    idx = np.abs(z[None, :] - z_common[:, None]).argmin(axis=1)
    if np.abs(z[idx] - z_common).max() > 1e-6:
        raise ValueError("a source lacks some of the common native planes")
    return {k: (v[0][..., idx], v[1][..., idx]) for k, v in fields.items() if k != "z"}


def stage_planes(means):
    levels = [lv for lv in (0.8, 0.6, 0.4, 0.2) if means.min() < lv < means.max()] or [float(np.median(means))]
    return [int(np.abs(means - lv).argmin()) for lv in levels]


def page(cone, truth, preds, z, out):
    """Gallery page; ``truth[field]`` cube, ``preds[name][field]`` cubes (field in ``truth``)."""
    names = list(preds)
    fields = [f for f in FIELDS if f in truth]
    means = truth["xhi"].mean((0, 1))
    planes = stage_planes(means)
    n_m = len(names)
    per = 1 + 2 * n_m
    fig, ax = plt.subplots(len(planes), per * len(fields), figsize=(2.1 * per * len(fields) + 0.6, 2.75 * len(planes)),
                           constrained_layout=True, squeeze=False)
    label = {"xhi": "x$_{HI}$", "tb": "T$_b$"}
    for i, k in enumerate(planes):
        for b, f in enumerate(fields):
            tr = truth[f][:, :, k]
            ps = [preds[n][f][:, :, k] for n in names]
            if f == "xhi":
                cm, v0, v1, dl = "viridis", 0.0, 1.0, 1.0
            else:
                cm = "magma"
                v0, v1 = np.percentile(tr, [1, 99.5])
                dl = max(float(np.percentile(np.abs(p - tr), 99)) for p in ps) or 1.0
            unit = "" if f == "xhi" else " [mK]"
            panels = [(f"{label[f]} truth{unit}", tr, cm, v0, v1, None)]
            panels += [(f"{label[f]} {n}", p, cm, v0, v1, p - tr) for n, p in zip(names, ps)]
            panels += [(f"{label[f]} {n}\n− truth{unit}", p - tr, "RdBu_r", -dl, dl, None) for n, p in zip(names, ps)]
            for j, (title, img, c, lo, hi, err) in enumerate(panels):
                a = ax[i, b * per + j]
                im = a.imshow(img, cmap=c, vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
                a.set_xticks([])
                a.set_yticks([])
                if i == 0:
                    a.set_title(title, fontsize=9)
                if err is not None:
                    e = float(np.sqrt(np.mean(err ** 2)))
                    a.text(0.02, 0.97, f"{e:.3f}" if f == "xhi" else f"{e:.1f} mK", transform=a.transAxes, va="top",
                           fontsize=8, color="w", bbox=dict(facecolor="k", alpha=0.55, lw=0))
                if j in (n_m, per - 1):
                    fig.colorbar(im, ax=a, shrink=0.8)
        ax[i, 0].set_ylabel(f"cone {cone}\nz = {z[k]:.2f}, <x$_{{HI}}$> = {means[k]:.2f}", fontsize=9)
    fig.suptitle(f"Test cone {cone}: transverse slices at four ionization stages; numbers are slice RMSE. "
                 f"{truth['xhi'].shape[0]}×{truth['xhi'].shape[1]} cells.", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(Path(out).with_suffix(f".{ext}"), dpi=170 if ext == "png" else None)
    plt.close(fig)


def active_band(means):
    active = np.flatnonzero((means > 0.02) & (means < 0.98))
    lo, hi = (active.min(), active.max()) if len(active) else (0, len(means) - 1)
    pad = max(4, (hi - lo) // 10)
    return slice(max(0, lo - pad), min(len(means), hi + pad + 1))


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
            z_common = loaded[manifests[0]]["z"]
        else:
            z = next(iter(loaded.values()))["z"]
            grid = np.linspace(max(z[0], 5.001), min(z[-1], 24.97), args.n_z)
            z_common = z[np.unique(np.abs(z[None, :] - grid[:, None]).argmin(axis=1))]
        aligned = {n: align(v, z_common) for n, v in loaded.items()}
        fields = [f for f in FIELDS if all(f in v for v in aligned.values())]
        first = next(iter(aligned.values()))
        truth = {f: first[f][0] for f in fields}
        for n, v in aligned.items():
            if not np.allclose(v["xhi"][0], truth["xhi"], atol=1e-6):
                raise ValueError(f"cone {cone}: truth of {n} differs from the other sources")
        preds = {n: {f: v[f][1] for f in fields} for n, v in aligned.items()}
        out = args.out / f"cone_{cone}.png"
        page(cone, truth, preds, z_common, out)
        band = active_band(truth["xhi"].mean((0, 1)))
        row = {"cone": cone, "planes": len(z_common),
               **{n: float(np.sqrt(np.mean((p["xhi"] - truth["xhi"]) ** 2))) for n, p in preds.items()},
               "band": {n: float(np.sqrt(np.mean((p["xhi"] - truth["xhi"])[..., band] ** 2))) for n, p in preds.items()}}
        if "tb" in fields:
            row["tb_mK"] = {n: float(np.sqrt(np.mean((p["tb"] - truth["tb"]) ** 2))) for n, p in preds.items()}
        table.append(row)
        print(f"cone {cone}: " + "  ".join(f"{n} {row[n]:.4f}" for n in preds) + f"  -> {out}")
    (args.out / "rmse.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()

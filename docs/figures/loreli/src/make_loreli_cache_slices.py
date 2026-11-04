"""Rows of a slice cache as the model sees them: target x_HI, T_b, the central density and LOS-velocity
bands, and the structured head's physical factor, for one cone at three stages plus a background slice.

    python docs/figures/loreli/src/make_loreli_cache_slices.py docs/figures/loreli/loreli_cache_slices
    python docs/figures/loreli/src/make_loreli_cache_slices.py docs/figures/21cmfast/21cmfast_cache_slices \
        --cache /pfs/10/work/hd_id260-fno_training/data/compressed/rhizome_mf_slices_all.h5 --split test

Reads either the LoReLi fine-tuning cache (``ebm21cm.data.loreli_cache``: one ``target`` dataset,
mirror-padded 280-cell rows, only the 206-cell interior shown) or the 21cmFAST multi-field cache
(``ebm21cm.data.mf_slices``: ``xhi``/``tb`` datasets on the periodic 140-cell box). Same layout and colour
scales for both. ``--cone`` picks the cone (default: LoReLi sim 12568; 21cmFAST: the first cone of
``--split``).
"""
import argparse
import json

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LORELI = "/pfs/10/work/hd_id260-fno_training/data/compressed/loreli_slices_l294_v3.h5"
SPLITS = {"train": 0, "val": 1, "validation": 1, "test": 2}


def rows_of(h, cone, mf):
    """Indices of the cone's rows, their slice-mean x_HI and background flags."""
    rows = np.flatnonzero(h["cone_id"][:] == cone)
    if mf:                                     # 21cmFAST cache: stages from the stored x_HI itself
        xm = np.array([float(np.asarray(h["xhi"][r], np.float32).mean()) for r in rows])
        lo, hi = json.loads(h.attrs.get("xhi_window", "[0.02, 0.98]"))
        bg = (xm < lo) | (xm > hi)
    else:
        xm, bg = h["xhi_mean"][:][rows], h["background"][:][rows].astype(bool)
    return rows, xm, bg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="output path without extension (writes .png and .pdf)")
    ap.add_argument("--cache", default=LORELI)
    ap.add_argument("--cone", type=int, default=None)
    ap.add_argument("--split", default="train", help="21cmFAST cache: split of the default cone")
    args = ap.parse_args()

    h = h5py.File(args.cache, "r")
    mf = h.attrs.get("schema", "") == "ebm21cm-mfslices-v1"
    bands = json.loads(h.attrs["bands"])
    if mf:
        n, pad = h["xhi"].shape[-1], 0
        tbn = json.loads(h.attrs["normalization"])["brightness_temp"]     # tb is stored normalized
        off, scale = float(tbn["offset"]), float(tbn["scale"])
        cone = args.cone if args.cone is not None else \
            int(h["cone_id"][:][np.flatnonzero(h["split"][:] == SPLITS[args.split])[0]])
        split_name = {v: k for k, v in SPLITS.items() if k != "validation"}[int(h["split"][:][h["cone_id"][:] == cone][0])]
        title = (f"21cmFAST cache {args.cache.rsplit('/', 1)[-1]} — cone {cone} ({split_name}); "
                 f"periodic {n}×{n} box (200 Mpc, 1.43 Mpc/cell)")
    else:
        off, scale = json.loads(h.attrs["tb_norm"])
        pad, n = int(h.attrs["pad"]), int(h.attrs["n_out"])
        cone = 12568 if args.cone is None else args.cone
        title = (f"LoReLi cache {args.cache.rsplit('/', 1)[-1]} — sim {cone} (train, block 428). "
                 f"valid {n}×{n} interior only (294 Mpc, 1.43 Mpc/cell).")
    z, los = h["z"][:], h["los_index"][:]
    rows, xm, bg = rows_of(h, cone, mf)
    pick = []
    for t in (0.85, 0.5, 0.2):
        act = np.flatnonzero(~bg)
        pick.append(act[np.argmin(np.abs(xm[act] - t))])
    if bg.any():
        pick.append(np.flatnonzero(bg)[0])
    pick = sorted(pick, key=lambda i: -xm[i])

    d0 = bands.index([0, 1])
    v0 = 13 + d0
    cols = [("target x_HI", "x", None), ("T_b [mK]", "tb", None),
            (f"density band {bands[d0]} (cond ch {d0})", "c", d0),
            (f"LOS velocity band {bands[d0]} (cond ch {v0})", "c", v0),
            ("phys A(z)(1+δ) [mK]", "phys", None)]
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(4 * len(cols), 3.6 * len(pick)),
                             constrained_layout=True)
    sl = np.s_[pad:pad + n, pad:pad + n]
    for i, k in enumerate(pick):
        r = rows[k]
        if mf:
            xhi = np.asarray(h["xhi"][r], np.float32)
            tb = np.asarray(h["tb"][r], np.float32) * scale + off
        else:
            tgt = h["target"][r][(slice(None),) + sl].astype(np.float32)
            xhi, tb = tgt[0], tgt[1] * scale + off
        cond = h["cond"][r][(slice(None),) + sl].astype(np.float32)
        ph = np.asarray(h["phys"][r][sl], np.float32)
        for j, (name, kind, ch) in enumerate(cols):
            ax = axes[i, j]
            if kind == "x":
                im = ax.imshow(xhi, cmap="Greys_r", vmin=0, vmax=1)
            elif kind == "tb":
                im = ax.imshow(tb, cmap="magma", vmin=np.percentile(tb, 1), vmax=np.percentile(tb, 99.5))
            elif kind == "c":
                a = cond[ch]
                s = np.percentile(np.abs(a), 99.5)
                im = ax.imshow(a, cmap="RdBu_r", vmin=-s, vmax=s)
            else:
                im = ax.imshow(ph, cmap="viridis", vmin=np.percentile(ph, 1), vmax=np.percentile(ph, 99.5))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
            if i == 0:
                ax.set_title(name, fontsize=11)
            if j == 0:
                tag = "background" if bg[k] else "active"
                ax.set_ylabel(f"row {r}  los {los[r]}\nz={z[r]:.2f}  <x_HI>={xm[k]:.2f}\n{tag}", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.savefig(args.out + ".png", dpi=200)
    fig.savefig(args.out + ".pdf")
    print(args.out, [(int(rows[k]), round(float(z[rows[k]]), 2), round(float(xm[k]), 3)) for k in pick])


if __name__ == "__main__":
    main()

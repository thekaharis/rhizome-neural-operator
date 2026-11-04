"""Render test-split predictions: truth vs zero-shot reference vs a fine-tuned run (emulated histories)."""
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")
from ebm21cm.train_loreli_finetune import load_reference
from ebm21cm.train_rhizome_mf2d import tb_normalized

W = Path("/pfs/10/work/hd_id260-fno_training")
CACHE = W / "data/compressed/loreli_slices_l294_v3.h5"
SIDE = W / "data/compressed/loreli_slices_l294_v3_emuhist.h5"
REF = W / "rhizome-neural-operator/runs/rhizome_mf2d_w64_structured_all"
RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
torch.set_num_threads(8)

with h5py.File(CACHE, "r") as h:
    split, cone, xm, bg, z, los = (h[k][:] for k in ("split", "cone_id", "xhi_mean", "background", "z", "los_index"))
    pad, n = int(h.attrs["pad"]), int(h.attrs["n_out"])
    off, scale = json.loads(h.attrs["tb_norm"])
    test_cones = np.unique(cone[split == 2])
    # a test cone whose active range spans the three stages
    best = max(test_cones, key=lambda c: (((cone == c) & ~bg & (xm < 0.3)).sum() > 0) + (((cone == c) & ~bg & (xm > 0.8)).sum() > 0)
               + 1e-4 * ((cone == c) & ~bg).sum())
    rows = np.flatnonzero((cone == best) & ~bg)
    pick = sorted({int(rows[np.argmin(np.abs(xm[rows] - t))]) for t in (0.85, 0.6, 0.35, 0.15)}, key=lambda r: -xm[r])
    batch = {k: torch.from_numpy(np.stack([h[k][r] for r in pick]).astype(np.float32)) for k in ("cond", "target", "phys")}
with h5py.File(SIDE, "r") as s:
    scal = torch.from_numpy(np.stack([s["scalars"][r] for r in pick]).astype(np.float32))

preds = {}
for name, path in (("reference (zero-shot)", REF), ("fine-tuned", RUN)):
    model, _, ck = load_reference(path)
    model.eval()
    keep = ck.get("scalar_keep")
    sc = scal if keep is None else scal[:, torch.as_tensor(keep)]
    with torch.no_grad():
        out = model(batch["cond"], sc)
        tb = tb_normalized(out, batch["phys"][:, None], ck.get("head", "structured"), *ck["tb_norm"])
    preds[name] = (out[:, 0].sigmoid().numpy(), (tb[:, 0] * scale + off).numpy())

sl = np.s_[pad:pad + n, pad:pad + n]
tx = batch["target"][:, 0].numpy()
tt = batch["target"][:, 1].numpy() * scale + off
names = list(preds)
fig, ax = plt.subplots(len(pick), 6, figsize=(19, 3.25 * len(pick)), constrained_layout=True)
summary = []
for i, r in enumerate(pick):
    tlo, thi = np.percentile(tt[i][sl], [1, 99.5])
    cols = [("x$_{HI}$ truth", tx[i][sl], None)] + [(f"x$_{{HI}}$ {nm}", preds[nm][0][i][sl], tx[i][sl]) for nm in names] \
        + [("T$_b$ truth [mK]", tt[i][sl], None)] + [(f"T$_b$ {nm}", preds[nm][1][i][sl], tt[i][sl]) for nm in names]
    row = {"row": int(r), "z": float(z[r]), "xhi_mean": float(xm[r])}
    for j, (title, img, truth) in enumerate(cols):
        is_x = j < 3
        im = ax[i, j].imshow(img, cmap="Greys_r" if is_x else "magma",
                             vmin=0 if is_x else tlo, vmax=1 if is_x else thi, origin="lower")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if truth is not None:
            rmse = float(np.sqrt(np.mean((img - truth) ** 2)))
            ax[i, j].text(0.02, 0.97, f"RMSE {rmse:.3f}" if is_x else f"RMSE {rmse:.1f} mK", transform=ax[i, j].transAxes,
                          va="top", fontsize=9, color="red" if is_x else "white",
                          bbox=dict(facecolor="white" if is_x else "black", alpha=0.6, lw=0))
            row[title] = rmse
        if i == 0:
            ax[i, j].set_title(title, fontsize=10)
        if j in (2, 5):
            fig.colorbar(im, ax=ax[i, j], shrink=0.8)
    ax[i, 0].set_ylabel(f"z = {z[r]:.2f}\n<x$_{{HI}}$> = {xm[r]:.2f}", fontsize=10)
    summary.append(row)
fig.suptitle(f"LoReLi test cone {best} (held-out block), v3 cache, emulated histories. Fine-tuned: {RUN.name}. "
             "Interior 206x206 (294 Mpc).", fontsize=11)
fig.savefig(OUT.with_suffix(".png"), dpi=200)
fig.savefig(OUT.with_suffix(".pdf"))
print(json.dumps({"cone": int(best), "rows": summary}, indent=1))

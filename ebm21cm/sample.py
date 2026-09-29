"""Draw an ensemble of samples per held-out slice and probe energies.

    python -m ebm21cm.sample --run-dir runs/base --split test \\
        --n-rows 64 --n-samples 8 --out runs/base/samples_test.h5

Rows are spread evenly in redshift over the split. For each row it stores
the truth, M samples (physical units) and, at each ``--probe-sigmas`` level,
the energy of the truth, of every sample and of the ensemble mean -- all
under the same conditioning. ``--candidates`` adds externally produced fields
(e.g. a deterministic FNO's predictions): an HDF5 file with datasets
``row`` (cache row ids), ``xhi`` and ``tb``, aligned by row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from .data.cache import SliceDataset
from .run import load_run, pick_device
from .sampling import sample


def _energy(model, x, sigma, cond, scal, seed):
    """Mean energy per pixel of clean fields x after adding probe noise sigma (fixed seed)."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(x.shape, generator=g).to(x.device)
    sig = torch.full((x.shape[0],), float(sigma), device=x.device)
    with torch.enable_grad():
        e = model.energy(x + noise * sigma, sig, cond, scal)
    return (e / x[0].numel()).detach().cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ebm21cm.sample", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="best", choices=("best", "last"))
    ap.add_argument("--cache", help="defaults to the cache the run was trained on")
    ap.add_argument("--split", default="test", choices=("train", "validation", "test"))
    ap.add_argument("--n-rows", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16, help="samples per forward pass")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--sigma-max", type=float, help="default: the run's train.sample_sigma_max (10)")
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--churn", type=float, default=0.0)
    ap.add_argument("--s-tmin", type=float, default=0.05, help="churn only for sigma >= this (EDM)")
    ap.add_argument("--s-noise", type=float, default=1.003, help="churn noise inflation (EDM)")
    ap.add_argument("--mala-steps", type=int, default=0)
    ap.add_argument("--mala-scale", type=float, default=0.05)
    ap.add_argument("--mala-sigma-max", type=float, default=1.0)
    ap.add_argument("--probe-sigmas", default="0.05,0.1,0.2,0.5")
    ap.add_argument("--candidates", help="HDF5 with row, xhi, tb to energy-probe as well")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    device = pick_device(a.device)
    model, meta, norm = load_run(a.run_dir, a.checkpoint, device)
    if a.sigma_max is None:
        a.sigma_max = float(meta["config"]["train"].get("sample_sigma_max", 10.0))
    ds = SliceDataset(a.cache or meta["cache"], a.split, meta["stats"])
    order = np.argsort(ds.z, kind="stable")
    pick = order[np.unique(np.linspace(0, len(order) - 1, min(a.n_rows, len(order))).astype(int))]
    probes = [float(s) for s in a.probe_sigmas.split(",") if s]
    M, B = a.n_samples, a.batch_size

    cand = {}
    if a.candidates:
        with h5py.File(a.candidates, "r") as h:
            for r, x, t in zip(h["row"][:], h["xhi"][:], h["tb"][:]):
                cand[int(r)] = (x, t)

    g = torch.Generator().manual_seed(a.seed)
    out = {k: [] for k in ("row", "z", "cone_id", "params", "delta_center", "xhi_true", "tb_true",
                           "xhi_samples", "tb_samples", "E_truth", "E_samples", "E_mean",
                           "E_candidate", "has_candidate")}
    accept = []
    center = ds.center_band
    for n, j in enumerate(pick):
        item = ds[int(j)]
        raw = ds.raw(int(j))
        cond1, targ1, scal1 = (item[k].to(device)[None] for k in ("cond", "target", "scalars"))
        xs = []
        for lo in range(0, M, B):
            m = min(B, M - lo)
            cond, scal = cond1.expand(m, -1, -1, -1), scal1.expand(m, -1)
            x, info = sample(model, cond, scal, n_steps=a.steps, sigma_min=a.sigma_min,
                             sigma_max=a.sigma_max, churn=a.churn, s_tmin=a.s_tmin, s_noise=a.s_noise,
                             mala_steps=a.mala_steps,
                             mala_scale=a.mala_scale, mala_sigma_max=a.mala_sigma_max, generator=g)
            xs.append(x)
            accept += info["mala_accept"]
        x = torch.cat(xs)
        xm = x.mean(0, keepdim=True)
        e_t, e_s, e_m, e_c = [], [], [], []
        c_all = cond1.expand(M, -1, -1, -1)
        s_all = scal1.expand(M, -1)
        has_c = item["row"] in cand
        if has_c:
            cx, ct = cand[item["row"]]
            xc = torch.from_numpy(norm.target(cx.astype(np.float64), ct.astype(np.float64))
                                  .astype(np.float32))[None].to(device)
        for p_i, s in enumerate(probes):
            seed = 10_000 * p_i + n
            e_t.append(_energy(model, targ1, s, cond1, scal1, seed)[0])
            e_s.append(_energy(model, x, s, c_all, s_all, seed))
            e_m.append(_energy(model, xm, s, cond1, scal1, seed)[0])
            e_c.append(_energy(model, xc, s, cond1, scal1, seed)[0] if has_c else np.nan)
        xhi_s, tb_s = (v.cpu().numpy() for v in norm.to_physical(x))
        out["row"].append(item["row"])
        out["z"].append(item["z"])
        out["cone_id"].append(item["cone_id"])
        out["params"].append(raw["params"])
        out["delta_center"].append(raw["delta"][center].astype(np.float32))
        out["xhi_true"].append(raw["xhi"].astype(np.float32))
        out["tb_true"].append(raw["tb"].astype(np.float32))
        out["xhi_samples"].append(xhi_s.astype(np.float32))
        out["tb_samples"].append(tb_s.astype(np.float32))
        out["E_truth"].append(e_t)
        out["E_samples"].append(np.stack(e_s, 1))
        out["E_mean"].append(e_m)
        out["E_candidate"].append(e_c)
        out["has_candidate"].append(has_c)
        print(f"[{n + 1}/{len(pick)}] row {item['row']} z={item['z']:.2f}  "
              f"E/px truth {e_t[0]:.3f} samples {np.mean(e_s[0]):.3f} mean {e_m[0]:.3f} (sigma={probes[0]})")

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        for k, v in out.items():
            h.create_dataset(k, data=np.asarray(v))
        h.attrs.update({
            "run_dir": str(Path(a.run_dir).resolve()), "checkpoint": a.checkpoint,
            "checkpoint_step": meta.get("checkpoint_step") or -1, "split": a.split,
            "cell_size_mpc": float(ds.attrs["cell_size_mpc"]), "probe_sigmas": json.dumps(probes),
            "sampler": json.dumps({k: getattr(a, k) for k in ("steps", "sigma_max", "sigma_min", "churn", "s_tmin", "s_noise",
                                                               "mala_steps", "mala_scale", "mala_sigma_max",
                                                               "seed", "n_samples")}),
            "mala_accept": json.dumps(accept),
        })
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

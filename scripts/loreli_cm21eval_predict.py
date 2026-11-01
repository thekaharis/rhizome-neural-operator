"""Full-lightcone predictions of rhizome models on LoReLi test cones, on one common z grid (for 21cmEVAL).

    python scripts/loreli_cm21eval_predict.py --models reference reference_oracle b8_decoder ... \
        [--groups runs/cm21eval/groups.json] [--truth-root data/eval_cubes/loreli_l294_v3_test] [--preds runs/cm21eval/preds]

21cmEVAL needs one geometry (shape and z grid) per run, but the cache only holds each cone's own
active window, and LoReLi flags the fully ionized low-z planes of early-reionizing cones as
invalid (no adapter input there).  The test cones are therefore split into geometry groups
(groups.json: {name: {"cones": [...], "z": [lo, hi]}}); a group's window starts at
max(lo, latest first-valid redshift of its cones).  Every plane of the window is rebuilt with the
cache builder's own code path (LoreliCone + LoreliAdapter(velocity="linear", device="cuda"),
conditioning cast to float16 like the cache) on the adapter's 1.4286 Mpc comoving target grid,
which is absolute, so the cones of a group share planes and redshifts.

Per cone it writes (once) the truth sample   <truth-root>/<group>/cone<N>.npz : xhi, tb [mK], density (1+delta), z_grid
and per model                               <preds>/<model>/cone<N>.npz : xhi, tb (float16, interior 206^2)

Histories: models named *_oracle get the cone's own histories (exactly the cache scalars);
all others get the v1 emulator's histories, computed as in ``loreli_history.write_scalars``.
Exactness: planes that are also cache rows are compared with the cache (conditioning, phys,
oracle scalars) and with the v3 side file (emulated scalars); the maxima go to check.json.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, ".")
from ebm21cm.data.loreli import (TRAINING_CELL_MPC, LoreliAdapter, LoreliCone, amplitude, loreli_files,
                                 loreli_redshifts, read_cube, training_stats)
from ebm21cm.data.loreli_cache import _resampled_truth
from ebm21cm.loreli_history import HIST_T, HIST_X, _sha256, history_scalars, load_globals
from ebm21cm.train_loreli_finetune import load_reference
from ebm21cm.train_rhizome_mf2d import tb_normalized

W = Path("/pfs/10/work/hd_id260-fno_training")
R = Path(__file__).resolve().parents[1]                      # repository root
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--groups", default=str(R / "runs/cm21eval/groups.json"))
ap.add_argument("--truth-root", default=str(W / "data/eval_cubes/loreli_l294_v3_test"))
ap.add_argument("--preds", default=str(R / "runs/cm21eval/preds"))
ap.add_argument("--cache", default=str(W / "data/compressed/loreli_slices_l294_v3.h5"))
ap.add_argument("--side", default=str(W / "data/compressed/loreli_slices_l294_v3_emuhist.h5"))
ap.add_argument("--data", default=str(W / "data/loreli_lc128"))
ap.add_argument("--tag", default="l294v3")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--subset", action="store_true", help="groups.json may list a subset of the test cones")
ap.add_argument("--skip-missing", action="store_true", help="skip models whose run has no best.pt (failed arm)")
a = ap.parse_args()
dev = torch.device("cuda")
preds = Path(a.preds)
groups = json.loads(Path(a.groups).read_text())

# ----------------------------------------------------------------------------- provenance from the cache
with h5py.File(a.cache, "r") as h:
    split, cone_col, los_col = h["split"][:], h["cone_id"][:], h["los_index"][:]
    attrs = dict(h.attrs)
    training_cache, ref_run = attrs["training_cache"], Path(attrs["reference_run"])
    pad, n_out, l_perp = int(attrs["pad"]), int(attrs["n_out"]), float(attrs["l_perp"])
side_attrs = dict(h5py.File(a.side, "r").attrs)
test_cones = [int(c) for c in np.unique(cone_col[split == 2])]
grouped = sorted(c for gr in groups.values() for c in gr["cones"])
if a.subset and set(grouped) <= set(test_cones) and len(set(grouped)) == len(grouped):
    test_cones = grouped                                   # smoke tests: a subset of the test cones
elif grouped != test_cones:
    raise ValueError(f"groups.json must list every test cone exactly once: missing {sorted(set(test_cones) - set(grouped))}, "
                     f"extra {sorted(set(grouped) - set(test_cones))}")
stats = training_stats(training_cache)
bands = stats["bands"]
sl = np.s_[pad:pad + n_out, pad:pad + n_out]

# ----------------------------------------------------------------------------- per-group absolute target grids
g = load_globals(side_attrs["globals"])
gi = {int(s): i for i, s in enumerate(g["sim"])}
z_native, dist_native = loreli_redshifts()
z_of = lambda d: np.interp(d, dist_native, z_native)
window, group_of = {}, {}
for gname, gr in groups.items():
    lo_abs, hi_abs = 0, 10 ** 9
    for c in gr["cones"]:
        v = np.flatnonzero(g["valid5"][gi[c]])
        lo_abs = max(lo_abs, int(np.ceil(dist_native[v[0]] / TRAINING_CELL_MPC - 1e-9)))
        hi_abs = min(hi_abs, int(np.floor(dist_native[v[-1]] / TRAINING_CELL_MPC + 1e-9)))
        group_of[c] = gname
    idx = np.arange(lo_abs, hi_abs + 1)
    zz = z_of(idx * TRAINING_CELL_MPC)
    window[gname] = idx[(zz >= gr["z"][0]) & (zz <= gr["z"][1])]
    print(f"group {gname}: {len(gr['cones'])} cones, {len(window[gname])} planes, "
          f"z {z_of(window[gname][0] * TRAINING_CELL_MPC):.3f}-{z_of(window[gname][-1] * TRAINING_CELL_MPC):.3f}", flush=True)
    (Path(a.truth_root) / gname).mkdir(parents=True, exist_ok=True)
if a.limit:
    test_cones = test_cones[:a.limit]

# ----------------------------------------------------------------------------- emulator histories
emu_dir = Path(side_attrs["emulator"])
if not (emu_dir / "predictions.h5").exists():                # side file written before the move into the repo
    emu_dir = R / "runs/loreli_emulator" / emu_dir.name
if _sha256(emu_dir / "predictions.h5") != side_attrs["predictions_sha256"]:
    raise ValueError(f"{emu_dir}/predictions.h5 is not the emulator the side file was built from")
with h5py.File(emu_dir / "predictions.h5", "r") as p:
    if not np.array_equal(p["sim"][:], g["sim"]):
        raise ValueError("emulator predictions and globals list different simulations")
    assigned = {q: p[f"{q}/assigned"][:] for q in ("x_HI", "T_b")}
node_bin = len(dist_native) // assigned["x_HI"].shape[1]
node_center = np.arange(assigned["x_HI"].shape[1]) * node_bin + (node_bin - 1) / 2
plane = np.arange(len(dist_native))

# ----------------------------------------------------------------------------- models
def run_dir(name):
    base = name[:-len("_oracle")] if name.endswith("_oracle") else name
    if base == "reference":
        return ref_run
    path = R / "runs/loreli_finetune" / f"{a.tag}_{base}_{'oracle' if name.endswith('_oracle') else 'emu'}_seed0"
    if not (path / "best.pt").exists():
        raise FileNotFoundError(f"model {name!r}: no {path}/best.pt")
    return path


models = {}
for name in a.models:
    try:
        run_dir(name)
    except FileNotFoundError as e:
        if not a.skip_missing:
            raise
        print(f"SKIP {e}", flush=True)
        continue
    m, _, ck = load_reference(run_dir(name))
    keep = ck.get("scalar_keep")
    models[name] = dict(model=m.to(dev).eval(), keep=None if keep is None else torch.as_tensor(keep, device=dev),
                        tbn=tuple(float(v) for v in ck["tb_norm"]), head=ck.get("head", "structured"),
                        run=str(run_dir(name)), step=int(ck.get("step", -1)), seconds=0.0, planes=0)
    (preds / name).mkdir(parents=True, exist_ok=True)

if not models:
    raise SystemExit("no models to run")
files, sims = loreli_files(a.data)
path_of = {int(s): p for p, s in zip(files, sims)}
check = {"cones": {}, "windows": {k: {"abs_first": int(v[0]), "abs_last": int(v[-1]), "n": int(len(v)),
                                     "z": [float(z_of(v[0] * TRAINING_CELL_MPC)), float(z_of(v[-1] * TRAINING_CELL_MPC))]}
                                 for k, v in window.items()}}
hc = h5py.File(a.cache, "r")
hs = h5py.File(a.side, "r")
t_start = time.time()
for ci, c in enumerate(test_cones):
    t0 = time.time()
    cone = LoreliCone(path_of[c], l_perp=l_perp, density_mean="slice", n_native=128, n_los=1536, data=read_cube(path_of[c]))
    ad = LoreliAdapter(cone, stats, velocity="linear", l_perp=l_perp, device="cuda", force_torch=True)
    abs_idx = window[group_of[c]]
    first = int(np.rint(ad.target_distance[0] / TRAINING_CELL_MPC))
    rel = abs_idx - first
    if rel[0] < 0 or rel[-1] >= len(ad.target_distance) or not np.allclose(ad.target_distance[rel], abs_idx * TRAINING_CELL_MPC):
        raise ValueError(f"cone {c}: common window outside its target grid")
    z_grid = ad.z_target[rel].astype(np.float64)
    # emulated history scalars for every plane of the window
    i = gi[c]
    ex = np.interp(plane, node_center, assigned["x_HI"][i])
    et = np.interp(plane, node_center, assigned["T_b"][i])
    emu_hist, _ = history_scalars(ex, et, g["valid5"][i], dist_native, rel, bands)

    tfile = Path(a.truth_root) / group_of[c] / f"cone{c}.npz"
    need_truth = not tfile.exists()
    if need_truth:
        x_t, tb_t = _resampled_truth(cone, ad)
        truth_x = np.clip(x_t[..., rel], 0, 1).astype(np.float32)
        truth_tb = tb_t[..., rel].astype(np.float32)
        density = np.empty_like(truth_x)
    out = {name: (np.empty((n_out, n_out, len(rel)), np.float32), np.empty((n_out, n_out, len(rel)), np.float32))
           for name in models}
    # cache rows of this cone inside the window -> exactness check
    crow = np.flatnonzero(cone_col == c)
    in_cache = {int(l): int(r) for l, r in zip(los_col[crow], crow)}
    probe = [j for j, rr in enumerate(rel) if int(rr) in in_cache]
    probe = set(probe[:: max(1, len(probe) // 6)][:6])
    ck = dict(cond=0.0, phys=0.0, oracle_scalars=0.0, emu_scalars=0.0, n=0)
    with torch.no_grad():
        for b0 in range(0, len(rel), a.batch):
            js = np.arange(b0, min(b0 + a.batch, len(rel)))
            ra, rb = int(rel[js[0]]), int(rel[js[-1]]) + 1          # contiguous
            inp = ad.inputs(ra, rb)
            cond = torch.as_tensor(inp["cond"], device=dev).half().float()    # cache stores float16
            phys = torch.as_tensor(inp["phys"], device=dev).float()
            sc_or = np.asarray(inp["scalars"], np.float32)
            sc_emu = sc_or.copy()
            sc_emu[:, HIST_X], sc_emu[:, HIST_T] = emu_hist[js, :13], emu_hist[js, 13:]
            if need_truth:
                density[..., js] = (phys[:, 0][:, sl[0], sl[1]].permute(1, 2, 0).cpu().numpy()
                                    / amplitude(z_grid[js])[None, None, :]).astype(np.float32)
            for k, j in enumerate(js):
                if j in probe:
                    r = in_cache[int(rel[j])]
                    ck["cond"] = max(ck["cond"], float(np.abs(cond[k].cpu().numpy() - hc["cond"][r].astype(np.float32)).max()))
                    ck["phys"] = max(ck["phys"], float(np.abs(phys[k, 0].cpu().numpy() - hc["phys"][r]).max()))
                    ck["oracle_scalars"] = max(ck["oracle_scalars"], float(np.abs(sc_or[k] - hc["scalars"][r]).max()))
                    ck["emu_scalars"] = max(ck["emu_scalars"], float(np.abs(sc_emu[k] - hs["scalars"][r]).max()))
                    ck["n"] += 1
            for name, M in models.items():
                tm = time.time()
                sc = torch.as_tensor(sc_or if name.endswith("_oracle") else sc_emu, device=dev)
                if M["keep"] is not None:
                    sc = sc[:, M["keep"]]
                y = M["model"](cond, sc)
                tbn = tb_normalized(y, phys, M["head"], *M["tbn"]) * M["tbn"][1] + M["tbn"][0]
                out[name][0][..., js] = y[:, 0, sl[0], sl[1]].sigmoid().permute(1, 2, 0).cpu().numpy()
                out[name][1][..., js] = tbn[:, 0, sl[0], sl[1]].permute(1, 2, 0).cpu().numpy()
                M["seconds"] += time.time() - tm
                M["planes"] += len(js)
    if need_truth:
        np.savez(tfile, xhi=truth_x, tb=truth_tb, density=density, z_grid=z_grid)
    for name, (px, pt) in out.items():
        np.savez(preds / name / f"cone{c}.npz", xhi=px.astype(np.float16), tb=pt.astype(np.float16), z_grid=z_grid)
    check["cones"][c] = ck
    print(f"[{ci + 1}/{len(test_cones)}] cone {c} ({group_of[c]}, {len(rel)} planes): {time.time() - t0:.0f}s; check {ck}", flush=True)

# ----------------------------------------------------------------------------- manifests + provenance
for gname, gr in groups.items():
    samples = [{"id": f"cone{c}", "path": f"cone{c}.npz", "split": "test", "cone_id": c} for c in gr["cones"]]
    manifest = {"box_mpc": l_perp, "omega_m": 0.308, "inputs": ["density"], "samples": samples,
                "group": gname, "window": check["windows"][gname], "cell_mpc": l_perp / n_out,
                "source": {"cones": a.data, "cache": a.cache, "builder": "LoreliAdapter(velocity=linear), float16 cond"},
                "note": "target not set on purpose: cm21eval --target xhi|tb (+ --set target=...)"}
    gdir = Path(a.truth_root) / gname
    (gdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (gdir / "manifest_tb.json").write_text(json.dumps({**manifest, "inputs": ["density", "xhi"]}, indent=1))
worst = {k: max(v[k] for v in check["cones"].values()) for k in ("cond", "phys", "oracle_scalars", "emu_scalars")}
check["max"] = worst
for name, M in models.items():
    info = {k: M[k] for k in ("run", "step", "seconds", "planes")}
    info["ms_per_plane"] = 1e3 * M["seconds"] / max(1, M["planes"])
    info["histories"] = "oracle" if name.endswith("_oracle") else f"emulated ({emu_dir})"
    (preds / name / "provenance.json").write_text(json.dumps({**info, "check": worst, "windows": check["windows"]}, indent=1))
(preds / f"check_{'-'.join(a.models)[:80]}.json").write_text(json.dumps(check, indent=1))
print("exactness vs cache (max |diff|):", worst, f"; total {time.time() - t_start:.0f}s", flush=True)

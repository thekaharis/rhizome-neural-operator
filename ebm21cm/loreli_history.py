"""LoReLi II global-history emulator and its scalar side file for the fine-tuning cache.

The field model sees LoReLi only through 26 history scalars (13 band means of the
global x_HI and T_b/100 histories); the 11 21cmFAST parameter slots are zero.  The
LoReLi cache fills them with the cone's *own* slice means, an oracle.  This module
replaces them with an emulator of the five LoReLi parameters
(f_X, tau, r_H/S, log M_min, f_esc), following fno-21cm's recipe
(``rhizome-neural-operator/runs/eval3d/emulator_v2/retrain.py``):

* ensembles of the history MLP (3 x 256, GELU), 5 members;
* x_HI on a logit scale with the monotone head (x_HI rising with z), T_b standardized;
* masked MSE with weight ``1 + ALPHA`` on nodes where the simulation's own box x_HI is
  in (0.02, 0.98);
* inputs log10 f_X, log10 tau, r_H/S, log M_min, f_esc, standardized on the pool.

Targets (``ebm21cm.data.loreli_globals``): x_HI = box diagnostics (9k simulations),
T_b = lightcone slice means (7.6k), both as 13 Mpc node means on z = 5..15.

Splits: the cache's val/test blocks are *held out* (never trained on).  The *pool* is
every other simulation.  A full fit on the pool predicts the held-out simulations; a
10-fold cross-fit grouped by block gives every pool simulation an out-of-fold history,
so fine-tuning rows see emulator errors of the same size as at test time.
Simulations the database authors pruned as faulty (``status=not_public``), off-grid
labels and T_b runs with corrupt planes are never trained on (they are still predicted).

    python -m ebm21cm.loreli_history train   --globals G.h5 --split-file S.json --out runs/loreli_emulator/v1
    python -m ebm21cm.loreli_history scalars --globals G.h5 --emulator runs/loreli_emulator/v1 \
        --cache C.h5 --out data/compressed/loreli_slices_l294_v2_emuhist.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from .data.loreli import TRAINING_CELL_MPC, HISTORY_TB_SCALE, band_profile, loreli_redshifts
from .data.loreli_splits import block_for_split, load_split

EPOCHS, ALPHA, EPS, MEMBERS, FOLDS = 400, 4.0, 1e-3, 5, 10
LR, WEIGHT_DECAY, INNER_VAL = 2e-3, 1e-4, 0.1
Z_BANDS = [(5, 6), (6, 7), (7, 8), (8, 10), (10, 15)]
PHASES = [("neutral >0.98", 0.98, 1.01), ("early 0.75-0.98", 0.75, 0.98), ("mid 0.25-0.75", 0.25, 0.75),
          ("late 0.02-0.25", 0.02, 0.25), ("ionized <0.02", -1.0, 0.02)]
HIST_X, HIST_T = slice(12, 25), slice(25, 38)     # history columns of the 38 scalars
SIDE_SCHEMA = "ebm21cm-loreli-scalars-v1"


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    """fno-21cm's history MLP; ``monotone`` returns b + cumsum softplus(d) - 10 (rising along the nodes)."""

    def __init__(self, n_in: int, n_nodes: int, monotone: bool):
        super().__init__()
        layers, w = [], n_in
        for _ in range(3):
            layers += [nn.Linear(w, 256), nn.GELU()]
            w = 256
        self.net = nn.Sequential(*layers, nn.Linear(256, n_nodes + (1 if monotone else 0)))
        self.monotone = monotone

    def forward(self, th):
        raw = self.net(th)
        if not self.monotone:
            return raw
        return raw[:, :1] + torch.cumsum(nn.functional.softplus(raw[:, 1:]), dim=1) - 10.0


def features(theta: np.ndarray) -> np.ndarray:
    """(n, 5) grid parameters -> log10 f_X, log10 tau, r_H/S, log M_min, f_esc."""
    th = np.asarray(theta, np.float64)
    return np.stack([np.log10(th[:, 0]), np.log10(th[:, 1]), th[:, 2], th[:, 3], th[:, 4]], 1)


def train_member(args):
    th, y, w, tr, va, monotone, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    th, y, w = (torch.as_tensor(a) for a in (th, y, w))
    model = Net(th.shape[1], y.shape[1], monotone)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = lambda p, b: (w[b] * (p - y[b]) ** 2).sum() / w[b].sum()
    best, best_state, tr_t, va_t = np.inf, None, torch.as_tensor(tr), torch.as_tensor(va)
    for ep in range(EPOCHS):
        model.train()
        for b in tr_t[torch.randperm(len(tr_t))].split(256):
            loss = loss_fn(model(th[b]), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep % 10 == 9 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                v = float(loss_fn(model(th[va_t]), va_t))
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(th).numpy(), best, best_state


# ----------------------------------------------------------------------------- data
def load_globals(path):
    with h5py.File(path, "r") as h:
        g = {k: h[k][:] for k in ("sim", "theta", "status", "lightcone", "block", "z_node", "distance_node",
                                  "distance_native", "z_native", "xhi_box", "tb_slice", "tb_bad_planes",
                                  "xhi_slice", "tb_slice5", "valid5", "xhi5_native", "tb5_native")}
        g["attrs"] = dict(h.attrs)
    return g


def eligible(g):
    """Training eligibility per quantity (independent of the split)."""
    clean = np.isin(g["status"], (0, 1))                       # published (or published with a broken label file)
    x = g["xhi_box"]
    xok = np.isfinite(x)
    reached = np.where(xok.any(1), np.nanmin(np.where(xok, x, np.inf), 1), 1.0)
    # complete histories: the simulation either reaches the end of the lightcone (z = 5) or ionizes (x_HI <= 0.05)
    complete = xok[:, 0] | (reached <= 0.05)
    e_x = clean & (xok.sum(1) >= 8) & complete
    e_t = clean & (np.isfinite(g["tb_slice"]).sum(1) >= 8) & (g["tb_bad_planes"] == 0)
    return e_x, e_t


def heldout_mask(g, split):
    """Simulations in the cache's val/test blocks."""
    names = {b: n for b, n in split["block_to_split"].items()}
    blocks = np.array([block_for_split(int(s), split) for s in g["sim"]])
    return np.array([names.get(int(b)) in ("val", "test") for b in blocks]), blocks


# ----------------------------------------------------------------------------- training
def _targets(quantity, g, rows_for_stats):
    if quantity == "x_HI":
        x = g["xhi_box"].astype(np.float64)
        mask = np.isfinite(x)
        xc = np.clip(np.where(mask, x, 0.5), EPS, 1 - EPS)
        y = np.log(xc / (1 - xc))
        to_phys = lambda p: 1.0 / (1.0 + np.exp(-p))
        norm = {}
    else:
        t = g["tb_slice"].astype(np.float64)
        mask = np.isfinite(t)
        vals = t[rows_for_stats][mask[rows_for_stats]]
        off, sc = float(vals.mean()), float(vals.std())
        y = (np.where(mask, t, off) - off) / sc
        to_phys = lambda p, off=off, sc=sc: p * sc + off
        norm = {"output_offset": off, "output_scale": sc}
    xb = g["xhi_box"]
    active = np.isfinite(xb) & (xb > 0.02) & (xb < 0.98)
    w = mask * (1.0 + ALPHA * active)
    return y.astype(np.float32), w.astype(np.float32), mask, to_phys, norm


def _inner_split(rows, blocks, rng):
    """Inner validation (for best-epoch selection): INNER_VAL of the training blocks."""
    ub = np.unique(blocks[rows])
    val_blocks = rng.choice(ub, max(1, int(round(INNER_VAL * len(ub)))), replace=False)
    va = rows[np.isin(blocks[rows], val_blocks)]
    return rows[~np.isin(blocks[rows], val_blocks)], va


def train(globals_path, split_file, out, workers=32, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g = load_globals(globals_path)
    split = load_split(split_file)
    held, blocks = heldout_mask(g, split)
    e_x, e_t = eligible(g)
    feats = features(g["theta"])
    pool = ~held
    rng = np.random.default_rng(0)
    pool_blocks = np.unique(blocks[pool])
    fold_of_block = dict(zip(rng.permutation(pool_blocks).tolist(),
                             np.arange(len(pool_blocks)) % FOLDS))
    fold = np.array([fold_of_block.get(int(b), -1) for b in blocks])
    log(f"{len(g['sim'])} simulations; held out {held.sum()} (cache val/test blocks); pool {pool.sum()} "
        f"in {len(pool_blocks)} blocks, {FOLDS} folds; trainable x_HI {int((e_x & pool).sum())}, "
        f"T_b {int((e_t & pool).sum())}")
    t0 = time.time()
    preds, report, ctx = {}, {}, mp.get_context("fork")
    with ctx.Pool(workers) as pool_mp:
        for quantity, elig in (("x_HI", e_x), ("T_b", e_t)):
            train_rows = np.flatnonzero(elig & pool)
            mu, sd = feats[train_rows].mean(0), feats[train_rows].std(0)
            th = ((feats - mu) / sd).astype(np.float32)
            y, w, mask, to_phys, norm = _targets(quantity, g, train_rows)
            monotone = quantity == "x_HI"
            jobs, meta = [], []
            for k in [-1] + list(range(FOLDS)):          # -1: full fit on the pool
                rows = train_rows if k < 0 else train_rows[fold[train_rows] != k]
                tr, va = _inner_split(rows, blocks, np.random.default_rng(100 + k))
                for m in range(MEMBERS):
                    jobs.append((th, y, w, tr, va, monotone, 1000 * (k + 1) + m))
                    meta.append(k)
            outs = pool_mp.map(train_member, jobs)
            full = [o for o, k in zip(outs, meta) if k < 0]
            p_full = to_phys(np.mean([o[0] for o in full], 0))
            p_oof = np.full_like(p_full, np.nan)
            for k in range(FOLDS):
                members = [o[0] for o, kk in zip(outs, meta) if kk == k]
                rows = fold == k
                p_oof[rows] = to_phys(np.mean(members, 0))[rows]
            spread = np.std([to_phys(o[0]) for o in full], 0)
            preds[quantity] = {"full": p_full.astype(np.float32), "oof": p_oof.astype(np.float32),
                               "spread": spread.astype(np.float32)}
            torch.save({"states": [o[2] for o in full], "theta_mean": mu, "theta_std": sd,
                        "features": "log10 f_x, log10 tau, rHS, logMmin, fesc", "z_node": g["z_node"],
                        "distance_node": g["distance_node"], "quantity": quantity, "monotone": monotone,
                        "alpha": ALPHA, "logit_eps": EPS, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS,
                        "train_sims": g["sim"][train_rows], **norm}, out / f"emulator_{quantity}.pt")
            log(f"  {quantity}: {len(jobs)} members trained, inner val loss "
                f"{np.mean([o[1] for o in full]):.5f} (full) / "
                f"{np.mean([o[1] for o, k in zip(outs, meta) if k >= 0]):.5f} (folds), {time.time() - t0:.0f} s")
            report[quantity] = {"n_train_pool": int(len(train_rows)), "theta_mean": mu.tolist(), "theta_std": sd.tolist()}
    # assigned history = out-of-fold for pool simulations, full fit for held-out ones
    assign = {q: np.where(pool[:, None], preds[q]["oof"], preds[q]["full"]) for q in preds}
    with h5py.File(out / "predictions.h5", "w") as h:
        h["sim"], h["heldout"], h["fold"] = g["sim"], held, fold
        h["eligible_x_HI"], h["eligible_T_b"] = e_x, e_t
        h["z_node"], h["distance_node"] = g["z_node"], g["distance_node"]
        for q, d in preds.items():
            for k, v in d.items():
                h[f"{q}/{k}"] = v
            h[f"{q}/assigned"] = assign[q]
        h.attrs["assigned"] = "out-of-fold for pool simulations, full fit for held-out (cache val/test blocks)"
        h.attrs["globals"] = str(Path(globals_path).resolve())
        h.attrs["split_file"] = str(Path(split_file).resolve())
        h.attrs["split_sha256"] = split["sha256"]
    report["scores"] = evaluate(g, preds, held, e_x, e_t, pool)
    report["seconds"] = round(time.time() - t0)
    json.dump(report, open(out / "results.json", "w"), indent=1)
    log(json.dumps(report["scores"], indent=1))
    return report


# ----------------------------------------------------------------------------- scoring
def _rms(e):
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else None


def midpoint(x, z):
    """Redshift where the history crosses x_HI = 0.5 (rising with z), NaN if it does not."""
    out = np.full(len(x), np.nan)
    for i, row in enumerate(x):
        ok = np.isfinite(row)
        r, zz = row[ok], z[ok]
        k = np.flatnonzero((r[:-1] < 0.5) & (r[1:] >= 0.5))
        if k.size:
            j = k[0]
            out[i] = zz[j] + (0.5 - r[j]) * (zz[j + 1] - zz[j]) / (r[j + 1] - r[j])
    return out


def score(pred, truth, xtruth, z, rows, unit=1.0):
    e = (pred - truth)[rows] * unit
    xt = xtruth[rows]
    s = {"n": int(len(rows)), "all": _rms(e),
         "z_band": {f"{a}-{b}": _rms(e[:, (z >= a) & (z < b)]) for a, b in Z_BANDS},
         "phase": {lab: _rms(np.where((xt >= lo) & (xt < hi), e, np.nan)) for lab, lo, hi in PHASES}}
    return s


def evaluate(g, preds, held, e_x, e_t, pool):
    z = g["z_node"]
    xb, tb = g["xhi_box"], g["tb_slice"]
    out = {}
    for q, truth, elig in (("x_HI", xb, e_x), ("T_b", tb, e_t)):
        P = preds[q]
        ho = np.flatnonzero(held & elig)
        po = np.flatnonzero(pool & elig)
        base = np.nanmean(np.where(np.isfinite(truth[po]), truth[po], np.nan), 0)[None].repeat(len(truth), 0)
        out[q] = {"heldout_full": score(P["full"], truth, xb, z, ho),
                  "pool_out_of_fold": score(P["oof"], truth, xb, z, po),
                  "pool_in_sample_full": score(P["full"], truth, xb, z, po),
                  "heldout_baseline_pool_mean": score(base, truth, xb, z, ho),
                  "heldout_ensemble_spread_mean": float(np.nanmean(P["spread"][ho]))}
    mt, mf, mo = midpoint(xb, z), midpoint(preds["x_HI"]["full"], z), midpoint(preds["x_HI"]["oof"], z)
    for name, rows, m in (("heldout", np.flatnonzero(held & e_x), mf), ("pool_oof", np.flatnonzero(pool & e_x), mo)):
        d = np.abs(m[rows] - mt[rows])
        d = d[np.isfinite(d)]
        out["x_HI"][f"midpoint_abs_dz_{name}"] = {"n": int(d.size), "median": float(np.median(d)) if d.size else None,
                                                   "p90": float(np.percentile(d, 90)) if d.size else None}
    # the 228 five-field cones: what the field model sees is the slice mean, not the box mean
    five = np.flatnonzero(g["lightcone"] == 2)
    assigned = {q: np.where(pool[:, None], preds[q]["oof"], preds[q]["full"]) for q in preds}
    xs, ts = g["xhi_slice"], g["tb_slice5"]
    out["five_field_vs_slice_means"] = {
        "n": int(len(five)),
        "x_HI_box_minus_slice (floor)": _rms((xb - xs)[five]),
        "x_HI_emulated_minus_slice": _rms((assigned["x_HI"] - xs)[five]),
        "T_b_emulated_minus_slice_mK": _rms((assigned["T_b"] - ts)[five]),
        "by_split": {name: {"x_HI": _rms((assigned["x_HI"] - xs)[five[sel]]),
                            "T_b_mK": _rms((assigned["T_b"] - ts)[five[sel]])}
                     for name, sel in (("pool(out-of-fold)", pool[five]), ("heldout(full)", held[five]))},
    }
    return out


# ----------------------------------------------------------------------------- side file
def history_scalars(hist_x, hist_tb, valid, distances, target_index, bands):
    """The 26 history scalars of ``LoreliAdapter.inputs`` for target indices ``target_index``.

    ``hist_*`` are native-grid histories (x_HI, T_b in mK), ``valid`` is ``LoreliCone.valid``.
    Reproduces the adapter: interpolate the valid native history onto the adapter's
    1.4286 Mpc target grid, T_b / 100, edge-pad by the band halo, band means.
    """
    native = np.flatnonzero(valid)
    first, last = native[[0, -1]]
    stop = int(np.ceil(distances[last] / TRAINING_CELL_MPC)) + 1
    grid = np.arange(stop) * TRAINING_CELL_MPC
    target = grid[(grid >= distances[first] - 1e-9) & (grid <= distances[last] + 1e-9)]
    hx = np.interp(target, distances[native], hist_x[native])
    ht = np.interp(target, distances[native], hist_tb[native]) / HISTORY_TB_SCALE
    halo = max(0, -min(lo for lo, _ in bands), max(hi for _, hi in bands) - 1)
    centers = np.asarray(target_index) + halo
    bx = band_profile(np.pad(hx, (halo, halo), mode="edge"), centers, bands)
    bt = band_profile(np.pad(ht, (halo, halo), mode="edge"), centers, bands)
    return np.concatenate([bx, bt], 1).astype(np.float32), len(target)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_scalars(globals_path, emulator_dir, cache, out, log=print):
    """Side file: the cache's scalars with the 26 history columns replaced by emulated histories."""
    g = load_globals(globals_path)
    emulator_dir = Path(emulator_dir)
    pred_path = emulator_dir / "predictions.h5"
    with h5py.File(pred_path, "r") as p:
        psim = p["sim"][:]
        held = p["heldout"][:]
        assigned = {q: p[f"{q}/assigned"][:] for q in ("x_HI", "T_b")}
    if not np.array_equal(psim, g["sim"]):
        raise ValueError("predictions and globals list different simulations")
    with h5py.File(cache, "r") as h:
        scalars = h["scalars"][:]
        cone, los, split_col = h["cone_id"][:], h["los_index"][:], h["split"][:]
        bands = json.loads(h.attrs["bands"])
        cache_attrs = {k: h.attrs[k] for k in ("created", "split_sha256", "n_rows", "checkpoint_sha256")}
    index = {int(s): i for i, s in enumerate(g["sim"])}
    # Reproduce the cache's LOS geometry exactly: loreli_redshifts() at the time the cache was built
    # (uniform z since 2026-11-01; the v2 cache used a uniform-comoving axis, so a v2 side file must be
    # built with the old code).  The oracle check below fails if the geometry does not match.
    # Emulator nodes map to native planes by index (NODE_BIN-plane bins).
    _, dist_native = loreli_redshifts()
    n_native = len(dist_native)
    node_bin = n_native // assigned["x_HI"].shape[1]
    node_center = np.arange(assigned["x_HI"].shape[1]) * node_bin + (node_bin - 1) / 2
    plane = np.arange(n_native)
    new = scalars.copy()
    oracle_err = np.zeros((len(scalars), 2), np.float32)
    source = np.zeros(len(scalars), np.int8)
    for c in np.unique(cone):
        rows = np.flatnonzero(cone == c)
        i = index[int(c)]
        valid = g["valid5"][i]
        # exactness check: the oracle histories through this pipeline must reproduce the cache
        truth, n_target = history_scalars(g["xhi5_native"][i], g["tb5_native"][i], valid, dist_native,
                                          los[rows], bands)
        oracle_err[rows, 0] = np.abs(truth[:, :13] - scalars[rows, HIST_X]).max(1)
        oracle_err[rows, 1] = np.abs(truth[:, 13:] - scalars[rows, HIST_T]).max(1)
        ex = np.interp(plane, node_center, assigned["x_HI"][i])
        et = np.interp(plane, node_center, assigned["T_b"][i])
        emu, _ = history_scalars(ex, et, valid, dist_native, los[rows], bands)
        new[rows, HIST_X], new[rows, HIST_T] = emu[:, :13], emu[:, 13:]
        source[rows] = 1 if held[i] else 0
    if oracle_err.max() > 1e-4:
        raise ValueError(f"oracle reproduction failed: max |diff| {oracle_err.max():.3g}")
    xhi_mean = None
    with h5py.File(cache, "r") as h:
        xhi_mean = h["xhi_mean"][:]
    k01 = bands.index([0, 1])
    stats = {name: {"rows": int((split_col == s).sum()),
                    "x_HI(0,1)_minus_slice_mean_rms": float(np.sqrt(np.mean((new[split_col == s, 12 + k01]
                                                                              - xhi_mean[split_col == s]) ** 2))),
                    "T_b(0,1)_emulated_minus_oracle_rms_mK": float(HISTORY_TB_SCALE * np.sqrt(np.mean(
                        (new[split_col == s, 25 + k01] - scalars[split_col == s, 25 + k01]) ** 2)))}
             for name, s in (("train", 0), ("val", 1), ("test", 2))}
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    with h5py.File(tmp, "w") as h:
        h["scalars"], h["cone_id"], h["los_index"], h["history_source"] = new, cone, los, source
        h.attrs.update({
            "schema": SIDE_SCHEMA, "created": datetime.now(timezone.utc).isoformat(),
            "cache": str(Path(cache).resolve()), **{f"cache_{k}": (v.item() if hasattr(v, "item") else v)
                                                   for k, v in cache_attrs.items()},
            "replaced_columns": "12-37 (x_HI_global and T_b_global band means)",
            "cache_los_geometry": "loreli_redshifts() at build time: uniform in z (checked by the oracle reproduction)",
            "history_source": "0 out-of-fold emulator (pool cone), 1 full-fit emulator (held-out cone)",
            "emulator": str(emulator_dir.resolve()), "predictions_sha256": _sha256(pred_path),
            "globals": str(Path(globals_path).resolve()),
            "oracle_reproduction_max_abs": float(oracle_err.max()), "stats": json.dumps(stats),
        })
    tmp.replace(out)
    log(f"wrote {out}: {len(new)} rows; oracle reproduction max |diff| {oracle_err.max():.2e}")
    log(json.dumps(stats, indent=1))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ebm21cm.loreli_history")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--globals", required=True)
    t.add_argument("--split-file", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--workers", type=int, default=32)
    s = sub.add_parser("scalars")
    s.add_argument("--globals", required=True)
    s.add_argument("--emulator", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)
    if a.cmd == "train":
        train(a.globals, a.split_file, a.out, a.workers, log=log)
    else:
        write_scalars(a.globals, a.emulator, a.cache, a.out, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

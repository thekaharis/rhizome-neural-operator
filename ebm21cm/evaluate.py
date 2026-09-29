"""Score a samples file written by ``ebm21cm.sample``.

    python -m ebm21cm.evaluate --samples runs/base/samples_test.h5 \\
        [--z-edges 5,7,9,12,16,25] [--out-dir runs/base/eval_test]

Writes metrics.json (overall and per redshift bin) and figures:
examples.png, spectra.png, bubble_sizes.png, energy.png.
Every per-slice metric compares against that slice's own truth; the
ensemble mean is scored alongside as the stand-in for a deterministic
conditional-mean regressor.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import h5py
import numpy as np

from . import metrics as M
from . import viz


def _nanmean(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(xs)) if np.isfinite(xs).any() else None


def evaluate(path, z_edges=None, out_dir=None, n_rays=2000, n_examples=4, seed=0):
    with h5py.File(path, "r") as h:
        d = {k: h[k][:] for k in h.keys()}
        attrs = dict(h.attrs)
    cell = float(attrs["cell_size_mpc"])
    probes = json.loads(attrs["probe_sigmas"])
    z = d["z"]
    n, m = d["xhi_samples"].shape[:2]
    if z_edges is None:
        z_edges = np.unique(np.quantile(z, np.linspace(0, 1, 5)))
        z_edges[-1] += 1e-9
    z_edges = np.asarray(z_edges, dtype=np.float64)
    shape = d["xhi_true"].shape[1:]
    edges = M.k_bins(shape, cell)
    kc = 0.5 * (edges[1:] + edges[:-1])
    rng = np.random.default_rng(seed)

    per = []
    spec = {"x_HI": [], "T_b": []}
    sizes = {"truth": [], "samples": [], "ens. mean": []}
    size_rows = []
    for i in range(n):
        xt, xs = d["xhi_true"][i], d["xhi_samples"][i]
        tt, ts = d["tb_true"][i], d["tb_samples"][i]
        xm = xs.mean(0)
        r = {"z": float(z[i])}
        r.update({
            "xhi_rmse_sample": float(np.mean([M.rmse(s, xt) for s in xs])),
            "xhi_rmse_mean": M.rmse(xm, xt), "xhi_crps": M.crps_ensemble(xs, xt),
            "xhi_spread_skill": M.spread_skill(xs, xt),
            "xhi_hedge_truth": M.hedging_fraction(xt), "xhi_hedge_samples": M.hedging_fraction(xs),
            "xhi_hedge_mean": M.hedging_fraction(xm),
            "xhi_mean_true": float(xt.mean()), "xhi_mean_samples": float(xs.mean()),
            "tb_rmse_sample": float(np.mean([M.rmse(s, tt) for s in ts])),
            "tb_rmse_mean": M.rmse(ts.mean(0), tt), "tb_crps": M.crps_ensemble(ts, tt),
            "tb_spread_skill": M.spread_skill(ts, tt),
        })
        # Bubble sizes only where both phases are present in the truth.
        ion_frac = float((xt < 0.5).mean())
        if 0.02 < ion_frac < 0.98:
            s_t = M.mfp_sizes(xt < 0.5, cell, n_rays, rng)
            s_s = np.concatenate([M.mfp_sizes(s < 0.5, cell, n_rays // m + 1, rng) for s in xs])
            s_m = M.mfp_sizes(xm < 0.5, cell, n_rays, rng)
            r["bsd_w1_samples"] = M.size_distance(s_s, s_t)
            r["bsd_w1_mean"] = M.size_distance(s_m, s_t)
            r["bsd_mean_ratio_samples"] = float(s_s.mean() / s_t.mean()) if s_s.size else None
            r["bsd_mean_ratio_mean"] = float(s_m.mean() / s_t.mean()) if s_m.size else None
            sizes["truth"].append(s_t)
            sizes["samples"].append(s_s)
            sizes["ens. mean"].append(s_m)
            size_rows.append(i)
        spec["x_HI"].append(M.spectra(xs, xt, cell, edges))
        spec["T_b"].append(M.spectra(ts, tt, cell, edges))
        # Energy gaps per pixel relative to the truth, same conditioning and probe noise.
        r["dE_samples"] = (d["E_samples"][i].mean(0) - d["E_truth"][i]).tolist()
        r["dE_mean"] = (d["E_mean"][i] - d["E_truth"][i]).tolist()
        if bool(d["has_candidate"][i]):
            r["dE_candidate"] = (d["E_candidate"][i] - d["E_truth"][i]).tolist()
        per.append(r)

    scalar_keys = sorted({k for r in per for k in r if not isinstance(r[k], list) and k != "z"})

    def summarize(idx):
        out = {"n_slices": int(len(idx))}
        for k in scalar_keys:
            vals = [per[i].get(k) for i in idx]
            out[k] = _nanmean([np.nan if v is None else v for v in vals])
        for k in ("dE_samples", "dE_mean", "dE_candidate"):
            rows = [per[i][k] for i in idx if k in per[i]]
            if rows:
                arr = np.asarray(rows)
                out[k] = {f"{s:g}": {"median": float(np.median(arr[:, j])),
                                     "frac_truth_lower": float((arr[:, j] > 0).mean())}
                          for j, s in enumerate(probes)}
        return out

    def spec_summary(idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return _spec_summary(idx)

    def _spec_summary(idx):
        return {f: {key: np.nanmean([spec[f][i][key] for i in idx], 0)
                    for key in ("ratio", "r", "ratio_ens_mean", "r_ens_mean")} for f in spec}

    bins = {}
    spec_bins = {}
    for lo, hi in zip(z_edges[:-1], z_edges[1:]):
        idx = [i for i in range(n) if lo <= z[i] < hi]
        if idx:
            label = f"{lo:.1f}-{hi:.1f}"
            bins[label] = summarize(idx)
            spec_bins[label] = spec_summary(idx)
    result = {
        "samples_file": str(Path(path).resolve()), "n_slices": n, "n_samples": m,
        "sampler": json.loads(attrs["sampler"]), "checkpoint_step": int(attrs["checkpoint_step"]),
        "probe_sigmas": probes, "overall": summarize(list(range(n))), "by_z": bins,
        "k_centers_per_mpc": kc.tolist(),
        "spectra_by_z": {b: {f: {k: np.asarray(v).tolist() for k, v in fs.items()} for f, fs in s.items()}
                         for b, s in spec_bins.items()},
    }

    out_dir = Path(out_dir or Path(path).with_suffix("")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    show = np.unique(np.linspace(0, n - 1, min(n_examples, n)).astype(int))
    viz.examples(out_dir / "examples.png", d["delta_center"][show], d["xhi_true"][show],
                 d["xhi_samples"][show], d["tb_true"][show], d["tb_samples"][show], z[show])
    viz.spectra(out_dir / "spectra.png", kc, spec_bins)
    if size_rows:
        viz.bubble_sizes(out_dir / "bubble_sizes.png", {k: np.concatenate(v) for k, v in sizes.items()})
    gaps = {"samples": np.asarray([r["dE_samples"] for r in per]),
            "ens. mean": np.asarray([r["dE_mean"] for r in per])}
    if any("dE_candidate" in r for r in per):
        gaps["candidate"] = np.asarray([r["dE_candidate"] for r in per if "dE_candidate" in r])
    viz.energy_gaps(out_dir / "energy.png", probes, gaps)
    return result, out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ebm21cm.evaluate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", required=True)
    ap.add_argument("--z-edges", help="comma-separated redshift bin edges (default: 4 quantile bins)")
    ap.add_argument("--out-dir")
    ap.add_argument("--n-rays", type=int, default=2000)
    ap.add_argument("--n-examples", type=int, default=4)
    a = ap.parse_args(argv)
    edges = [float(v) for v in a.z_edges.split(",")] if a.z_edges else None
    res, out = evaluate(a.samples, edges, a.out_dir, a.n_rays, a.n_examples)
    o = res["overall"]
    keys = ("xhi_rmse_sample", "xhi_rmse_mean", "xhi_crps", "xhi_spread_skill", "xhi_hedge_truth",
            "xhi_hedge_samples", "xhi_hedge_mean", "bsd_w1_samples", "bsd_w1_mean", "tb_rmse_sample",
            "tb_rmse_mean", "tb_crps", "tb_spread_skill")
    for k in keys:
        if o.get(k) is not None:
            print(f"{k:22s} {o[k]:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

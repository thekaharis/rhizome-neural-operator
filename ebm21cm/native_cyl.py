"""Cylindrical (k_perp, k_par) power at native LOS resolution, accumulated during export.

fno-21cm's cylindrical maps are computed on the 512-point evaluation grid (uniform
in z), whose LOS cells are 4-14 Mpc, so k_par stops at 0.23-0.77 Mpc^-1 while
k_perp reaches 2.2 Mpc^-1. Here each map uses a chunk of ``n_cells`` native slices
(1.43 Mpc each) centred on the native slice nearest each ``z_centers`` entry. With
the default 140 cells (= 200 Mpc, the transverse box) k_par has exactly the
transverse fundamental and Nyquist (0.031-2.2 Mpc^-1). k_perp uses fno-21cm's 15 log
bins (``KBinner``); k_par keeps every discrete LOS frequency as its own row (linear
spacing, edges halfway between, as fno-21cm's ``build_chunks``), or with
``kpar_bins="log"`` the k_perp log bins.

Per chunk, as in fno-21cm's ``_cylindrical_sums``: per-slice mean removed, Hann
window along the LOS (the chunk is not periodic), 3-D FFT, |F|^2 and Re(F_t F_p*)
summed into (k_par, k_perp) bins; the k_par = 0 plane is excluded. Per cone: ratio
P_pred/P_true and r per bin, masked when the truth is ~uniform (transverse std
<= 1e-3); then the median over cones. Saved per field in ``ebm21cm.plot_cyl_error``'s
layout (``<model>/cyl_ratio_med`` ...), one npz per field.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

FNO_ROOT = "/pfs/10/work/hd_id260-fno_training/fno-21cm"
BOX_MPC, N_PERP, N_BINS, MIN_TRUTH_STD = 200.0, 140, 15, 1e-3


class NativeCyl:
    def __init__(self, z_centers, n_cells=140, fields=("xhi", "tb"), kpar_bins="linear"):
        sys.path.insert(0, FNO_ROOT)
        from viz.power_spectrum_evaluation import KBinner

        self.z_centers, self.n_cells, self.fields = tuple(z_centers), int(n_cells), tuple(fields)
        self.binner = KBinner(N_PERP, N_PERP, BOX_MPC, N_BINS)
        self.cell = BOX_MPC / N_PERP
        kpar = 2 * np.pi * np.fft.rfftfreq(self.n_cells, d=self.cell)
        if kpar_bins == "linear":  # one row per frequency, DC dropped
            self.kpar_matrix = np.eye(len(kpar))[:, 1:]
            df = kpar[1]
            self.kpar_edges = np.concatenate([[kpar[1] - df / 2], kpar[1:] + df / 2])
        elif kpar_bins == "log":
            idx = np.digitize(kpar, self.binner.edges) - 1
            valid = (idx >= 0) & (idx < N_BINS) & (kpar > 0)
            self.kpar_matrix = np.zeros((len(kpar), N_BINS))
            self.kpar_matrix[np.flatnonzero(valid), idx[valid]] = 1.0
            self.kpar_edges = self.binner.edges
        else:
            raise ValueError("kpar_bins must be 'linear' or 'log'")
        self.window = np.hanning(self.n_cells)
        self.records = {f: [] for f in self.fields}
        self.cone_ids = {f: [] for f in self.fields}

    def chunk_slices(self, z_native):
        """Native index slice per z centre (None when the chunk leaves the cone)."""
        z = np.asarray(z_native)
        out = []
        for zc in self.z_centers:
            start = int(np.argmin(np.abs(z - zc))) - self.n_cells // 2
            out.append(slice(start, start + self.n_cells) if 0 <= start and start + self.n_cells <= len(z) else None)
        return out

    def needed(self, z_native):
        """Sorted native slice indices covered by the chunks."""
        return np.unique(np.concatenate([np.arange(s.start, s.stop) for s in self.chunk_slices(z_native) if s]))

    def _sums(self, truth, pred):
        b = self.binner

        def spectrum(x):
            x = (x - x.mean(axis=(0, 1), keepdims=True)) * self.window
            return np.fft.fftn(np.fft.rfft(x, axis=2), axes=(0, 1))

        def binned(x):
            flat = x.reshape(-1, x.shape[2])[b.valid]          # (modes, n_kpar_freq)
            return (b.matrix.T @ flat @ self.kpar_matrix).T    # (n_kpar_bins, n_perp_bins)

        tf, pf = spectrum(truth), spectrum(pred)
        return binned(np.abs(tf) ** 2), binned(np.abs(pf) ** 2), binned(np.real(tf * np.conj(pf)))

    def add(self, field, cone_id, z_native, truth, pred):
        """``truth``/``pred``: (H, W, n) on the native slices ``needed(z_native)`` (or the whole cone)."""
        truth, pred = np.asarray(truth, np.float64), np.asarray(pred, np.float64)
        slices = self.chunk_slices(z_native)
        have = self.needed(z_native) if truth.shape[2] != len(z_native) else np.arange(len(z_native))
        pos = {int(i): k for k, i in enumerate(have)}
        rec = []
        for s in slices:
            if s is None:
                rec.append(None)
                continue
            cols = [pos[i] for i in range(s.start, s.stop)]
            t, p = truth[..., cols], pred[..., cols]
            pt, pp, cross = self._sums(t, p)
            ok = bool(t.std(axis=(0, 1)).mean() > MIN_TRUTH_STD)
            rec.append((pt, pp, cross, ok))
        self.records[field].append(rec)
        self.cone_ids[field].append(int(cone_id))

    def save(self, prefix, model):
        """Write ``<prefix>_<field>.npz`` for each field; return the paths."""
        paths = []
        for field, recs in self.records.items():
            ratios, rs, n = [], [], []
            for c in range(len(self.z_centers)):
                rc, rr = [], []
                for rec in recs:
                    if rec[c] is None:
                        continue
                    pt, pp, cross, ok = rec[c]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ratio = np.where(pt > 0, pp / pt, np.nan)
                        r = np.where(pt * pp > 0, cross / np.sqrt(pt * pp), np.nan)
                    if not ok:
                        ratio[:], r[:] = np.nan, np.nan
                    rc.append(ratio)
                    rr.append(r)
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ratios.append(np.nanmedian(np.stack(rc), axis=0))
                    rs.append(np.nanmedian(np.stack(rr), axis=0))
                n.append(len(rc))
            m = model
            path = Path(f"{prefix}_{field}.npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{
                f"{m}/cyl_ratio_med": np.stack(ratios), f"{m}/cyl_r_med": np.stack(rs),
                f"{m}/cyl_z_centers": np.array(self.z_centers, float),
                f"{m}/cyl_kpar_edges": np.stack([self.kpar_edges] * len(self.z_centers)),
                f"{m}/k_edges": self.binner.edges, f"{m}/n_cones": np.array(n),
                # per cone and chunk: total truth power, to check that models saw identical truth
                f"{m}/cone_ids": np.array(self.cone_ids[field]),
                f"{m}/truth_power": np.array([[np.nan if r is None else float(r[0].sum()) for r in rec]
                                              for rec in recs]),
                f"{m}/meta": np.array(json.dumps({"n_cells": self.n_cells, "cell_mpc": self.cell, "field": field,
                                                  "kpar_rows": len(self.kpar_edges) - 1,
                                                  "resolution": "native LOS"}))})
            paths.append(path)
        return paths

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

The transverse geometry defaults to the 21cmFAST test cones (200 Mpc, 140 cells); LoReLi
lightcones pass ``box_mpc=294, n_perp=206`` (same 1.43 Mpc cells, so the same k_par rows; k_perp
uses ``KBinner``'s 15 log bins of the larger box).

Chunk placement: at fixed redshifts (``z_centers``), or with ``stages`` per cone at an
ionization stage, so that simulations that reionize at different redshifts are compared like
for like. A stage chunk is centred on the native slice where the cone's true slice-mean x_HI
(``history=(z, mean x_HI)`` passed to ``needed``/``add``, on any z grid) is nearest the stage;
the cone is skipped for that stage when its history comes no closer than ``STAGE_TOL``. A chunk
that would leave the cone is shifted inside it, and dropped if the shift moves its centre to a
mean x_HI more than ``STAGE_SHIFT_TOL`` away. The saved ``cyl_z_centers`` are then the median
chunk-centre redshifts, and ``cyl_stage`` holds the stages.

Resolution: when the truth was interpolated onto the evaluation grid from a coarser native one
(LoReLi: 2.3 Mpc transverse cells, 1.6-3.1 Mpc LOS planes), pass ``kperp_max`` (transverse
Nyquist of the native truth) and ``los_spacing`` (callable z -> native LOS plane spacing in Mpc);
``kperp_max`` and per chunk column ``cyl_kpar_max`` (median over cones of pi / spacing at the
chunk centre) are saved so ``ebm21cm.plot_cyl_error`` can mark the unresolved wavenumbers.

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
STAGE_TOL, STAGE_SHIFT_TOL = 0.05, 0.1


class NativeCyl:
    def __init__(self, z_centers=None, n_cells=140, fields=("xhi", "tb"), kpar_bins="linear", box_mpc=BOX_MPC,
                 n_perp=N_PERP, stages=None, kperp_max=None, los_spacing=None):
        sys.path.insert(0, FNO_ROOT)
        from viz.power_spectrum_evaluation import KBinner

        if (z_centers is None) == (stages is None):
            raise ValueError("give exactly one of z_centers and stages")
        self.stages = None if stages is None else tuple(float(s) for s in stages)
        self.z_centers = () if z_centers is None else tuple(z_centers)
        self.n_columns = len(self.stages or self.z_centers)
        self.kperp_max, self.los_spacing = kperp_max, los_spacing
        self.n_cells, self.fields = int(n_cells), tuple(fields)
        self.box_mpc, self.n_perp = float(box_mpc), int(n_perp)
        self.binner = KBinner(self.n_perp, self.n_perp, self.box_mpc, N_BINS)
        self.cell = self.box_mpc / self.n_perp
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
        self.chunk_z = {f: [] for f in self.fields}

    def chunk_slices(self, z_native, history=None):
        """Native index slice per column (None when the cone has no valid chunk there)."""
        z = np.asarray(z_native)
        half, out = self.n_cells // 2, []
        if self.stages is None:
            for zc in self.z_centers:
                start = int(np.argmin(np.abs(z - zc))) - half
                out.append(slice(start, start + self.n_cells) if 0 <= start and start + self.n_cells <= len(z) else None)
            return out
        if history is None:
            raise ValueError("stage chunks need the cone's truth history (z, mean x_HI)")
        hz, hx = (np.asarray(v, np.float64) for v in history)
        order = np.argsort(hz)
        hz, hx = hz[order], hx[order]
        for stage in self.stages:
            j = int(np.argmin(np.abs(hx - stage)))
            if abs(hx[j] - stage) > STAGE_TOL or len(z) < self.n_cells:
                out.append(None)
                continue
            start = min(max(int(np.argmin(np.abs(z - hz[j]))) - half, 0), len(z) - self.n_cells)
            shifted = float(np.interp(z[start + half], hz, hx))
            out.append(slice(start, start + self.n_cells) if abs(shifted - stage) <= STAGE_SHIFT_TOL else None)
        return out

    def needed(self, z_native, history=None):
        """Sorted native slice indices covered by the chunks."""
        chunks = [np.arange(s.start, s.stop) for s in self.chunk_slices(z_native, history) if s]
        return np.unique(np.concatenate(chunks)) if chunks else np.zeros(0, int)

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

    def add(self, field, cone_id, z_native, truth, pred, history=None):
        """``truth``/``pred``: (H, W, n) on the native slices ``needed(z_native, history)`` (or the whole cone)."""
        truth, pred = np.asarray(truth, np.float64), np.asarray(pred, np.float64)
        slices = self.chunk_slices(z_native, history)
        have = self.needed(z_native, history) if truth.shape[2] != len(z_native) else np.arange(len(z_native))
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
        self.chunk_z[field].append([np.nan if s is None else float(np.asarray(z_native)[s.start + self.n_cells // 2])
                                    for s in slices])

    def save(self, prefix, model):
        """Write ``<prefix>_<field>.npz`` for each field; return the paths."""
        paths = []
        for field, recs in self.records.items():
            ratios, rs, n = [], [], []
            empty = np.full((len(self.kpar_edges) - 1, N_BINS), np.nan)
            for c in range(self.n_columns):
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
                    ratios.append(np.nanmedian(np.stack(rc), axis=0) if rc else empty)
                    rs.append(np.nanmedian(np.stack(rr), axis=0) if rr else empty)
                n.append(len(rc))
            m = model
            chunk_z = np.array(self.chunk_z[field], float).reshape(-1, self.n_columns)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                z_cols = np.array(self.z_centers, float) if self.stages is None else np.nanmedian(chunk_z, axis=0)
            extra = {}
            if self.stages is not None:
                extra[f"{m}/cyl_stage"] = np.array(self.stages, float)
            if self.kperp_max is not None:
                extra[f"{m}/kperp_max"] = np.array(float(self.kperp_max))
            if self.los_spacing is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    extra[f"{m}/cyl_kpar_max"] = np.nanmedian(
                        np.where(np.isfinite(chunk_z), np.pi / np.vectorize(self.los_spacing)(np.nan_to_num(chunk_z, nan=1.0)),
                                 np.nan), axis=0)
            path = Path(f"{prefix}_{field}.npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{
                f"{m}/cyl_ratio_med": np.stack(ratios), f"{m}/cyl_r_med": np.stack(rs),
                f"{m}/cyl_z_centers": z_cols, f"{m}/chunk_z": chunk_z,
                f"{m}/cyl_kpar_edges": np.stack([self.kpar_edges] * self.n_columns),
                f"{m}/k_edges": self.binner.edges, f"{m}/n_cones": np.array(n),
                # per cone and chunk: total truth power, to check that models saw identical truth
                f"{m}/cone_ids": np.array(self.cone_ids[field]),
                f"{m}/truth_power": np.array([[np.nan if r is None else float(r[0].sum()) for r in rec]
                                              for rec in recs]),
                f"{m}/meta": np.array(json.dumps({"n_cells": self.n_cells, "cell_mpc": self.cell, "field": field,
                                                  "box_mpc": self.box_mpc, "n_perp": self.n_perp,
                                                  "kpar_rows": len(self.kpar_edges) - 1,
                                                  "chunking": "fixed z" if self.stages is None else "stage",
                                                  "resolution": "native LOS"})), **extra})
            paths.append(path)
        return paths

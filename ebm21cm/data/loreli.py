"""LoReLi II (Licorice) lightcones on the slice-wise rhizome's input grid.

A five-field file holds ``(128, 128, 1536, 5)`` little-endian float32 with the
fields interleaved per voxel: ``xa, tpHI, xion, dens, dtb``. Index 0 is the
low-redshift end. The line of sight is uniform in comoving distance from z = 5
to z = 15 (1536 cells, 1.639 Mpc); ``dtb`` is in mK and has no velocity-gradient
term. Empty tails below the redshift at which the simulation stopped have
``dens = 0`` and ``dtb = NaN``.

This module turns one cone into the inputs of ``train_rhizome_mf2d``'s slice
cache: 13 LOS density bands, 13 LOS velocity bands (linear theory, in the
training convention; see :func:`linear_velocity`), the scalar row (redshift,
11 parameters at the training mean, and the cone's own global x_HI / T_b
histories band-averaged over the same 13 bands), and the structured head's
physical factor. The transverse plane is resampled from 128 cells over
``L_perp`` to the training cell (200/140 Mpc) with mirror boundaries (LoReLi
is not periodic), then mirror-padded to 280 cells = 400 Mpc = twice the
training box for the remapped spectral model (:mod:`ebm21cm.model.remap`).

Everything is derived from the cone itself; no cache is rebuilt and no
training code changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ..cosmology import FlatLCDM
from ..model.rhizome3d import StructuredBrightness3d

FIELDS = ("xa", "tpHI", "xion", "dens", "dtb")
LORELI_H = 0.678
LORELI_OMEGA_M = 0.308
LORELI_Z = (5.0, 15.0)
LORELI_N_LOS = 1536
LORELI_NATIVE = 128
LORELI_L_PERP = 294.0            # LoReLi I Table 1; LoReLi II writes 300
DOMAIN = 280                     # 400 Mpc, twice the 200 Mpc training box
HISTORY_TB_SCALE = 100.0         # fno-21cm's T_b history channel is T_b [mK] / 100
TRAINING_CELL_MPC = 200.0 / 140   # the slice cache's 1.4286 Mpc transverse/LOS cell


def loreli_redshifts(z_range=LORELI_Z, n=LORELI_N_LOS, omega_m=LORELI_OMEGA_M, h=LORELI_H):
    """(z, distance from z_min) of the uniform-in-comoving-distance LOS axis."""
    cosmo = FlatLCDM(omega_m, h)
    z0, z1 = z_range
    chi = np.linspace(cosmo.comoving_distance(z0), cosmo.comoving_distance(z1), n)
    distance = chi - chi[0]
    z = cosmo.z_at_distance(chi)
    return z, distance


def loreli_files(root):
    """Sorted five-field files and their simulation numbers."""
    root = Path(root)
    files = sorted(root.glob("lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu*.dat"))
    ids = [int(re.findall(r"\d+", p.stem)[-1]) for p in files]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate simulation numbers in {root}")
    return files, ids


def _mirror_index(n, pad):
    """Length-2*pad+n axis indices, mirror-extended without repeating the edge."""
    if n == 1:
        return np.zeros(2 * pad + 1, dtype=np.int64)
    period = 2 * n - 2
    i = np.abs(np.arange(-pad, n + pad)) % period
    return np.where(i >= n, period - i, i)


def mirror_pad(values, pad, axis=-1):
    """Mirror-pad one axis with :func:`_mirror_index` (works past the array length)."""
    return np.take(values, _mirror_index(values.shape[axis], pad), axis=axis)


def pad_to(plane, size=DOMAIN):
    """Mirror-pad the first two axes of ``plane`` to ``size x size``, centred."""
    ny, nx = plane.shape[:2]
    if ny > size or nx > size:
        raise ValueError(f"plane {plane.shape[:2]} does not fit {size}x{size}")
    py, px = (size - ny) // 2, (size - nx) // 2
    return mirror_pad(mirror_pad(plane, py, 0), px, 1)


def _grid(n_in, n_out):
    """``grid_sample`` coordinates mapping cell centres of n_in to n_out (align_corners=False)."""
    t = (torch.arange(n_out, dtype=torch.float32) + 0.5) * (2.0 / n_out) - 1.0
    return torch.stack([t[None, :].expand(n_out, n_out), t[:, None].expand(n_out, n_out)], -1)[None]


def resample_transverse(plane, n_out):
    """(ny, nx, n) -> (n_out, n_out, n) bicubic resample with mirror boundaries.

    Cell centres are mapped linearly in physical coordinates, so the output
    spans the same ``L_perp``. LoReLi is not periodic; ``grid_sample`` reflects
    at the edges instead of wrapping.
    """
    y = torch.as_tensor(plane, dtype=torch.float32).permute(2, 0, 1)[:, None]
    out = F.grid_sample(y, _grid(plane.shape[0], n_out).expand(y.shape[0], -1, -1, -1),
                        mode="bicubic", padding_mode="reflection", align_corners=False)
    return out[:, 0].permute(1, 2, 0).numpy()


def resample_los(cube, distance, target_distance):
    """(ny, nx, n) -> (ny, nx, len(target)) linear in comoving distance, edge-clamped."""
    position = np.interp(target_distance, distance, np.arange(len(distance)))
    i0 = np.floor(position).astype(np.int64).clip(0, min(len(distance), cube.shape[-1]) - 1)
    i1 = np.minimum(i0 + 1, cube.shape[-1] - 1)
    t = (position - i0).astype(np.float32)
    return cube[..., i0] * (1 - t) + cube[..., i1] * t


def band_planes(values, centers, bands):
    """(ny, nx, n) -> (len(centers), len(bands), ny, nx) LOS band means in float64 arithmetic."""
    if max(centers) + max(hi for _, hi in bands) > values.shape[-1] or \
            min(centers) + min(lo for lo, _ in bands) < 0:
        raise ValueError("band lies outside the LOS axis")
    csum = np.concatenate([np.zeros(values.shape[:2] + (1,), np.float64),
                           np.cumsum(values, -1, dtype=np.float64)], -1)
    out = np.empty((len(centers), len(bands), *values.shape[:2]), np.float32)
    for k, c in enumerate(centers):
        for b, (lo, hi) in enumerate(bands):
            out[k, b] = ((csum[:, :, c + hi] - csum[:, :, c + lo]) / (hi - lo)).astype(np.float32)
    return out


def band_profile(profile, centers, bands):
    """(n,) -> (len(centers), len(bands)) LOS band means of a 1-D profile."""
    csum = np.concatenate([[0.0], np.cumsum(profile, dtype=np.float64)])
    return np.stack([(csum[c + hi] - csum[c + lo]) / (hi - lo)
                     for c in centers for lo, hi in bands]).reshape(len(centers), len(bands))


def hubble_growth(z, omega_m=LORELI_OMEGA_M, h=LORELI_H):
    """H(z) f(z) in 1/s, the factor that turns the velocity shape into comoving Mpc/s."""
    om = omega_m * (1.0 + np.asarray(z, np.float64)) ** 3
    hubble = 100.0 * h / 3.0856775814913673e19 * np.sqrt(om + 1 - omega_m)
    growth = (om / (om + 1 - omega_m)) ** 0.55
    return hubble * growth


def velocity_shape(delta, cell=TRAINING_CELL_MPC, cell_los=None):
    """Geometric part of the linear-theory LOS velocity: ``i k_par delta_k / k^2`` (1/Mpc).

    Multiply by :func:`hubble_growth` at the slice redshift for comoving Mpc/s.
    ``delta`` is (ny, nx, n) and is treated as periodic; the transverse and LOS
    spacings may differ (LoReLi's native grid is 2.297 x 1.6397 Mpc).
    """
    ny, nx, nz = delta.shape
    cell_los = cell if cell_los is None else cell_los
    kx = 2 * np.pi * np.fft.fftfreq(ny, d=cell)
    ky = 2 * np.pi * np.fft.fftfreq(nx, d=cell)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, d=cell_los)
    k2 = (kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(np.float32)
    k2[0, 0, 0] = 1.0
    spectrum = np.fft.rfftn(np.asarray(delta, np.float32))
    return np.fft.irfftn((1j * kz[None, None, :] * spectrum / k2).astype(np.complex64),
                         s=(ny, nx, nz), axes=(0, 1, 2)).astype(np.float32)


def linear_velocity(delta, z, cell=TRAINING_CELL_MPC, cell_los=None, omega_m=LORELI_OMEGA_M, h=LORELI_H):
    """Linear-theory LOS velocity in the 21cmFAST training convention (comoving Mpc/s).

    ``v = i H(z) f(z) k_par delta_k / k^2``. This is the convention of the
    training caches: the stored ``los_velocity`` of a 21cmFAST cone matches it
    with slope 1.00 and r = 0.9996 at z = 6.5-12 (a factor 1+z above
    ``a H f k delta_k / k^2``, i.e. dx/deta).
    """
    return velocity_shape(delta, cell, cell_los) * np.float32(hubble_growth(z, omega_m, h))


def amplitude(z, omega_m=LORELI_OMEGA_M, h=LORELI_H):
    """Structured head's A(z) in mK: 27 (Omega_b h^2 / 0.023) sqrt(0.15/(Omega_m h^2) (1+z)/10)."""
    return 27.0 * (StructuredBrightness3d.OMEGA_B_H2 / 0.023) * np.sqrt(
        0.15 / (omega_m * h ** 2) * (1 + np.asarray(z, np.float64)) / 10.0)


class LoreliCone:
    """One memory-mapped five-field cone; index 0 is the low-redshift end."""

    def __init__(self, path, l_perp=LORELI_L_PERP, z_range=LORELI_Z, omega_m=LORELI_OMEGA_M, h=LORELI_H,
                 density_mean="slice", n_native=LORELI_NATIVE, n_los=LORELI_N_LOS):
        if float(l_perp) not in (294.0, 300.0):
            raise ValueError("l_perp must be 294 or 300 Mpc (the two published values)")
        if density_mean not in ("slice", "cone"):
            raise ValueError("density_mean must be 'slice' or 'cone'")
        self.path = Path(path)
        self.sim = int(re.findall(r"\d+", self.path.stem)[-1])
        self.l_perp, self.density_mean = float(l_perp), density_mean
        self.n_native, self.n_los = int(n_native), int(n_los)
        self.data = np.memmap(self.path, dtype="<f4", mode="r").reshape(
            self.n_native, self.n_native, self.n_los, len(FIELDS))
        self.z, self.distances = loreli_redshifts(z_range, self.n_los, omega_m, h)
        self.valid, self.n_slices = self._validity()
        if not self.n_slices:
            raise ValueError(f"{self.path}: no valid slices")
        self._mean = None
        self._history = None

    def _validity(self):
        """Nonempty slices of the one contiguous run that holds the cone, with edge artefacts removed."""
        dens = np.array(self.data[..., FIELDS.index("dens")], dtype=np.float32)
        mean = dens.mean((0, 1))
        valid = np.isfinite(np.array(self.data[..., FIELDS.index("dtb")], np.float32)).all((0, 1)) & (mean > 0)
        if valid.any():
            # The first/last slice of a particle lightcone can hold a partly filled plane (18% low in density);
            # interior slice means fluctuate by a few percent (large-scale modes), so a 10% cut removes the
            # artefact rather than signal.
            valid &= np.abs(mean / np.median(mean[valid]) - 1) <= 0.10
        idx = np.flatnonzero(valid)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        keep = np.zeros(len(mean), bool)
        if any(len(r) for r in runs):
            keep[max(runs, key=len)] = True
        return keep, int(keep.sum())

    def mean_density(self):
        if self._mean is None:
            dens = np.array(self.data[..., FIELDS.index("dens")], np.float32)
            good = dens > 0
            self._mean = float(dens[good].mean())
        return self._mean

    def history(self):
        """(x_HI, T_b [mK]) means over all cells per native slice."""
        if self._history is None:
            xhi = 1.0 - np.array(self.data[..., FIELDS.index("xion")], np.float32)
            tb = np.array(self.data[..., FIELDS.index("dtb")], np.float32)
            self._history = (xhi.mean((0, 1)), np.nanmean(tb, axis=(0, 1)))
        return self._history

    def read(self, start, stop, fields=("dens", "xion", "dtb", "los_velocity")):
        """Native (128, 128, stop-start) fields; ``los_velocity`` is None for LoReLi."""
        out = {}
        for name in fields:
            if name == "los_velocity":
                out[name] = None
            elif name in FIELDS:
                out[name] = np.array(self.data[:, :, start:stop, FIELDS.index(name)], np.float32)
            else:
                raise KeyError(name)
        return out

    def delta(self, dens):
        """delta = rho / rho_bar - 1 over populated cells, slice (default) or cone mean."""
        good = dens > 0
        if self.density_mean == "cone":
            rho = self.mean_density()
        else:
            rho = np.where(good, dens, 0).sum((0, 1), keepdims=True) / np.maximum(
                good.sum((0, 1), keepdims=True), 1)
        ratio = np.divide(dens, rho, out=np.zeros_like(dens), where=np.asarray(rho) > 0)
        return np.where(good, ratio - 1.0, 0.0).astype(np.float32)


def area_matrix(n_native, n_out):
    """(n_native, n_out) overlap weights: native cell averages of a finer periodic plane."""
    edges_in = np.linspace(0.0, 1.0, n_native + 1)
    edges_out = np.linspace(0.0, 1.0, n_out + 1)
    weights = np.zeros((n_native, n_out), np.float32)
    for i in range(n_native):
        j0, j1 = np.searchsorted(edges_out, edges_in[i], "right") - 1, np.searchsorted(edges_out, edges_in[i + 1], "left")
        for j in range(max(0, j0), min(n_out, j1 + 1)):
            overlap = min(edges_in[i + 1], edges_out[j + 1]) - max(edges_in[i], edges_out[j])
            if overlap > 0:
                weights[i, j] = overlap / (edges_in[i + 1] - edges_in[i])
    return weights


def training_stats(cache):
    """The slice cache's bands and normalization, as the adapter needs them."""
    with h5py.File(cache, "r") as h:
        return {"bands": [tuple(b) for b in json.loads(h.attrs["bands"])],
                "normalization": json.loads(h.attrs["normalization"]),
                "param_names": [str(s) for s in json.loads(h.attrs["param_names"])],
                "scalar_names": [str(s) for s in json.loads(h.attrs["scalar_names"])]}


class LoreliAdapter:
    """Training-cache-shaped slice inputs for one cone on the 280-cell domain."""

    def __init__(self, cone, stats, domain=DOMAIN, velocity="linear", l_perp=None):
        if velocity not in ("linear", "stored", "zero"):
            raise ValueError("velocity must be 'linear', 'stored' or 'zero'")
        self.cone, self.stats, self.domain = cone, stats, int(domain)
        self.velocity = velocity
        self.bands = stats["bands"]
        self.norm = stats["normalization"]
        self.halo = max(0, -min(lo for lo, _ in self.bands), max(hi for _, hi in self.bands) - 1)
        self.l_perp = float(l_perp or cone.l_perp)
        self.n_out = int(round(self.l_perp / TRAINING_CELL_MPC))
        self.cell = self.l_perp / self.n_out
        self.cell_los = TRAINING_CELL_MPC
        first, last = np.flatnonzero(cone.valid)[[0, -1]]
        stop = int(np.ceil(cone.distances[last] / self.cell_los)) + 1
        grid = np.arange(stop) * self.cell_los
        self.target_distance = grid[(grid >= cone.distances[first] - 1e-9) & (grid <= cone.distances[last] + 1e-9)]
        cosmo = FlatLCDM(LORELI_OMEGA_M, LORELI_H)
        self.z_target = cosmo.z_at_distance(cosmo.comoving_distance(LORELI_Z[0]) + self.target_distance)
        xhi_hist, tb_hist = cone.history()
        native = np.flatnonzero(cone.valid)
        self.history = (np.interp(self.target_distance, cone.distances[native], xhi_hist[native]),
                        np.interp(self.target_distance, cone.distances[native], tb_hist[native]) / HISTORY_TB_SCALE)
        self._vel_shape, self._vel0 = None, 0
        if velocity == "linear":
            self._vel_shape, self._vel0 = self._precompute_velocity()

    def _precompute_velocity(self):
        """Geometric velocity shape over the valid native run; one FFT, mirror-padded in LOS.

        The z-dependent H(z) f(z) factor is applied per target slice after
        resampling, so the amplitude follows the lightcone without chunk seams.
        """
        first, last = np.flatnonzero(self.cone.valid)[[0, -1]]
        lo, hi = max(0, first - self.halo), min(self.cone.n_los, last + self.halo + 1)
        dense = self.cone.delta(self.cone.read(lo, hi, fields=("dens",))["dens"])
        shape = velocity_shape(dense, cell=self.l_perp / self.cone.n_native,
                               cell_los=float(np.median(np.diff(self.cone.distances))))
        return shape, lo

    def _native(self, start, stop):
        raw = self.cone.read(start, stop)
        velocity = (self._vel_shape[:, :, start - self._vel0:stop - self._vel0]
                    if self.velocity == "linear" else raw["los_velocity"])
        return {"delta": self.cone.delta(raw["dens"]), "xhi": 1.0 - raw["xion"], "tb": raw["dtb"],
                "los_velocity": velocity}

    def _target(self, a, b):
        """Resampled fields for target positions [lo, hi) that contain [a, b) plus the band halo."""
        lo = max(0, a - self.halo)
        hi = min(len(self.target_distance), b + self.halo + 1)
        distance = self.target_distance[lo:hi]
        grid = np.arange(self.cone.n_los)
        n0 = int(np.floor(np.interp(distance[0], self.cone.distances, grid))) - 1
        n1 = int(np.ceil(np.interp(distance[-1], self.cone.distances, grid))) + 2
        n0, n1 = max(0, n0), min(self.cone.n_los, n1)
        native = self._native(n0, n1)
        out = {}
        for key in ("delta", "xhi", "tb", "los_velocity"):
            value = native[key]
            if value is not None:
                out[key] = resample_transverse(
                    resample_los(value, self.cone.distances[n0:n1], distance), self.n_out)
        if self.velocity == "linear":
            out["los_velocity"] *= hubble_growth(self.z_target[lo:hi]).astype(np.float32)[None, None, :]
        elif self.velocity == "zero":
            out["los_velocity"] = np.zeros_like(out["delta"])
        elif out.get("los_velocity") is None:
            raise ValueError("stored velocity requested but this cone has none")
        return out, lo, hi

    def inputs(self, a, b):
        """Inputs for target indices [a, b): cond (n,26,280,280), scalars (n,38), phys (n,1,280,280)."""
        fields, lo, hi = self._target(a, b)
        n = b - a
        centers = np.arange(a, b) - lo + self.halo
        delta, vel = fields["delta"], fields["los_velocity"]
        delta_p = np.pad(delta, ((0, 0), (0, 0), (self.halo, self.halo)), mode="edge")
        vel_p = np.pad(vel, ((0, 0), (0, 0), (self.halo, self.halo)), mode="edge")
        dn, vn = self.norm["density"], self.norm["los_velocity"]
        cond = np.concatenate([band_planes((delta_p - dn["offset"]) / dn["scale"], centers, self.bands),
                               band_planes((vel_p - vn["offset"]) / vn["scale"], centers, self.bands)], 1)
        # Band means commute with the transverse mirror padding: pad the (H, W, C, n) stack once.
        cond = pad_to(cond.transpose(2, 3, 1, 0), self.domain).transpose(3, 2, 0, 1)
        phys = pad_to(np.stack([amplitude(z) * (1.0 + delta[:, :, a - lo + i])
                                for i, z in enumerate(self.z_target[a:b])], -1), self.domain)
        phys = phys.transpose(2, 0, 1)[:, None]
        hist_x = band_profile(np.pad(self.history[0][lo:hi], (self.halo, self.halo), mode="edge"),
                              centers, self.bands)
        hist_t = band_profile(np.pad(self.history[1][lo:hi], (self.halo, self.halo), mode="edge"),
                              centers, self.bands)
        scalars = np.concatenate([(1.0 / (1.0 + self.z_target[a:b]))[:, None],
                                  np.zeros((n, len(self.stats["param_names"])), np.float32),
                                  hist_x, hist_t], 1).astype(np.float32)
        return {"cond": cond.astype(np.float32), "scalars": scalars, "phys": phys.astype(np.float32),
                "z": self.z_target[a:b].astype(np.float32), "target": np.arange(a, b)}

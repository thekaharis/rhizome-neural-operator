"""A 21cmFAST lightcone re-expressed in LoReLi's 128-cell / 1536-slice geometry.

For the resolution-only part of the LoReLi transfer error: the same adapter is
run on a 21cmFAST cone that has been degraded to 2.297 Mpc transverse and
1.6397 Mpc along the line of sight, then resampled back. Density, x_HI, T_b and
the stored LOS velocity are all passed through the same degradation, so the
remaining difference to a native 21cmFAST run is the lost small-scale
information (plus the mirror padding of the 294 Mpc plane). The 200 Mpc
21cmFAST box sits in the centre of the 294 Mpc LoReLi box; the outer 47 Mpc on
each side are a mirror extension.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from .loreli import LORELI_NATIVE, LORELI_N_LOS, LORELI_Z, TRAINING_CELL_MPC, area_matrix, loreli_redshifts, pad_to, resample_los

FIELDS = ("density", "neutral_fraction", "brightness_temp", "los_velocity")


class Degraded21cmCone:
    """Read-only LoReLi-shaped view of one 21cmFAST lightcone (mirror fields)."""

    def __init__(self, path, l_perp=294.0, z_range=LORELI_Z, n_native=LORELI_NATIVE, n_los=LORELI_N_LOS):
        self.path = Path(path)
        self.sim = int(re.findall(r"\d+", self.path.stem)[-1])
        self.l_perp = float(l_perp)
        self.n_native, self.n_los = int(n_native), int(n_los)
        self.z, self.distances = loreli_redshifts(z_range, self.n_los)
        with h5py.File(self.path, "r") as f:
            group = f["lightcone"] if "lightcone" in f else f
            z = np.asarray(group["lightcone_redshifts"])
            chi = np.asarray(group["lightcone_distances"])
            order = np.argsort(z, kind="stable")
            z, chi = z[order], chi[order]
            keep = (z >= z_range[0] - 0.05) & (z <= z_range[1] + 0.05)
            z, chi = z[keep], chi[keep]
            fields = {name: np.asarray(group[name], np.float32)[..., order][..., keep] for name in FIELDS}
        if len(z) < 2:
            raise ValueError(f"{self.path}: no lightcone slices in z {z_range}")
        # Interpolate in redshift, not comoving distance: the 21cmFAST cone uses its own cosmology, whose
        # chi(z) differs from LoReLi's by a few percent. Same z is the physical slice to compare.
        self.fields = {}
        for name, value in fields.items():
            los = resample_los(value, z, self.z)
            self.fields[name] = self._degrade(los)
        self.valid = np.ones(self.n_los, bool)
        self.n_slices = self.n_los
        self.density_mean = "slice"

    def _degrade(self, cube):
        """Mirror-pad each 140-cell plane to 294 Mpc, then area-average to the 128-cell grid."""
        n_out = int(round(self.l_perp / TRAINING_CELL_MPC))
        weights = area_matrix(self.n_native, n_out)
        planes = np.stack([pad_to(cube[..., i], n_out) for i in range(cube.shape[-1])], -1)
        return np.einsum("ij,jkn,lk->iln", weights, planes, weights, optimize=True).astype(np.float32)

    def mean_density(self):
        return None

    def history(self):
        return self.fields["neutral_fraction"].mean((0, 1)), self.fields["brightness_temp"].mean((0, 1))

    def read(self, start, stop, fields=("dens", "xion", "dtb", "los_velocity")):
        out = {"dens": 1.0 + self.fields["density"][..., start:stop],
               "xion": 1.0 - self.fields["neutral_fraction"][..., start:stop],
               "dtb": self.fields["brightness_temp"][..., start:stop],
               "los_velocity": self.fields["los_velocity"][..., start:stop]}
        return {k: out[k] for k in fields}

    def delta(self, dens):
        good = dens > 0
        rho = np.where(good, dens, 0).sum((0, 1), keepdims=True) / np.maximum(
            good.sum((0, 1), keepdims=True), 1)
        ratio = np.divide(dens, rho, out=np.zeros_like(dens), where=rho > 0)
        return np.where(good, ratio - 1.0, 0.0).astype(np.float32)

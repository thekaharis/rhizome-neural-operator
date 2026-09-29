"""Read 21cmFAST lightcones in either of the two on-disk schemas.

``raw_v2`` (the cluster ensemble, ``raw_lightcone_v2.0``)::

    /lightcone/<field>                 (H, W, n_los) float32
    /lightcone/lightcone_distances     (n_los,)  comoving Mpc
    /lightcone/lightcone_redshifts     (n_los,)
    /params                  attrs: F_STAR10, L_X, OMm, SIGMA_8, ...
    /params/fixed_cosmo_params  attrs: hlittle, OMb, POWER_INDEX
    file attrs: box_len_mpc

``native`` (``py21cmfast.LightCone.save`` as used by local runs)::

    /lightcones/<field>                (H, W, n_los) float32
    /lightcone_distances               (n_los,)  comoving Mpc
    /InputParameters/{astro_params,cosmo_params,cosmo_tables,
                      simulation_options}  attrs

Native files store no redshift axis; it is recovered from the distances with
the file's own cosmology. Both schemas are exposed with the LOS axis ordered
by increasing redshift and fields returned as ``(n, H, W)`` float32.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..cosmology import FlatLCDM

FIELDS = ("density", "neutral_fraction", "brightness_temp")

# The 11 parameters varied in the cluster design (21cmfast_11d_design.json).
DEFAULT_PARAM_NAMES = (
    "F_ESC10", "F_STAR10", "ALPHA_ESC", "ALPHA_STAR", "L_X", "NU_X_THRESH",
    "M_TURN", "t_STAR", "X_RAY_SPEC_INDEX", "OMm", "SIGMA_8",
)


def _scalar(v):
    return v.item() if isinstance(v, np.generic) else v


class Lightcone:
    """Lazy reader. Use as a context manager."""

    def __init__(self, path: str | Path, chunk_cache_bytes: int | None = None):
        """``chunk_cache_bytes`` enlarges HDF5's per-dataset chunk cache (default 1 MB).

        The cluster files store (9, 9, 293) gzip chunks, so one transverse slice
        touches a full 140x140x293 slab. A cache holding a few slabs lets
        neighbouring LOS reads reuse it instead of decompressing it again.
        """
        self.path = Path(path)
        self.chunk_cache_bytes = chunk_cache_bytes
        self._f: h5py.File | None = None

    def __enter__(self):
        kw = {}
        if self.chunk_cache_bytes:
            kw = {"rdcc_nbytes": int(self.chunk_cache_bytes), "rdcc_nslots": 1_000_003}
        self._f = h5py.File(self.path, "r", **kw)
        self._detect()
        return self

    def __exit__(self, *exc):
        if self._f is not None:
            self._f.close()
            self._f = None

    # ------------------------------------------------------------ schema
    def _detect(self):
        f = self._f
        if "lightcone" in f and isinstance(f["lightcone"], h5py.Group) and "density" in f["lightcone"]:
            self.schema = "raw_v2"
            self._fields = f["lightcone"]
            dist = np.asarray(f["lightcone/lightcone_distances"], dtype=np.float64)
            z = np.asarray(f["lightcone/lightcone_redshifts"], dtype=np.float64)
        elif "lightcones" in f and isinstance(f["lightcones"], h5py.Group):
            self.schema = "native"
            self._fields = f["lightcones"]
            dist = np.asarray(f["lightcone_distances"], dtype=np.float64)
            cosmo = f["InputParameters/cosmo_params"].attrs
            # py21cmfast's convention: Planck15 neutrinos/T_cmb, the file's Om0 and h.
            z = FlatLCDM(float(cosmo["OMm"]), float(cosmo["hlittle"])).z_at_distance(dist)
        else:
            raise ValueError(f"{self.path}: neither /lightcone (raw_v2) nor /lightcones (native)")

        shape = self._fields["density"].shape
        if len(shape) != 3 or shape[2] != dist.size or z.size != dist.size:
            raise ValueError(f"{self.path}: density {shape} does not match LOS axis {dist.size}")
        # Present every field in increasing-redshift order.
        self._flip = bool(z[0] > z[-1])
        self.distances = dist[::-1].copy() if self._flip else dist
        self.redshifts = z[::-1].copy() if self._flip else z
        if np.any(np.diff(self.redshifts) <= 0):
            raise ValueError(f"{self.path}: redshift axis is not monotonic")
        self.shape_hw = (int(shape[0]), int(shape[1]))
        self.n_los = int(shape[2])

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(k for k, v in self._fields.items() if isinstance(v, h5py.Dataset) and v.ndim == 3)

    @property
    def box_len(self) -> float:
        f = self._f
        if self.schema == "raw_v2":
            if "box_len_mpc" in f.attrs:
                return float(f.attrs["box_len_mpc"])
            for g in ("params", "params/fixed_simulation_options", "params/fixed_matter_options"):
                if g in f and "BOX_LEN" in f[g].attrs:
                    return float(f[g].attrs["BOX_LEN"])
            raise KeyError(f"{self.path}: no box_len_mpc / BOX_LEN attribute")
        return float(f["InputParameters/simulation_options"].attrs["BOX_LEN"])

    @property
    def cell_size(self) -> float:
        return self.box_len / self.shape_hw[0]

    # ------------------------------------------------------------ params
    def _param_groups(self):
        f = self._f
        names = (["params", "params/fixed_cosmo_params"] if self.schema == "raw_v2" else
                 ["InputParameters/astro_params", "InputParameters/cosmo_params",
                  "InputParameters/cosmo_tables"])
        return [f[n].attrs for n in names if n in f]

    def param(self, name: str) -> float:
        for attrs in self._param_groups():
            if name in attrs:
                return float(_scalar(attrs[name]))
        # py21cmfast normalizes by sigma_8 through cosmo_tables.ps_norm.
        if name == "SIGMA_8":
            for attrs in self._param_groups():
                if "ps_norm" in attrs:
                    return float(_scalar(attrs["ps_norm"]))
        raise KeyError(f"{self.path}: parameter {name!r} not found")

    def params(self, names=DEFAULT_PARAM_NAMES, missing: str = "raise") -> np.ndarray:
        out = []
        for n in names:
            try:
                out.append(self.param(n))
            except KeyError:
                if missing == "raise":
                    raise
                out.append(np.nan)
        return np.asarray(out, dtype=np.float64)

    # ------------------------------------------------------------ history
    def xhi_history(self, block: int = 256) -> np.ndarray:
        """Mean x_HI at every LOS index (increasing z), shape (n_los,).

        Uses the stored global history (``global_quantities/neutral_fraction``
        at ``node_redshifts``, raw_v2 only) interpolated to the lightcone
        redshifts; otherwise averages the neutral-fraction field slice by slice.
        The global history is a coeval-box mean, so it tracks the slice mean
        only up to cosmic variance across the transverse plane.
        """
        f = self._f
        if self.schema == "raw_v2" and "lightcone/global_quantities/neutral_fraction" in f \
                and "lightcone/node_redshifts" in f:
            zn = np.asarray(f["lightcone/node_redshifts"], dtype=np.float64)
            xn = np.asarray(f["lightcone/global_quantities/neutral_fraction"], dtype=np.float64)
            if zn.shape == xn.shape and zn.size > 1 and np.all(np.isfinite(xn)):
                order = np.argsort(zn)
                return np.interp(self.redshifts, zn[order], xn[order])
        return np.concatenate([self.read_range("neutral_fraction", a, min(a + block, self.n_los)).mean(axis=(1, 2))
                               for a in range(0, self.n_los, block)])

    # ------------------------------------------------------------ fields
    def read_range(self, field: str, lo: int, hi: int) -> np.ndarray:
        """Slices ``lo <= i < hi`` of ``field`` (increasing-z indices), as (n, H, W)."""
        if not 0 <= lo < hi <= self.n_los:
            raise IndexError(f"range [{lo}, {hi}) outside [0, {self.n_los})")
        ds = self._fields[field]
        if self._flip:
            a = ds[:, :, self.n_los - hi:self.n_los - lo][:, :, ::-1]
        else:
            a = ds[:, :, lo:hi]
        return np.ascontiguousarray(np.moveaxis(np.asarray(a, dtype=np.float32), 2, 0))

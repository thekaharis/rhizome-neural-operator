"""Toy excursion-set lightcones in the real on-disk schemas.

Not a physical simulation: it exists so the whole pipeline (cache, training,
sampling, metrics) can run end to end on a laptop, on data that has what the
real task has -- sharp binary bubbles set by a nonlocal, threshold-over-scales
rule on the density, and a T_b that mixes x_HI, density and a redshift-dependent
spin-temperature factor.

Recipe, per cone:
  * g: periodic Gaussian random field, P(k) ~ k^-2 exp(-(k R_s)^2), zero below
    the transverse fundamental 2 pi / (H cell).
  * delta(z) = exp(a g - a^2/2) - 1,  a = sigma_8-like amplitude x 7/(1+z).
  * nu_R = (g smoothed with a top-hat of radius R) / std, for R in 1,2,4,8 cells.
  * ionized where max_R nu_R > B(z) = (z - z_mid) / w  (z_mid from F_STAR10).
  * T_b = 27 x_HI (1+delta) sqrt((1+z)/10) (1 - A exp(-(z-z_abs)^2 / 2 w_abs^2)),
    A from L_X, so there is an absorption trough at high z.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from ..cosmology import FlatLCDM
from .lightcone import DEFAULT_PARAM_NAMES


def _tophat_k(kr):
    kr = np.where(kr == 0, 1e-12, kr)
    return 3.0 * (np.sin(kr) - kr * np.cos(kr)) / kr ** 3


def make_fields(rng, H, n_los, z, cell, f_star10, l_x, sigma_8):
    kx = 2 * np.pi * np.fft.fftfreq(H, d=cell)
    kz = 2 * np.pi * np.fft.rfftfreq(n_los, d=cell)
    k = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    # No power below the transverse fundamental: the cone is far longer along
    # the LOS than across, and those long LOS modes would otherwise make whole
    # slices uniformly neutral or ionized.
    k_min = 2 * np.pi / (H * cell)
    pk = np.where(k >= k_min, (k + 1e-6) ** -2.0 * np.exp(-(k * 1.5 * cell) ** 2), 0.0)
    white = np.fft.rfftn(rng.standard_normal((H, H, n_los)))
    gk = white * np.sqrt(pk)
    g = np.fft.irfftn(gk, s=(H, H, n_los), axes=(0, 1, 2))
    g /= g.std()

    amp = sigma_8 / 0.8 * 7.0 / (1.0 + z)  # growth ~ 1/(1+z), per LOS slice
    delta = np.exp(amp * g - 0.5 * amp ** 2) - 1.0

    nu_max = np.full(g.shape, -np.inf)
    for r_cells in (1, 2, 4, 8):
        gr = np.fft.irfftn(gk * _tophat_k(k * r_cells * cell), s=g.shape, axes=(0, 1, 2))
        nu_max = np.maximum(nu_max, gr / gr.std())
    z_mid = 9.0 + 1.5 * (f_star10 + 1.5)
    barrier = (z - z_mid) / 0.8
    xhi = (nu_max <= barrier).astype(np.float32)

    a_abs = 1.0 + 3.0 * (l_x - 38.0) / 4.0
    spin = 1.0 - a_abs * np.exp(-0.5 * ((z - 12.5) / 1.2) ** 2)
    tb = 27.0 * xhi * (1.0 + delta) * np.sqrt((1.0 + z) / 10.0) * spin
    return delta.astype(np.float32), xhi, tb.astype(np.float32)


def write_toy(path, seed, H=32, cell=2.0, z_lo=6.0, z_hi=14.0, schema="raw_v2", params=None):
    rng = np.random.default_rng(seed)
    cosmo = FlatLCDM(0.31, 0.68)
    d0, d1 = cosmo.comoving_distance([z_lo, z_hi])
    n_los = int((d1 - d0) // cell)
    dist = d0 + cell * np.arange(n_los)
    z = cosmo.z_at_distance(dist)
    if params is None:
        params = {
            "F_ESC10": rng.uniform(-3, 0), "F_STAR10": rng.uniform(-3, 0),
            "ALPHA_ESC": rng.uniform(-1, 1), "ALPHA_STAR": rng.uniform(-0.5, 1),
            "L_X": rng.uniform(38, 42), "NU_X_THRESH": rng.uniform(100, 1500),
            "M_TURN": rng.uniform(8, 10), "t_STAR": rng.uniform(0.01, 1),
            "X_RAY_SPEC_INDEX": rng.uniform(-1, 3), "OMm": 0.31,
            "SIGMA_8": rng.uniform(0.7, 0.9),
        }
    delta, xhi, tb = make_fields(rng, H, n_los, z[None, None, :], cell,
                                 params["F_STAR10"], params["L_X"], params["SIGMA_8"])
    vel = np.zeros_like(delta)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        if schema == "raw_v2":
            g = f.create_group("lightcone")
            for name, arr in (("density", delta), ("neutral_fraction", xhi),
                              ("brightness_temp", tb), ("los_velocity", vel)):
                g.create_dataset(name, data=arr, chunks=(H, H, 16))
            g.create_dataset("lightcone_distances", data=dist)
            g.create_dataset("lightcone_redshifts", data=z)
            p = f.create_group("params")
            for k in DEFAULT_PARAM_NAMES:
                p.attrs[k] = params[k]
            fc = p.create_group("fixed_cosmo_params")
            fc.attrs.update({"hlittle": 0.68, "OMb": 0.049, "POWER_INDEX": 0.965})
            p.create_group("fixed_matter_options")
            f.attrs["box_len_mpc"] = H * cell
            f.attrs["schema_version"] = "raw_lightcone_v2.0"
            f.attrs["toy"] = True
        elif schema == "native":
            g = f.create_group("lightcones")
            for name, arr in (("density", delta), ("neutral_fraction", xhi),
                              ("brightness_temp", tb), ("los_velocity", vel)):
                g.create_dataset(name, data=arr)
            f.create_dataset("lightcone_distances", data=dist)
            ip = f.create_group("InputParameters")
            ap = ip.create_group("astro_params")
            for k in DEFAULT_PARAM_NAMES:
                if k not in ("OMm", "SIGMA_8"):
                    ap.attrs[k] = params[k]
            ip.create_group("cosmo_params").attrs.update(
                {"OMm": params["OMm"], "hlittle": 0.68, "OMr": 8.6e-5})
            ip.create_group("cosmo_tables").attrs["ps_norm"] = params["SIGMA_8"]
            ip.create_group("simulation_options").attrs.update({"BOX_LEN": H * cell, "HII_DIM": H})
        else:
            raise ValueError(schema)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ebm21cm.data.toy", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--cell", type=float, default=2.0)
    ap.add_argument("--z-lo", type=float, default=6.0)
    ap.add_argument("--z-hi", type=float, default=14.0)
    ap.add_argument("--schema", choices=("raw_v2", "native"), default="raw_v2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    for i in range(a.n):
        p = write_toy(Path(a.out) / f"toy_sample{i:06d}.h5", a.seed * 100_003 + i, a.H, a.cell,
                      a.z_lo, a.z_hi, a.schema)
        print(p)


if __name__ == "__main__":
    main()

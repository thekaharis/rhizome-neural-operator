"""Flat LCDM comoving distance and its inverse, without astropy.

Used only to label native py21cmfast lightcones, which store comoving
distances but no redshift axis. Raw-v2 files carry their own redshifts and
never go through this module.

py21cmfast (4.x) builds its cosmology as astropy's Planck15 cloned with the
file's H0, Om0, Ob0 and Neff=3.044, i.e. T_cmb = 2.7255 K and neutrino masses
(0, 0, 0.06) eV. The defaults below reproduce that, including astropy's
neutrino density fit (Komatsu et al. 2011, eq. 26), so the radiation and
massive-neutrino terms match what generated the distances. Ignoring them
shifts z by ~1e-2 (radiation) to ~0.5 (massive neutrinos) at z ~ 30.
"""

from __future__ import annotations

import numpy as np

C_KM_S = 299_792.458
_A_B_C2 = 4 * 5.670374419e-5 / 2.99792458e10 ** 3  # 4 sigma_SB / c^3, g cm^-3 K^-4
_G_CGS = 6.67430e-8
_H100_S = 100.0 / 3.0856775814913673e19             # 100 km/s/Mpc in 1/s
_KB_EV = 8.617333262e-5


class FlatLCDM:
    """Flat LCDM with photons, (massive) neutrinos and a cosmological constant.

    ``omega_m`` excludes massive neutrinos, as in astropy's ``Om0``.
    """

    def __init__(self, omega_m: float, h: float, t_cmb: float = 2.7255, n_eff: float = 3.044,
                 m_nu=(0.0, 0.0, 0.06), z_max: float = 200.0, n_grid: int = 400_001):
        self.omega_m = float(omega_m)
        self.h = float(h)
        rho_crit = 3 * (self.h * _H100_S) ** 2 / (8 * np.pi * _G_CGS)
        self.omega_gamma = _A_B_C2 * t_cmb ** 4 / rho_crit if t_cmb > 0 else 0.0
        m_nu = np.asarray(m_nu, dtype=np.float64)
        self._n_massless = int((m_nu == 0).sum())
        self._neff_per_nu = n_eff / max(len(m_nu), 1)
        t_nu = (4.0 / 11.0) ** (1.0 / 3.0) * t_cmb
        self._nu_y = m_nu[m_nu > 0] / (_KB_EV * t_nu) if t_cmb > 0 else np.zeros(0)
        self.omega_nu = self.omega_gamma * self._nu_rel(0.0)
        self.omega_l = 1.0 - self.omega_m - self.omega_gamma - self.omega_nu

        z = np.linspace(0.0, z_max, n_grid)
        inv_e = 1.0 / self._e(z)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * np.diff(z))])
        self._z = z
        self._d = cum * C_KM_S / (100.0 * self.h)

    def _nu_rel(self, z):
        """rho_nu / rho_gamma, astropy FLRW.nu_relative_density."""
        z = np.asarray(z, dtype=np.float64)
        prefac = 0.22710731766  # 7/8 (4/11)^(4/3)
        if self._nu_y.size == 0:
            return prefac * self._neff_per_nu * self._n_massless * np.ones_like(z)
        y = self._nu_y[None, :] / (1.0 + z.reshape(-1, 1))
        massive = ((1.0 + (0.3173 * y) ** 1.83) ** (1 / 1.83)).sum(1)
        return (prefac * self._neff_per_nu * (massive + self._n_massless)).reshape(z.shape)

    def _e(self, z):
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        rad = self.omega_gamma * (1.0 + self._nu_rel(z))
        return np.sqrt(self.omega_m * zp1 ** 3 + rad * zp1 ** 4 + self.omega_l)

    def comoving_distance(self, z) -> np.ndarray:
        """Comoving distance in Mpc (not Mpc/h)."""
        return np.interp(np.asarray(z, dtype=np.float64), self._z, self._d)

    def z_at_distance(self, d) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64)
        if np.any(d < 0) or np.any(d > self._d[-1]):
            raise ValueError("distance outside the tabulated range")
        return np.interp(d, self._d, self._z)

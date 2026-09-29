"""Figures. Headless (Agg); every function writes one PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def examples(path, delta, xhi_true, xhi_samples, tb_true, tb_samples, z, n_show_samples=2):
    """Rows = slices. Columns: delta | truth | samples... | ensemble mean | ensemble std, per field."""
    n = len(z)
    m = min(n_show_samples, xhi_samples.shape[1])
    cols = 1 + 2 * (1 + m + (2 if xhi_samples.shape[1] > 1 else 0))
    fig, axes = plt.subplots(n, cols, figsize=(1.6 * cols, 1.7 * n + 0.4), squeeze=False)
    for i in range(n):
        tb_lim = max(float(np.percentile(np.abs(tb_true[i]), 99.5)), 1.0)  # mK; never zoom into sub-mK noise
        panels = [(delta[i], "delta", "cividis", None, None)]
        for name, t, s, cmap, vmin, vmax in (
                ("x_HI", xhi_true[i], xhi_samples[i], "Greys", 0, 1),
                ("T_b", tb_true[i], tb_samples[i], "RdBu_r", -tb_lim, tb_lim)):
            panels.append((t, f"{name} truth", cmap, vmin, vmax))
            panels += [(s[j], f"{name} sample {j + 1}", cmap, vmin, vmax) for j in range(m)]
            if s.shape[0] > 1:  # with one sample, mean == sample and std == 0: omit
                panels.append((s.mean(0), f"{name} ens. mean ({s.shape[0]})", cmap, vmin, vmax))
                panels.append((s.std(0), f"{name} ens. std", "magma", 0, None))
        for j, (img, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7)
        axes[i, 0].set_ylabel(f"z = {z[i]:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def spectra(path, k_centers, by_bin: dict):
    """by_bin[label][field] = {"ratio", "r", "ratio_ens_mean", "r_ens_mean"} arrays."""
    fields = ("x_HI", "T_b")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(by_bin), 1)))
    for c, (label, d) in zip(colors, by_bin.items()):
        for j, f in enumerate(fields):
            axes[0, j].plot(k_centers, d[f]["ratio"], color=c, label=f"{label} samples")
            axes[0, j].plot(k_centers, d[f]["ratio_ens_mean"], color=c, ls="--", lw=1)
            axes[1, j].plot(k_centers, d[f]["r"], color=c)
            axes[1, j].plot(k_centers, d[f]["r_ens_mean"], color=c, ls="--", lw=1)
    for j, f in enumerate(fields):
        axes[0, j].axhline(1, color="k", lw=0.6)
        axes[0, j].set_title(f"{f}: P_pred / P_true (solid: samples, dashed: ens. mean)", fontsize=8)
        axes[1, j].set_title(f"{f}: cross-correlation r(k)", fontsize=8)
        axes[1, j].set_xlabel("k [1/Mpc]")
        axes[0, j].set_ylim(0, 1.6)
        axes[1, j].set_ylim(-0.1, 1.05)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def bubble_sizes(path, sizes: dict):
    """sizes[label] = 1-D array of MFP sizes in Mpc."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    allv = np.concatenate([v for v in sizes.values() if len(v)]) if sizes else np.ones(1)
    bins = np.logspace(np.log10(max(allv.min(), 1e-2)), np.log10(allv.max() + 1e-9), 30)
    for label, v in sizes.items():
        if len(v):
            ax.hist(v, bins=bins, histtype="step", density=True, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("ionized region size (MFP) [Mpc]")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def energy_gaps(path, sigmas, gaps: dict):
    """gaps[label] = (rows, n_sigma) of E(candidate) - E(truth) per pixel."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, g in gaps.items():
        med = np.nanmedian(g, 0)
        lo, hi = np.nanpercentile(g, [16, 84], axis=0)
        ax.plot(sigmas, med, marker="o", label=label)
        ax.fill_between(sigmas, lo, hi, alpha=0.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("sigma (probe noise level)")
    ax.set_ylabel("[E(candidate) - E(truth)] / pixel")
    ax.set_title("> 0: the model finds the truth more plausible", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)

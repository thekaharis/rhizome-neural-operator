"""Rhizome interaction maps of the multi-field slice-wise model on real lightcone slices.

    python -m ebm21cm.mechanism_mf --run runs/rhizome_mf2d_w64_structured_all \\
        --out runs/eval3d/mechanism --targets 0.2 0.4 0.6 0.8

The multi-field counterpart of ``scripts/make_rhizome_mechanism.py``. For test-split
slices of the multi-field cache whose true mean x_HI is closest to each ``--targets``
value (one slice per cone), and one receiver cell i on an ionization front (the
front cell closest to the slice centre), it plots:

* the input density band [0, 1], the true and the predicted x_HI;
* the learned kernel norm ||kappa(r_i - r_j)||_F (impulse response of the trained
  SpectralConv2d, so the model's own weights);
* the channel-RMS receiver and source gates A, B at the last update;
* for every update t, how the gates re-weight the incoming connections of cell i
  relative to the ungated kernel,

      M_t(j) = ||diag(A_i) kappa(r_i - r_j) diag(B_j)||_F / ||kappa(r_i - r_j)||_F,

  shown as M_t / mean_j M_t, with the mean gain <M_t>_j printed.

For a multi-pathway model (MultiPathIntegral2d, e.g. the spectral + disc hybrid) the
figure has one row per pathway instead: that pathway's kernel norm and its M_t for
every update, computed from its own gates and (depthwise) kernel. The top row then
shows the input, truth, prediction and each pathway's gates at the last update.
For a disc kernel M_t is defined only inside the disc (blank outside).
A second figure (``*_messages.png``) shows what each pathway actually transmits at
every update: the channel-RMS message ||m_r|| = ||A_r * kappa_r(B_r * V_r)||, and its
contribution to the update, ||W_r m_r||, with W_r the proposal weights acting on that
pathway's channels (pre-tanh; bias, state and forcing terms excluded). Both use one
colour scale for all pathways and updates.

The M_t are gains on the transmitted value V_j, not a Jacobian of the output and not
calibrated physical couplings. Contours: true x_HI = 0.5. Panels are cropped to +-``--crop``
cells around cell i (periodic; 0 = whole slice). One PNG + JSON per slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .data.lightcone import LC_CELL_MPC, node_weight
from .data.mf_slices import cache_omega_m
from .model.rhizome import MultiPathIntegral2d, RhizomeOperator2d
from .train_rhizome_mf2d import node_output, node_scalars

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def kernel_impulse(kernel, width, h, w):
    """K[o, i, dy, dx]: output channel o at displacement (dy, dx) from a unit source in channel i."""
    x = torch.zeros(width, width, h, w, dtype=torch.float64)
    x[torch.arange(width), torch.arange(width), 0, 0] = 1.0
    with torch.no_grad():
        return kernel.double()(x).permute(1, 0, 2, 3).numpy()


def depthwise_impulse(kernel, h, w):
    """K[c, dy, dx]: channel c at displacement (dy, dx) from a unit source in the same channel (no channel mixing)."""
    x = torch.zeros(1, kernel.channels, h, w, dtype=torch.float64)
    x[0, :, 0, 0] = 1.0
    with torch.no_grad():
        return kernel.double()(x)[0].numpy()


def pathway_kernels(inter, h, w):
    """[(label, K[c, dy, dx]), ...] per pathway of a depthwise MultiPathIntegral2d, in channel order."""
    c = inter.width
    out = []
    if inter.kernel is not None:
        if inter.kernel.groups != inter.kernel.channels:
            raise ValueError("pathway maps need depthwise spectral kernels (kernel_groups = width)")
        K = depthwise_impulse(inter.kernel, h, w)
        out += [(f"spectral {r + 1}", K[r * c:(r + 1) * c]) for r in range(K.shape[0] // c)]
    if inter.disc is not None:
        K = depthwise_impulse(inter.disc, h, w)
        out += [(f"disc R={inter.disc.radius:g} {r + 1}" if K.shape[0] > c else f"disc R={inter.disc.radius:g}",
                 K[r * c:(r + 1) * c]) for r in range(K.shape[0] // c)]
    return out


def pick_receiver(truth):
    """Front cell (neutral next to ionized) closest to the slice centre."""
    xh = truth > 0.5
    front = xh & (~np.roll(xh, 1, 0) | ~np.roll(xh, -1, 0) | ~np.roll(xh, 1, 1) | ~np.roll(xh, -1, 1))
    ys, xs = np.nonzero(front)
    h, w = truth.shape
    k = np.argmin((ys - h / 2) ** 2 + (xs - w / 2) ** 2)
    return int(ys[k]), int(xs[k])


def pick_rows(h, split, targets):
    """Cache rows of ``split`` whose true mean x_HI is closest to each target, from distinct cones."""
    rows = np.flatnonzero(h["split"][:] == SPLIT_CODE[split])
    means = np.array([float(h["xhi"][r].astype(np.float32).mean()) for r in rows])
    cones = h["cone_id"][:][rows]
    picked, used = [], set()
    for t in targets:
        for k in np.argsort(np.abs(means - t)):
            if int(cones[k]) not in used:
                picked.append(int(rows[k]))
                used.add(int(cones[k]))
                break
    return picked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=Path("runs/rhizome_mf2d_w64_structured_all"))
    ap.add_argument("--cache", default=None, help="slice cache (default: the run's training cache)")
    ap.add_argument("--split", default="test", choices=list(SPLIT_CODE))
    ap.add_argument("--targets", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--rows", type=int, nargs="*", default=None, help="explicit cache rows instead of --targets")
    ap.add_argument("--band", default="[0, 1]", help="density band shown in panel (a)")
    ap.add_argument("--crop", type=int, default=32, help="half-width of the shown window in cells (0: whole slice)")
    ap.add_argument("--out", type=Path, default=Path("runs/eval3d/mechanism"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    meta = json.loads((a.run / "metadata.json").read_text())
    best = torch.load(a.run / "best.pt", map_location="cpu", weights_only=False)
    model = RhizomeOperator2d(**best["model_config"]).eval()
    model.load_state_dict(best["model"])
    model = model.double()
    keep = best.get("scalar_keep")
    node = best.get("node_phase")
    inter = model.cells[0].interaction
    width = model.width
    bands = [list(b) for b in json.loads(meta["cache_attrs"]["bands"])]
    band = bands.index(json.loads(a.band))
    a.out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.cache or meta["cache"], "r") as h:
        rows = a.rows or pick_rows(h, a.split, a.targets)
        data = [{k: h[k][r] for k in ("cond", "scalars", "xhi", "z", "cone_id", "los_index")} for r in rows]
    K = None
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.linewidth": 0.5, "mathtext.fontset": "cm"})
    for row, d in zip(rows, data):
        cond = torch.from_numpy(d["cond"].astype(np.float64))[None]
        scal = torch.from_numpy(d["scalars"].astype(np.float64))[None]
        if keep is not None:
            scal = scal[:, keep]
        w = None
        if node is not None:
            omm = cache_omega_m(d["scalars"][None], meta["cache_attrs"])
            w = torch.from_numpy(node_weight(np.atleast_1d(d["z"]), omm, node["w_mode"],
                                             los_distance=np.atleast_1d(d["los_index"]) * LC_CELL_MPC))
        scal = node_scalars(scal, w, node)
        truth = d["xhi"].astype(np.float64)
        with torch.no_grad():
            out, states = model(cond, scal, return_states=True)
            out = node_output(out, w, node)
            forcing = model.lift(cond) + model.scalar_embed(scal)[:, :, None, None]
            factors = [inter.factors(s, forcing) for s in states[:-1]]
        pred = out[0, 0].sigmoid().numpy()
        h_, w_ = truth.shape
        if isinstance(inter, MultiPathIntegral2d):
            if K is None:
                K = pathway_kernels(inter, h_, w_)
            pathway_figure(a, plt, TwoSlopeNorm, d, row, best, bands, band, truth, pred, factors, K, width)
            with torch.no_grad():
                messages = [inter(s, forcing)[0].numpy() for s in states[:-1]]
            weights = model.cells[0].proposal.weight[:, width:width + inter.pathways * width, 0, 0].detach().numpy()
            message_figure(a, plt, d, bands, band, truth, pred, messages, weights, [k[0] for k in K], width)
            continue
        if K is None:
            K = kernel_impulse(inter.kernel, width, h_, w_)
        iy, ix = pick_receiver(truth)
        yy, xx = np.meshgrid(np.arange(h_), np.arange(w_), indexing="ij")
        Kd = K[:, :, (iy - yy) % h_, (ix - xx) % w_]           # (C_out, C_in, H, W): kappa(r_i - r_j)
        kernel_norm = np.sqrt((Kd ** 2).sum((0, 1)))
        mod, rms_a, rms_b, eff = [], [], [], []
        for source, receiver, _ in factors:
            B, Afield = source[0].numpy(), receiver[0].numpy()
            A = Afield[:, iy, ix]
            e = np.sqrt(np.einsum("o,oiyx,iyx->yx", A ** 2, Kd ** 2, B ** 2))
            eff.append(e)
            mod.append(e / kernel_norm)
            rms_a.append(np.sqrt((Afield ** 2).mean(0)))
            rms_b.append(np.sqrt((B ** 2).mean(0)))
        gains = [float(m.mean()) for m in mod]
        rel = [m / m.mean() for m in mod]
        T = len(factors)
        c = a.crop

        def view(img):
            """Window of +-crop cells centred on cell i (periodic), or the whole slice."""
            if not c:
                return img
            return np.roll(img, (c - iy, c - ix), axis=(0, 1))[:2 * c + 1, :2 * c + 1]
        cy, cx = (c, c) if c else (iy, ix)
        truth_v = view(truth)
        shown = [view(r) for r in rel]
        lo, hi = min(r.min() for r in shown), max(r.max() for r in shown)

        fig, axes = plt.subplots(2, max(6, T), figsize=(1.75 * max(6, T) + 0.6, 3.6), constrained_layout=True)

        def show(ax, img, cmap, title, norm=None, cbar=True, label=None, ring="#009E73", vmin=None, vmax=None):
            im = ax.imshow(view(img), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, origin="lower",
                           interpolation="nearest")
            ax.contour(truth_v, levels=[0.5], colors="w" if cmap in ("gray", "viridis") else "k", linewidths=0.45,
                       alpha=0.75)
            ax.plot(cx, cy, marker="o", ms=5, mfc="none", mec=ring, mew=1.3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, pad=3)
            if cbar:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=6, width=0.4, length=2)
                cb.outline.set_linewidth(0.4)
                if label:
                    cb.set_label(label, fontsize=6.5)

        top = axes[0]
        show(top[0], d["cond"][band].astype(np.float64), "gray", rf"(a) density $\delta$, band [{bands[band][0]},{bands[band][1]}]",
         label="normalized")
        show(top[1], truth, "viridis", r"(b) true $x_{\rm HI}$", vmin=0, vmax=1)
        show(top[2], pred, "viridis", r"(c) predicted $x_{\rm HI}$", vmin=0, vmax=1)
        show(top[3], kernel_norm / kernel_norm.max(), "Blues", r"(d) kernel $\|\kappa_\theta(\mathbf{r}_i-\mathbf{r}_j)\|_F$",
             label="rel. to max")
        show(top[4], rms_a[-1], "Greens", rf"(e) receiver gate $\|\mathbf{{A}}\|_{{\rm rms}}$, $t$={T}",
             label=r"$\in(0,2)$", ring="k")
        show(top[5], rms_b[-1], "Oranges", rf"(f) source gate $\|\mathbf{{B}}\|_{{\rm rms}}$, $t$={T}",
             label=r"$\in(0,2)$")
        for ax in top[6:]:
            ax.axis("off")
        rnorm = TwoSlopeNorm(1.0, min(lo, 0.99), max(hi, 1.01))
        letters = "ghijklmnop"
        for t, ax in enumerate(axes[1]):
            if t >= T:
                ax.axis("off")
                continue
            show(ax, rel[t], "PuOr_r", rf"({letters[t]}) $M_t$, update $t$={t + 1}", norm=rnorm, cbar=t == T - 1,
                 label=r"$M_t(j)\,/\,\langle M_t\rangle_j$")
            ax.text(0.03, 0.04, rf"gain $\langle M_t\rangle$ = {gains[t]:.2f}", transform=ax.transAxes, fontsize=6.5,
                    bbox=dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.8))
        z, cone = float(d["z"]), int(d["cone_id"])
        fig.suptitle(rf"Test cone {cone}, $z$ = {z:.2f}, true $\bar x_{{\rm HI}}$ = {truth.mean():.2f}; "
                     rf"receiver cell $i$ at (y, x) = ({iy}, {ix}); "
                     + (f"window ±{c} cells = ±{c * 200 / 140:.0f} Mpc" if c else "whole 200 Mpc slice"), fontsize=8.5)
        stem = a.out / f"mechanism_cone{cone}_z{z:.2f}".replace(".", "p")
        fig.savefig(stem.with_suffix(".png"), dpi=220)
        plt.close(fig)
        info = {"run": str(a.run), "checkpoint_step": best.get("step"), "cache_row": row, "cone_id": cone,
                "redshift": z, "true_mean_xhi": float(truth.mean()), "receiver_yx": [iy, ix],
                "mean_gate_gain_per_update": gains,
                "reweighting_range_per_update": [[float(r.min()), float(r.max())] for r in rel],
                "receiver_gate_rms_at_i": [float(r[iy, ix]) for r in rms_a],
                "effective_shape_change_first_to_last": float(np.linalg.norm(
                    eff[-1] / np.linalg.norm(eff[-1]) - eff[0] / np.linalg.norm(eff[0]))),
                "prediction_rmse": float(np.sqrt(((pred - truth) ** 2).mean())),
                # channel-RMS gates at the last update, averaged over ionized (x_HI < 0.5) and neutral cells
                "gate_rms_last_update": {f"{g}_{phase}": float(f[sel].mean())
                                         for g, f in (("A", rms_a[-1]), ("B", rms_b[-1]))
                                         for phase, sel in (("ionized", truth < 0.5), ("neutral", truth >= 0.5))}}
        stem.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        print(stem.with_suffix(".png"), json.dumps({k: info[k] for k in ("redshift", "true_mean_xhi",
                                                                          "mean_gate_gain_per_update")}), flush=True)


def pathway_figure(a, plt, TwoSlopeNorm, d, row, best, bands, band, truth, pred, factors, kernels, width):
    """One row per pathway: kernel norm and gate re-weighting M_t for every update (see module docstring)."""
    h_, w_ = truth.shape
    iy, ix = pick_receiver(truth)
    yy, xx = np.meshgrid(np.arange(h_), np.arange(w_), indexing="ij")
    T, P = len(factors), len(kernels)
    paths = []
    for p, (label, Kc) in enumerate(kernels):
        Kd = Kc[:, (iy - yy) % h_, (ix - xx) % w_]           # (C, H, W): kappa_c(r_i - r_j)
        norm = np.sqrt((Kd ** 2).sum(0))
        support = norm > 1e-9 * norm.max()                     # a disc kernel is exactly zero outside its radius
        sl = slice(p * width, (p + 1) * width)
        rel, gains, rms_a, rms_b = [], [], [], []
        for source, receiver, _ in factors:
            B, Afield = source[0, sl].numpy(), receiver[0, sl].numpy()
            m = np.sqrt(np.einsum("c,cyx,cyx->yx", Afield[:, iy, ix] ** 2, Kd ** 2, B ** 2)) / np.where(support, norm, 1)
            m = np.where(support, m, np.nan)
            gains.append(float(np.nanmean(m)))
            rel.append(m / gains[-1])
            rms_a.append(np.sqrt((Afield ** 2).mean(0)))
            rms_b.append(np.sqrt((B ** 2).mean(0)))
        paths.append({"label": label, "norm": np.where(support, norm, np.nan), "rel": rel, "gains": gains,
                      "rms_a": rms_a, "rms_b": rms_b})
    c = a.crop

    def view(img):
        if not c:
            return img
        return np.roll(img, (c - iy, c - ix), axis=(0, 1))[:2 * c + 1, :2 * c + 1]
    cy, cx = (c, c) if c else (iy, ix)
    truth_v = view(truth)
    ncol = max(T + 1, 3 + 2 * P)
    fig, axes = plt.subplots(1 + P, ncol, figsize=(1.75 * ncol + 0.6, 1.8 * (1 + P)), constrained_layout=True)

    def show(ax, img, cmap, title, norm=None, cbar=True, label=None, ring="#009E73", vmin=None, vmax=None):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#d9d9d9")
        im = ax.imshow(view(img), cmap=cm, norm=norm, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        ax.contour(truth_v, levels=[0.5], colors="w" if cmap in ("gray", "viridis") else "k", linewidths=0.45,
                   alpha=0.75)
        ax.plot(cx, cy, marker="o", ms=5, mfc="none", mec=ring, mew=1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=3)
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6, width=0.4, length=2)
            cb.outline.set_linewidth(0.4)
            if label:
                cb.set_label(label, fontsize=6.5)

    top = axes[0]
    show(top[0], d["cond"][band].astype(np.float64), "gray", rf"density $\delta$, band [{bands[band][0]},{bands[band][1]}]",
         label="normalized")
    show(top[1], truth, "viridis", r"true $x_{\rm HI}$", vmin=0, vmax=1)
    show(top[2], pred, "viridis", r"predicted $x_{\rm HI}$", vmin=0, vmax=1)
    for p, path in enumerate(paths):
        show(top[3 + 2 * p], path["rms_a"][-1], "Greens", rf"{path['label']}: $\|\mathbf{{A}}\|_{{\rm rms}}$, $t$={T}",
             label=r"$\in(0,2)$", ring="k")
        show(top[4 + 2 * p], path["rms_b"][-1], "Oranges", rf"{path['label']}: $\|\mathbf{{B}}\|_{{\rm rms}}$, $t$={T}",
             label=r"$\in(0,2)$")
    for ax in top[3 + 2 * P:]:
        ax.axis("off")
    for p, path in enumerate(paths):
        ax_row = axes[1 + p]
        show(ax_row[0], path["norm"] / np.nanmax(path["norm"]), "Blues",
             rf"{path['label']}: $\|\kappa(\mathbf{{r}}_i-\mathbf{{r}}_j)\|$", label="rel. to max")
        shown = [view(r) for r in path["rel"]]
        lo, hi = min(np.nanmin(r) for r in shown), max(np.nanmax(r) for r in shown)
        rnorm = TwoSlopeNorm(1.0, min(lo, 0.99), max(hi, 1.01))
        for t in range(T):
            ax = ax_row[1 + t]
            show(ax, path["rel"][t], "PuOr_r", rf"{path['label']}: $M_t$, $t$={t + 1}", norm=rnorm, cbar=t == T - 1,
                 label=r"$M_t(j)\,/\,\langle M_t\rangle_j$")
            ax.text(0.03, 0.04, rf"gain {path['gains'][t]:.2f}", transform=ax.transAxes, fontsize=6.5,
                    bbox=dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.8))
        for ax in ax_row[1 + T:]:
            ax.axis("off")
    z, cone = float(d["z"]), int(d["cone_id"])
    fig.suptitle(rf"{a.run.name}: test cone {cone}, $z$ = {z:.2f}, true $\bar x_{{\rm HI}}$ = {truth.mean():.2f}; "
                 rf"receiver cell $i$ at (y, x) = ({iy}, {ix}); "
                 + (f"window ±{c} cells = ±{c * 200 / 140:.0f} Mpc" if c else "whole 200 Mpc slice"), fontsize=8.5)
    stem = a.out / f"mechanism_cone{cone}_z{z:.2f}".replace(".", "p")
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)
    info = {"run": str(a.run), "checkpoint_step": best.get("step"), "cache_row": row, "cone_id": cone, "redshift": z,
            "true_mean_xhi": float(truth.mean()), "receiver_yx": [iy, ix],
            "prediction_rmse": float(np.sqrt(((pred - truth) ** 2).mean())),
            "pathways": {path["label"]: {
                "mean_gate_gain_per_update": path["gains"],
                "reweighting_range_per_update": [[float(np.nanmin(r)), float(np.nanmax(r))] for r in path["rel"]],
                "receiver_gate_rms_at_i": [float(r[iy, ix]) for r in path["rms_a"]],
                "gate_rms_last_update": {f"{g}_{phase}": float(f[sel].mean())
                                         for g, f in (("A", path["rms_a"][-1]), ("B", path["rms_b"][-1]))
                                         for phase, sel in (("ionized", truth < 0.5), ("neutral", truth >= 0.5))}}
                for path in paths}}
    stem.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(stem.with_suffix(".png"), json.dumps({k: v["mean_gate_gain_per_update"] for k, v in info["pathways"].items()}),
          flush=True)


def message_figure(a, plt, d, bands, band, truth, pred, messages, weights, labels, width):
    """Per pathway and update: channel-RMS message ||m_r|| and its proposal contribution ||W_r m_r||."""
    T, P = len(messages), len(labels)
    iy, ix = pick_receiver(truth)
    size = [[np.sqrt((m[p * width:(p + 1) * width] ** 2).mean(0)) for m in messages] for p in range(P)]
    contrib = [[np.sqrt((np.einsum("oc,cyx->oyx", weights[:, p * width:(p + 1) * width],
                                   m[p * width:(p + 1) * width]) ** 2).mean(0)) for m in messages] for p in range(P)]
    c = a.crop

    def view(img):
        if not c:
            return img
        return np.roll(img, (c - iy, c - ix), axis=(0, 1))[:2 * c + 1, :2 * c + 1]
    cy, cx = (c, c) if c else (iy, ix)
    truth_v = view(truth)
    ncol = max(T, 6)
    fig, axes = plt.subplots(1 + 2 * P, ncol, figsize=(1.75 * ncol + 0.6, 1.8 * (1 + 2 * P)), constrained_layout=True)

    def show(ax, img, cmap, title, cbar=True, label=None, vmin=None, vmax=None):
        im = ax.imshow(view(img), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        ax.contour(truth_v, levels=[0.5], colors="w", linewidths=0.45, alpha=0.75)
        ax.plot(cx, cy, marker="o", ms=5, mfc="none", mec="#009E73", mew=1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=3)
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6, width=0.4, length=2)
            cb.outline.set_linewidth(0.4)
            if label:
                cb.set_label(label, fontsize=6.5)

    top = axes[0]
    show(top[0], d["cond"][band].astype(np.float64), "gray", rf"density $\delta$, band [{bands[band][0]},{bands[band][1]}]",
         label="normalized")
    show(top[1], truth, "viridis", r"true $x_{\rm HI}$", vmin=0, vmax=1)
    show(top[2], pred, "viridis", r"predicted $x_{\rm HI}$", vmin=0, vmax=1)
    steps = np.arange(1, T + 1)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for ax, field, title in ((top[3], size, r"slice mean $\|m_r\|$"), (top[4], contrib, r"slice mean $\|W_r m_r\|$")):
        for p in range(P):
            ax.plot(steps, [f.mean() for f in field[p]], "o-", ms=3, lw=1, color=colors[p % 4], label=labels[p])
        ax.set_xticks(steps)
        ax.set_xlabel("update $t$", fontsize=7)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6)
        ax.set_title(title, pad=3)
        ax.legend(fontsize=6, frameon=False)
    for ax in top[5:]:
        ax.axis("off")
    for q, (field, name, cmap) in enumerate(((size, r"\|m_r\|", "magma"), (contrib, r"\|W_r m_r\|", "cividis"))):
        vmax = float(np.percentile(np.concatenate([view(f).ravel() for fp in field for f in fp]), 99.5))
        for p in range(P):
            ax_row = axes[1 + 2 * q + p]
            for t in range(ncol):
                ax = ax_row[t]
                if t >= T:
                    ax.axis("off")
                    continue
                show(ax, field[p][t], cmap, rf"{labels[p]}: ${name}$, $t$={t + 1}", cbar=t == T - 1,
                     label=f"channel RMS (shared scale)", vmin=0, vmax=vmax)
                ax.text(0.03, 0.04, f"mean {field[p][t].mean():.3f}", transform=ax.transAxes, fontsize=6.5,
                        bbox=dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.8))
    z, cone = float(d["z"]), int(d["cone_id"])
    fig.suptitle(rf"{a.run.name}: test cone {cone}, $z$ = {z:.2f}, true $\bar x_{{\rm HI}}$ = {truth.mean():.2f}" "\n"
                 r"message $m_r=A_r\,\kappa_r(B_r V_r)$ and its update contribution $W_r m_r$ (pre-tanh); "
                 + (f"window ±{c} cells around cell $i$; " if c else "whole 200 Mpc slice; ")
                 + "panel means and line plots over the whole slice", fontsize=8.5)
    stem = a.out / f"mechanism_cone{cone}_z{z:.2f}".replace(".", "p")
    fig.savefig(stem.parent / (stem.name + "_messages.png"), dpi=220)
    plt.close(fig)
    info = {labels[p]: {key: {"slice_mean": [float(f.mean()) for f in field[p]],
                              "ionized_mean": [float(f[truth < 0.5].mean()) for f in field[p]],
                              "neutral_mean": [float(f[truth >= 0.5].mean()) for f in field[p]]}
                        for key, field in (("message_rms", size), ("proposal_contribution_rms", contrib))}
            for p in range(P)}
    (stem.parent / (stem.name + "_messages.json")).write_text(json.dumps(info, indent=2) + "\n")
    print(stem.parent / (stem.name + "_messages.png"),
          json.dumps({k: [round(x, 3) for x in v["proposal_contribution_rms"]["slice_mean"]] for k, v in info.items()}),
          flush=True)


if __name__ == "__main__":
    main()

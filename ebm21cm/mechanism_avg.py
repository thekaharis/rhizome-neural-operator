"""Gate re-weighting M_t averaged over many receiver cells and slices, as radial profiles.

    python -m ebm21cm.mechanism_avg --run plain=runs/rhizome_mf2d_w64_structured_all \\
        --run "both power terms=runs/rhizome_mf2d_w64_structured_all_pw0.03_px0.03" --out runs/eval3d/mechanism_avg

Same quantity as ``ebm21cm.mechanism_mf``, for every receiver cell i of a sample:

    M_t(j) = ||diag(A_i) kappa(r_i - r_j) diag(B_j)||_F / ||kappa(r_i - r_j)||_F,

reduced to a radial profile in |r_i - r_j| (periodic distance, 1-cell annuli) of the
re-weighting M_t(j) / <M_t>_j, plus the mean gain <M_t>_j and the receiver gate
||A_i||_rms per update. Receivers per slice (true x_HI, 0.5 threshold): up to
``--front`` front cells (neutral next to ionized) and ``--interior`` cells each deep
inside neutral and inside ionized regions (Euclidean distance to the other phase
> ``--depth`` cells). Slices: ``--slices`` test slices with true mean x_HI in
[0.1, 0.9], one per cone, chosen once and used for every model.

Writes ``profiles.npz`` (per receiver) and figures: profiles per update and receiver
class, the last-update profile by reionization stage, and gains / receiver gates per
update. As for the single-slice maps, M_t is a gain on the transmitted value, not a
Jacobian of the output.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .mechanism_mf import SPLIT_CODE, kernel_impulse
from .model.rhizome import RhizomeOperator2d

CELL_MPC = 200.0 / 140
CLASSES = ("front", "neutral interior", "ionized interior")
STAGES = ((0.1, 0.4), (0.4, 0.6), (0.6, 0.9))


def pick_slices(h, split, n, seed):
    """Up to ``n`` (row, x̄) pairs: one active slice per cone, cones chosen at random."""
    rows = np.flatnonzero(h["split"][:] == SPLIT_CODE[split])
    cones = h["cone_id"][:][rows]
    rng = np.random.default_rng(seed)
    picked = []
    for cone in rng.permutation(np.unique(cones)):
        cand = rows[cones == cone]
        means = np.array([float(h["xhi"][r].astype(np.float32).mean()) for r in cand])
        ok = np.flatnonzero((means >= 0.1) & (means <= 0.9))
        if len(ok):
            k = rng.choice(ok)
            picked.append((int(cand[k]), float(means[k])))
        if len(picked) == n:
            break
    return picked


def pick_receivers(truth, n_front, n_interior, depth, rng):
    from scipy.ndimage import distance_transform_edt

    neutral = truth >= 0.5
    front = neutral & (~np.roll(neutral, 1, 0) | ~np.roll(neutral, -1, 0) | ~np.roll(neutral, 1, 1)
                       | ~np.roll(neutral, -1, 1))
    # Distance to the other phase on a 3x3 periodic tiling, cropped back to the slice.
    h, w = truth.shape
    d_neutral = distance_transform_edt(np.tile(neutral, (3, 3)))[h:2 * h, w:2 * w]     # inside neutral
    d_ionized = distance_transform_edt(np.tile(~neutral, (3, 3)))[h:2 * h, w:2 * w]    # inside ionized
    out = []
    for cls, mask, n in (("front", front, n_front), ("neutral interior", neutral & (d_neutral > depth), n_interior),
                         ("ionized interior", ~neutral & (d_ionized > depth), n_interior)):
        ys, xs = np.nonzero(mask)
        if len(ys):
            k = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
            out += [(int(ys[q]), int(xs[q]), CLASSES.index(cls)) for q in k]
    return out


def analyse(run, slices, cache, args, device):
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if best.get("node_phase"):
        raise SystemExit(f"{run}: node-phase models are not supported here (see mechanism_mf)")
    model = RhizomeOperator2d(**best["model_config"]).eval()
    model.load_state_dict(best["model"])
    model = model.to(device)
    keep = best.get("scalar_keep")
    inter = model.cells[0].interaction
    width = model.width
    with h5py.File(cache, "r") as h:
        H, W = h["xhi"].shape[1:]
    # kernel_impulse converts the module it is given to float64, so use a copy.
    K = torch.from_numpy(kernel_impulse(copy.deepcopy(inter.kernel).cpu(), width, H, W)).float()
    knorm = K.square().sum((0, 1)).sqrt().to(device)                      # ||kappa(d)||_F, d = r_i - r_j
    K2 = K.square().reshape(width, -1).to(device)                          # (C_out, C_in*H*W)
    del K
    yy, xx = torch.arange(H, device=device), torch.arange(W, device=device)
    n_bins = int(np.ceil(np.hypot(H / 2, W / 2))) + 1
    rng = np.random.default_rng(args.seed)
    rec = {k: [] for k in ("slice", "cls", "xbar", "y", "x", "rel", "raw", "gain", "a_rms")}
    with h5py.File(cache, "r") as h:
        for s_idx, (row, xbar) in enumerate(slices):
            truth = h["xhi"][row].astype(np.float64)
            cond = torch.from_numpy(h["cond"][row].astype(np.float32))[None].to(device)
            scal = torch.from_numpy(h["scalars"][row].astype(np.float32))[None].to(device)
            if keep is not None:
                scal = scal[:, keep]
            receivers = pick_receivers(truth, args.front, args.interior, args.depth, rng)
            ry = torch.tensor([r[0] for r in receivers], device=device)
            rx = torch.tensor([r[1] for r in receivers], device=device)
            R = len(receivers)
            gy = (ry[:, None] - yy[None]) % H                                  # (R, H): displacement rows
            gx = (rx[:, None] - xx[None]) % W
            dy = torch.minimum((yy[None] - ry[:, None]).abs(), H - (yy[None] - ry[:, None]).abs())
            dx = torch.minimum((xx[None] - rx[:, None]).abs(), W - (xx[None] - rx[:, None]).abs())
            dist = torch.sqrt(dy[:, :, None].float() ** 2 + dx[:, None, :].float() ** 2)   # (R, H, W)
            bins = dist.floor().long().clamp(max=n_bins - 1)
            counts = torch.zeros(R, n_bins, device=device).scatter_add_(1, bins.flatten(1),
                                                                         torch.ones_like(dist).flatten(1))
            kn = knorm[gy[:, :, None], gx[:, None, :]]                        # (R, H, W)
            with torch.no_grad():
                _, states = model(cond, scal, return_states=True)
                forcing = model.lift(cond) + model.scalar_embed(scal)[:, :, None, None]
                rel_t, raw_t, gain_t, arms_t = [], [], [], []
                for state in states[:-1]:
                    source, receiver, _ = inter.factors(state, forcing)
                    B2 = source[0].square()                                   # (C, H, W)
                    A = receiver[0][:, ry, rx].T                              # (R, C)
                    Wmap = (A.square() @ K2).view(R, width, H, W)             # sum_o A_io^2 K_oc(d)^2
                    e2 = torch.zeros(R, H, W, device=device)
                    for c0 in range(0, width, 16):                            # gather in channel chunks
                        c = torch.arange(c0, min(c0 + 16, width), device=device)
                        Wd = Wmap[torch.arange(R, device=device)[:, None, None, None], c[None, :, None, None],
                                  gy[:, None, :, None], gx[:, None, None, :]]
                        e2 += (Wd * B2[c][None]).sum(1)
                    M = e2.sqrt() / kn
                    gain = M.mean((1, 2))
                    rel = M / gain[:, None, None]
                    prof = lambda f: torch.zeros(R, n_bins, device=device).scatter_add_(
                        1, bins.flatten(1), f.flatten(1)) / counts.clamp(min=1)
                    rel_t.append(prof(rel))
                    raw_t.append(prof(M))
                    gain_t.append(gain)
                    arms_t.append(A.square().mean(1).sqrt())
            rec["rel"].append(torch.stack(rel_t, 1).cpu().numpy())             # (R, T, n_bins)
            rec["raw"].append(torch.stack(raw_t, 1).cpu().numpy())
            rec["gain"].append(torch.stack(gain_t, 1).cpu().numpy())           # (R, T)
            rec["a_rms"].append(torch.stack(arms_t, 1).cpu().numpy())
            rec["slice"].append(np.full(R, s_idx))
            rec["cls"].append(np.array([r[2] for r in receivers]))
            rec["xbar"].append(np.full(R, xbar))
            rec["y"].append(ry.cpu().numpy())
            rec["x"].append(rx.cpu().numpy())
            if (s_idx + 1) % 20 == 0:
                print(f"[{run.name}] {s_idx + 1}/{len(slices)} slices", flush=True)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    # Bare-kernel radial profile (relative to its maximum), for reference.
    d0y = torch.minimum(yy, H - yy)[:, None].float()
    d0x = torch.minimum(xx, W - xx)[None, :].float()
    b0 = torch.sqrt(d0y ** 2 + d0x ** 2).floor().long().clamp(max=n_bins - 1).flatten()
    kp = torch.zeros(n_bins, device=device).scatter_add_(0, b0, knorm.flatten()) / torch.zeros(
        n_bins, device=device).scatter_add_(0, b0, torch.ones_like(knorm.flatten())).clamp(min=1)
    out["kernel_profile"] = (kp / kp.max()).cpu().numpy()
    return out


def peak_radius(profile):
    """Annulus centre (cells) of the profile maximum."""
    return int(np.argmax(profile)) + 0.5


def half_width(profile):
    """Distance (cells) beyond the peak where the boost above 1 first drops below half its maximum."""
    k = int(np.argmax(profile))
    half = 1 + 0.5 * (profile[k] - 1)
    below = np.flatnonzero(profile[k:] < half)
    return (k + below[0] + 0.5) if len(below) else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="Label=run_dir")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--split", default="test", choices=list(SPLIT_CODE))
    ap.add_argument("--slices", type=int, default=120)
    ap.add_argument("--front", type=int, default=32)
    ap.add_argument("--interior", type=int, default=16)
    ap.add_argument("--depth", type=float, default=4.0, help="interior cells: distance to the other phase > depth")
    ap.add_argument("--r-max-mpc", type=float, default=60.0, help="profile range shown")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/eval3d/mechanism_avg"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.run]
    cache = a.cache or json.loads((runs[0][1] / "metadata.json").read_text())["cache"]
    with h5py.File(cache, "r") as h:
        slices = pick_slices(h, a.split, a.slices, a.seed)
    print(f"{len(slices)} slices", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    results = {}
    for label, run in runs:
        results[label] = analyse(run, slices, cache, a, device)
        np.savez_compressed(a.out / f"profiles_{run.name}.npz", **results[label],
                            slice_rows=np.array([s[0] for s in slices]))

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "mathtext.fontset": "cm"})
    T = next(iter(results.values()))["rel"].shape[1]
    n_bins = next(iter(results.values()))["rel"].shape[2]
    r = (np.arange(n_bins) + 0.5) * CELL_MPC
    show = r <= a.r_max_mpc
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, T))
    summary = {}

    # 1. Mean re-weighting profile per update, per receiver class and model.
    fig, axes = plt.subplots(len(CLASSES), len(results), figsize=(3.6 * len(results), 2.5 * len(CLASSES)),
                             sharex=True, sharey="row", squeeze=False, constrained_layout=True)
    for col, (label, res) in enumerate(results.items()):
        for row, cls in enumerate(CLASSES):
            ax = axes[row][col]
            sel = res["cls"] == row
            for t in range(T):
                prof = res["rel"][sel, t]
                m, se = prof.mean(0), prof.std(0) / np.sqrt(max(sel.sum(), 1))
                ax.plot(r[show], m[show], color=colors[t], lw=1.1, label=f"$t$={t + 1}")
                ax.fill_between(r[show], (m - se)[show], (m + se)[show], color=colors[t], alpha=0.2, lw=0)
            ax2 = ax.twinx()
            ax2.plot(r[show], res["kernel_profile"][show], "k--", lw=0.8)
            ax2.set_ylim(0, 1.05)
            ax2.set_yticks([0, 0.5, 1]) if col == len(results) - 1 else ax2.set_yticks([])
            if col == len(results) - 1:
                ax2.set_ylabel(r"bare kernel $\|\kappa\|_F$ (dashed, rel.)", fontsize=7)
            ax.axhline(1, color="0.6", lw=0.5)
            ax.set_title(f"{label}: {cls} receivers (n = {sel.sum()})")
            if col == 0:
                ax.set_ylabel(r"$\langle M_t(j)/\langle M_t\rangle_j\rangle$ at $|r_i-r_j|$")
            if row == len(CLASSES) - 1:
                ax.set_xlabel(r"distance $|r_i - r_j|$  [Mpc]")
    axes[0][0].legend(ncol=2, fontsize=6.5, frameon=False)
    fig.suptitle(f"Gate re-weighting of incoming connections, mean ± s.e. over receivers "
                 f"({len(slices)} test slices, x̄_HI 0.1–0.9)")
    fig.savefig(a.out / "mt_profiles_by_update.png", dpi=200)
    plt.close(fig)

    # 2. Last-update front profile by reionization stage, models overlaid.
    fig, axes = plt.subplots(1, len(STAGES), figsize=(3.4 * len(STAGES), 2.6), sharey=True, constrained_layout=True)
    for k, (lo, hi) in enumerate(STAGES):
        ax = axes[k]
        for (label, res), ls in zip(results.items(), ("-", "--", ":")):
            sel = (res["cls"] == 0) & (res["xbar"] >= lo) & (res["xbar"] < hi)
            if not sel.any():
                continue
            m = res["rel"][sel, -1].mean(0)
            se = res["rel"][sel, -1].std(0) / np.sqrt(sel.sum())
            pk, hw = peak_radius(m), half_width(m)
            summary.setdefault(label, {})[f"front t={T}, x̄ {lo}-{hi}: peak radius [Mpc]"] = pk * CELL_MPC
            summary[label][f"front t={T}, x̄ {lo}-{hi}: peak value"] = float(m[int(pk)])
            summary[label][f"front t={T}, x̄ {lo}-{hi}: half-width [Mpc]"] = hw * CELL_MPC
            ax.plot(r[show], m[show], ls, lw=1.2,
                    label=f"{label} (peak {m[int(pk)]:.1f}× at {pk * CELL_MPC:.0f} Mpc, half-width "
                          f"{hw * CELL_MPC:.0f} Mpc, n = {sel.sum()})")
            ax.fill_between(r[show], (m - se)[show], (m + se)[show], alpha=0.2, lw=0)
        ax.axhline(1, color="0.6", lw=0.5)
        ax.set_title(f"front receivers, $t$={T}, true x̄_HI {lo}–{hi}")
        ax.set_xlabel(r"distance $|r_i - r_j|$  [Mpc]")
        ax.legend(fontsize=6.5, frameon=False)
    axes[0].set_ylabel(r"$\langle M_t(j)/\langle M_t\rangle_j\rangle$")
    fig.savefig(a.out / "mt_front_profile_by_stage.png", dpi=200)
    plt.close(fig)

    # 3. Mean gain and receiver gate per update.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    ts = np.arange(1, T + 1)
    for (label, res), ls in zip(results.items(), ("-", "--", ":")):
        for c, cls in enumerate(CLASSES):
            sel = res["cls"] == c
            col = f"C{c}"
            axes[0].plot(ts, res["gain"][sel].mean(0), ls, color=col, marker="o", ms=3, label=f"{label}, {cls}")
            axes[1].plot(ts, res["a_rms"][sel].mean(0), ls, color=col, marker="o", ms=3)
            summary.setdefault(label, {})[f"gain per update, {cls}"] = res["gain"][sel].mean(0).round(4).tolist()
            summary[label][f"receiver gate per update, {cls}"] = res["a_rms"][sel].mean(0).round(4).tolist()
    axes[0].set_ylabel(r"mean gain $\langle M_t\rangle_j$")
    axes[1].set_ylabel(r"receiver gate $\|\mathbf{A}_i\|_{\rm rms}$")
    for ax in axes:
        ax.set_xlabel("update $t$")
    axes[0].legend(fontsize=6, frameon=False, ncol=1)
    fig.savefig(a.out / "mt_gain_and_receiver_gate.png", dpi=200)
    plt.close(fig)

    for label, res in results.items():
        for c, cls in enumerate(CLASSES):
            sel = res["cls"] == c
            summary[label][f"{cls}: n receivers"] = int(sel.sum())
            for t in (0, 2, T - 1):
                m = res["rel"][sel, t].mean(0)
                pk = peak_radius(m)
                summary[label][f"{cls}: t={t + 1} peak {m[int(pk)]:.2f}x at [Mpc]"] = pk * CELL_MPC
                summary[label][f"{cls}: t={t + 1} half-width [Mpc]"] = half_width(m) * CELL_MPC
            m = res["rel"][sel, -1].mean(0)
            summary[label][f"{cls}: profile at 40-60 Mpc, t={T}"] = float(m[(r > 40) & (r < 60)].mean())
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

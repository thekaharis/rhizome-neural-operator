"""Does the rollout settle? Run trained rhizomes past their training length.

    python -m ebm21cm.rollout_diagnostic --run T3=runs/rhizome_mf2d_w64_structured_all_T3 \\
        --run T6=runs/rhizome_mf2d_w64_structured_all --max-steps 30 --out runs/eval3d/rollout_diagnostic

For every validation slice of the multi-field cache, the tied update is applied
``--max-steps`` times (training used T of them) and after each update t:

* the relative state change  ||h_t - h_{t-1}|| / ||h_{t-1}||  (per slice, then averaged);
* the two-step change ||h_t - h_{t-2}|| / ||h_{t-2}|| and the cosine between successive changes
  d_t = h_t - h_{t-1} and d_{t-1}: a 2-cycle shows as a small two-step change and a cosine near -1,
  a monotone drift as a cosine near +1;
* the decoded prediction's x_HI RMSE, T_b RMSE (mK) and the validation score
  0.5 (MSE x_HI + MSE normalized T_b), as in training.

A step size that is too large would show as state changes that do not shrink, or that
oscillate, and as a score that degrades right after the trained T. Writes
``rollout.json`` and ``rollout.png``.

Two-level models (RhizomeUNet2d) give two sweeps, each extending one level with the other
at its trained length: ``fine`` extends the full-resolution rollout (coarse state fixed after
its trained updates); ``coarse`` extends the coarse rollout and, after every coarse update t,
runs the fine level for its trained length from that coarse state. The state measures refer
to the extended level's own state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data.mf_slices import MFMemorySplit
from .model.rhizome import RhizomeOperator2d, RhizomeUNet2d, build_model, periodic_upsample
from .train_rhizome_mf2d import tb_normalized


class Tracker:
    """Per-update sums over slices: state changes (one-, two-step, cosine) and decoded errors."""

    def __init__(self, max_steps):
        self.sums = {k: np.zeros(max_steps) for k in ("dx", "dx2", "cos", "x_se", "t_se", "x_n")}
        self.n_slices = 0

    def states(self, t, prev2, prev, new):
        diff = (new - prev).flatten(1)
        self.sums["dx"][t] += float((diff.norm(dim=1) / prev.flatten(1).norm(dim=1).clamp(min=1e-12)).sum())
        if prev2 is not None:
            two = (new - prev2).flatten(1).norm(dim=1) / prev2.flatten(1).norm(dim=1).clamp(min=1e-12)
            self.sums["dx2"][t] += float(two.sum())
            self.sums["cos"][t] += float(torch.nn.functional.cosine_similarity(
                diff, (prev - prev2).flatten(1), dim=1).sum())

    def errors(self, t, out, batch, head, off, scale):
        y = batch["target"]
        tb = tb_normalized(out, batch["phys"], head, off, scale)
        self.sums["x_se"][t] += float((out[:, :1].sigmoid() - y[:, :1]).square().sum())
        self.sums["t_se"][t] += float((tb - y[:, 1:]).square().sum())
        self.sums["x_n"][t] += y[:, :1].numel()

    def result(self, trained_T, step_size, scale):
        s, n = self.sums, self.n_slices
        x_mse, t_mse = s["x_se"] / s["x_n"], s["t_se"] / s["x_n"]
        return {"trained_T": trained_T, "step_size": step_size, "rel_change": (s["dx"] / n).tolist(),
                "rel_change_two_step": [float("nan")] + (s["dx2"][1:] / n).tolist(),
                "cos_successive_changes": [float("nan")] + (s["cos"][1:] / n).tolist(),
                "xhi_rmse": np.sqrt(x_mse).tolist(), "tb_rmse_mk": (np.sqrt(t_mse) * scale).tolist(),
                "score": (0.5 * (x_mse + t_mse)).tolist(), "slices": n}


def rollout(run, max_steps, split, device, batch_size, max_slices=0, seed=0):
    """{sweep: result}: "" for single-level models, "fine" and "coarse" for two-level ones."""
    meta = json.loads((run / "metadata.json").read_text())
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(best["model_config"]).to(device).eval()
    model.load_state_dict(best["model"])
    two_level = isinstance(model, RhizomeUNet2d)
    if not two_level and (not isinstance(model, RhizomeOperator2d) or model.untied):
        raise SystemExit(f"{run}: only tied rhizome models can be rolled out further")
    data = MFMemorySplit(meta["cache"], split, "cpu", device)
    keep = best.get("scalar_keep")
    if keep is not None:
        data.scalars = data.scalars[:, keep]
    off, scale = best["tb_norm"]
    head = best["head"]
    sweeps = {"fine": Tracker(max_steps), "coarse": Tracker(max_steps)} if two_level else {"": Tracker(max_steps)}

    def iterate(tracker, state, forcing, cell, batch, n):
        """n updates of ``cell``; with a tracker, record state changes and decoded errors after each."""
        prev2 = None
        for t in range(n):
            new = cell(state, forcing)
            if tracker is not None:
                tracker.states(t, prev2, state, new)
                tracker.errors(t, model.decode(new), batch, head, off, scale)
            prev2, state = state, new
        return state

    with torch.no_grad():
        positions = None
        if max_slices and max_slices < len(data):
            positions = np.sort(np.random.default_rng(seed).choice(len(data), max_slices, replace=False))
        for batch in data.loader(batch_size, positions):
            cond, scal = batch["cond"], batch["scalars"]
            if not two_level:
                forcing = model.lift(cond) + model.scalar_embed(scal)[:, :, None, None]
                iterate(sweeps[""], forcing.tanh(), forcing, model.cells[0], batch, max_steps)
            else:
                f = model.factor
                forcing_c = model._forcing(model.lift_coarse, model.scalar_coarse,
                                           torch.nn.functional.avg_pool2d(cond, f), scal)
                fine_in = model._forcing(model.lift_fine, model.scalar_fine, cond, scal)

                def fine_forcing(coarse):
                    return model.merge(torch.cat([fine_in, periodic_upsample(coarse, f)], dim=1))

                # Coarse sweep: extend the coarse rollout; after each update run the fine level as trained.
                coarse, prev2, trained_coarse = forcing_c.tanh(), None, None
                for t in range(max_steps):
                    new = model.coarse(coarse, forcing_c)
                    sweeps["coarse"].states(t, prev2, coarse, new)
                    ff = fine_forcing(new)
                    state = iterate(None, ff.tanh(), ff, model.fine, batch, model.n_steps)
                    sweeps["coarse"].errors(t, model.decode(state), batch, head, off, scale)
                    prev2, coarse = coarse, new
                    if t + 1 == model.coarse_steps:
                        trained_coarse = coarse
                if trained_coarse is None:
                    raise SystemExit("--max-steps must reach the trained coarse length")
                # Fine sweep: coarse state after its trained updates; extend the fine rollout.
                ff = fine_forcing(trained_coarse)
                iterate(sweeps["fine"], ff.tanh(), ff, model.fine, batch, max_steps)
            for tracker in sweeps.values():
                tracker.n_slices += len(cond)
    if not two_level:
        return {"": sweeps[""].result(model.n_steps, model.cells[0].step_size, scale)}
    return {"fine": sweeps["fine"].result(model.n_steps, model.fine.step_size, scale),
            "coarse": sweeps["coarse"].result(model.coarse_steps, model.coarse.step_size, scale)}


def report(label, r, max_steps):
    T = r["trained_T"]
    if T > max_steps:
        print(f"{label}: trained T={T} exceeds --max-steps", flush=True)
        return
    cos = r["cos_successive_changes"]
    print(f"{label}: trained T={T}, step {r['step_size']}; score at T {r['score'][T - 1]:.5f}, "
          f"best {min(r['score']):.5f} at t={int(np.argmin(r['score'])) + 1}, at t={max_steps} "
          f"{r['score'][-1]:.5f}; rel. change t=1/T/last {r['rel_change'][0]:.3g} / {r['rel_change'][T - 1]:.3g} / "
          f"{r['rel_change'][-1]:.3g}; two-step at t={max_steps} {r['rel_change_two_step'][-1]:.3g}; "
          f"cos(d_t, d_t-1) at T/last {cos[T - 1]:.3f} / {cos[-1]:.3f}", flush=True)


def plot(results, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(1, a.max_steps + 1)
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.2), constrained_layout=True)
    for k, (label, r) in enumerate(results.items()):
        c = f"C{k}"
        T = r["trained_T"]
        axes[0].semilogy(t, r["rel_change"], "-o", ms=2.5, color=c, label=f"{label} (trained T={T})")
        axes[0].semilogy(t, r["rel_change_two_step"], "--", lw=0.8, color=c)
        axes[1].plot(t, r["score"], "-o", ms=2.5, color=c)
        axes[2].plot(t, r["tb_rmse_mk"], "-o", ms=2.5, color=c)
        axes[3].plot(t, r["cos_successive_changes"], "-o", ms=2.5, color=c)
        for ax in axes:
            ax.axvline(T, color=c, lw=0.6, ls=":")
    axes[0].set_ylabel(r"$\|h_t - h_{t-1}\| / \|h_{t-1}\|$")
    axes[1].set_ylabel("validation score")
    axes[2].set_ylabel(r"$T_b$ RMSE [mK]")
    axes[3].set_ylabel(r"$\cos(d_t, d_{t-1})$, $d_t = h_t - h_{t-1}$")
    axes[3].axhline(0, color="0.6", lw=0.5)
    axes[3].set_ylim(-1.05, 1.05)
    axes[0].set_title("one-step (solid) and two-step (dashed) change", fontsize=8)
    lo = min(min(r["score"]) for r in results.values())
    axes[1].set_ylim(lo * 0.97, lo * 1.5)
    lo = min(min(r["tb_rmse_mk"]) for r in results.values())
    axes[2].set_ylim(lo * 0.97, lo * 1.4)
    for ax in axes:
        ax.set_xlabel("update $t$ of the extended level (dotted: trained $T$)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Rollout beyond the trained length ({a.split} split, "
                 f"{next(iter(results.values()))['slices']} slices)")
    fig.savefig(a.out / "rollout.png", dpi=180)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="Label=run_dir")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-slices", type=int, default=0, help="random subset of the split (same for every run); 0: all")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/eval3d/rollout_diagnostic"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if a.max_steps < 2:
        raise SystemExit("--max-steps must be at least 2")
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in a.run:
        label, run = spec.split("=", 1)
        for sweep, r in rollout(Path(run), a.max_steps, a.split, torch.device(a.device), a.batch_size,
                                a.max_slices).items():
            name = f"{label} ({sweep} level)" if sweep else label
            results[name] = r
            report(name, r, a.max_steps)
        (a.out / "rollout.json").write_text(json.dumps(results, indent=1))
    plot(results, a)


if __name__ == "__main__":
    main()

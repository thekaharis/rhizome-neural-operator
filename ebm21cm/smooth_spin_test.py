"""Post-hoc LOS smoothing of the structured head's spin variable u (no retraining).

    python -m ebm21cm.smooth_spin_test --run runs/rhizome_mf2d_w64_structured_all \\
        --reference $CUBES/rhizome3d_mf_C/manifest.json --tag rhizome2d_mf_all \\
        --out-dir $CUBES --sigmas 0 2 4 8 16

The slice-wise multi-field model predicts u = log(T_CMB/T_S) independently for
every slice, so slice-to-slice scatter in u becomes small-scale T_b power before
reionization. Here every native slice of each reference test cone is predicted
once; u is Gaussian-smoothed along the LOS by ``sigma`` native cells per variant
and T_b = x_HI * phys * (1 - exp(u)) is rebuilt (x_HI is unchanged). Each variant
is written as its own eval_cubes tag ``<tag>_u<sigma>`` on the reference slices,
with truth checked against the reference cubes; ``sigma 0`` must reproduce the
plain slice-wise export.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from .data.mf_slices import FNO_ROOT, PREPARATION, fno_dataset
from .export_slicewise_mf import slice_inputs
from .model.rhizome import RhizomeOperator2d
from .train_rhizome_mf2d import U_MAX


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 2, 4, 8, 16])
    ap.add_argument("--preparation", default=PREPARATION)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-cones", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    meta = json.loads((args.run / "metadata.json").read_text())
    attrs = meta["cache_attrs"]
    bands = [tuple(b) for b in json.loads(attrs["bands"])]
    dataset, rows, LOSWindowConfig, head, _, param_names = fno_dataset(args.preparation, FNO_ROOT, attrs["history"])
    ip = [list(dataset.channel_names).index(p) for p in param_names]
    best = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    if best["head"] != "structured":
        raise SystemExit("needs a structured-head run")
    device = torch.device(args.device)
    model = RhizomeOperator2d(**best["model_config"]).to(device).eval()
    model.load_state_dict(best["model"])
    tb_off, tb_scale = best["tb_norm"]
    row_of = {int(re.findall(r"\d+", Path(dataset.file_paths[r]).stem)[-1]): int(r) for r in rows["test"]}

    ref = json.loads(args.reference.read_text())
    (ref_tag, entries), = ref["models"].items()
    entries = entries[:args.max_cones]
    tags = {s: f"{args.tag}_u{s:g}" for s in args.sigmas}
    for t in tags.values():
        (args.out_dir / t).mkdir(parents=True, exist_ok=True)
    written = {s: [] for s in args.sigmas}
    for k, entry in enumerate(entries):
        sid, row = int(entry["cone_id"]), int(entry["row"])
        if row_of.get(sid) != row:
            raise ValueError(f"cone {sid}: reference row {row} is not this sample's test row")
        with np.load(entry["npz"]) as d:
            ref_d = {key: d[key] for key in ("truth", "tb_truth", "density", "z_native")}
        z = np.asarray(dataset.redshifts[row])
        pick = np.abs(z[None, :] - ref_d["z_native"][:, None]).argmin(axis=1)
        inp = slice_inputs(dataset, row, np.arange(len(z)), head, LOSWindowConfig, bands, ip)
        n = len(z)
        x = np.empty((n, *inp["cond"].shape[-2:]), np.float32)
        u = np.empty_like(x)
        with torch.no_grad():
            for a in range(0, n, args.batch_size):
                b = slice(a, a + args.batch_size)
                out = model(torch.from_numpy(inp["cond"][b]).to(device), torch.from_numpy(inp["scalars"][b]).to(device))
                x[b] = out[:, 0].sigmoid().float().cpu().numpy()
                u[b] = out[:, 1].float().cpu().numpy()
        x, u, phys = (np.moveaxis(v, 0, -1) for v in (x, u, inp["phys"].astype(np.float32)))
        truth = np.moveaxis(inp["xhi"], 0, -1)[..., pick]
        truth_tb = (np.moveaxis(inp["tb"], 0, -1) * tb_scale + tb_off)[..., pick]
        if not (np.allclose(truth, ref_d["truth"], atol=1e-5) and np.allclose(truth_tb, ref_d["tb_truth"], atol=1e-2)):
            raise ValueError(f"cone {sid}: targets differ from the reference truth")
        line = []
        for s, tag in tags.items():
            us = u if s == 0 else gaussian_filter1d(u, s, axis=-1, mode="nearest")
            tb = (x * phys * (1.0 - np.exp(np.minimum(us, U_MAX))))[..., pick]
            target = args.out_dir / tag / f"cone_{sid:06d}.npz"
            np.savez(target, pred=x[..., pick], truth=ref_d["truth"], tb_pred=tb.astype(np.float32),
                     tb_truth=ref_d["tb_truth"], density=ref_d["density"], z_native=ref_d["z_native"])
            written[s].append({**entry, "npz": str(target.resolve())})
            line.append(f"u{s:g} {np.sqrt(np.mean((tb - ref_d['tb_truth']) ** 2)):.2f}")
        print(f"[{k + 1}/{len(entries)}] sample {sid}: T_b rmse mK " + "  ".join(line), flush=True)
    for s, tag in tags.items():
        manifest = {"z_grid": ref["z_grid"], "models": {tag: written[s]}, "split": ref.get("split"),
                    "checkpoint": str((args.run / "best.pt").resolve()), "fields": ref.get("fields", {}),
                    "selection": f"slice-wise 2-D at the slices of {ref_tag}; u smoothed along LOS, sigma {s:g} native cells"}
        (args.out_dir / tag / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", ", ".join(tags.values()))


if __name__ == "__main__":
    main()

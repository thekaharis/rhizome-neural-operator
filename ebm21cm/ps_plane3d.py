"""Placement vs amount of small-scale structure for 3-D x_HI models.

    python -m ebm21cm.ps_plane3d --out runs/eval3d/ps_plane3d.png \\
        --group "fno-21cm 3-D matrix=$FNO/figures/shared/eval/final_eval/*/ps/ps_metrics.csv" \\
        --group "fno LOS windows=runs/eval3d/a/power_spectrum/ps_metrics.csv:fno_cnn_whno_es" \\
        --group "rhizome=runs/eval3d/a/power_spectrum/ps_metrics.csv:rhizome3d_w48"

The 3-D counterpart of fno-21cm's ``viz.ps_coherence_highk`` summary plane, built
from ``viz.power_spectrum_evaluation``'s ``ps_metrics.csv`` (same transverse k
bins for every run): small-scale coherence ``r`` (k > 1 Mpc^-1, x-axis, higher
is better) against the small-scale power error ``|P_pred/P_true - 1|`` (k > 1,
y-axis, lower is better), on the active stage (x_HI 0.05-0.95).

``--group label=glob[;glob...][:model,model...]``: every model row of the
matching csv files, or only the named ones. A model appearing in several files of one group
is plotted once (first file wins).
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"
PALETTE = ["#8a8880", "#eb6834", "#2a78d6", "#1baf7a", "#9d4edd", "#d4a017"]
COH, POWER = "one_minus_r_small_k>1", "abs_ratio_err_small_k>1"


def parse_group(spec: str):
    label, _, rest = spec.partition("=")
    pattern, _, names = rest.partition(":")
    return label, pattern, [n for n in names.split(",") if n]


def read_group(pattern: str, names: list[str]) -> dict[str, dict]:
    files = sorted(f for part in pattern.split(";") for f in glob.glob(part))
    if not files:
        raise SystemExit(f"no csv matches {pattern}")
    rows = {}
    for f in files:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["stage"].startswith("active") and r["model"] not in rows and (not names or r["model"] in names):
                rows[r["model"]] = {"r": 1.0 - float(r[COH]), "err": float(r[POWER]), "cones": int(r["n_cones"])}
    missing = set(names) - set(rows)
    if missing:
        raise SystemExit(f"{pattern}: no active-stage rows for {sorted(missing)}")
    return rows


def place_labels(ax, labels, fixed=None):
    """Label points, choosing per label the first offset whose text box (estimated
    in display pixels) overlaps no earlier label and no point."""
    ax.figure.canvas.draw()
    to_px = ax.transData.transform
    dpi = ax.figure.dpi / 72
    points = [to_px((x, y)) for _, x, y, _ in labels]
    fixed = fixed or {}
    boxes = []
    candidates = [(7, 4, "left"), (7, -12, "left"), (-7, 4, "right"), (-7, -12, "right"),
                  (7, 14, "left"), (-7, 14, "right"), (7, -22, "left"), (-7, -22, "right")]
    for text, x, y, hollow in labels:
        size = 7.5 if hollow else 8.5
        w, h = 0.6 * size * len(text) * dpi, size * 1.2 * dpi
        px, py = to_px((x, y))
        own = [(fixed[text][0], fixed[text][1], "right" if fixed[text][0] < 0 else "left")] if text in fixed else []
        for dx, dy, ha in own or candidates:
            x0 = px + dx * dpi - (w if ha == "right" else 0)
            y0 = py + dy * dpi
            box = (x0, y0, x0 + w, y0 + h)
            clash = any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes)
            clash |= any(box[0] - 4 < qx < box[2] + 4 and box[1] - 4 < qy < box[3] + 4 for qx, qy in points)
            if not clash or own:
                break
        boxes.append(box)
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=size,
                    color=INK_MUTED if hollow else INK_2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--group", action="append", required=True, type=parse_group)
    ap.add_argument("--rename", nargs="*", default=[], help="model=label pairs for the point labels")
    ap.add_argument("--label-offset", nargs="*", default=[], metavar="LABEL=DX,DY",
                    help="fixed label offsets in points (negative DX: label left of the point)")
    ap.add_argument("--legend-loc", default="upper left")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rename = dict(s.split("=", 1) for s in args.rename)

    fig, ax = plt.subplots(figsize=(9.5, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lines = ["group,model,coherence_r_small,power_err_small,n_cones"]
    groups = [(label, read_group(pattern, names)) for label, pattern, names in args.group]
    labels = []
    for i, (label, rows) in enumerate(groups):
        color = PALETTE[i % len(PALETTE)]
        hollow = i == 0
        ax.scatter([v["r"] for v in rows.values()], [v["err"] for v in rows.values()], s=70, label=label,
                   zorder=3, linewidths=1.6, facecolors="none" if hollow else color, edgecolors=color)
        for name, v in rows.items():
            labels.append((rename.get(name, name), v["r"], v["err"], hollow))
            lines.append(f"{label},{name},{v['r']:.4f},{v['err']:.4f},{v['cones']}")
    place_labels(ax, labels, {k: tuple(map(float, v.split(","))) for k, v in
                              (o.split("=", 1) for o in args.label_offset)})
    ax.set_xlabel("small-scale coherence  r(k > 1 Mpc$^{-1}$)  (higher better)", color=INK)
    ax.set_ylabel("small-scale power error  |P$_{pred}$/P$_{true}$ − 1|, k > 1  (lower better)", color=INK)
    ax.set_title("Placement vs amount of small-scale structure (3-D x$_{HI}$, active stage)",
                 loc="left", color=INK)
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, loc=args.legend_loc, fontsize=9)
    if args.note:
        fig.text(0.01, 0.01, args.note, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06 if args.note else 0, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    args.out.with_suffix(".csv").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

# LoReLi II integration figures

PDF for LaTeX, PNG (200-300 dpi) for slides. Scripts in `src/`; run them from the repository
root (`rhizome-neural-operator`) with the project environment (`source slurm/_env.sh`) on a
compute node (`srun -p compute -c 8 --mem=80G`).

| File | Content | Data |
|---|---|---|
| `fig_los_geometry` | The LOS axis is uniform in redshift: z(plane) inferred from the 21ssd box histories vs the three axis hypotheses; one cone's slice-mean x_HI vs its box history; plane spacing and the LOS lag-1 density correlation | globals v1, five-field cones |
| `fig_cache_rows` | Redshift of the cache rows (labelled on the old comoving axis vs true) and rows per stage and split | **v2 cache** |
| `fig_history_leakage` | History scalar x_HI(0,1) vs target slice mean: v2 oracle, emulated side file, 21cmFAST training cache | **v2 cache** + v2 side file |
| `fig_emulator_examples` | History emulator on four held-out simulations (x_HI, T_b, ±2σ ensemble spread) | `runs/loreli_emulator/v1` |
| `fig_emulator_errors` | Emulator RMS by redshift band: held-out, out-of-fold, in-sample | `runs/loreli_emulator/v1/results.json` |
| `fig_finetune_curves` | Validation score vs step, 44- and 8-block ladders | **v2 runs** (`runs/loreli_finetune/l294_b*`, oracle histories) |
| `loreli_cache_slices` | One training cone (sim 12568) at four stages: x_HI, T_b, density and velocity bands, phys; interior only | v3 cache |
| `decoder_b8_v3/dec_gallery` | Two test cones (12316, 12317; same block, differ only in r_H/S) at <x_HI> 0.75 and 0.40: truth, reference, fine-tuned, residuals for x_HI and T_b | v3 cache + v3 side file |
| `decoder_b8_v3/dec_lightcone_<cone>` | Lightcone cut (centre line of every contiguous row, z 5.0-7.1) for truth / reference / fine-tuned | v3 |
| `decoder_b8_v3/dec_pixel_stats` | 1.27 M random interior cells from 4,971 test rows (every 3rd): 2-D histograms, PDFs by stage, calibration | v3 |
| `decoder_b8_v3/dec_metrics` | RMSE vs z, ionized-region IoU vs stage, slice-mean x_HI/T_b vs truth, per-row RMSE gain | v3 |
| `decoder_b8_v3/dec_power` | Transverse P_pred/P_true and cross-correlation r(k) by stage | v3 |
| `decoder_b8_v3/dec_decoder_change` | Relative weight change of the 1x1-conv decoder; learned remap reference -> fine-tuned (x_HI, logit, T_b) | v3 |
| `loreli_v3_b8_decoder_test_predictions` | Test cone 12316: truth vs zero-shot reference vs fine-tuned (8 blocks, decoder arm), emulated histories, per-panel RMSE | v3 cache + v3 side file |
| `cm21eval/cm21_scorecard` | 21cmEVAL headline metrics per model: pooled x_HI / T_b RMSE, IoU, r(k=0.2), power ratio, Δτ, Δz(0.5), front width; median over the 30 test cones with 16-84 % and per-cone dots (numbers in `cm21_scorecard.json`) | `runs/cm21eval` |
| `cm21eval/cm21_per_cone_rmse` | x_HI and T_b RMSE per test lightcone, connected across models, coloured by redshift group | `runs/cm21eval` |
| `cm21eval/cm21_spectra_{xhi,tb}` | Transverse P_pred/P_true and r(k) for the five x_HI stages + active, median over cones | `runs/cm21eval` |
| `cm21eval/cm21_bubbles` | Mean-free-path bubble-size distributions per stage, truth vs models | `runs/cm21eval` |
| `cm21eval/cm21_history` | τ, z(0.5), duration pred vs truth per cone; history errors; one example history per group | `runs/cm21eval` |
| `cm21eval/cm21_fronts` | Mean x_HI and RMSE vs distance to the true ionization front; 10-90 % front width per cone | `runs/cm21eval` |
| `eval3d/native_cyl/loreli_v3_*_cyl_error_{xhi,tb}` | Cylindrical power error P_pred/P_true - 1 (white within +-0.1), native LOS resolution, cone median, rows = models: `ebm21cm.plot_cyl_error` on `ebm21cm.native_cyl` output, the same tool and parameters as the 21cmFAST maps in `runs/eval3d/native_cyl/` | `data/eval_cubes/loreli_v3_<model>` |
| `eval3d/loreli_cones/cone_<id>` | Gallery per cone: rows = stages (x_HI-bar 0.8/0.6/0.4/0.2); x_HI (viridis) and T_b (magma) truth, models, model - truth, slice RMSE per panel: `ebm21cm.compare_cones3d` | `data/eval_cubes/loreli_v3_<model>` |
| `eval3d/loreli_deltas/<group>/cone_<id>_deltas`, `overview_deltas` | Prediction - truth per slice at x_HI-bar 0.2/0.4/0.6/0.8: `ebm21cm.delta_slices` | `data/eval_cubes/loreli_v3_<model>` |
| `cm21eval_ref_vs_b8_pointwise/cm21_*` | The same eight 21cmEVAL figures restricted to the reference and the 8-block pointwise arm (`--models reference b8_pointwise`) | `runs/cm21eval` |
| `cm21eval/cm21_eft_morphology` | EFT transfer / stochasticity / r(k) relative to truth (k <= 0.5); 3-D IoU vs ionized fraction; region counts and radii | `runs/cm21eval` |

Figures marked **v2** document the state before the LOS-axis fix (wrong per-row redshifts,
oracle histories); they belong to `LoReLi_integration_report.pdf` (1 Nov 2026) and are
superseded once the v3 ladders finish. `summary.json` holds the numbers printed by
`make_loreli_report_figs.py`.

## Rebuild

```bash
python docs/figures/loreli/src/make_loreli_report_figs.py docs/figures/loreli
python docs/figures/loreli/src/make_loreli_cache_slices.py docs/figures/loreli/loreli_cache_slices
python docs/figures/loreli/src/make_loreli_test_predictions.py \
    runs/loreli_finetune/l294v3_b8_decoder_emu_seed0 \
    docs/figures/loreli/loreli_v3_b8_decoder_test_predictions.png
```

The `cm21eval/` set (CPU, < 1 min; skips models whose 21cmEVAL runs are incomplete, so rerun it as
the ladders finish): `python docs/figures/loreli/src/make_loreli_cm21eval_figs.py [runs/cm21eval] [docs/figures/loreli/cm21eval]`.
21cmEVAL pools nothing across the four redshift groups; the script concatenates per-cone curves over all
groups (median, 16-84 %) and pools voxel RMSE exactly.

`eval3d/` figures come from the repository's standard viz tools (rendered into `runs/eval3d/`, copied
here). Export a model once (`scripts/loreli_export_eval_cubes.py --model <m> --native-cyl
runs/eval3d/native_cyl/loreli_v3_<m>`; one manifest per redshift group), then e.g.

```bash
python -m ebm21cm.plot_cyl_error --out runs/eval3d/native_cyl --prefix loreli_v3_cyl_error_tb \
    --field '$T_b$ (native LOS resolution)' --cones '<n> LoReLi II test cones' \
    --model runs/eval3d/native_cyl/loreli_v3_reference_tb.npz:loreli_v3_reference=Reference ...
python -m ebm21cm.compare_cones3d --cone 12316 --out runs/eval3d/loreli_cones \
    --model Reference=data/eval_cubes/loreli_v3_reference/manifest_G1_z5.0.json ...
python -m ebm21cm.delta_slices --cone 12316 12317 --out runs/eval3d/loreli_deltas/G1_z5.0 \
    --model Reference=data/eval_cubes/loreli_v3_reference/manifest_G1_z5.0.json ...
```

The `decoder_b8_v3/` set: `python docs/figures/loreli/src/make_loreli_decoder_viz.py <run_dir> <out_dir> [--stride 3]`
(GPU, ~6 min on an A30; writes `decoder_viz_data.npz`, then `--plot-only` re-plots on CPU). Works for any ladder run.

`make_loreli_report_figs.py` builds the old comoving axis explicitly (`comoving_axis`), because
`loreli_redshifts()` now returns the true uniform-z axis.

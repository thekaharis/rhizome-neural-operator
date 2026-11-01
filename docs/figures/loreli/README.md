# LoReLi II integration figures

PDF for LaTeX, PNG (200-300 dpi) for slides. Scripts in `src/`; run them from the worktree
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

The `decoder_b8_v3/` set: `python docs/figures/loreli/src/make_loreli_decoder_viz.py <run_dir> <out_dir> [--stride 3]`
(GPU, ~6 min on an A30; writes `decoder_viz_data.npz`, then `--plot-only` re-plots on CPU). Works for any ladder run.

`make_loreli_report_figs.py` builds the old comoving axis explicitly (`comoving_axis`), because
`loreli_redshifts()` now returns the true uniform-z axis.

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
| `cm21eval_viz/<set>/summary_*` | 21cmEVAL `cm21eval plot summary`: scorecard (pooled x_HI / T_b RMSE, IoU, r(k=0.2), power ratio, Δτ, Δz(0.5), front width; `summary_scorecard.json`), per-cone RMSE, transverse spectra per stage, bubble sizes, history, fronts, EFT + morphology; pooled over the four redshift groups | `runs/cm21eval/<model>/<group>/{xhi,tb}` |
| `cm21eval_viz/<set>/cyl_error_{xhi,tb}` | 21cmEVAL `cylindrical` metric + `cm21eval plot cyl`: P_pred/P_true - 1 over (k_perp, k_par), stage chunks x_HI-bar 0.9...0.1, hatched beyond LoReLi's native resolution; identical (1e-7) to `native_cyl` on the same chunks | `runs/cm21eval/<model>/viz/...` |
| `cm21eval_viz/ref_vs_b8_pointwise/{gallery,deltas}/` | 21cmEVAL `gallery` metric + `cm21eval plot gallery/deltas`: per-cone gallery (all 30 test cones) and prediction - truth pages, the `compare_cones3d` / `delta_slices` layouts | `runs/cm21eval/<model>/viz/...` |
| sets | `ref_vs_b8_pointwise/` (every figure), `all_models/` (cyl + summary, all 13 models incl. the 44-block arms); labels/colours from `src/cm21eval_style.json` | |
| `eval3d/native_cyl/loreli_v3_*_cyl_error_{xhi,tb}` | Cylindrical power error P_pred/P_true - 1 (white within +-0.1), native LOS resolution, cone median, rows = models: `ebm21cm.plot_cyl_error` on `ebm21cm.native_cyl` output, the same tool and parameters as the 21cmFAST maps in `runs/eval3d/native_cyl/` | `data/eval_cubes/loreli_v3_<model>` |
| `eval3d/native_cyl/loreli_v3_cyl_error_stage_{xhi,tb}` | The same error maps with chunks centred per cone at x_HI-bar 0.9/0.7/0.5/0.3/0.1 (`native_cyl` stage mode); hatched: beyond LoReLi's native resolution (2.3 Mpc transverse, 1.6-3.1 Mpc LOS), not counted | `runs/eval3d/native_cyl/stage/` |
| `eval3d/native_cyl/native_cyl_error_stage_{xhi,tb}` | 21cmFAST baseline in the same stage layout: plain `rhizome_mf2d_w64_structured_all` (the LoReLi reference) and its `pw0.03_px0.03` power-loss variant, 199 test cones | `runs/eval3d/native_cyl/stage/` |
| `eval3d/loreli_cones/cone_<id>` | Gallery per cone: rows = stages (x_HI-bar 0.8/0.6/0.4/0.2); x_HI (viridis) and T_b (magma) truth, models, model - truth, slice RMSE per panel: `ebm21cm.compare_cones3d` | `data/eval_cubes/loreli_v3_<model>` |
| `eval3d/loreli_deltas/<group>/cone_<id>_deltas`, `overview_deltas` | Prediction - truth per slice at x_HI-bar 0.2/0.4/0.6/0.8: `ebm21cm.delta_slices` | `data/eval_cubes/loreli_v3_<model>` |

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

The `cm21eval_viz/` sets come from 21cmEVAL itself (`eval21 plot`, ~2 min on a compute node). The
`cylindrical` and `gallery` metrics run as an extra pass next to the full one (merged by the plots):

```bash
export XHI_METRICS=cylindrical,gallery TB_METRICS=cylindrical,gallery
export EXTRA="--set omega_m=0.308 --set cyl_native_cell_mpc=2.296875 --set cyl_native_los_dz=0.0065146580 --set gallery_samples=100"
MODELS=<m> OUT=runs/cm21eval/<m>/viz sbatch --export=ALL slurm/loreli_cm21eval.sbatch     # ~3 min per model
PYTHONPATH=$HOME/21cmEVAL/src python -m eval21 plot all --run runs/cm21eval \
    --out docs/figures/loreli/cm21eval_viz/ref_vs_b8_pointwise --models reference,b8_pointwise \
    --style docs/figures/loreli/src/cm21eval_style.json --title "LoReLi II v3 test"
```

(`sbatch --export=A=x,y` splits at the comma: export the metric lists in the shell and pass `--export=ALL`.)
21cmEVAL pools nothing across the four redshift groups; the script concatenates per-cone curves over all
groups (median, 16-84 %) and pools voxel RMSE exactly.

`eval3d/` figures come from the repository's standard viz tools (rendered into `runs/eval3d/`, copied
here). Export a model once (`scripts/loreli_export_eval_cubes.py --model <m> --native-cyl
runs/eval3d/native_cyl/loreli_v3_<m>`; one manifest per redshift group), then e.g.

```bash
python -m ebm21cm.plot_cyl_error --out runs/eval3d/native_cyl --prefix loreli_v3_cyl_error_tb \
    --field '$T_b$ (native LOS resolution)' --cones '<n> LoReLi II test cones' \
    --model runs/eval3d/native_cyl/loreli_v3_reference_tb.npz:loreli_v3_reference=Reference ...
# stage-matched chunks (LoReLi; 21cmFAST: export_slicewise_mf ... --cyl-only --cyl-stages ...)
python scripts/loreli_export_eval_cubes.py --model <m> --cyl-only --cyl-stages 0.9 0.7 0.5 0.3 0.1 \
    --native-cyl runs/eval3d/native_cyl/stage/loreli_v3_<m>
python -m ebm21cm.compare_cones3d --cone 12316 --out runs/eval3d/loreli_cones \
    --model Reference=data/eval_cubes/loreli_v3_reference/manifest_G1_z5.0.json ...
python -m ebm21cm.delta_slices --cone 12316 12317 --out runs/eval3d/loreli_deltas/G1_z5.0 \
    --model Reference=data/eval_cubes/loreli_v3_reference/manifest_G1_z5.0.json ...
```

The `decoder_b8_v3/` set: `python docs/figures/loreli/src/make_loreli_decoder_viz.py <run_dir> <out_dir> [--stride 3]`
(GPU, ~6 min on an A30; writes `decoder_viz_data.npz`, then `--plot-only` re-plots on CPU). Works for any ladder run.

`make_loreli_report_figs.py` builds the old comoving axis explicitly (`comoving_axis`), because
`loreli_redshifts()` now returns the true uniform-z axis.

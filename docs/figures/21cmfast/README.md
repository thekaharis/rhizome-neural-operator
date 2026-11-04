# 21cmFAST test-set figures

The slice-wise reference model `runs/rhizome_mf2d_w64_structured_all` on the 199 21cmFAST test cones,
next to fno-21cm's 3-D multi-field model `mf_cnn_whno_es_tbs_histtb_tbclean_ep26` (CNN + Walsh–Hadamard
local operator). Both models' predictions are the `eval_cubes` exports on fno-21cm's 512-plane
evaluation grid (`$WORK/data/eval_cubes/{rhizome2d_mf_w64_all,mf_cnn_whno_es_tbs_histtb_tbclean_ep26}`).
PDF for LaTeX, PNG for slides.

| File | Content |
|---|---|
| `21cmfast_cache_slices` | Test cone 24 as the model sees it: target x_HI, T_b [mK], central density and LOS-velocity bands, phys, at three stages and one background slice (`rhizome_mf_slices_all.h5`) |
| `cone_24` | Gallery of test cone 24 at x_HI-bar 0.8/0.6/0.4/0.2: x_HI and T_b truth, both models, residuals, slice RMSE (`ebm21cm.compare_cones3d`) |
| `cm21eval_viz/` | 21cmEVAL figures (scorecard, spectra, cylindrical power error, bubbles, history, fronts, …), labels from `cm21eval_style.json` |

## Rebuild

```bash
source slurm/_env.sh
python docs/figures/loreli/src/make_loreli_cache_slices.py docs/figures/21cmfast/21cmfast_cache_slices \
    --cache $WORK/data/compressed/rhizome_mf_slices_all.h5 --split test --cone 24
python -m ebm21cm.compare_cones3d --cone 24 --out docs/figures/21cmfast \
    --model Rhizome=$WORK/data/eval_cubes/rhizome2d_mf_w64_all/manifest.json \
    --model "fno-21cm 3-D=$WORK/data/eval_cubes/mf_cnn_whno_es_tbs_histtb_tbclean_ep26/manifest.json"

# 21cmEVAL: chunk manifests over the exports, one CPU array task per 25 cones (~15 min each)
python scripts/eval_cubes_cm21eval.py --cubes $WORK/data/eval_cubes/rhizome2d_mf_w64_all \
    --out $WORK/data/eval_cubes/21cmfast_test_eval21 --chunk 25 \
    --check $WORK/data/eval_cubes/mf_cnn_whno_es_tbs_histtb_tbclean_ep26
export MODELS="rhizome=rhizome2d_mf_w64_all ufno=mf_cnn_whno_es_tbs_histtb_tbclean_ep26" \
       OUT=runs/cm21eval_21cmfast/main \
       XHI_METRICS=voxel,edge,spectrum,bubble,morphology,history,observable,eft,front,parity,anisotropy,cylindrical \
       TB_METRICS=voxel,edge,spectrum,cylindrical
sbatch --array=0-7 --export=ALL slurm/cm21eval_eval_cubes.sbatch
PYTHONPATH=$HOME/21cmEVAL/src python -m eval21 plot all --run runs/cm21eval_21cmfast/main \
    --out docs/figures/21cmfast/cm21eval_viz --style docs/figures/21cmfast/cm21eval_style.json \
    --title "21cmFAST test"
```

The stage of a T_b slice is set by the true x_HI (`--set stage_field=truth`; the T_b manifests carry
the truth x_HI as an input). The predictions come from files, so 21cmEVAL's `efficiency` metric
does not apply.

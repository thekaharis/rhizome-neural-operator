# Darcy test: Rhizome with hidden width 8

Completed 2026-09-30 on CPU, using the Darcy implementation associated with
the supplied technical note. The new model achieved mean test relative L2
**7.998%** on 1,000 held-out fields after 3,000 optimizer steps. Its error is
close to the earlier width-8 FNO's **7.928%**, with **15,113** versus **59,201**
real parameters. This is a single-seed, fixed-budget comparison.

## Protocol

Only hidden width changed from the report's width-16 Rhizome recipe:

| Setting | Value |
|---|---|
| Architecture | Shared recurrent Rhizome integral operator |
| Input | Encoded coefficient plus two nominal grid coordinates |
| Grid | 32 x 32 |
| Retained mode setting | 8 |
| Hidden width | 8 |
| Updates | 4, with shared parameters |
| Damping | 0.5 |
| Boundary convention | Zero-extended messages on the doubled domain |
| Objective | MSE of training-only standardized solutions |
| Optimizer | Adam, initial LR 0.003, gradient clipping 1 |
| Batch size | 32 |
| Optimizer steps | 3,000 |
| Training seed / split seed | 0 / 42 |
| Training / validation / test | 4,500 / 500 / 1,000 fields |
| Validation cadence | Every 200 steps |
| Selection | Minimum mean per-field validation relative L2 |
| Runtime | PyTorch 2.9.1, CPU, two threads |

The runner's validation-driven LR scheduler was retained; it did not reduce
the LR. Training presented 95,748 examples, or 21.277 epochs. Validation
selected the final step, 3,000, with mean relative L2 7.931%. The best
checkpoint was then evaluated on the test fields. Recorded training time,
including validation, was 531.697 seconds (8.862 minutes); final test
evaluation and plotting are outside that timer.

The dataset is the NeuralOperator Team's encoded Darcy-32 release, Zenodo
record 12784353 (CC BY 4.0), as in the technical note. Metrics use stored
solution units; the archive's physical scaling was not reconstructed. There
is no sigmoid, clipping or imposed boundary-pixel mask on the solution.

## Test results and comparisons

The width-16 Rhizome and width-8 FNO columns are the saved runs from the
technical note, not newly trained baselines.

| Metric | Rhizome width 8, new | Rhizome width 16, prior | FNO width 8, prior |
|---|---:|---:|---:|
| Real parameters | 15,113 | 60,305 | 59,201 |
| Best optimizer step | 3,000 | 2,600 | 3,000 |
| Mean relative L2 | **7.998%** | **5.556%** | **7.928%** |
| Median relative L2 | 7.310% | 5.127% | 7.262% |
| 95th percentile relative L2 | 13.295% | 9.152% | 13.773% |
| RMSE, stored units | 0.043699 | 0.030315 | 0.044098 |
| MAE, stored units | 0.028459 | 0.019576 | 0.028558 |
| Recorded CPU training minutes | 8.862 | 15.710 | 7.425 |

Against width-8 FNO, the new Rhizome has 74.47% fewer parameters and a mean
relative L2 error 0.06954 percentage points higher (0.877% higher relative
error). Rhizome has lower per-field relative error on 471 test fields; FNO
has lower error on 529. Rhizome's global RMSE and 95th-percentile relative
error are slightly lower. These metrics aggregate errors differently and
need not rank the models identically.

Recorded training time is 1.194 times the earlier FNO run. The parameter
saving therefore does not establish faster training. Timings were recorded
in separate runs and are observational, not a controlled hardware-throughput
benchmark.

Reducing Rhizome width from 16 to 8 increased mean relative L2 by 43.95%,
while retaining 25.06% of its parameters. The width-16 model has lower
per-field relative error on 937 of 1,000 test fields. The wider shared model
remains substantially more accurate under this recipe.

The useful result is that the compact shared architecture reaches approximately
the width-8 FNO's accuracy at about one quarter of its parameter count. This
does not establish a statistical tie, general architectural superiority or
the isolated contribution of the gates. The originally reported accuracy
advantage at a similar parameter budget compared a width-16 Rhizome with a
width-8 FNO; this new equal-width run makes that distinction explicit.

## Diagnostics and verification

The training-mean solution baseline has mean relative L2 48.690%. Cyclically
mismatching coefficient fields and targets increases the new model's error to
70.623%, supporting input dependence rather than prediction of a typical
solution shape.

Before training, all 48 targeted tests for the report's Rhizome, spectral
operator and Darcy pipeline passed. The current Darcy source matches the
recorded FNO run's source hashes. The original width-16 Rhizome predates a
shared spectral refactor; loading its saved checkpoint in the current Darcy
implementation reproduced its recorded validation error exactly:
0.0553750486213652.

The comparison script checked equality of both dataset file hashes, train and
validation row IDs, training-only normalization, test coefficients, test
targets, row ordering, shared training settings and example presentations.
It independently recomputed the saved prediction metrics with NumPy. Reloading
the new checkpoint reproduced all 1,000 exported predictions bitwise, and the
training source hashes still matched. Example figures were inspected.

The dataset hashes are:

```text
darcy_train_32.pt: f44a1151802eaf2eac65ddaded1ecf105bac019add0e3d277a3e9939cca1cd8b
darcy_test_32.pt:  f9fbe30065908327f132425d760d573ed7ce29cb2a54cfdb3fcefd3d80d7ac83
```

## Artifacts and reproduction

New run artifacts are under `runs/darcy_rhizome_w8/`: metadata, validation log,
selected `best.pt`, results JSON, test predictions, learning curve and example
fields. The comparison is in `runs/darcy_width_comparison/results.json`, with
paired errors in `per_field_relative_l2.npz`. These large run artifacts are
ignored by Git. The documentation and comparison helper can be versioned.

The current root package lacks the report's Darcy additions. Run the command
from `.delta/worktrees/e49xfk01xnej/ebm-21cm`, using a **new empty** output
directory for a repeat:

```bash
python -m ebm21cm.train_darcy \
  --data-root ../../jav8yay7g827/ebm-21cm/runs/darcy_data \
  --run-dir ../../../../runs/darcy_rhizome_w8_repeat \
  --architecture rhizome --width 8 \
  --resolution 32 --n-val 500 --steps 3000 --batch-size 32 \
  --modes 8 --updates 4 --step-size 0.5 \
  --lr 0.003 --val-every 200 --seed 0 --split-seed 42 \
  --threads 2 --device cpu --boundary zero_extension
```

From the root checkout, regenerate the comparison using:

```bash
python scripts/compare_darcy_widths.py \
  --rhizome8 runs/darcy_rhizome_w8 \
  --rhizome16 .delta/worktrees/jav8yay7g827/ebm-21cm/runs/darcy_rhizome \
  --fno8 .delta/worktrees/jav8yay7g827/ebm-21cm/runs/darcy_fno_matched \
  --out runs/darcy_width_comparison
```

## Limits

One training seed was used, with the already-inspected public test set. Equal
width and optimizer steps do not match trainable parameters or compute. Both
width-8 models selected their final step, so neither run establishes convergence.
Further seeds, longer independently chosen budgets and gate ablations would
be needed to explain mechanisms or generalize this comparison. This run does
not validate physical PDE residuals or resolution transfer.

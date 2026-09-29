# Long rhizome training with an explicit stopping rule

This run measures **training/validation convergence**, not fixed-point
convergence of the recurrent state. It keeps the integrated architecture and
four state updates from the earlier pilot. Increasing optimizer steps does not
automatically make extra recurrent updates accurate.

## Dataset and protocol

The larger synthetic dataset uses 128 new toy-lightcone random seeds, with
32 slices per cone, at 32×32 resolution. The cache has 4,096 slices:

- 3,264 training slices from 102 cones;
- 416 validation slices from 13 cones;
- 416 test slices from 13 cones.

The toy generator seed is **1**, distinct from the earlier seed-0 pilot.
Simulation-level splits and normalization are provided by the existing cache.
The task remains deterministic `x_HI` prediction using density bands, redshift,
and the 11 simulation parameters. This is not training on real 21cmFAST fields.

The architecture is unchanged: width 16, four retained modes, four shared
recurrent updates, step size 0.5, and 15,617 real-valued parameters. Training
uses Adam, batch size 16, gradient clipping at 1, and training seed 0.

### Predefined plateau rule

`ebm21cm/train_rhizome.py` implements the following:

1. Evaluate BCE on the **entire validation split every 200 optimizer steps**.
2. Start at learning rate `0.003`. After four checks without a significant
   improvement, halve it, down to `0.00003`.
3. A significant improvement is a relative reduction of **more than 0.2%**
   from the last significant reference loss. Smaller cumulative improvements
   can eventually pass that threshold.
4. Reset the patience counter after each learning-rate reduction.
5. Stop only after at least **4,000 optimizer steps**, at the minimum learning
   rate, with **eight further checks** without significant improvement.
6. Always retain the checkpoint with the lowest absolute validation BCE,
   even if a small improvement does not reset the patience counter.

The initial safety budget is **20,000 steps**. Exhausting the budget is reported
as `max_steps`, **not convergence**; it can be extended with `--resume`.
Test fields are not evaluated at a budget stop. Only after the plateau criterion
is met is the validation-selected checkpoint evaluated on the test split.

This is an empirical generalization-plateau rule, not a proof that the
optimizer found a global minimum. It may stop when training loss is still
falling but validation no longer improves.

## Reproduce

Use the project's PyTorch environment (the recorded run uses `FNO_env`).

```bash
python -m ebm21cm.data.toy \
  --out runs/rhizome_long_seed1/lightcones --n 128 --H 32 \
  --cell 4 --z-lo 6 --z-hi 12 --seed 1

python -m ebm21cm.data.cache build \
  --data runs/rhizome_long_seed1/lightcones \
  --out runs/rhizome_long_seed1/slices.h5 --slices-per-cone 32

python -m ebm21cm.train_rhizome \
  --cache runs/rhizome_long_seed1/slices.h5 \
  --run-dir runs/rhizome_long_seed1/fit \
  --max-steps 20000 --min-steps 4000 \
  --batch-size 16 --width 16 --modes 4 --updates 4 --step-size 0.5 \
  --lr 0.003 --min-lr 0.00003 --lr-factor 0.5 \
  --lr-patience 4 --stop-patience 8 --relative-min-delta 0.002 \
  --val-every 200 --train-probe-rows 256 --seed 0 --threads 2 --device cpu
```

Choose a new run directory when starting a fresh run. To resume an interrupted
run, pass the same options plus `--resume`. To extend a budget-limited run, also
increase `--max-steps`; other training options must remain identical.

```bash
# All unlisted options here use exactly the defaults in the full command above.
python -m ebm21cm.train_rhizome \
  --cache runs/rhizome_long_seed1/slices.h5 \
  --run-dir runs/rhizome_long_seed1/fit \
  --max-steps 40000 --resume
```

A stopped, plateau-complete run does not restart optimization on resume.

## Checkpoints, isolation, and reproducibility

`last.pt` is saved at each completed validation boundary and at a clean budget
stop. It contains:

- current model and optimizer;
- best model and validation-plateau state;
- shuffled training order, batch position, and augmentation RNG;
- PyTorch RNG state, including CUDA RNG state when applicable;
- partial training-loss window, examples seen, and the complete metric history.

Saving uses temporary files followed by atomic replacement. `last.pt` also
contains the best weights, allowing `best.pt` and the JSON log to be reconstructed
if interruption occurs between file writes. A keyboard interruption resumes
from the last completed checkpoint; up to one validation interval can be lost.
If setup was interrupted before the first checkpoint, `--resume` can restart
from the original seed when only matching metadata and an optional checkpoint
temporary file exist.
The source, cache SHA-256, training configuration, PyTorch version, and device
must match on resume. This intentionally refuses silent experiment changes.
Exact interrupted/uninterrupted equivalence is tested on CPU; no cross-device
bitwise reproducibility claim is made.

Reported training time is accumulated through the final checkpoint snapshot;
it excludes learning-curve plotting, test evaluation, and that snapshot's own
serialization. It is an operational runtime estimate, not a kernel benchmark.

The learning curve distinguishes:

- `train_window_bce`: pixel-weighted mean of augmented training minibatch losses
  since the preceding validation;
- `train_probe_bce`: loss on 256 fixed, unaugmented training rows;
- `validation_bce`: loss on the entire unaugmented validation split.

Probe row IDs and all split cone IDs are saved in `metadata.json`. Test pixel
values are not accessed while fitting, scheduling, or deciding when to stop.

## Outputs

All large artifacts remain under the ignored run directory:

- `metadata.json`: configuration, normalization, data/source fingerprints,
  split IDs, and train-probe rows.
- `last.pt`: authoritative resumable state.
- `best.pt`: inference checkpoint compatible with
  `ebm21cm.train_recurrent.load_checkpoint`.
- `metrics.jsonl`, `learning_curves.png`: training/validation history.
- `results.json`: exact stop reason, criterion status, best step, rates,
  examples/epochs seen, runtime, and (only after plateau) held-out metrics.
- `test_predictions.npz`, `examples.png`: generated only after plateau.

The tests cover plateau transitions, minimum-LR waiting, cumulative small
improvements, budget stops without test access, strict resume checks, and exact
CPU resume across a partial validation interval.

## Completed run: manual stop at step 8,200

The user requested wrapping up the run before the automatic stopping rule
fired. Finalization used the last completed checkpoint at **step 8,200**, which
was also the validation-selected best checkpoint. No further optimizer steps
were taken. The report records `stop_reason: "user_requested_stop"` and
`criterion_met: false`, rather than claiming convergence.

Training completed 131,200 example presentations, approximately **40.20 epochs**,
in **1,180 seconds (19.7 minutes)** of recorded training time. Three learning-rate
reductions had taken place, leaving the LR at **0.000375**, still above the
predefined minimum of 0.00003.

Validation BCE decreased from **0.66202** initially to **0.007491**. For context,
on this same dataset it was 0.048656 at step 400. The final fixed training-probe
BCE was **0.006854**. Learning had slowed considerably, but validation was still
improving: the last two checks, at 8,000 and 8,200, both established new bests.

### Held-out results

At the user's request to finalize, the selected checkpoint was evaluated on
the previously uninspected test fields. This was an explicit manual evaluation,
not a change to the automatic trainer's plateau-gated test policy.

| Metric | All 416 test slices | 166 mixed-phase test slices |
|---|---:|---:|
| x_HI RMSE | 0.05148 | 0.07668 |
| Pixel accuracy at threshold 0.5 | 99.626% | 99.173% |
| Ionized-mask IoU | 0.99451 | 0.98507 |
| Slice-mean absolute error | 0.00219 | 0.00469 |
| Intermediate pixels, 0.1 < x_HI < 0.9 | 1.058% | 2.390% |

Test cone IDs: `20, 24, 25, 51, 76, 85, 86, 93, 103, 117, 121, 124, 125`.
Truth is binary in this toy task; predictions were not hard-thresholded for
RMSE, slice means, or example figures.

The final artifacts are under `runs/rhizome_long_seed1/fit/`:

- `best.pt`: inference checkpoint at step 8,200.
- `last.pt`: unchanged optimizer/RNG/data-order state for resumption.
- `results.json`: manual-stop status and held-out metrics.
- `learning_curves.png`, `examples.png`, `test_predictions.npz`.

The model learns the synthetic mapping substantially better with longer
training. The dataset also differs from the earlier 40-cone pilot, so those
test RMSEs are not a controlled short-versus-long comparison. These results do
not establish real 21cmFAST accuracy, calibrated stochastic generation, or
fixed-point convergence. The test set has now been inspected; further model
development using these results would require another untouched holdout for
a confirmatory assessment.

All **89 tests** passed before this run, and cache/source hashes were verified
against the checkpoint before final evaluation.

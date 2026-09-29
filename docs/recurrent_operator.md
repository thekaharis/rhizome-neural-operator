# Recurrent local–Fourier field updates

This guide documents the **original additive baseline**. For the integrated,
state-dependent interaction operator now selected by default, see
[the rhizome guide](rhizome_operator.md). The explicit commands below continue
to run the original comparison.

## What is implemented

`ebm21cm/model/recurrent.py` implements `RecurrentFNO2d`, separate from the EBM.
The model represents every cell with a vector-valued hidden state. Each update
uses the previous state of the **whole slice**, not a sequential traversal or
parent–child hierarchy:

```text
forcing = lift(density bands) + embed(redshift, simulation parameters)
h_0     = tanh(forcing)
h_{t+1} = (1 - dt) h_t
          + dt tanh(pointwise(h_t) + local(h_t) + Fourier(h_t) + forcing)
logits  = decode(h_T)
x_HI    = sigmoid(logits)
```

- `local`: circular spatial convolution, for nearby interactions and fronts.
- `Fourier`: learned mixing of retained spatial Fourier modes, for nonlocal
  communication across the periodic plane.
- Shared update parameters across iterations: cells exchange **states** during
  inference; training updates the shared parameters.
- No spatial downsampling or positional coordinates. The operator respects
  arbitrary transverse rolls, not only stride-aligned rolls.
- Redshift is conditioning, not the recurrence index. Iterations are neither
  physical time evolution nor separate LOS slices.
- For `0 < dt <= 1`, the hidden states remain bounded by one. This is **not**
  a proof of convergence, contraction, or physical stability.

The Fourier weights and pointwise layers can be applied on different supported
grid sizes. That does not establish resolution-independent physical accuracy:
the local stencil spans cells, so its physical footprint changes when cell
size changes. Fourier mode interpretation likewise assumes a consistent
physical domain. Cross-resolution use needs its own validation.

## Scope of this first experiment

The training command predicts only `x_HI`, using binary cross entropy on logits.
It accepts binary or fractional neutral-fraction targets in `[0,1]`. It uses
the existing density normalization and scalar conditioning from the slice
cache; the other target channel, `T_b`, is not trained here.

This deliberately isolates whether recurrent local/global interaction can learn
the mapping. **It does not solve conditional distribution learning.** There is
no latent-noise input or sampling distribution in this experiment. With partial
conditioning, BCE can still learn a hedged conditional mean. Calling an output
sharp does not establish calibration or correct uncertainty.

The existing toy lightcones have nonlocal threshold-over-scales ionization
patterns. Their full 3-D generation is deterministic given their underlying
random field and parameters, but the slice cache retains only partial LOS
context. The toy is a pipeline/architecture test, not 21cmFAST physics.

## Reproduce the toy experiment

Use an environment with this project's dependencies, for example:

```bash
conda activate FNO_env   # use your own environment name
pip install -e '.[test]'

python -m ebm21cm.data.toy \
  --out runs/recurrent_toy/lightcones --n 40 --H 32 \
  --cell 4 --z-lo 6 --z-hi 12 --seed 0

python -m ebm21cm.data.cache build \
  --data runs/recurrent_toy/lightcones \
  --out runs/recurrent_toy/slices.h5 --slices-per-cone 16

python -m ebm21cm.train_recurrent \
  --cache runs/recurrent_toy/slices.h5 \
  --run-dir runs/recurrent_toy/experiment \
  --variants recurrent untied pointwise \
  --steps 400 --batch-size 16 --width 16 --modes 4 \
  --updates 4 --step-size 0.5 --lr 0.003 --seed 0 \
  --val-every 100 --threads 2 --device cpu
```

Use a new experiment directory for each run; existing outputs are never
silently overwritten. CPU is the default because the complex FFT implementation
is not validated on MPS. CUDA can be selected explicitly when available.
The same command accepts an existing real-data slice cache.

Splits and normalization come from the shared cache. The runner checks that
train, validation, and test are nonempty and contain disjoint cone IDs.
Checkpoints are selected by validation BCE only. The test set is evaluated
after fitting, not used for optimizer updates or checkpoint selection.

### Comparisons

| Variant | Update weights | Spatial communication |
|---|---|---|
| `recurrent` | shared | local + Fourier |
| `untied` | independent at each step | local + Fourier |
| `local_only` | shared | local |
| `spectral_only` | shared | Fourier |
| `pointwise` | shared | none beyond the supplied density bands |

The untied model has the same width and number of updates, approximately
matching forward work, **not parameter count**. Counts include both real and
imaginary components of complex parameters. Identical seeds and batch/augmentation
ordering support reproducibility in the documented CPU environment, but a single
seed is not a significance test. CUDA determinism is not enforced, and results
need not be bitwise identical across hardware or PyTorch versions.
Training time includes validation and checkpoint writes; prediction time
includes data loading, so neither is a pure kernel benchmark.

### Outputs

The experiment directory contains:

- `metadata.json`: configuration, normalization, exact split cone IDs, cache
  path, device, PyTorch version, Git revision, and source-file SHA-256 hashes
  (so uncommitted model/trainer changes are distinguishable).
- `results.json`: held-out RMSE, ionized-mask IoU, pixel accuracy, slice-mean
  error, and hedging. Metrics are reported for all slices and separately for
  mixed slices (`0.05 < mean(x_HI_truth) < 0.95`), so trivial phases cannot
  hide performance around fronts. A constant training-mean baseline is included.
- `examples.png`: density, truth, and unthresholded predictions on the same
  mixed-phase test slices.
- `<variant>/best.pt`: validation-selected model weights, constructor arguments,
  training step, and normalization statistics.
- `<variant>/metrics.jsonl`: training and validation history.
- `<variant>/test_predictions.npz`: physical `x_HI` predictions and truth, cache
  row IDs, cone IDs, and redshift.
- `<tied-variant>/rollout.png`: decoded predictions at different update counts.
  `results.json` also records held-out iteration sweeps and hidden-state update
  magnitudes. Longer rollout is diagnostic, not a test-time tuning recommendation.

## Observed toy pilot

The command above was run on CPU with PyTorch 2.9.1, two threads, training seed
0, 400 optimizer steps, and four field updates. The cache contained 640
32×32 slices: 512 from 32 training cones, 64 from four validation cones, and
64 from four test cones. Test cone IDs were **4, 5, 16, 37**. Of the test
slices, 25 were mixed-phase by the criterion above.

All three learned variants selected step 400 using validation BCE. The recurrent
model's validation BCE decreased from **0.74394 to 0.04442**.

| Model | Real parameter count | Test RMSE, all | Test RMSE, mixed | Ionized IoU, mixed |
|---|---:|---:|---:|---:|
| Recurrent local + Fourier | 15,857 | 0.13464 | 0.20654 | 0.89572 |
| Untied local + Fourier | 61,265 | 0.11299 | 0.17391 | 0.92743 |
| Recurrent pointwise | 993 | 0.17360 | 0.26303 | 0.83067 |
| Constant training mean | 0 | 0.44665 | 0.50056 | 0.56594 |

The recurrent model learned useful spatial structure and outperformed the
pointwise model, but the **untied model was more accurate** in this pilot.
The recurrent model used about 26% as many parameters as the untied baseline,
with similar observed training time (41.5 s versus 42.1 s, including validation).
The pointwise comparison is not parameter-matched either; it does not isolate
the effect of communication from the effect of capacity.

### What happens when updates are repeated?

The **same** validation-selected recurrent checkpoint gave:

| Updates at inference | Test RMSE, all | Test RMSE, mixed |
|---|---:|---:|
| 1 | 0.27786 | 0.37397 |
| 2 | 0.20889 | 0.30387 |
| 4 (training horizon) | 0.13464 | 0.20654 |
| 8 | 0.26874 | 0.41747 |

Hidden states remained bounded (maximum absolute value 0.99998 in the rollout
probe), but doubling the update count **degraded accuracy substantially**.
The network learned a finite-horizon computation, not an equilibrium solver.
Variable-horizon training or explicitly contractive updates would be separate
experiments, selected on validation data rather than this test sweep.

Predictions were not hard-thresholded for RMSE or example images. On mixed
slices, 18.5% of recurrent predictions were intermediate (`0.1 < x_HI < 0.9`),
whereas the toy truth was binary. Thus this run does not solve the earlier
conditional-mean/hedging problem.

These are results from **one seed and four test simulations**, not evidence
of statistical superiority or physical accuracy. The 62-test suite passed,
including the legacy EBM pipeline and new operator/training tests. No GPU or
real 21cmFAST accuracy benchmark was performed.

### Load a checkpoint

```python
from ebm21cm.train_recurrent import load_checkpoint
from ebm21cm.data.cache import SliceDataset

model, checkpoint = load_checkpoint("runs/recurrent_toy/experiment/recurrent/best.pt")
ds = SliceDataset("runs/recurrent_toy/slices.h5", "test", checkpoint["stats"])
item = ds[0]
xhi = model(item["cond"][None], item["scalars"][None]).sigmoid()
logits, states = model(
    item["cond"][None], item["scalars"][None],
    n_steps=8, return_states=True,
)
```

## Tests and interpretation

```bash
python -m pytest -q
```

The new tests exercise gradients through both communication paths, parameter
sharing, periodic roll equivariance, nonlocal influence, grid-size handling,
bounded rollout, invalid configuration, checkpoint round trips, and the complete
train/validation/test path. The legacy EBM tests remain applicable.

The next scientific steps are multiple-seed comparisons, local/Fourier
ablations, real 21cmFAST validation, and a distribution-learning objective if
multiple realizations are required. Independent slice updates still do not
produce a coherent 3-D lightcone.

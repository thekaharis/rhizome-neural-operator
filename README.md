# ebm-21cm

Energy-based generative model for 21cmFAST reionization fields: given the matter
density around a lightcone slice, its redshift and the 11 simulation parameters,
**sample** `(x_HI, T_b)` transverse slices and **score** candidate fields with an
explicit energy.

It is independent of `fno-21cm` (no imports or shared files), but reads the same
raw lightcones.

## Rhizome neural operator

Alongside the EBM, `ebm21cm.model.rhizome.RhizomeOperator2d` makes the recurrent
cell-to-cell interaction itself a **state-dependent integral operator**.
Learned source and receiver gates surround a periodic interaction kernel;
Fourier coefficients parameterize that kernel, and an FFT evaluates the
connections efficiently. There is **no separate additive FNO or local-convolution
message branch**. All cells update synchronously with shared weights.

The original additive local + Fourier model, `RecurrentFNO2d`, remains available
as a baseline. Both architectures have an untied, same-width/update-count
comparison, and older additive checkpoints still load.

`python -m ebm21cm.train_recurrent` trains this model on the same slice cache,
selects checkpoints using validation cones, and evaluates held-out test cones.
This first experiment is **deterministic x_HI regression**, not a calibrated
stochastic generator or a coherent 3-D lightcone model. It leaves the EBM
training and sampling commands below unchanged.

See [the rhizome guide](docs/rhizome_operator.md) for the interaction equations,
reproducible toy comparison, results, and limitations. The
[original recurrent-operator guide](docs/recurrent_operator.md) documents the
additive baseline and its first pilot.

For long training, `python -m ebm21cm.train_rhizome` adds validation-driven
learning-rate reductions, an explicit plateau stopping rule, and resumable
checkpoints. See [the long-training protocol](docs/rhizome_long_training.md).

## Why

A deterministic regressor trained with a pointwise loss predicts the conditional
*mean*. Where the position of an ionization front is uncertain, that mean is a
ramp. It is sharp nowhere, and no pointwise output map can fix it (the
`fno-21cm` contrast-map study reached this conclusion: edges are *hedged in
position*, not blurred). A generative model avoids the averaging. Each sample is
one plausible sharp configuration, and the ensemble mean recovers the regressor.

## Model

A U-Net `D(x; σ | c)` with EDM preconditioning (Karras et al. 2022) defines a
scalar energy per sample:

```
E(x; σ | c) = ‖x − D(x; σ | c)‖² / (2σ²)          s(x) = −∇ₓE          D_eff = x + σ² s(x)
```

- **Conservative score.** The score is the exact gradient of a scalar, so the
  learned field is conservative by construction.
- **Training.** Denoising score matching with the EDM loss on `D_eff`. This
  differentiates through the network twice, costing about 2–3× a plain score
  model.
- **Energy values.** `E` is a number you can compare between candidate fields
  that share the same `(σ, c)`.
- **Metropolis sampling.** The explicit energy allows Metropolis-adjusted
  Langevin (MALA) corrections during sampling.
- **Reduction.** If the Jacobian of `D` is small, the score reduces to
  `(D − x)/σ²`, and `D` is then the denoiser itself. Salimans & Ho (2021),
  *Should EBMs model the energy or the score?*, discuss this family of
  parameterizations.

### Inputs and targets

- **Conditioning `c`:** 13 channels of LOS density *bands* around the target
  slice:
  - single slices within ±3 native cells;
  - band means over `[4,8)`, `[8,16)` and `[16,32)` cells on each side, about
    ±46 Mpc at the cluster's 1.43 Mpc cells.

  Density enters as standardized `log(1+δ)`. The conditioning also includes
  standardized `log(1+z)` and the 11 parameters mapped to [−1, 1] via AdaGN.
- **Targets:** `2·x_HI − 1` and standardized `T_b`.
- **Normalization:** from training cones only.
- **Periodicity:** every convolution pads circularly, because transverse planes
  come from periodic boxes. The energy is therefore exactly invariant to
  transverse rolls by multiples of 4 (tested).
- **Augmentation:** training uses the transverse dihedral group plus random
  rolls.

### Noise schedule

Training draws log σ ~ N(−0.4, 1.4), covering about 0.04 ≤ σ ≤ 11 at ±2 std.
EDM's image defaults, N(−1.2, 1.2), almost never train σ > 3. That is exactly
the regime where a conditional sampler decides the *global* layout, i.e. which
regions end up neutral and which ionized.

On toy data those defaults gave a near-perfect denoiser at σ ≤ 1 but x_HI RMSE
0.58 at σ = 10. The resulting samples were sharp but ignored the conditioning:
the slice-mean x_HI was off by 0.2 on average. Changing where sampling starts
(σ_max 40 → 5) did not help; training at high σ is what matters. Sampling
starts at `train.sample_sigma_max = 10`, which lies inside the trained range.

On the same toy budget (5k steps), the new schedule lowered the fixed-grid
validation loss at every checkpoint, e.g. 0.100 vs 0.126 at step 2500. It also
put bubbles in the right places. The deterministic ODE sampler, however, still
left pixel-level noise.

A trajectory trace shows why. The state stays on-manifold
(|x − D|/σ ≈ 1) down to σ ≈ 1. Below that, the ratio falls to about 0.65:
pixel decisions go wrong where the denoiser is weakest, and the ODE cannot
undo them.

**Use stochastic sampling.** On 16 toy validation slices:

| sampler | x_HI RMSE | hedged px | slice-mean error |
|---|---|---|---|
| ODE | 0.35 | 0.45 | 0.17 |
| `--churn 40` | **0.23** | 0.19 | **0.06** |
| `--churn 40` + `--mala-steps 4` | 0.26 | **0.13** | 0.07 |

### Sampling

The sampler is EDM Heun on the probability-flow ODE, with optional churn. With
`--mala-steps k`, each step with `σ ≤ --mala-sigma-max` is followed by `k` MALA
moves that target the model's own `p_σ ∝ exp(−E)`. Step sizes adapt towards a
0.574 acceptance rate, so the corrector repairs predictor error. It is not a
certified reversible chain.

## Data

`ebm21cm.data.lightcone.Lightcone` reads both on-disk schemas:

| schema | layout | redshifts |
|---|---|---|
| `raw_v2` (cluster, `raw_lightcone_v2.0`) | `/lightcone/<field>`, `/params` attrs, `box_len_mpc` | stored |
| `native` (`py21cmfast.LightCone.save`) | `/lightcones/<field>`, `/InputParameters/*` | recovered from `lightcone_distances` |

Native files store no redshift axis. `cosmology.FlatLCDM` reproduces
py21cmfast 4.x's astropy cosmology: Planck15 with T_cmb = 2.7255 K,
m_ν = (0, 0, 0.06) eV, and the file's H0, Ωm and Ωb, with Neff = 3.044. It
matches astropy to 1e-6 in z. Ignoring the massive neutrinos shifts z by about
0.5 at z ≈ 34. The LOS axis is always presented in increasing-z order.

### The slice cache

The cache (`ebm21cm.data.cache`) has one row per (cone, native LOS index):

- **Stored per row:** density bands, `x_HI`, `T_b`, `z`, parameters, cone id and
  split, in physical units, float16 by default. Normalization lives in `/stats`.
- **Splits** are assigned per simulation (seed 42, 80/10/10), before sharding.
- **No invented padding:** indices whose bands would leave the cone are never
  sampled.
- **Rejected cones:** those with non-finite sampled slices, or a transverse
  geometry different from the first file, are skipped and listed in the
  `skipped` attribute.

## Quickstart (local, toy data)

The toy generator writes real-schema lightcones with sharp binary bubbles from a
threshold-over-scales rule. It exists to test the pipeline, not as physics.

```bash
conda activate fno-env        # or any env with torch, h5py, scipy, matplotlib
pip install -e .              # or: export PYTHONPATH=$PWD
python -m ebm21cm.data.toy --out /tmp/toy/lc --n 80 --H 32
python -m ebm21cm.data.cache build --data /tmp/toy/lc --out /tmp/toy/slices.h5 --slices-per-cone 32
python -m ebm21cm.train --cache /tmp/toy/slices.h5 --run-dir /tmp/toy/run --config configs/toy.json
python -m ebm21cm.sample --run-dir /tmp/toy/run --n-rows 32 --n-samples 8 --out /tmp/toy/run/samples.h5
python -m ebm21cm.evaluate --samples /tmp/toy/run/samples.h5
```

## Cluster workflow

Submit from the repository root. `slurm/_env.sh` activates `$CONDA_ENV`
(default `fno-env`) and sets `WORK=/pfs/10/work/hd_id260-fno_training`.

```bash
mkdir -p logs
AID=$(sbatch --parsable slurm/build_cache.sbatch)           # 33 shards, CPU
sbatch --dependency=afterok:"$AID" slurm/merge_cache.sbatch  # + train-only stats
sbatch --export=ALL,RUN_DIR=runs/base slurm/train.sbatch     # resubmit to resume
sbatch --export=ALL,RUN_DIR=runs/base slurm/sample_eval.sbatch
```

- **Build:** `build_cache.sbatch` reads `LIGHTCONES` (default `$WORK/data/data`)
  with `PATTERN=21cmfast_11d_sample*.h5`, `SLICES_PER_CONE=8` and z in
  5.001–24.97.
- **Size:** 6,600 cones × 8 slices × 15 stored 140² maps in float16 comes to
  about 31 GB.
- **Pilot first:** try the build on a subset with `--limit` on the command line
  before the full array.
- **Overrides:** `train.sbatch` accepts `CONFIG` and `OVERRIDES` (e.g.
  `OVERRIDES="train.lr=1e-4 model.ch=96"`).

## Evaluation

`ebm21cm.sample` draws `M` samples per held-out slice, spread evenly in z, and
records energies at several probe noise levels σ. `ebm21cm.evaluate` then
reports, overall and per z-bin:

| metric | what it tells you |
|---|---|
| `*_rmse_sample` vs `*_rmse_mean` | a single sample is *expected* to have higher RMSE than the ensemble mean (≈√2 × at best); the mean plays the deterministic regressor |
| `*_crps` | proper score for the ensemble; lower is better |
| `*_spread_skill` | ≈ 1 when the ensemble spread is calibrated; < 1 over-confident |
| `xhi_hedge_{truth,samples,mean}` | fraction of pixels with 0.1 < x_HI < 0.9; samples should match the truth, the mean should not |
| P_pred/P_true, r(k) | samples should keep power at bubble scales where the mean loses it |
| `bsd_w1_*` | Wasserstein distance of log MFP ionized-region sizes vs truth |
| `dE_*` | [E(candidate) − E(truth)] per pixel at the same σ and conditioning |

Figures: `examples.png`, `spectra.png`, `bubble_sizes.png`, `energy.png`.

### Energy probe on other models' outputs

`--candidates preds.h5` (datasets `row`, `xhi`, `tb`, with `row` = cache row id)
adds any external prediction, e.g. an `fno-21cm` output resampled to the same
slice, to the energy probe. `dE > 0` means the model finds the truth more
plausible than the candidate. Energies are only comparable at the **same σ and
conditioning**. Clean fields get fixed-seed probe noise at that σ before
evaluation, since `E(·; σ)` is trained on noised inputs.

## Tests

```bash
python -m pytest -q
```

The suite covers:

- **Data:** schema agreement, LOS reversal, py21cmfast cosmology, the cache
  against the source files, train-only stats, and shard/merge equivalence.
- **Model:** score = −∇E by finite differences, per-sample energies and roll
  invariance.
- **Sampling:** determinism and MALA bounds.
- **Metrics:** CRPS against brute force, calibrated spread/skill, white-noise
  P(k), and MFP on a disk.
- **End to end:** train → resume → sample → evaluate on toy data.

## Limitations

- **2-D slices.** LOS context beyond ±32 cells is not seen, and neighbouring
  slices are sampled independently, so a stack of samples is not a consistent
  3-D cone.
- **Plain (no-RSD) fields only.**
- **Single GPU.** DDP is untested with the double backward.
- **fp16 storage.** The cache stores fp16 (T_b resolution ~0.03 mK at 30 mK);
  use `--dtype float32` if that matters.
- **Toy data.** It checks mechanics and the sharpness claim qualitatively; it
  says nothing about accuracy on 21cmFAST.

# Operator-defined rhizome interactions

For a longer run with validation-driven learning-rate reductions, explicit
stopping criteria, and resumable checkpoints, see
[the long-training protocol](rhizome_long_training.md). The comparison below
remains the original short architecture pilot.

## The architecture

`ebm21cm/model/rhizome.py` implements `RhizomeOperator2d`. Unlike the original
additive `RecurrentFNO2d`, it does not add a Fourier message to a separate local
message. The connections between cells **are** a state-dependent neural
integral operator:

$$
m_i^t = \frac{1}{N}\sum_j
\operatorname{diag}(A_\theta(h_i^t,c_i))\,
\kappa_\theta(r_i-r_j)\,
\operatorname{diag}(B_\theta(h_j^t,c_j))\,
V_\theta(h_j^t,c_j).
$$

- $B$ is a learned source transmission gate.
- $A$ is a learned receiver response gate.
- $\kappa$ is a learned periodic, channel-mixing spatial kernel.
- $V$ constructs the transmitted value from state and conditioning.
- $N=H W$ gives uniform quadrature for normalized measure on the periodic box.

The diagonal gates lie in `(0,2)`. They are recomputed from the previous state
and the fixed conditioning at **every iteration**. The effective connections
therefore change as the field evolves, although the parameters themselves are
updated only during training.

The only cross-site communication is this one interaction:

```text
c = lift(density bands) + embed(redshift, simulation parameters)
h = tanh(c)
repeat T times, reusing the same weights:
    source, receiver = split(2 * sigmoid(gates(concat(h, c))))
    value = value_projection(concat(h, c))
    message = receiver * IFFT(R * FFT(source * value))
    proposal = tanh(pointwise_projection(concat(h, message, c)))
    h = (1 - dt) * h + dt * proposal
x_HI = sigmoid(decode(h))
```

Here `R` parameterizes the Fourier coefficients of $\kappa$. The FFT is an
efficient evaluator of the connections, not a separate predictor. Matching
orthonormal forward/inverse FFTs already produce the `1/N` quadrature; an
additional area factor would incorrectly shrink messages as resolution grows.
The real DC and conjugate frequency constraints ensure a real-valued kernel.

The source/receiver factors are channel-wise, not arbitrary pair-specific
matrices. This factorization avoids storing an `N × N` edge tensor. It is more
restricted than a general kernel depending jointly on every pair of states.
Interactions may be signed and asymmetric: neither reciprocal edges,
conservation laws, nor radiative-transfer physics are enforced.

## Recurrence and operator assumptions

- Updates are synchronous: all messages use the whole previous state, without
  a cell visitation order. All spatially mixing operations live inside the
  integral interaction; lifting, proposal, and decoding are pointwise.
- `rhizome` reuses one interaction/update block. `rhizome_untied` gives each
  iteration independent parameters as a feed-forward comparison.
- For `0 < dt <= 1`, the hidden state remains bounded by one. This is not a
  convergence or contraction guarantee.
- There are no cell-index parameters, coordinates identifying special cells,
  spatial pooling, or cell-sized convolution stencils. Transverse rolls commute
  with the model up to floating-point error.
- Fourier modes refer to the same normalized periodic domain. Different grids
  should represent the **same physical box** for resolution-transfer experiments.
  Both grid dimensions must be at least `2*modes`; changing physical box size
  without providing that information is not supported as a physical invariance.
- Resolution-consistent quadrature does not imply accurate transfer: nonlinear
  state/gate operations can alias on coarse grids, and learned predictions
  still require cross-resolution validation.

The shared rollout implementation preserves the original additive model's state
dictionary names and initialization order. Checkpoints carry an `architecture`
tag; old checkpoints without one load as the additive baseline.

## Reproduce the toy comparison

Use a Python environment containing the project dependencies:

```bash
# Generate the same synthetic lightcones and cache as the original pilot.
# Skip these two commands if this cache already exists.
python -m ebm21cm.data.toy \
  --out runs/recurrent_toy/lightcones --n 40 --H 32 \
  --cell 4 --z-lo 6 --z-hi 12 --seed 0
python -m ebm21cm.data.cache build \
  --data runs/recurrent_toy/lightcones \
  --out runs/recurrent_toy/slices.h5 --slices-per-cone 16

python -m ebm21cm.train_recurrent \
  --cache runs/recurrent_toy/slices.h5 \
  --run-dir runs/rhizome_toy/experiment \
  --variants rhizome recurrent rhizome_untied \
  --steps 400 --batch-size 16 --width 16 --modes 4 \
  --updates 4 --step-size 0.5 --lr 0.003 --seed 0 \
  --val-every 100 --threads 2 --device cpu
```

Choose a new run directory for a repeat. When `--variants` is omitted, the
runner compares `rhizome` and `recurrent`. The original `untied`, `local_only`,
`spectral_only`, and `pointwise` variants remain available as additive-model
baselines, not alternative branches inside the rhizome.

Training uses the existing cache's density normalization, redshift, and 11
simulation parameters. It fits **deterministic x_HI** with BCE; it does not
train `T_b` or introduce latent noise. Checkpoints are selected on validation
cones, then evaluated on the full test split. Splits must have disjoint cone
IDs. The existing toy generator is not a physical 21cmFAST simulation.

The runner writes configuration/source hashes, training logs, checkpoints,
physical-unit predictions, examples, and iteration sweeps; see the
[shared output description](recurrent_operator.md#outputs). Large outputs
remain under ignored `runs/`, not in version control.

## Observed comparison

The command above ran with PyTorch 2.9.1 on CPU, two threads, training seed 0.
It used the same 640 synthetic 32×32 slices as the earlier pilot:
512 training slices from 32 cones, 64 validation slices from four cones,
and 64 test slices from cones **4, 5, 16, 37**. Twenty-five test slices were
mixed-phase (`0.05 < mean(x_HI_truth) < 0.95`).

| Model | Real parameters | Selected step | Test RMSE, all | Test RMSE, mixed | Ionized IoU, mixed |
|---|---:|---:|---:|---:|---:|
| Integrated rhizome, tied | 15,617 | 300 | 0.12880 | 0.19609 | 0.90427 |
| Original additive recurrent | 15,857 | 400 | 0.13464 | 0.20654 | 0.89572 |
| Integrated rhizome, untied | 60,305 | 400 | 0.11732 | 0.18136 | 0.91982 |

The tied rhizome's validation BCE decreased from 0.65125 to 0.03460 at step 300;
step 400 was worse on validation and was not selected. The additive baseline
reproduced its earlier predictions/metrics. The integrated model improved the
reported test errors modestly at a similar parameter count, but was still
less accurate than its larger untied counterpart.

Observed training times, including validation/checkpoint writing, were 46.0 s
for tied rhizome, 37.2 s for additive recurrent, and 47.9 s for untied rhizome.
Comparable parameter counts do not make these kernels compute-identical.

The same tied rhizome checkpoint gave the following iteration diagnostic:

| Updates | Test RMSE, all | Test RMSE, mixed |
|---|---:|---:|
| 1 | 0.18653 | 0.25966 |
| 2 | 0.14213 | 0.20481 |
| 4 (training horizon) | 0.12880 | 0.19609 |
| 8 | 0.17097 | 0.26494 |

Extra iterations still degraded accuracy, despite bounded hidden states. On
mixed test slices, 13.4% of predictions lay between 0.1 and 0.9, versus 18.5%
for the additive recurrent model and 0% for the binary toy truth. Sharper
predictions are not necessarily more accurate: at eight updates the rhizome
hedged less but had higher error.

These are a **single-seed architecture pilot**, reusing the earlier test cones,
not a fresh confirmatory test or evidence of general superiority. There is no
stochastic calibration, real 21cmFAST accuracy, 3-D coherence, or fixed-point
convergence claim.

## Load and inspect the model

```python
import torch
from ebm21cm.data.cache import SliceDataset
from ebm21cm.train_recurrent import load_checkpoint

model, checkpoint = load_checkpoint("runs/rhizome_toy/experiment/rhizome/best.pt")
ds = SliceDataset("runs/recurrent_toy/slices.h5", "test", checkpoint["stats"])
item = ds[0]
with torch.no_grad():
    logits, states = model(item["cond"][None], item["scalars"][None], return_states=True)
    xhi = logits.sigmoid()
```

For research diagnostics, `model.cells[0].interaction.factors(state, forcing)`
returns the source gates, receiver gates, and values. `forcing` is the lifted
density plus embedded scalars, identical to that used for the rollout.

## Verification

`python -m pytest -q` passed **74 tests** after this implementation.
New coverage includes:

- Output and gradient agreement with an **independent dense all-pairs
  Fourier-series quadrature**, on even and odd grids.
- Source and receiver factors responding to state and conditioning, and
  independently controlling transmission/reception.
- Exactly one spatial interaction path: zeroing the kernel removes all
  influence between distinct cells.
- Shared synchronous updates, gradients throughout the model, bounded rollout,
  periodic translation equivariance, and constant-field resolution consistency.
- Execution of every untied block, both model families' training/checkpoint
  round trips, and loading old additive checkpoints without an architecture tag.

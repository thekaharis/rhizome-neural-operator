# Multi-pathway rhizome interaction

`MultiPathIntegral2d` (in `ebm21cm/model/rhizome.py`) replaces the single dense spectral interaction with R pathways:

```
m_r = A_r(h, c) * kappa_r( B_r(h, c) * V_r(h, c) ),   r = 1..R
h'  = (1 - dt) h + dt tanh( P [h, m_1, ..., m_R, c] )
```

Each pathway has its own gates, values and Fourier kernel. A kernel consists of `kernel_groups` dense channel
blocks: 1 gives dense C x C, and `width` gives a depthwise kernel with one global filter per channel. The messages
are concatenated, not summed. The pointwise proposal `P` already acts as the signed output mixing `W_o`: a
separate linear `W_o` in front of it would multiply into `P` and add no expressiveness, so the model omits it.
`pathways=1, kernel_groups=1` is the existing `StateDependentIntegral2d` (tested), and old checkpoints still
load unchanged.

## What R means

When pathways share nothing, R depthwise pathways of C channels are exactly one depthwise pathway of R*C
channels. Concatenation makes the pathway label arbitrary. R therefore **expands the message width** (an
inverted bottleneck around the spectral transport) while the hidden state stays at width C. "Two
context-selected spatial laws" only becomes a distinct hypothesis when something is tied across pathways
(`share_pathways gates|kernels`), or when the comparison widens the state instead of the message (F below).

The dense R=1 model can already route between sites through different spatial laws: it has C^2 filters
kappa_ij, and the per-channel gates choose among them. The depthwise variants have far fewer filters (C or RC).

## Runs and cost

All runs use the hyperparameters of `runs/rhizome_mf2d_w64_structured_all` (w64, modes 24, T=6, batch 32,
lr 2e-3 with plateau reductions to 2e-5) on the all-cones cache `rhizome_mf_slices_all.h5`, with one seed each.
Times are measured A100 training steps (elapsed time / steps, validation included). Memory is the autograd-saved
activations at batch 32 plus weights, gradients and Adam, counted on the CPU (the GPU peak is somewhat higher).

| run | flags | params | ms/step | train memory |
|---|---|---|---|---|
| A dense R=1 (existing `..._all`) | | 9.09M | 152 | 14.9 GiB |
| B depthwise R=1 | `--kernel-groups 64` | 0.19M | 143 | 14.2 GiB |
| C depthwise R=2 | `--pathways 2 --kernel-groups 64` | 0.36M | 244 | 23.2 GiB |
| D R=2 shared gates | C + `--share-pathways gates` | 0.34M | 224 | 21.4 GiB |
| E R=2 shared kernels | C + `--share-pathways kernels` | 0.22M | 234 | 23.2 GiB |
| F state-widened R=1 | `--width 128 --kernel-groups 128` | 0.46M | 309 | 28.3 GiB |

Spectral parameters fall from C^2 M to RCM (9.05M -> 0.14M at R=1), but time and memory do not. The FFT is about
10-15 ms of a ~150 ms step: the compact R=1 kernel with no FFT at all ran at 139 ms. Cost follows the per-cell
work: gates, values, the proposal and the elementwise gating on every message channel, over 6 updates and the
backward pass. Two pathways double the message channels and so cost ~1.6x in time and memory. F has the same
message width as C with a doubled state, so it is the "only width helps" control. A parameter-matched dense
R=1 would need width ~13 and would not be informative.

The parameter saving pays off where the kernel multiplies. Dense at all 2-D modes would be ~80M parameters.
The 3-D model at modes (36, 36, 24) would be ~560M, which is why it already uses `rank=48`. Kernels conditioned
on z or the astrophysical parameters, and untied updates, also become affordable.

**Collapse check.** `kernel_overlap(model.cells[0].interaction.kernel)` returns the |cos| Gram matrix of the
learned filters and their effective rank. Rank near 1 means identical kernels. Gate collapse requires data: compare
`factors()` across pathways on validation slices (for example by extending `mechanism_mf.py`).

## Disc kernel applied by FFT

`--disc-pathways D --disc-radius R` makes the last D pathways use `DiscFFTConv2d`. This is a depthwise kernel
whose weights live on a disc of R cells, as in `LocalConv2d`. The weights are placed into a full-grid kernel image
and applied by FFT across every mode. The result equals the direct periodic sum (tested), but the cost does not grow
with R, and the kernel carries every scale down to the grid's finest wavelength. The 24-mode Fourier kernel stops
at about 6 cells. With T updates, information still travels at most T*R cells.

Measured step times show the FFT is about 10-15 ms of a ~150 ms step, so a direct large-R convolution would not save time.

| run | flags | params | compare with |
|---|---|---|---|
| G | `--disc-pathways 1 --disc-radius 8` | 0.058M | B (depthwise spectral, R=1) |
| H | `--disc-pathways 1 --disc-radius 16` | 0.097M | B |
| I | `--pathways 2 --kernel-groups 64 --disc-pathways 1 --disc-radius 8` | 0.228M | C (two spectral pathways) |

## Results

Best validation score (mean normalized MSE of x_HI and T_b; lower is better), single seed:

| run | job | best score | vs A | x_HI RMSE | T_b RMSE (mK) |
|---|---|---|---|---|---|
| A dense | 5572600 | 0.02136 | | 0.0981 | 10.21 |
| B depthwise | 5836707 | 0.02248 | +5.2% | 0.1001 | 10.49 |
| C depthwise R=2 | 5836708 | 0.02056 | -3.7% | 0.0964 | 10.02 |
| D shared gates | 5836709 | 0.02071 | -3.1% | 0.0972 | 10.03 |
| E shared kernels | 5836710 | 0.02035 | -4.7% | 0.0963 | 9.95 |
| F w128 depthwise | 5836711 | 0.02047 | -4.2% | 0.0960 | 10.00 |
| G disc R=8 | 5841517 | 0.02270 | +6.2% | 0.1019 | 10.50 |
| H disc R=16 | 5841518 | 0.02237 | +4.7% | 0.1006 | 10.45 |
| **I spectral + disc8** | 5841519 | **0.02013** | **-5.7%** | **0.0958** | **9.90** |

For reference, the best earlier standard-input models are the two-level unetC (0.02036) and unetB (0.02065).
The dense compact R=8 rank-16 kernel scored 0.02276 at 214 ms/step. G and H ran at 136 and 133 ms/step.

- Dropping dense channel mixing in transport (B) costs 5% with 2% of the parameters.
- Any wider message beats the dense baseline by 3-5%, and how the pathways are tied (C, D, E) matters by under
  2%. F matches them with one pathway on a wider state. The gain is width, not richer routing between
  independent pathways.
- On its own, a local full-resolution disc kernel is no better than the global 24-mode one (H ~ B). Combined
  as two pathways (I), it beats two spectral pathways (C) by 2% at the same cost: global smooth plus local
  sharp transport complement each other.
- A's two seeds agree to 1e-5, but the C/D/E/F spread (~1%) is near the noise for one seed; I's lead needs
  a second seed.

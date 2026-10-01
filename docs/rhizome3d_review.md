# Review of the current 3-D Rhizome implementation

Reviewed 2026-09-30 against root-checkout commit
`fe4ba2b0f1634bfe947658ac9b1a03a9160641dd`, with exact source hashes in
[the numerical results](rhizome3d_audit_results.json). Model and trainer source
were not changed. The attached PDF describes an earlier, separate 2-D/Darcy
worktree; its 2-D guarantees are not automatically guarantees of this extension.

## Result

The FFT implementation agrees with an independent real-space Fourier-series
sum, including input and parameter gradients. The optional factorization is
implemented consistently. However, the default LOS configuration restricts a
short padded-circle kernel to the window, with forced equality of some
opposite signed separations. It is not an unrestricted nonperiodic interaction
on that window. Quadrature, spectral parameter redundancy, anisotropic shape
validation and an external API dependency also need attention before making
3-D operator-transfer or reproducibility claims.

This review separates confirmed implementation errors, modeling restrictions
and scientific validation gaps. It does not infer 3-D predictive accuracy from
passing numerical tests.

## Reproduction and evidence

From the repository root, use a Python environment with PyTorch and the project
dependencies:

```bash
python scripts/rhizome3d_audit.py --out docs/rhizome3d_audit_results.json
python -m pytest -q -p no:cacheprovider tests/test_rhizome3d.py tests/test_rhizome.py tests/test_recurrent.py
```

The optional `--fno-root` argument reads external Python signatures through
AST inspection, without importing that project or accessing its data. The
saved audit also inspected the available sibling `../FNO v3` checkout; its API
need not be the API installed on the cluster.

The independent calculation explicitly sums all retained Fourier series terms
over all physical source/receiver pairs. It uses no FFT, no inverse FFT and no
sampled impulse-response construction. It handles the real-FFT kz=0 projection
and the conjugate partners for positive kz. Twelve cases cover two even/odd
transverse grids, unpadded and padded even/odd transform lengths, and dense and
factorized channel kernels. On this CPU run, maximum forward discrepancy was
`1.11e-15` and maximum gradient discrepancy was `5.00e-16`.

Existing tests also check finite backpropagation, probability outputs,
transverse rolls and checkpointed/non-checkpointed gradient agreement. CUDA,
bf16, full-size GPU memory and real-data training were not exercised here.
The targeted root-checkout test command above passed all 43 tests on CPU with
PyTorch 2.9.1. The 12 additional dense-series cases and diagnostic probes are
saved separately in the audit JSON; they are not counted as pytest cases.

## Findings

### 1. Default LOS padding aliases signed separations (high scientific impact)

In `SpectralConv3d.forward`, messages of length Z are padded to `L=Z+P`, mixed
on that periodic box and cropped. A connection sees a LOS displacement only
modulo L. All physical pair displacements lie in `[-(Z-1), Z-1]`; they are
distinct modulo L only if `L >= 2Z-1`, equivalently `P >= Z-1`.

The trainer defaults to Z=256 and P=64, giving L=320. The connections from
source 0 to receiver 160 and from source 255 to receiver 95 have signed
displacements +160 and -160, but both are residue 160. Both receivers lie
inside the supervised core with halo 32. The audit's nonzero kernel gives the
same response, approximately -0.00625, to both impulses (difference below
`9e-19`). Gates may subsequently change effective strengths; the base kernel
cannot independently represent these signed offsets.

This does not mean opposing physical edge pixels become adjacent: padding
does prevent that immediate identification. A restricted circular kernel is
also a Toeplitz matrix on the cropped window, so it can formally be described
as a linear convolution with a *constrained* signed-offset kernel. The actual
limitation is the imposed equality of different physical offsets, not that
every zero-extended calculation is invalid.

For one update restricted to core receivers, offsets occupy
`[-(Z-halo-1), Z-halo-1]`, so uniqueness requires
`P >= Z-2*halo-1` (191 here). That weaker condition is insufficient for the
whole recurrent rollout: halo states also update and can affect the core in
later rounds. Use P>=255 for all states; P=256 gives a convenient 512-cell FFT.

Recommendation: make the distinction explicit in configuration and metadata.
Use full signed-offset padding for the proposed nonperiodic scientific model,
or deliberately retain P=64 as a periodic-embedding approximation with an
ablation. Increasing padding changes the Fourier period and frequency grid,
so retrain; do not treat it as a harmless correction to existing predictions.
For a comparable physical maximum frequency, consider the new padded extent
when selecting mz. Matching an approximate cutoff does not map all old
frequency coefficients exactly to the new period.

### 2. The boundary test does not detect this aliasing (test gap)

`test_full_los_padding_is_a_linear_convolution` shifts an impulse one cell
away from the last plane and compares translated interior responses. The
audit reproduces that assertion for P=9, P=4 **and P=0**. Translation covariance
inside a cropped circulant is enough to pass; the check cannot establish
distinct signed offsets. The other LOS test establishes lack of physical-window
roll equivariance, which also does not establish absence of long-distance
aliasing.

Recommendation: test a dense signed-offset reference and explicit modular
collisions on adversarial asymmetric kernels. Include a test that fails the
intended unrestricted-offset requirement when P<Z-1. Zero extension remains
bidirectional and nonlocal; it does not imply a causal LOS kernel or vanishing
communication from later to earlier redshift planes.

### 3. Integration uses the padded measure (normalization decision)

Matching forward/inverse FFT normalizations yield a denominator
`Nx * Ny * (Z+P)`. The current code crops without converting this to the
physical-window denominator `Nx * Ny * Z`.

With only a unit DC multiplier and a constant physical input, the output is
`Z/(Z+P)`: the audit measures 1 for (Z,P)=(8,0), 2/3 for (8,4), 0.8 for (16,4),
and 2/3 for (16,8). The report's 2-D bounded-domain operator instead explicitly
restores physical-domain quadrature after padding.

This is a valid padded-domain convention and a learned kernel can absorb the
constant gain on a fixed grid. It becomes a scientific consistency issue if
the intended definition is the same normalized physical-domain integral as
2-D, or if padding fraction changes during transfer.

Recommendation: declare the integration measure. To retain the normalized
physical-window definition while keeping this Fourier-series parameterization,
multiply the cropped message by `(Z+P)/Z`; add DC and dense value/gradient
checks for that convention. Such a change affects old checkpoints. Merely
scaling the result does not solve the separate frequency-period issue.

### 4. kz=0 has redundant complex parameters (efficiency and interpretation)

The transverse signed-frequency plane at kz=0 is self-conjugate. Real output
requires `R(-kx,-ky,0)=conj(R(kx,ky,0))`, and DC must be real. The 3-D code
stores every member independently as a complex tensor; `irfftn` implicitly
projects it to a Hermitian plane. Its effective coefficient is

`R_eff(kx,ky,0) = (R(kx,ky,0) + conj(R(-kx,-ky,0))) / 2`.

The audit changes imaginary DC by 10 without changing output, and changes
an anti-Hermitian frequency-pair direction without meaningful output change
(the isolated probe uses complex128 storage to avoid perturbation rounding).
Thus forward realness is correct, but raw coefficients are non-identifiable.
Finite gradients to a tensor do not establish that every stored direction
affects the output. The careful 2-D parameterization avoids this redundancy.

For the default dense kernel, the kz=0 plane has 5,089,536 redundant real
directions, about 3.1% of the 162,865,152 stored real components. The remaining
positive-kz planes do not have this same conjugacy constraint; Nyquist modes
are already excluded by the accepted-grid bounds.

Recommendation: parameterize real DC and independent conjugate pairs on kz=0,
or explicitly symmetrize while acknowledging remaining redundant parameters.
For factorized kernels, conjugacy must be designed jointly with the shared
channel projections. Preserving low-rank structure and removing every null
direction requires more than making the DC scalar weights real.

### 5. Anisotropic shape validation rejects valid inputs (confirmed bug)

The guard is `min(nx//2, ny//2) < max(mx,my)`. That incorrectly demands both
transverse dimensions support the larger of the two mode counts. For
grid (6,10,8) and modes (3,4,2), all documented per-axis inequalities hold,
but the current code raises ValueError.

Recommendation: check `nx < 2*mx or ny < 2*my or length < 2*mz` separately,
and add accepted rectangular anisotropic grids plus axis-specific rejection
tests. This is a shape-validation bug, not evidence of wrong messages on
currently accepted grids.

### 6. Factorization is stronger than independent per-mode rank reduction

The code forms `R_io(k) = sum_r P_in[i,r] w_r(k) P_out[r,o]`, with the same
complex channel projections at every frequency. For positive kz, rank is at
most r. All frequencies share those channel subspaces; this is more restrictive
than assigning arbitrary rank-r matrices independently to every mode.

On kz=0, Hermitian projection can increase the effective complex rank to at
most 2r (capped by the channel count). DC becomes the real part of the
factorized matrix: the audit constructs rank 1 before projection but real
DC rank 2 afterward. The factorization still saves parameters and its messages
match the dense series; the claimed per-mode rank bound needs this qualification.

The default width-48, (24,24,16) dense kernel stores 162,865,152 real parameters.
Rank 48 stores 3,402,240. Even r equal to channel width imposes shared-frequency
structure and is not generally equivalent to the dense model. The kernel
count alone excludes lift, gates, proposal and decoder. Dense float32 weights,
gradients and two Adam moments total approximately 2.61 GB (decimal), before
activations, FFT buffers and optimizer temporaries. Activation checkpointing
does not remove those parameter/optimizer allocations.

Recommendation: report rank and dense/factorized convention explicitly, count
complex components as two real parameters, and study rank independently of
the gate ablation. Correct the module comment's leading-order dense estimate:
the stored real count is approximately `8 mx my mz C^2`, not `4 mx my mz C^2`.

### 7. The output always becomes float32 (precision/testing limitation)

`RhizomeOperator3d.forward` applies `.float()` to decoded output even for a
double model and double input. The audit observes `torch.float32` logits.
This is sensible for bf16 CUDA training/BCE but discards requested float64
precision on CPU and weakens interpretation of the double-model tests.

Recommendation: promote bf16/float16 output for the training objective while
preserving float64 when AMP is disabled. For numerical tests, verify dtype as
well as closeness, and compare pre-cast intermediates. No harmful float32
training behavior was established by this finding.

### 8. External pipeline compatibility and resumption are not fully established

The trainer passes `augment=...` to `LOSWindowDataset` and calls
`fm.validation_subset`. The available sibling `FNO v3` constructor accepts only
source, rows, config and seed, and its fno_multifield module does not define or
import validation_subset. The static audit records both mismatches and source
hashes. Supplying `--fno-root` for that particular local checkout will not meet
the trainer's expected API. The cluster default points to another checkout,
which was not inspected or executed; this is not a claim that its jobs fail.

The current checkpoint signature records argument values and Git commits in
metadata but does not enforce hashes of Rhizome source, external source or
the preparation file. Editing code or preparation in place leaves the same
CLI signature and can silently change a resumed experiment. Seeds and sampler
position preserve declared sample ordering only under an unchanged data/API
contract. Exact CPU/GPU resume equivalence has not been tested here.

Recommendation: pin and check the external API before allocating GPU memory;
run a tiny end-to-end fit/evaluation in the intended checkout; persist and
validate source/preparation/data fingerprints on resume; and include interrupted
versus uninterrupted checks with augmentation and worker sampling. Construct
a real-data fixture or a faithful stub rather than assuming a neighboring
project version is interchangeable. Preserve existing checkpoints as their
original version when fixing these issues.

## Boundaries, resolution and physical interpretation

Appended FFT padding transmits exactly zero because the code pads *transmitted
values*. Physical-window cells whose `native_valid` channel is zero are a
different matter: the external dataset edge-pads its fields and supplies a
validity channel. Rhizome has no hard transmission mask for those cells; it can
learn to use or suppress that extrapolated context. Record whether this is the
desired boundary convention, and distinguish it from zero extension outside
the entire window.

A halo of 32 cells excludes border predictions from the loss, but does not
bound the receptive field of a global spectral interaction. Changing tiling
origins can change even core predictions. Evaluate overlapping predictions and
seam errors for multiple tiling origins at fixed window size/physical extent.
Do not claim tiled predictions are equivalent to a whole-cone pass. Applying
the same kernel to a differently sized physical cone also changes its spatial
period and is a separate operator-transfer question.

The physical spectral wavenumbers depend on transverse box lengths and the
padded LOS length times cell spacing. A fixed `los_pad=64` does not maintain a
fixed physical extension when resolution changes. For fixed-domain transfer,
scale padding with resolution, preserve physical domain conventions, exclude
unsupported Nyquist modes and inspect aliasing from gate/nonlinear products.
Position and redshift conditioning can help adaptation but do not establish
consistent quadrature or resolution accuracy by themselves.

The damped tanh update bounds hidden states for step_size in (0,1]. It does
not make the field update contractive: kernel norms, gate derivatives and
proposal weights still affect the Jacobian. Validate extra update counts by
training/evaluation experiments before claiming equilibrium convergence. The
model imposes neither conservation nor a radiative-transfer equation.

## Recommended repair and validation order

1. Fix the rectangular-grid guard and preserve float64 outside AMP; cover the
   accepted/rejected axes and dtypes with targeted tests.
2. Specify the desired LOS signed-offset and integration-measure conventions;
   implement them with explicit checkpoint/version metadata and independent
   dense value/gradient/DC/collision tests. Retrain changed conventions.
3. Refine the kz=0 parameterization and document the factorization's actual
   constraints; assess rank and memory on the intended GPU.
4. Verify the exact external data API, normalization, validity semantics,
   source/preparation fingerprints and interrupted training behavior.
5. Evaluate real-data tiling seams, coherent LOS statistics and fixed-physical-
   domain resolution transfer, then execute the gate ablations.

The unchanged current code can produce real, differentiable messages, and the
audit verifies their precise Fourier meaning. The findings concern the intended
boundary/measure contract, avoidable implementation restrictions and the
scientific claims those messages can support.

# Rhizome ablation study

Designed 2026-09-30. This is a prospective protocol, not a new experimental
result. Architecture changes and training runs below have not been executed.
The associated technical note's Darcy implementation lives in
`.delta/worktrees/e49xfk01xnej/ebm-21cm/`; the current root checkout has the
periodic 2-D model and a newer 3-D extension. Do not silently mix these versions.

## Question and primary hypothesis

Does recomputing source and receiver gates from evolving hidden states improve
prediction over the same recurrent integral operator without that dependence?

The primary comparison is the full Rhizome against initial-state gates,
at equal hidden width and training exposure. The ungated comparison establishes
whether gates help at all. Conditioning-only gates are an additional control,
but their smaller pointwise gate function also changes local expressiveness.
A gate advantage would not by itself establish a
benefit from recurrence, parameter sharing, or a new operator family.

For context `z_t=[h_t;c]`, all arms retain

```text
V_t = W_V [h_t;c]
m_t = A_t * K(B_t * V_t)
h_next = (1-eta) h_t + eta tanh(W_U [h_t;m_t;c] + b_U)
```

Only the gates change in the first stage. Retrain every arm from scratch;
turning off gates in a trained full model is an intervention diagnostic, not a
fair architecture comparison.

## Stage A: gate mechanism, 35 runs

Use seeds 0, 1, 2, 3, 4 for each of seven arms. The counts below assume the
report's Darcy model: width 16, mode setting 8, four shared updates, three input
channels, one output, no scalar embedding, and its Hermitian 2-D kernel.

| ID | Source B | Receiver A | Question | Active real parameters |
|---|---|---|---|---:|
| full | `2 sigmoid(W_B[h_t;c]+b_B)` | same form, independent A weights | Reference | 60,305 |
| no_gates | ones | ones | Does gating help? | 59,249 |
| source_only | full | ones | Is sender selection sufficient? | 59,777 |
| receiver_only | ones | full | Is receiver selection sufficient? | 59,777 |
| conditioning_only | `2 sigmoid(W_B c+b_B)` | same form | Does evolving state add useful control? | 59,793 |
| initial_state | full evaluated once at `[h_0;c]` | same form, cached for all updates | Does recomputation matter? | 60,305 |
| static_channels | learned `2 sigmoid(b_B)` broadcast over space | independent broadcast A bias | Is improvement merely channel rescaling? | 59,281 |

The initial-state arm retains differentiable initial gates, recomputed for each
new example/forward pass. Do not detach them or freeze their trained weights.
The cached gates are constant only across the updates within that pass.

Parameter accounting: both full gates use `4d^2+2d`; a single gate uses
`2d^2+d`; conditioning-only gates use `2d^2+2d`; static gates use `2d`.
The static gate factors can be absorbed into a learned linear kernel, so this
arm is largely a reparameterization/optimization control, not a fundamentally
more expressive model than the ungated arm.

Equal width keeps the spectral channel capacity fixed. Counts intentionally
differ slightly; do not add disconnected dummy parameters to claim a match.
If inactive gate modules are kept for implementation convenience, report both
allocated and active parameters and exclude unused parameters from Adam.

Use an explicit state dictionary for common lifting, value, kernel, proposal
and decoder initialization within each seed. Use independent RNG streams for
initialization and sample order; constructing fewer modules must not change
the order of minibatches. Full/source/receiver/initial-state arms can copy the
corresponding gate rows from a common initialization. Conditioning-only gates
can copy the conditioning columns. Save these initialization hashes.

## Data, stopping and measurement

Use the checksum-verified Darcy-32 data and retain the original training-only
normalization and zero-extension convention. For a new study, derive a new
development split from the 5,000 training-file rows: seed 20260930, first 500
permuted rows as validation, remaining 4,500 as training. Persist the exact IDs.
This is a new validation protocol, not a repeat of the old recorded result.

The public 1,000-row test file has already been inspected. Keep it closed
during design, tuning and fitting; report its eventual scores as benchmark
evaluation on a previously inspected distribution. A confirmatory claim needs
new independently generated realizations with documented physical encoding,
or an untouched external task. A newly chosen split alone cannot undo past
test inspection.

For Stage A, freeze a 6,000-step budget, Adam, batch size 32, peak LR 0.003,
gradient clipping 1, validation every 200 steps and the report's plateau LR
rule. Pick the minimum mean per-field validation relative L2 independently for
each arm. This extends the earlier 3,000-step budget but does not certify
optimization convergence. Log whether the best checkpoint is the last one.

Primary metric: mean of per-field relative L2 in stored output units after
inverse normalization. Also save median, 95th percentile, RMSE, MAE, per-field
errors, examples seen, active/allocated real parameters, optimizer steps,
training/evaluation seconds, peak accelerator memory and device details.
Do not pool norms before dividing. Include validation time in an end-to-end
timing table and also measure pure training throughput separately.

One common recipe isolates behavior under that recipe, not each arm's maximum
achievable performance. Reserve separate equal-budget validation tuning for
the architectural baselines in Stage B.

## Analysis and decision rule

Predeclare the full-versus-initial-state comparison as primary. For each
paired seed compute the relative reduction in the mean error,
`1 - error_full/error_control`. Report all five values, their mean and SD,
and the absolute error differences. Treat the other six-arm contrasts as
exploratory; if testing them formally, declare multiplicity correction first.

For field uncertainty, resample common field IDs jointly for paired models;
for training uncertainty, resample paired seed IDs at the outer level and
field IDs at the inner level (10,000 bootstrap draws). Do not count the
thousands of seed-field combinations as independent training trials. With
only five seeds, bootstrap intervals are descriptive and potentially unstable;
retain the individual seed results prominently.

Use a prospective engineering threshold of at least 5% mean relative error
reduction over initial-state gates, with improvement in at least four of
five seeds, before prioritizing state dependence. This is a chosen practical
threshold, not a significance theorem. If uncertainty overlaps zero, report
the evidence as inconclusive rather than asserting a mechanism advantage.

Diagnostic interpretations:

| Pattern | Supported interpretation |
|---|---|
| full beats conditioning-only and initial-state | Updating gates from evolving states is useful under this protocol |
| full approximately equals conditioning-only | Fixed input context may explain the gain |
| source-only matches full | Receiver gate may be unnecessary at this capacity |
| static gate matches full | Reparameterization/channel scaling merits investigation |
| all gating arms match no_gates | No demonstrated gate benefit at this task and budget |

Measure gate distributions and near-saturation fractions on validation only,
using thresholds below 0.1 and above 1.9. Channelwise gain and kernel magnitude
have scaling ambiguities; gate pictures are not calibrated physical connection
strengths. A separate input-Jacobian or perturbation diagnostic is more useful
for identifying actual influence. Gate maps alone do not establish causal
physics or sparse topology.

## Stage B: sharing and FNO, at least 20 runs

Five seeds each for tied Rhizome, untied Rhizome, tied ungated integral model,
and a conventional untied FNO. At width 16, four updates and mode setting 8,
untied Rhizome has 240,161 real parameters in the report's 2-D convention.
Equal width addresses channel capacity; report its parameter cost explicitly.
Use the same boundary evaluator, input channels and pointwise decoder width
for a controlled FNO arm, plus a separately identified external/reference FNO
with its recommended projection and domain-padding recipe.

Perform two analyses, never conflate them:

1. Equal example/step budget, including width-16 FNO and the earlier width-8
   compact FNO for continuity with the report.
2. Equal end-to-end training-time budget on one fixed device, allowing the
   faster model more updates. Choose the time budget from training-only
   profiling before inspecting performance. Log both seconds and examples;
   equal wall time is not a hardware-independent FLOP match.

Give each baseline the same validation-tuning allocation (for example, peak
LR in {0.001, 0.003} and 6,000 or 12,000 steps, two development seeds), without
opening test outputs. Freeze the selected recipes before the five-seed study.
For parameter-budget comparison, use a declared width grid and plot error
against measured active parameter counts; do not select widths using test
scores. Any extra reference/compact configurations add to the 20-run minimum.

Holding forcing only in the initializer, replacing tanh by another activation,
or changing damping should be later one-variable studies. Combining those
changes with a gate ablation would weaken attribution.

## Stage C: operator transfer and 21cm validation

Train at one resolution and evaluate the same continuous-domain input
realizations with independently computed targets at additional resolutions.
Darcy's undocumented archive transformations make a controlled solver dataset
preferable: record physical coefficients, boundary sampling, units, solver
tolerances, anti-aliasing and restriction. Interpolating old coarse targets is
not evidence of high-resolution accuracy.

For 3-D Rhizome, resolve the boundary, measure, spectral and external-pipeline
findings in [the implementation review](rhizome3d_review.md) first. Keep the
physical box/window extent and padding fraction fixed under resolution changes.
Scaling fixed cell counts without scaling physical extents changes the kernel
being evaluated. State/gate products also require an aliasing diagnostic.

On real 21cmFAST cones, split by simulation/initial-condition identity, not
windows or slices. Keep an untouched cone set for final assessment. Use BCE
for all compared x_HI backbones in the controlled experiment; comparing Rhizome
BCE with an existing FNO MSE recipe combines objective and architecture effects.
Include voxel RMSE, mixed-phase RMSE, global neutral-fraction histories,
brightness-temperature power spectra using consistent physics assumptions,
bubble-size/connectivity statistics, LOS correlations and window-seam error.
Bootstrap cones, not individual correlated pixels or windows.

## Implementation and execution checklist

The current runners do **not** expose these gate variants. Required work before
launch: explicit variant configuration/checkpoint fields, active-parameter
accounting, shared initialization, a test-isolated evaluation command and a
manifest of split IDs, data hashes and budgets. Do not pass invented CLI flags.

For each implemented arm verify: a fresh model/checkpoint round trip; finite
forward/backward; expected state dependence (or independence); correct active
count; identical sampler order; identical default full-model output; and a
small overfit smoke test. Checkpoint removal/intervention must not access test
data. A unit test of dependency structure is necessary but does not replace
the retrained comparison.

Save configuration, source/data hashes, split IDs, normalizers, initialization
hashes, per-field validation histories, best checkpoint, per-field final errors
and timing records. Write one machine-readable row per arm/seed/budget. Keep
failed runs and report the cause; rerun technical failures with the same seed,
never discard a seed because its accuracy is poor.

Recommended order: implement the seven Stage A arms; run one-seed training-only
smoke checks; freeze the study; execute 35 fits; inspect validation; then decide
whether the larger Stage B/C budget is warranted.

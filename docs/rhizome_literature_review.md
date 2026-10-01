# Rhizome: architectural positioning and prior work

Reviewed 2026-09-30. This is a focused architectural review, not an exhaustive
priority search. Full texts were checked where available; limited-access leads
are explicitly distinguished below. No new comparative training was performed.

## Assessment

The Rhizome has a clear mechanism, but the broad primitive of input-dependent
gates surrounding an efficiently evaluated convolution has substantial prior
art. Graph/integral operators and shared recurrent Fourier updates also predate
this implementation. The defensible candidate contribution is a particular
combination and its empirical value on scientific fields, rather than the
first distributed, gated, recurrent or nonlocal neural operator.

This narrows the initial assessment: a novelty discussion limited to FNO and
DeepONet would miss closely related gate/convolution mechanisms outside the
PDE literature.

## The implemented mechanism

Flatten spatial sites and channels, and define block-diagonal site/channel
multiplication operators `D_A(h,c)` and `D_B(h,c)`. One message is

`m = D_A(h,c) K_theta D_B(h,c) V_theta(h,c)`.

The base K is block-circulant on periodic transverse grids and a cropped
convolution under the bounded-domain convention. Both positive gates depend
pointwise on the hidden state and fixed conditioning. A shared damped tanh
proposal updates h; the gates are recomputed from the new h next round.

This is a nonlinear map in h. It is linear in the transmitted V only when
the gates are held fixed. Calling the whole interaction a learned *linear*
operator without this qualification would be misleading.

## Closest verified precedents

| Work | Mechanism verified in source | Overlap | Distinction in this implementation |
|---|---|---|---|
| [Graph Kernel Network, Li et al., 2020](https://arxiv.org/pdf/2003.03485), Eq. 8 | Iterated integral message with kernel depending on positions and input values | Spatial field states, learned connections, integral/message-passing view | Rhizome restricts pair dependence to source/receiver factors around a convolution and adds evolving-state gates |
| [FNO, Li et al., 2020/2021](https://arxiv.org/abs/2010.08895) | Fourier-parameterized integral kernel | Retained modes and efficient nonlocal convolution | Gates modulate the convolution's input and output; the update is shared and damped |
| [IFNO, You et al., 2022](https://arxiv.org/pdf/2203.08205), Eq. 3.4 | Fourier integral increments with layer-independent parameters | Repeated shared Fourier update, depth-independent parameter count | IFNO's stated increment has no Rhizome source/receiver gates; its update and fixed-point framing differ |
| [NKN, You et al., 2022](https://arxiv.org/pdf/2201.02217), Sec. 2 | Shared nonlocal diffusion/reaction update, with a stability analysis | Nonlocal recurrent field and parameter sharing | Rhizome does not impose the diffusion/reaction structure; the NKN stability conclusions do not transfer |
| [Attention Free Transformer, Zhai et al., 2021](https://arxiv.org/html/2105.14103v1), Eqs. 2, 6 | Key/value weighting by position followed by sigmoid query gating; normalized aggregation | Sender weighting and receiver gating; AFT-conv also shares spatial offsets | AFT is normalized and uses constrained weighting; Rhizome uses a signed channel-mixing spectral kernel and a shared latent-state update |
| [Hyena, Poli et al., ICML 2023](https://proceedings.mlr.press/v202/poli23a/poli23a.pdf), Sec. 3 | Alternating input-controlled diagonal gates and long Toeplitz convolutions evaluated by FFT | Very close diagonal/convolution algebra without an explicit edge matrix | Original projections/gates are computed from the input; Rhizome recomputes gates from an evolving state with persistent conditioning |

The [official FNO theory guide](https://github.com/neuraloperator/neuraloperator/blob/main/doc/source/theory_guide/fno.rst)
also describes the spectral/pointwise architecture. Synchronous evaluation
without a spatial visitation order is not a distinguishing feature over FNO.

## AFT comparison: close mechanism, not identical architecture

AFT Eq. 2 can be written schematically as

`Y_i = sigmoid(Q_i) * sum_j exp(w_ij) exp(K_j) V_j / sum_j exp(w_ij) exp(K_j)`.

Thus source weighting, positional interaction and receiver gating already
coexist. Under relative-position biases, the aggregation admits a convolution
interpretation. Its denominator introduces global input dependence into the
effective receiver factor. Rhizome omits that normalization and allows signed,
cross-channel kernels. AFT's feed-forward architecture is not the full shared
Rhizome rollout. This algebraic comparison is an inference from AFT's published
equations, not a claim made by its authors about Rhizome.

## Hyena/H3 comparison: the central novelty risk

Hyena Sec. 3 expresses its data-controlled operator as a product
`D_xN S_hN ... D_x1 S_h1`; its H3 discussion gives `D_q S_psi D_k S_phi`.
In the scalar/per-channel setting, replacing the first convolution by the
identity gives the same `D_A K D_B V` message pattern. This is a restricted
algebraic comparison, not identity of the networks: channel mixing, gate
parameterization, domains, filter representations and update rules differ.

Most notably, Hyena's internal recurrence consumes projections from the input,
whereas Rhizome evolves a latent field and recomputes its gates using that
field and fixed forcing. That distinction deserves an ablation against gates
computed once from the initializer. Without such an experiment, a new name
does not demonstrate a new useful mechanism. This mapping is our inference
from the published equations.

## Additional leads and scope limits

- [GAFNO, ICDM 2023 official program](https://www.cloud-conf.net/icdm2023/schedule.html)
  lists *Gated Adaptive Fourier Neural Operator for Task-Agnostic Time Series
  Modeling* (DOI [10.1109/ICDM58522.2023.00136](https://doi.org/10.1109/ICDM58522.2023.00136)).
  The publisher full text was not accessible in this review. Its equations have
  not been verified; this is a related-work lead, not evidence of exact overlap.
- [FV-PIFNO/Gated-FNO](https://www.sciencedirect.com/science/article/pii/S0309170825002015)
  explicitly introduces a gated Fourier architecture in its publisher abstract.
  Only the indexed abstract was available. That supports the existence of
  prior gated FNO work, not an equivalence to the two-sided Rhizome kernel.
- [Nonlocal Attention Operator, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ce5b4f79f4752b7f8e983a80ebcd9c7a-Abstract-Conference.html)
  provides an attention-based neural operator and is relevant to positioning
  learned nonlocal interactions. It is not used here as evidence for the exact
  diagonal/convolution factorization.
- [Recurrent Neural Operators, Ye et al., 2025](https://arxiv.org/abs/2505.20721)
  address recurrent *training over physical temporal predictions*. That is a
  different recurrence axis from Rhizome's internal computational updates.
  Similar names should not be treated as architectural equivalence.

This search covered graph/integral kernels, gated Fourier models, nonlocal
updates, AFT, Hyena/H3 and shared Fourier layers. No exhaustive assertion is
made about every paper, unpublished implementation or patent. Inaccessible
papers remain unresolved comparisons.

## Claims to use and avoid

Supported description:

> We investigate a recurrent neural integral operator whose source and receiver
> gates are recomputed from evolving latent states and persistent conditioning.
> A Fourier kernel evaluates the modulated interactions efficiently. We test
> whether this feedback improves accuracy at controlled parameter and compute
> budgets on heterogeneous scientific fields.

This is a prospective research statement. The existing recorded comparison
shows an exploratory advantage over one compact FNO configuration, not the
controlled mechanism result proposed above.

Avoid claiming: the first network-like neural operator; an alternative to
iteration; arbitrary learned graph topology; first gates around global
convolution; established equilibrium convergence; inherited FNO/NKN universal
approximation or stability guarantees; or superior compute efficiency from
parameter matching alone.

The positive diagonal gates rescale existing channelwise couplings. At fixed
state they cannot change the sign of an individual base-kernel entry, and they
do not independently create arbitrary pairwise edges. Kernel/channel mixing
and evolving values can still make the complete nonlinear model expressive.
No general graph-structure learning theorem follows from the gate factorization.

## Experiments motivated by prior work

1. Full versus conditioning-only and initial-state gates: identify the value
   of feedback through evolving h, the distinction from fixed input gates.
2. Ungated shared integral model and IFNO-style shared baseline: determine
   whether parameter sharing/recurrence explains the compact-model advantage.
3. A normalized AFT-inspired scientific-field control or a carefully adapted
   gated-convolution baseline: test whether the specific unnormalized signed
   spectral interaction matters. Identify it as an adaptation, not an exact
   reproduction of the original sequence/image model.
4. Compute/width sweeps and real field/resolution transfer: determine whether
   the mechanism is useful beyond one short toy/low-resolution recipe.

The [ablation protocol](rhizome_ablation_study.md) defines the first two stages.
The most credible contribution would combine an explicit relationship to this
prior art, controlled evidence for latent-state feedback, and a useful
scientific accuracy/resource tradeoff.

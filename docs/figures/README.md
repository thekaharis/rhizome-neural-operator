# Rhizome paper figures

| File | Content | Source |
|---|---|---|
| `rhizome_architecture.pdf` | Full model and one synchronous update | `rhizome_architecture.tex` (TikZ, standalone) |
| `rhizome_mechanism.pdf` | Learned kernel, gates and gate re-weighting of a trained checkpoint | `scripts/make_rhizome_mechanism.py` |
| `rhizome_network.pdf` | How cells exchange information: tree (U-Net), fixed global kernel, rhizome | `scripts/make_rhizome_network.py` |

PNG copies (2400 px / 300 dpi) are for slides; use the PDFs in LaTeX. All
figures are sized for a full text width of about 7 in (`\textwidth` in a
two-column layout, or `\linewidth` in a single-column thesis).

## Rebuild

```bash
cd docs/figures && pdflatex rhizome_architecture.tex
```

```bash
python scripts/make_rhizome_mechanism.py \
  --checkpoint runs/rhizome_long_seed1/fit/best.pt \
  --cache runs/rhizome_long_seed1/slices.h5 \
  --out docs/figures/rhizome_mechanism
```

```bash
python scripts/make_rhizome_network.py \
  --checkpoint runs/rhizome_long_seed1/fit/best.pt \
  --cache runs/rhizome_long_seed1/slices.h5 \
  --out docs/figures/rhizome_network
```

The recorded figures used the long-run checkpoint (step 8,200) and cache from
the `.delta/worktrees/jav8yay7g827` worktree, with PyTorch 2.9.1 and
matplotlib 3.10.8 (`FNO_env`) and a TeX Live 2024 install for `usetex`. The
script picks the validation slice whose true mean x_HI is closest to 0.5
(row 252, cone 63, z = 10.55) and the ionization-front cell nearest the box
centre (16, 16). `rhizome_mechanism.json` records these choices and the
plotted summary numbers. Pass `--row` to show another slice.

## Captions

Architecture:

```latex
\caption{Rhizome neural operator. \emph{Top:} density bands $\delta$ and the
scalars $(z,\bm\vartheta)$ are lifted pointwise to the conditioning
$\mathbf c$, the state is initialised as $\mathbf h_0=\tanh\mathbf c$, one
update $\mathcal U_\theta$ is applied $T=4$ times with shared weights, and a
pointwise decoder gives $\hat x_{\mathrm{HI}}$. \emph{Bottom:} a single
update. Channel-wise gates $\mathbf A,\mathbf B\in(0,2)^C$ and the value
$\mathbf V$ are $1{\times}1$ projections of $[\mathbf h_t;\mathbf c]$. The
message $\mathbf m_i=\mathbf A_i\odot\frac1N\sum_j\kappa_\theta(\mathbf
r_i-\mathbf r_j)(\mathbf B_j\odot\mathbf V_j)$ is the only operation that
couples cells; it is evaluated with an FFT and the retained Fourier
multipliers $R_{\mathbf k}$, so no $N\times N$ edge tensor is stored.}
```

Mechanism:

```latex
\caption{Learned interaction of the trained rhizome (toy lightcones,
validation slice at $z=10.55$, $\langle x_{\mathrm{HI}}\rangle=0.50$).
The ring marks the receiver cell $i$ on an ionization front; black lines are
the true $x_{\mathrm{HI}}=0.5$ contour. (a)~Input density band. (b)~Norm of
the learned kernel $\kappa_\theta(\mathbf r_i-\mathbf r_j)$, which is
translation invariant. (c,d)~Channel-RMS receiver and source gates at the
last update. (e--h)~Gate re-weighting of the connections into $i$,
$M_t(j)=\|\mathrm{diag}(\mathbf A_i)\,\kappa_\theta(\mathbf r_i-\mathbf
r_j)\,\mathrm{diag}(\mathbf B_j)\|_F/\|\kappa_\theta(\mathbf r_i-\mathbf
r_j)\|_F$, divided by its spatial mean; the mean gain is printed in each
panel. The gates boost a ring of sources at intermediate distance and damp
distant ones by up to ${\approx}20\%$, while the overall gain grows from
$0.95$ to $1.67$ over the four updates. These are gains on the transmitted
value, not output sensitivities.}
```

Network:

```latex
\caption{How cells exchange information. Nodes are an $8\times8$ lattice of
cells on the same validation slice; grey is the true neutral region.
(a)~A U-Net-style hierarchy: information moves through coarse parent nodes,
so two neighbouring cells on either side of a block boundary (green) meet only
at the root. (b)~An ungated global kernel, as in a Fourier layer, here the
trained rhizome's own $\kappa_\theta$ with $\mathbf A=\mathbf B=\mathbf 1$:
every cell is connected to every other, with the same pattern around each
cell. (c)~The trained rhizome at update $t=4$: edge $(i,j)$ carries
$\|\mathrm{diag}(\mathbf A_i)\,\kappa_\theta(\mathbf r_i-\mathbf
r_j)\,\mathrm{diag}(\mathbf B_j)\|_F$, so the same all-to-all web is
re-weighted cell by cell from the current state. Ring size shows the receiver
gate, fill the source gate. Green edges are the incoming connections of one
cell. Edge width and opacity show the geometric mean of the two directed
weights relative to the panel maximum; edges crossing the periodic boundary
are omitted for clarity.}
```

## What the mechanism figure shows, and what it does not

- The receiver gates vary strongly in space (panel c); the source-gate RMS
  varies little (panel d, 1.04 to 1.12) because individual channels move in
  different directions and partly cancel.
- Most of the change across updates is a growing overall gain. After
  normalising each effective-connection map, its shape changes by only about
  4% (L2) from update 1 to update 4, so this checkpoint re-weights its
  connections moderately rather than re-wiring them.
- The ring pattern in (e–h) comes from the receiver's channel weights
  selecting kernel channels with different spatial profiles, plus
  source-dependent speckle from B.
- These are gains on V_j for a single slice and cell, on synthetic toy data.
  An input-Jacobian or perturbation study would be needed to claim causal
  influence, as noted in `docs/rhizome_ablation_study.md`.

## What the network figure shows, and what it does not

- Total incoming strength varies across cells by a coefficient of variation
  of 0.32 in the rhizome, against exactly 0 for the ungated kernel. Outgoing
  strength varies far less (0.04): in this checkpoint the heterogeneity comes
  mostly from the receiver gates, not the senders.
- Directed weights differ: the median of |w_ij − w_ji| / (w_ij + w_ji) is
  0.17, which panel (c) does not show because it draws undirected edges.
- Panel (a) is a schematic of 2×2 pooling, not a trained U-Net. Real U-Nets
  also have skip connections and overlapping convolution stencils, so
  neighbouring cells do interact at the finest level through the stencil;
  the panel shows the coarse-path structure the name "tree" refers to.
- All values are in `rhizome_network.json`.

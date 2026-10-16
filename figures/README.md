# Publication figures

From the repository root:

```bash
python figures/reproduce.py --workers 4
```

The command produces four figures, each as **vector PDF**, editable SVG, and
220 dpi PNG in `generated/`. It also saves the samples and exact loss arrays in
`data/`, a long-format `gap_values.csv`, and a manifest containing every seed,
grid point, repetition, software version, and SHA-256 checksum. Each numerical
task has its own restartable cache. Re-running the command reuses matching
tasks; it never changes the random samples because of execution order.

`--render-only` rebuilds all figures from the saved arrays and assignment
permutations without any assignment solves. `--quick` produces a clearly marked preview with
fewer samples and grid points under `preview/`; these files are never substituted
for the publication figures.

Run `python figures/validate.py` to check saved-file checksums, deterministic sampling
sequences, paired-plan optimality at the true parameter, numerical convexity,
and agreement with a separate implementation of the assignment objective on
all small-sample tasks.

Cache specification 2 records the numerical module's SHA-256 checksum and
rejects caches from a different numerical implementation. Change the
`SPEC_VERSION` in the figure generator whenever sampling or calculation logic
outside that module changes.

| Output | Content | Source and numerical specification |
|---|---|---|
| `figure_1_cycles` | Three feasible discrete four-cycles | Matplotlib vector rendering of the included [TikZ source](source/discrete_identifiability_cycles.tex). No random data. |
| `figure_2_cosine` | Paired Gaussian/cosine empirical gap | Three perturbations and four sample sizes, with the seeds, grids and repetitions specified below. |
| `figure_3_annulus` | Paired Gaussian/annulus empirical gap | Four sample sizes, five repetitions each, with a refined evaluation grid around the true cost. |
| `figure_4_annulus_maps` | Source/target and three empirical assignments | All 1000 arrows shown, seed 1000, paired observations, radii 0.8 and 0.85. Uses repetition zero of the n = 1000 annulus experiment. |

## Mathematical setup

All samples use `numpy.random.default_rng(seed)`. The publication experiments
draw **paired observations**

$$
X_i\sim\mathcal N(0,I_2),\qquad Y_i=T(X_i),\qquad
\widehat\pi_n=\frac1n\sum_{i=1}^n\delta_{(X_i,Y_i)}.
$$

On the positive semidefinite trace-one slice
$A_u=\operatorname{diag}(u,1-u)$, the plotted loss is

$$
\mathcal L_{n,0}(A_u)=
\frac1n\max_{\sigma\in S_n}\sum_iX_i^\top A_uY_{\sigma(i)}
-\frac1n\sum_iX_i^\top A_uY_i.
$$

Every interior grid value is computed with SciPy's exact maximum-weight
assignment solver. The endpoints use the rearrangement inequality (sorting)
for their rank-one scores, giving the same exact optimum. Endpoint objectives
are cross-checked against assignment solves on every small task. The center
is also solved explicitly and checked against its theoretically zero gap.
No entropic regularization, fitted curves or hidden rescaling is used.
Straight segments connect the evaluated grid points; intervening loss values
are not evaluated by the figure renderer.

### Cosine perturbation

$$
T_\varepsilon(x)=\nabla\Psi_\varepsilon(x),\qquad
\Psi_\varepsilon(x)=\tfrac12|x|^2+
\varepsilon[\cos x_1+\cos x_2+\cos(x_1+x_2)].
$$

The bound $D^2\Psi_\varepsilon\succeq(1-3\varepsilon)I$ proves
strong convexity for $0\le\varepsilon<1/3$. Thus the paired plan is an OT
plan at $u=1/2$. Use $\varepsilon=0.01,0.1,0.3$,
$n=50,100,1000,2000$, and respectively 5, 5, 1, 1 repetitions. For each
$n$, seed equals $n$; successive repetitions consume successive
$(n,2)$ Gaussian arrays. **The same source array is reused across all three
perturbations for each repetition.** The grid is exactly the sorted unique
union of `linspace(0,1,40)` and `linspace(.4,.6,11)` (51 points).

### Radial annulus

$$
T(x)=\frac{x}{|x|}\sqrt{a^2+(b^2-a^2)(1-e^{-|x|^2/2})},
\qquad a=0.8,\ b=0.85.
$$

This monotone radial quantile map sends the standard planar Gaussian to
uniform area measure on the annulus. Its arbitrary value at the Gaussian
null point $x=0$ is defined as $(a,0)$. The implementation uses
`-expm1(-r*r/2)` for numerical stability and divides by the actual positive
radius rather than replacing it by a fixed threshold.

There are 5 repetitions for each $n=50,100,1000,2000$, using seed $n$
and successive $(n,2)$ Gaussian arrays for the repetitions. The grid is exactly
the sorted unique union of `linspace(0,1,41)` and `linspace(.4,.6,21)`.
Floating-point values that are close but not identical remain distinct;
no tolerance-based grid merging is applied.

### Bands and arrow interpretation

Curves show repetition means. Bands show mean plus/minus the empirical
standard deviation across repetitions (`ddof=0`), clipped below at zero. For the single-run large
cosine samples the band has zero width; it is not a confidence interval.
Sample sizes have a consistent color across the loss figures.

The last figure shows maximum-score **empirical assignments between the paired
empirical marginals** at $u=0.5,0.9,0.1$. At the center the paired reference
is optimal. Away from it, the source atoms are reassigned to target atoms.
These arrows do not depict a smooth continuum map or an independently sampled
reference coupling.

## Explore the sampling model

All article figures use paired observations. The exploration notebooks and
browser experiments also support independent empirical marginals matched at
the true parameter. In that alternative protocol, independent samples are
drawn from the same two population marginals and matched once at $u=1/2$ to
define a fixed reference coupling. This is a different finite-sample
observation model, even when the population marginals are unchanged. See
[Notebook 02](../notebooks/02_sampling_and_plateaus.ipynb) for a direct comparison.

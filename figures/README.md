# Publication figures

[Browse the figures](https://gpeyre.github.io/iot-curvature/#figures) · [Interactive demo](https://gpeyre.github.io/iot-curvature/demo.html) · [Project homepage](https://gpeyre.github.io/iot-curvature/)

The current manuscript uses section-dependent numbering: the four files named
`figure_1` through `figure_4` correspond to **Figures 5.1–5.4**, respectively.
Filenames retain their ordinal identifiers.

## Installation and reproduction

Use **Python 3.12** for the recorded numerical environment. From a new checkout:

```sh
git clone https://github.com/gpeyre/iot-curvature.git
cd iot-curvature
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python figures/reproduce.py --render-only
```

On Windows activate with `.venv\Scripts\activate`. A normal
`python -m pip install -e .` also installs the supported dependencies; the lock
file records the exact NumPy, SciPy and Matplotlib versions used for the
committed data.

`--render-only` rebuilds all four figures from the saved loss arrays and
assignment permutations without any assignment solves. Outputs are **vector
PDF**, editable SVG, and 220 dpi PNG in `figures/generated/`.

To compute missing experiments from the deterministic samples, run:

```sh
python figures/reproduce.py --workers 4
```

Each numerical task has its own restartable cache under `figures/data/`.
Re-running the command reuses matching tasks; it never changes the random
samples because of execution order. To recompute a cached task, remove that
task's `.npz` file and rerun. To recompute everything, move the existing data
directory aside first. The complete run includes repeated dense assignments
up to **n = 2000** and is substantially slower than rendering the included data.

The computation saves samples and exact loss arrays, a long-format
`gap_values.csv`, and a [manifest](data/manifest.json) containing every seed,
grid point, repetition, software version, and SHA-256 checksum.

`--quick` produces a clearly marked preview with fewer samples and grid points
under `figures/preview/`; these files are never substituted for the publication
figures.

Run `python figures/validate.py` to check saved-file checksums, deterministic sampling
sequences, paired-plan optimality at the true parameter, numerical convexity,
and agreement with a separate implementation of the assignment objective on
all small-sample tasks.

The gap experiments use `SPEC_VERSION = 2`; the separate Figure 5.4
assignment archive uses `ARROW_SPEC_VERSION = 3`. Both record the numerical
module's SHA-256 checksum and reject caches from a different numerical
implementation or sampling specification. Change the corresponding version
whenever its sampling or calculation logic outside that module changes.

| Article figure | Output | Content | Source and numerical specification |
|---|---|---|---|
| 5.1 | `figure_1_cycles` | Three feasible discrete four-cycles | Matplotlib vector rendering of the included [TikZ source](source/discrete_identifiability_cycles.tex). No random data. |
| 5.2 | `figure_2_cosine` | Paired Gaussian/cosine empirical gap | Three perturbations and four sample sizes, with the seeds, grids and repetitions specified below. |
| 5.3 | `figure_3_annulus` | Paired Gaussian/annulus empirical gap | Four sample sizes, five repetitions each, with a refined evaluation grid around the true cost. |
| 5.4 | `figure_4_annulus_maps` | Source/target and three empirical assignments | All 1000 arrows shown, seed 42, independent Gaussian and annulus samples, radii 0.8 and 0.85; plasma colors show displacement magnitude. |

## Mathematical setup

The bilinear cost is $c_A(x,y)=-x^\top Ay$.

All samples use `numpy.random.default_rng(seed)`. The loss experiments in
Figures 5.2 and 5.3 draw **paired observations**

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

The perturbation is denoted **ε in the article** and **δ in the exploratory
controls**; these are the same parameter, never an entropic regularization
parameter here. At δ = 0 the paired loss vanishes identically on the whole PSD
slice for every sample size. The exploratory option to divide by δ² changes
only the vertical scale and preserves empirical zero plateaus. A visibly flat
small-δ curve is not by itself evidence of a numerical precision failure.

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

### Bands

Curves show repetition means. Bands show mean plus/minus the empirical
standard deviation across repetitions (`ddof=0`), clipped below at zero. For the single-run large
cosine samples the band has zero width; it is not a confidence interval.
Sample sizes have a consistent color across the loss figures.

### Figure 5.4: assignments between independent clouds

Figure 5.4 is a separate forward transport illustration. Draw the source and
annulus target independently, using the following exact generator sequence:

```python
rng = np.random.default_rng(42)
n = 1000
X = rng.standard_normal((n, 2))
r = np.sqrt(rng.uniform(0.8**2, 0.85**2, n))
theta = rng.uniform(0, 2*np.pi, n)
Y = np.column_stack((r*np.cos(theta), r*np.sin(theta)))
```

Thus $X_i\sim\mathcal N(0,I_2)$ and $Y_j$ are independently uniform in area
on the annulus. The square root in the radius sampler is essential for uniform
area measure. These are not the paired observations from Figure 5.3.

For each $u=0.5,0.9,0.1$, independently solve

$$
\sigma_u\in\operatorname*{argmax}_{\sigma\in S_n}
\sum_{i=1}^n X_i^\top A_uY_{\sigma(i)},
\qquad X_i\longmapsto Y_{\sigma_u(i)}.
$$

The two point clouds remain fixed across the three costs. Every one of the
1000 assignment arrows is shown, with plasma colors encoding
$\lVert Y_{\sigma_u(i)}-X_i\rVert_2$ on a shared scale. Blue and orange points
show the source and target atoms. These arrows depict discrete optimal
assignments, rather than the exact smooth population map or samples from a
population coupling. In particular, at $u=1/2$ they need not be exactly radial.

## Explore the sampling model

The loss figures use paired observations. Figure 5.4 uses independent empirical
marginals solely to display how their optimal assignment changes with the cost.
The exploration notebooks and browser experiments also support independent
empirical marginals matched at the true parameter: samples from the same two
population marginals are matched once at $u=1/2$ to define a fixed reference
coupling for a gap curve. This differs from observing genuine population-plan
pairs, even when the population marginals are unchanged. See
[Notebook 02](../notebooks/02_sampling_and_plateaus.ipynb) for a direct comparison.

"""Create the three executable, self-contained exploration notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BOOT = '''from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "iot_gap").is_dir())
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib.pyplot as plt
from iot_gap import (cosine_map, annulus_map, sample_pairs, sample_independent_marginals,
                     optimal_assignment, gap_curve, sampling_gap_curve)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
US = np.linspace(0, 1, 51)
'''

def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=[(nbf.v4.new_markdown_cell if kind == 'md' else nbf.v4.new_code_cell)(text.strip()) for kind,text in cells])
    notebook.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python', 'version': '3.12'}}
    nbf.write(notebook, ROOT / 'notebooks' / name)

write('01_models_and_gap.ipynb', [
('md', r'''# From an optimal map to an inverse gap

This notebook reconstructs the two continuous models in Figures 2–4. It runs quickly at small sample sizes. The exact article settings and cached large runs are explored in notebook 03.

The cost is $c_A(x,y)=-x^\top Ay$, with $A(u)=\operatorname{diag}(u,1-u)$ and $0\leq u\leq1$. For observed pairs $Y_i=T(X_i)$ the empirical mean gap is
$$L_n(u)=\max_{\sigma\in\mathfrak S_n}\frac1n\sum_i X_i^\top A(u)Y_{\sigma(i)}-\frac1n\sum_i X_i^\top A(u)Y_i.$$
Every value below is computed by exact assignment, without entropy. Straight line segments join evaluated points; they do not assert that all intervening assignment breakpoints have been found.'''),
('code', BOOT),
('md', r'''## A convex gradient perturbation
$$\Psi_\delta(x)=\frac12|x|^2+\delta[\cos x_1+\cos x_2+\cos(x_1+x_2)],\qquad T_\delta=\nabla\Psi_\delta.$$
For $0\leq\delta<1/3$, $\nabla^2\Psi_\delta\succeq(1-3\delta)I$. Thus $Y=T_\delta(X)$ with $X\sim N(0,I_2)$ samples the true OT plan. Positive $\delta$ identifies the population matrix up to scale, whereas $\delta=0$ leaves the whole PSD slice compatible. Uniform convexity is lost at the boundary $\delta=1/3$.'''),
('code', '''def cosine_hessian(X, delta):
    H = np.tile(np.eye(2), (len(X), 1, 1))
    common = np.cos(X.sum(axis=1))
    H[:, 0, 0] -= delta * (np.cos(X[:, 0]) + common)
    H[:, 1, 1] -= delta * (np.cos(X[:, 1]) + common)
    H[:, 0, 1] = H[:, 1, 0] = -delta * common
    return H

X, Y = sample_pairs("cosine", 100, seed=100, delta=.32)
minimum_eigenvalue = np.linalg.eigvalsh(cosine_hessian(X, .32)).min()
assert minimum_eigenvalue >= 1 - 3*.32 - 1e-12
print("Sample minimum Hessian eigenvalue:", minimum_eigenvalue)
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(*X.T, s=12, label="X: standard Gaussian")
ax.scatter(*Y.T, s=12, label="Y = Tδ(X)")
ax.quiver(*X.T, *(Y-X).T, angles="xy", scale_units="xy", scale=1, alpha=.3)
ax.set_aspect("equal"); ax.legend(); ax.set_title("Observed true-plan pairs, δ = 0.32")
plt.show()'''),
('code', '''fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for delta in [0, .01, .1, .3, .329]:
    X, Y = sample_pairs("cosine", 100, seed=100, delta=delta)
    loss = gap_curve(X, Y, US)
    assert loss.min() >= 0 and abs(loss[len(US)//2]) < 1e-12
    axes[0].plot(US, loss, label=f"δ={delta:g}")
    if delta > 0:
        axes[1].plot(US, loss/delta**2, label=f"δ={delta:g}")
    else:
        assert np.max(loss) < 1e-12
axes[0].set(title="Raw mean gap", xlabel="u", ylabel="Lₙ(u)")
axes[1].set(title="Same losses divided by δ²", xlabel="u", ylabel="Lₙ(u) / δ²")
for ax in axes: ax.legend(fontsize=8); ax.axvline(.5, color="grey", lw=.7, ls=":")
fig.tight_layout(); plt.show()'''),
('md', r'''Dividing by $\delta^2$ changes the vertical scale. It cannot remove a finite-sample zero plateau. The normalized curves need not coincide: the sample assignment combinatorics and the perturbation size both matter. The case $\delta=0$ is deliberately omitted from this normalization.'''),
('md', r'''## A radial map to an annulus
For $a=0.8$, $b=0.85$,
$$T(x)=\frac{x}{|x|}\sqrt{a^2+(b^2-a^2)(1-e^{-|x|^2/2})}.$$
The squared target radius is uniform on $[a^2,b^2]$. The map is defined almost everywhere and is discontinuous at the origin; the implementation chooses a measurable value at that null point. No $C^1$ claim at the origin is needed.'''),
('code', '''X, Y = sample_pairs("annulus", 100, seed=100)
quantiles = (np.sum(Y**2, axis=1) - .8**2)/(.85**2-.8**2)
assert np.allclose(quantiles, -np.expm1(-np.sum(X**2, axis=1)/2))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].scatter(*X.T, s=12, label="Gaussian source")
axes[0].scatter(*Y.T, s=12, label="Paired annulus target")
axes[0].set_aspect("equal"); axes[0].legend(fontsize=8)
axes[1].plot(US, gap_curve(X, Y, US), color="#7851a9")
axes[1].set(xlabel="u", ylabel="Mean gap", title="Radial population identification; empirical plateau")
fig.tight_layout(); plt.show()'''),
('md', '''## Optional live controls
The next cell creates a widget in Jupyter. Change n, δ and the model, then press **Run experiment**. It uses small sample sizes for exploration; it does not overwrite article data.'''),
('code', '''try:
    import ipywidgets as widgets
    from IPython.display import display
    def explore(model="cosine", n=80, delta=.1, seed=10):
        X, Y = sample_pairs(model, n, seed, delta=delta)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(US, gap_curve(X, Y, US)); ax.set(xlabel="u", ylabel="Mean gap", title=f"{model}; n={n}; δ={delta:g}")
        plt.show()
    display(widgets.interact_manual(explore, model=["cosine", "annulus"], n=[30, 80, 160],
                                   delta=widgets.FloatSlider(value=.1, min=0, max=.329, step=.001), seed=(1, 50)))
except ImportError:
    print("Install the notebooks extra to enable widgets.")'''),
])

write('02_sampling_and_plateaus.ipynb', [
('md', r'''# True-plan samples versus an empirical OT reference

These are different experiments, even though both reference couplings are optimal at $u=1/2$.

1. **Paired:** draw $X_i\sim\alpha$ and set $Y_i=T(X_i)$. The observations are iid from the population OT plan. The observed permutation is the identity.
2. **Independent marginals:** draw $X_i\sim\alpha$ and $Z_i\sim\alpha$ independently, set $Y_i=T(Z_i)$, then compute a reference permutation $\tau_*$ by OT at $u=1/2$. This empirical coupling is not an iid sample from the population plan.

For the second experiment subtract $n^{-1}\sum_iX_i^\top A(u)Y_{\tau_*(i)}$. Keep $\tau_*$ fixed while varying $u$. Both finite-sample gaps are convex and piecewise linear. Their curve shapes and finite-sample offsets can differ greatly.'''),
('code', BOOT),
('code', '''fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
for ax, delta in zip(axes, [0, .1, .32]):
    Xp, Yp = sample_pairs("cosine", 100, seed=7, delta=delta)
    Xi, Yi = sample_independent_marginals("cosine", 100, seed=7, delta=delta)
    ref = optimal_assignment(Xi, Yi, .5)
    lp = gap_curve(Xp, Yp, US)
    li = gap_curve(Xi, Yi, US, reference=ref)
    ax.plot(US, lp, label="True-plan paired")
    ax.plot(US, li, label="Independent + empirical OT", ls="--")
    assert abs(lp[25]) < 1e-12 and abs(li[25]) < 1e-12
    ax.set(title=f"δ = {delta:g}", xlabel="u", ylabel="Mean gap")
axes[0].legend(fontsize=7); fig.tight_layout(); plt.show()'''),
('md', r'''At $\delta=0$, the paired loss is identically zero on the PSD slice for every $n$. The independent-marginal loss need not be zero: it describes the sensitivity of a finite empirical assignment between two different clouds. Apparent curvature in the latter is not evidence of population identifiability.'''),
('code', '''fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
delta, seed = .1, 12
# Use nested draws here to isolate point count. This differs from the article,
# which deliberately preserves the original seed=n and replicate protocol.
rng = np.random.default_rng(seed)
Xmax, Zmax = rng.standard_normal((160, 2)), rng.standard_normal((160, 2))
for n in [30, 80, 160]:
    X = Xmax[:n]
    paired = gap_curve(X, cosine_map(X, delta), US)
    Y = cosine_map(Zmax[:n], delta)
    independent = gap_curve(X, Y, US, reference=optimal_assignment(X, Y, .5))
    axes[0].plot(US, paired/delta**2, label=f"n={n}")
    axes[1].plot(US, independent/delta**2, label=f"n={n}")
axes[0].set(title="Paired: Lₙ / δ²", xlabel="u", ylabel="Normalized mean gap")
axes[1].set(title="Independent + OT: Lₙ / δ²", xlabel="u")
for ax in axes: ax.legend()
fig.tight_layout(); plt.show()'''),
('md', r'''## Convexity and a sampled zero-interval diagnostic
The assignment maximum is a maximum of affine functions of $u$. Subtracting the fixed observed score leaves a convex function. On an evenly spaced grid its successive slopes must be nondecreasing, up to floating-point error.

Below we report only the extent of near-zero **evaluated grid nodes** around $1/2$. This is not an exact plateau endpoint computation. A smaller tolerance or finer grid can change this diagnostic; no monotonicity in $n$ is asserted.'''),
('code', '''def zero_grid_interval(us, losses, tolerance=1e-12):
    k = np.argmin(abs(us-.5)); left = right = k
    while left > 0 and losses[left-1] <= tolerance: left -= 1
    while right+1 < len(us) and losses[right+1] <= tolerance: right += 1
    return us[left], us[right]

for n in [30, 80, 160]:
    X, Y = sample_pairs("cosine", n, seed=n, delta=.1)
    losses = gap_curve(X, Y, US)
    slopes = np.diff(losses)/np.diff(US)
    assert np.diff(slopes).min() > -1e-10
    print(f"n={n}: near-zero grid interval {zero_grid_interval(US, losses)}")'''),
('md', r'''## Why a large-$n$ paired curve can still look flat
Positive $\delta$ breaks the population degeneracy, but weak perturbations produce a small signal. Finite assignments also retain a plateau around the reference. The paired and independently matched schemes remove different reference scores, so they need not resemble one another at a visually large but finite $n$.

Use the article's full $n=1000,2000$ data in notebook 03 rather than extrapolating from this quick exploration. Rescaling by $\delta^{-2}$ helps read the vertical scale; it does not change the zero set or establish a convergence rate.'''),
])

write('03_article_data_and_figures.ipynb', [
('md', r'''# Inspect and reproduce the article figures

The committed data were generated from the complete article experiment, including $n=1000$ and $2000$. This notebook reads those data; it does not recompute the expensive assignments by default.

Figure 1: discrete support cycles. Figure 2: cosine perturbations. Figure 3: radial annulus gap. Figure 4: empirical annulus assignments on the same article model. Read `figures/README.md` for the mathematical setup, seeds, grids and repetition protocol.'''),
('code', BOOT),
('code', '''import json
DATA = ROOT / "figures" / "data"
files = sorted(DATA.glob("*.npz"))
assert files, "Run python figures/reproduce.py first, or use the committed data."
print(f"{len(files)} data archives available")
for path in files[:6]:
    with np.load(path, allow_pickle=False) as data:
        print(path.name, {key: data[key].shape for key in data.files})
for path in sorted(DATA.glob("*.json")):
    print("Metadata:", path.name)'''),
('md', r'''## Raw arrays and normalization
Each gap archive stores exact evaluated grid nodes and loss values. Both source and target arrays are included so that values can be checked independently. The long CSV exports allow analysis without Python-specific serialization. Standard-deviation bands in the figures summarize repeated samples, with denominator equal to the number of repetitions; they are not confidence intervals. The two large cosine curves are single realizations.

Select any data archive below to inspect. The cell deliberately tolerates extra metadata keys while showing the shapes and numerical ranges.'''),
('code', '''selected = files[0]
with np.load(selected, allow_pickle=False) as archive:
    for key in archive.files:
        value = archive[key]
        if np.issubdtype(value.dtype, np.number):
            print(key, "shape", value.shape, "range", (value.min(), value.max()))
        else:
            print(key, value)
print("Selected:", selected.name)'''),
('md', '''## View the exact publication assets
The SVG files below contain the same plotted data as the vector PDFs used in the manuscript. Use the PDF files for publication; PNG companions are provided for previews.'''),
('code', '''from IPython.display import SVG, display
for path in sorted((ROOT / "figures" / "generated").glob("*.svg")):
    print(path.name)
    display(SVG(filename=str(path)))'''),
('md', '''## Regenerate rendering from cached values
This runs the plotting stage only and does not rerun the large assignment problems. Any changes to fonts, line widths or colors can be explored in the figure script and rendered again.'''),
('code', '''import subprocess
# Uncomment to regenerate all vector and preview assets from committed raw data.
# subprocess.run([sys.executable, str(ROOT / "figures" / "reproduce.py"), "--render-only"], cwd=ROOT, check=True)
print("Rendering command: python figures/reproduce.py --render-only")'''),
('md', '''## Recompute from samples
Run `python figures/reproduce.py` from the repository root to compute missing cached experiments and render. To recompute one existing experiment, remove its `.npz` cache from `figures/data/` and rerun. To recompute everything, move `figures/data/` aside first. Full generation is substantially more expensive than these notebooks because exact dense assignments are evaluated at every article grid point, repeatedly up to n=2000. Check the generated manifest for the parameter settings, software versions and file checksums.

The browser experiments under `docs/` use smaller point counts and a different random generator. They implement the same mathematical experiments; their individual random clouds are not the publication samples.'''),
])
print('Wrote three notebooks.')

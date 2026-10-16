# Project website and interactive inverse optimal transport

[Project homepage](https://gpeyre.github.io/iot-curvature/) · [Interactive demo](https://gpeyre.github.io/iot-curvature/demo.html) · [Figures](https://gpeyre.github.io/iot-curvature/#figures) · [Notebooks](https://gpeyre.github.io/iot-curvature/#resources)

A static, dependency-free companion to *Second-order geometry of optimal transport: cost identifiability and stability*. The homepage, `index.html`, introduces the project and links to the demo, article figures, Jupyter notebooks and repository source. The interactive experiment is in `demo.html`.

No installation, build step or backend is needed. All required assets are committed in this directory. The site does not upload experiment data or load external fonts, scripts or services; links to GitHub open the corresponding repository content.

## Run locally

From the repository root:

```sh
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000/> for the homepage or <http://localhost:8000/demo.html> for the demo. Direct `file://` opening of the demo does not work reliably because browsers restrict module workers. The app uses modern browser ES modules, SVG, and web workers.

## Publish with GitHub Pages

After uploading the repository to `gpeyre/iot-curvature`:

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Choose branch **`main`**, folder **`/docs`**, then click **Save**.
4. Leave this project's **Custom domain** blank and do not add a `CNAME` file.
5. Wait for the Pages deployment to succeed. The settings page displays the published address and deployment status.

The intended homepage is <https://gpeyre.github.io/iot-curvature/> and the demo is <https://gpeyre.github.io/iot-curvature/demo.html>. These addresses become available when the first deployment succeeds; this repository does not enable Pages on your behalf. Once enabled, later pushes to `main` update the site. Relative asset and navigation links support the `/iot-curvature/` project subpath.

All project links and canonical URLs use `gpeyre.github.io`. GitHub Pages can inherit a custom domain from the account's user site even when this repository has no `CNAME`. If Pages redirects the project to a personal domain, removing that inherited domain requires changing the user-site Pages setting; editing project links alone does not change the hosting redirect. See [GitHub's domain inheritance documentation](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages). Review that account-wide setting separately because it also affects the user site and other project sites.

## What is computed

The candidate cost is `c_u(x,y) = -xᵀ diag(u,1-u)y`, for `0 ≤ u ≤ 1`. For each `u`, the Hungarian algorithm finds the exact combinatorial assignment, up to floating-point arithmetic. There is no entropy regularization. We plot the **mean** score improvement over a fixed observed pairing:

```
L_n(u) = [ max_sigma Σ_i X_iᵀ A(u) Y_sigma(i)
           - Σ_i X_iᵀ A(u) Y_reference(i) ] / n.
```

The reference is never recomputed as `u` varies. It is the identity for true paired samples, or an optimal assignment computed once at `u=1/2` for independent marginals.

The two maps match the manuscript formulas:

- **Cosine:** `T_delta(x) = x - delta*(sin(x1)+sin(x1+x2), sin(x2)+sin(x1+x2))`, with `0 ≤ delta ≤ .329 < 1/3`.
- **Annulus:** the radial quantile map from `N(0,I_2)` to the uniform annulus with radii `.8` and `.85`.

In the paired scheme, `X_i ~ N(0,I_2)` and `Y_i=T(X_i)`. In the independent scheme, `Y_i=T(Z_i)` for an independent Gaussian sample `Z`. Comparisons use a common source sample. Each scheme has its own reference assignment. These are different finite-sample observation models, even though their target marginals agree in distribution.

Each curve contains **41 exact evaluations on a uniform grid**, joined by straight lines. The selected slider value is also evaluated exactly, independently of that grid. The joined curve can miss corners or narrow plateaus and carries **no interpolation error certificate**. A linearly interpolated chord of a convex loss lies above the true loss between its endpoints. All `n` points enter every assignment and all are displayed. The browser cap of 160 keeps exploration responsive; the Python figure scripts support the larger manuscript sample sizes.

The `delta=0` paired baseline is analytically zero on the entire PSD slice. Optional division by `delta²` changes only the vertical scale and is unavailable when `delta=0`. It does not remove empirical plateaus or assert an asymptotic theorem.

Floating-point negative gaps within `5e-13*(1+|optimal mean score|+|reference mean score|)` are displayed as zero. A larger negative gap triggers an error. Tied assignments can change the displayed pair count without changing the gap.

The deterministic browser generator is Mulberry32 plus Box–Muller, with an independent-marginal seed derived by XOR with `0x9E3779B9`. It is intentionally independent of NumPy's generator: the same numeric seed **does not** produce the Python manuscript samples. Downloaded CSV files retain the model, sampling scheme, seed, `n`, `delta`, raw mean loss, and displayed normalization.

## Files and numerical API

| File | Purpose |
| --- | --- |
| `index.html`, `landing.css` | Project homepage with access to the demo and GitHub content |
| `demo.html`, `styles.css` | Accessible, responsive experiment interface and mathematical description |
| `assets/` | Committed images used by the project homepage |
| `app.mjs` | SVG rendering, controls, status, CSV export |
| `worker.mjs` | Background sampling and exact assignment evaluations |
| `math.mjs` | Pure deterministic mathematical kernel, no DOM dependencies |
| `test_math.mjs` | Brute-force assignment validation and model checks |
| `favicon.svg`, `.nojekyll` | Site icon and static GitHub Pages configuration |

```js
import {sampleData, solveAt, curve} from './docs/math.mjs';
const data = sampleData({
  model: 'cosine', // or 'annulus'
  sampling: 'paired', // or 'independent'
  n: 60, delta: 0.3, seed: 1
});
const result = solveAt(data, 0.7); // gap, rawGap, assignment, optimalScore, referenceScore
const values = curve(data, 41); // array of {u, gap}
```

Other exports: `random`, `normalPoints`, `transform`, `scoreMatrix`, and `hungarianMax`. The latter accepts a row-major `Float64Array` and its square dimension and returns `{assignment,score}`.

Run the dependency-free kernel checks with:

```sh
node docs/test_math.mjs
```

They compare 84 random assignment problems of sizes 1–7 against exhaustive permutations; check convexity and optimality at `u=1/2` in all four model/protocol combinations; check the exactly degenerate paired identity baseline; verify annulus radii and Gaussian sampling moments; and time one 160-point curve.

The interaction is inspired by [Gabriel Peyré's post on the convexity of the inverse gap loss](https://www.gpeyre.com/blog/2026/10/03/inverse-optimisation-gap-loss/). This implementation is independently written and uses the manuscript's cosine and radial transport models.

## Update the website assets

The committed website is ready to serve. After changing the publication figures, refresh the copied previews and the small numerical illustration with:

```sh
python scripts/prepare_site.py
python scripts/check_site.py
```

The illustration uses an annulus sample with `n=60`, NumPy seed 7, and the candidate cost `u=0.85`. Its gap curve contains 41 exact assignment evaluations. The four publication previews are direct copies of their generated PNG exports. The static-site check verifies their checksums, local links, page anchors and module dependencies, including compatibility with the GitHub Pages project subpath.

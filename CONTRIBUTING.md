# Local exploration and development

The [project website](https://gpeyre.github.io/iot-curvature/) is the main entry point for the demo, figures and notebooks. The [figure guide](figures/README.md#installation-and-reproduction) gives the Python environment and reproduction instructions.

## Explore in Jupyter

From the installed repository:

```sh
python -m pip install -e '.[notebooks]'
jupyter notebook notebooks/
```

The notebooks find this repository from their own location. The first two run at modest sample sizes; the third reads the committed large-sample results. Optional widgets allow changes to the sample size, perturbation, model and seed. To execute all notebooks without opening a server:

```sh
python scripts/check_notebooks.py
```

## Run the checks

Use Python 3.12 and Node.js 24, matching the GitHub workflow:

```sh
python -m unittest discover -s tests -v
python scripts/check_data.py
python figures/validate.py
node docs/test_math.mjs
python scripts/check_browser_math.py
python scripts/check_site.py
python scripts/check_notebooks.py
```

These checks cover small exact assignments, the loss sign and normalization, zero baseline and true-cost gaps, convexity, radial sampling, deterministic seeds, committed-data integrity, and agreement between browser and SciPy objectives. The workflow runs checks and notebook execution; it does not recompute the expensive full experiments on every push.

## Update the website

The project homepage URL is **https://gpeyre.github.io/iot-curvature/**. Preview it locally from the repository root:

```sh
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000/>. See the [website guide](docs/README.md) for GitHub Pages setup, the browser model and updating figure previews.

## Use the figures in a manuscript

```sh
python scripts/update_manuscript_figures.py --paper /path/to/iOT_JEMS_revised
```

This copies the four publication PDFs to `Pictures/generated/` in the chosen manuscript and records their matching checksums. Include those files directly from LaTeX. The [figure guide](figures/README.md) maps the stable output filenames to the article's section-based figure numbers.

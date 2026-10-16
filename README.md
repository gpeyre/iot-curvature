# Inverse optimal transport: the geometry of the gap

Numerical companion to *Second-order geometry of optimal transport: cost identifiability and stability*, by Gabriel Peyré, Clarice Poon and Oscar Tron.

**[Project website](https://gpeyre.github.io/iot-curvature/)** · **[Interactive demo](https://gpeyre.github.io/iot-curvature/demo.html)**

Start on the **project website** to explore the models, view the [article figures](https://gpeyre.github.io/iot-curvature/#figures), and access the [code and Jupyter notebooks](https://gpeyre.github.io/iot-curvature/#resources). The demo runs directly in your browser.

## Install and reproduce

With Python 3.12, render all four article figures from the included data:

```sh
git clone https://github.com/gpeyre/iot-curvature.git
cd iot-curvature
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python figures/reproduce.py --render-only
```

Outputs include vector PDF and SVG, plus PNG previews. For environment setup, full recomputation, equations, seeds and sampling protocols, see the [figure guide](figures/README.md). The website also has a [reproduction quick start](https://gpeyre.github.io/iot-curvature/#reproduce).

## Repository contents

| Directory | Content |
|---|---|
| [`figures/`](figures/) | Reproduction scripts, recorded data and publication figures |
| [`notebooks/`](notebooks/) | Three guided explorations of models, sampling and article data |
| [`docs/`](docs/) | Source of the project website and interactive demo |
| [`src/iot_gap/`](src/iot_gap/) | Shared numerical implementation |

[Website and deployment guide](docs/README.md) · [Local exploration and checks](CONTRIBUTING.md) · [Citation](CITATION.cff)

Released under the [MIT License](LICENSE).

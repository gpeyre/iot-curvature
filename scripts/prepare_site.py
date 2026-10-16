#!/usr/bin/env python3
"""Refresh committed website images from publication assets and exact OT data.

Run from any directory with the numerical package's dependencies installed:
    python /path/to/iot-curvature/scripts/prepare_site.py

The static website already includes the resulting assets; visitors and GitHub
Pages do not need to run this helper or install the numerical package.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from iot_gap import gap_curve, optimal_assignment, sample_pairs


def _nice_upper(value: float) -> float:
    """Round a positive maximum up to a readable axis limit."""
    magnitude = 10 ** math.floor(math.log10(value))
    return next(s for s in (1, 2, 2.5, 5, 10) if s * magnitude >= value) * magnitude


def experiment_preview() -> str:
    """Draw actual paired observations, one optimum, and the exact grid loss."""
    n, seed, selected_u = 60, 7, .85
    source, target = sample_pairs("annulus", n, seed=seed)
    us = np.linspace(0., 1., 41)
    gaps = gap_curve(source, target, us)
    assignment = optimal_assignment(source, target, selected_u)
    selected_gap = float(gap_curve(source, target, [selected_u])[0])

    blue, orange, teal = "#547eb8", "#de8b4e", "#087f80"
    ink, muted, line = "#253b3d", "#647477", "#dce4e2"
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 490" '
        'role="img" aria-labelledby="preview-title preview-desc">',
        '<title id="preview-title">Paired annulus transport and its empirical gap</title>',
        '<desc id="preview-desc">Sixty independent Gaussian source points and their '
        'radial annulus images, using seed seven. Lines show the exact optimal '
        'assignment for the cost matrix diag(0.85, 0.15). Source and target '
        'clouds use the same coordinate scale and are translated apart. The '
        'lower graph shows the mean score improvement over the observed pairing '
        'at forty-one equally spaced cost parameters, with the selected '
        'parameter highlighted. Lines between evaluated losses are visual guides.</desc>',
        '<style>text{font-family:Arial,Helvetica,sans-serif} '
        '.title{font-family:Georgia,"Times New Roman",serif} '
        '.tick{font-size:10px;fill:#647477} '
        '.label{font-size:10px;letter-spacing:1.5px;fill:#647477}</style>',
        '<rect width="560" height="490" rx="18" fill="#fafbf8"/>',
        f'<text class="title" x="30" y="34" font-size="21" fill="{ink}">'
        'An observed pairing, a changing cost</text>',
        f'<text x="30" y="57" font-size="12" fill="{muted}">'
        'n = 60 · u = 0.85</text>',
        '<text class="label" x="139" y="87" text-anchor="middle">SOURCE</text>',
        '<text class="label" x="413" y="87" text-anchor="middle">TARGET</text>',
    ]

    # Both clouds use the same scale. Only horizontal translation separates
    # them; each connection is the actual optimum at the selected parameter.
    scale, center_y = 34., 177.
    px = np.column_stack((139 + scale * source[:, 0], center_y - scale * source[:, 1]))
    py = np.column_stack((413 + scale * target[:, 0], center_y - scale * target[:, 1]))
    parts.append(f'<g stroke="{teal}" stroke-width=".85" opacity=".27">')
    for i, j in enumerate(assignment):
        parts.append(f'<line x1="{px[i, 0]:.3f}" y1="{px[i, 1]:.3f}" '
                     f'x2="{py[j, 0]:.3f}" y2="{py[j, 1]:.3f}"/>')
    parts.append('</g>')
    parts.append(f'<g fill="{blue}" stroke="#fafbf8" stroke-width=".6">')
    for x, y in px:
        parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3.1"/>')
    parts.append('</g>')
    parts.append(f'<g fill="{orange}" stroke="#fafbf8" stroke-width=".6">')
    for x, y in py:
        parts.append(f'<rect x="{x-2.8:.3f}" y="{y-2.8:.3f}" width="5.6" '
                     'height="5.6" rx=".55"/>')
    parts.extend([
        '</g>',
        f'<text x="280" y="270" text-anchor="middle" font-size="11" fill="{muted}">'
        'Links show the optimal assignment for the selected cost.</text>',
        '<rect x="30" y="286" width="500" height="168" rx="10" fill="#fff" '
        f'stroke="{line}"/>',
        f'<text x="46" y="308" font-size="12" fill="{ink}">Mean gap Lₙ(u)</text>',
    ])

    left, right, top, bottom = 67., 505., 326., 420.
    ymax = _nice_upper(float(gaps.max()) * 1.06)
    gx = lambda u: left + (right - left) * u
    gy = lambda g: bottom - (bottom - top) * g / ymax
    for value in (0., ymax / 2, ymax):
        y = gy(value)
        parts.append(f'<line x1="{left}" y1="{y:.3f}" x2="{right}" '
                     f'y2="{y:.3f}" stroke="{line}" stroke-width=".8"/>')
        label = "0" if value == 0 else f"{value:.3g}"
        parts.append(f'<text class="tick" x="{left-9}" y="{y+3:.3f}" '
                     f'text-anchor="end">{label}</text>')
    parts.append(f'<line x1="{gx(.5):.3f}" y1="{top}" x2="{gx(.5):.3f}" '
                 f'y2="{bottom}" stroke="{line}" stroke-dasharray="3 4"/>')
    for u, label in ((0., "0"), (.5, "0.5"), (1., "1")):
        x = gx(u)
        parts.append(f'<line x1="{x:.3f}" y1="{bottom}" x2="{x:.3f}" '
                     f'y2="{bottom+4}" stroke="{muted}"/>')
        parts.append(f'<text class="tick" x="{x:.3f}" y="{bottom+18}" '
                     f'text-anchor="middle">{label}</text>')
    parts.append(f'<text class="tick" x="{right+12}" y="{bottom+18}">u</text>')
    points = " ".join(f"{gx(float(u)):.3f},{gy(float(g)):.3f}" for u, g in zip(us, gaps))
    parts.append(f'<polyline points="{points}" fill="none" stroke="{teal}" '
                 'stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(f'<g fill="{teal}">')
    for u, g in zip(us, gaps):
        parts.append(f'<circle cx="{gx(float(u)):.3f}" cy="{gy(float(g)):.3f}" r="1.7"/>')
    parts.append('</g>')
    x, y = gx(selected_u), gy(selected_gap)
    parts.extend([
        f'<line x1="{x:.3f}" y1="{y:.3f}" x2="{x:.3f}" y2="{bottom}" '
        f'stroke="{teal}" stroke-width="1" stroke-dasharray="3 3" opacity=".55"/>',
        f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4.6" fill="{teal}" stroke="#fff" stroke-width="1.8"/>',
        f'<text x="{x-8:.3f}" y="{y-13:.3f}" text-anchor="end" '
        f'font-size="10" fill="{teal}">u = 0.85</text>',
        f'<text x="280" y="478" text-anchor="middle" font-size="11" fill="{muted}">'
        'Paired annulus sample · exact empirical assignments</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def main() -> None:
    assets = ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    names = (
        "figure_1_cycles.png", "figure_2_cosine.png",
        "figure_3_annulus.png", "figure_4_annulus_maps.png",
    )
    for name in names:
        output = assets / name
        shutil.copyfile(ROOT / "figures" / "generated" / name, output)
        print(output.relative_to(ROOT))
    output = assets / "experiment-preview.svg"
    output.write_text(experiment_preview(), encoding="utf-8")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()

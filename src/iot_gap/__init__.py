"""Exact empirical inverse-optimal-transport gap experiments.

The score is x.T @ diag(u, 1-u) @ y. All empirical measures have
equal weights 1/n; assignment objectives and observed scores use this weight.
"""

from .core import (
    annulus_map, cosine_map, gap_curve, identity_map, optimal_assignment,
    sample_independent_marginals, sample_pairs, score, sampling_gap_curve,
)

__all__ = [
    "annulus_map", "cosine_map", "gap_curve", "identity_map",
    "optimal_assignment", "sample_independent_marginals", "sample_pairs",
    "score", "sampling_gap_curve",
]

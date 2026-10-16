"""Numerical definitions shared by figures, notebooks and tests."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment


def _points(X: ArrayLike, name: str = "X") -> NDArray[np.float64]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError(f"{name} must be a nonempty finite array with shape (n, 2)")
    return X


def identity_map(X: ArrayLike) -> NDArray[np.float64]:
    return _points(X).copy()


def cosine_map(X: ArrayLike, delta: float) -> NDArray[np.float64]:
    """Gradient of |x|²/2 + delta*(cos x1+cos x2+cos(x1+x2)).

    The publication uses 0 <= delta < 1/3, where the potential is strongly
    convex. The formula itself is defined for every finite delta.
    """
    X = _points(X)
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    s = np.sin(X[:, 0] + X[:, 1])
    return X - delta * (np.sin(X) + s[:, None])


def annulus_map(X: ArrayLike, a: float = .8, b: float = .85) -> NDArray[np.float64]:
    """Radial Gaussian-to-uniform-annulus quantile map, defined a.e.

    T(x) = x/|x| sqrt(a²+(b²-a²)(1-exp(-|x|²/2))).
    At the measure-zero input x=0 choose the measurable value (a,0).
    expm1 avoids cancellation for small radii; no positive-radius clipping.
    """
    X = _points(X)
    if not (0 <= a < b and np.isfinite(b)):
        raise ValueError("annulus radii must satisfy 0 <= a < b < infinity")
    r = np.linalg.norm(X, axis=1)
    q = -np.expm1(-.5 * r * r)
    rho = np.sqrt(a*a + (b*b-a*a) * q)
    Y = np.zeros_like(X)
    nonzero = r > 0
    Y[nonzero] = X[nonzero] * (rho[nonzero] / r[nonzero])[:, None]
    Y[~nonzero, 0] = a
    return Y


def _map(X: NDArray, model: str, delta: float, a: float, b: float) -> NDArray:
    if model == "cosine":
        return cosine_map(X, delta)
    if model == "annulus":
        return annulus_map(X, a, b)
    if model == "identity":
        return identity_map(X)
    raise ValueError("model must be 'cosine', 'annulus', or 'identity'")


def sample_pairs(model: str, n: int, seed: int, delta: float = .1,
                 a: float = .8, b: float = .85) -> tuple[NDArray, NDArray]:
    """i.i.d. pairs from pi=(Id,T)#N(0,I₂), using an explicit local RNG."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    X = np.random.default_rng(seed).standard_normal((n, 2))
    return X, _map(X, model, delta, a, b)


def sample_independent_marginals(model: str, n: int, seed: int,
                                 delta: float = .1, a: float = .8,
                                 b: float = .85) -> tuple[NDArray, NDArray]:
    """Independent empirical marginals; these are not paired observations."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    Z = rng.standard_normal((n, 2))
    return X, _map(Z, model, delta, a, b)


def _data(X: ArrayLike, Y: ArrayLike) -> tuple[NDArray, NDArray]:
    X, Y = _points(X), _points(Y, "Y")
    if X.shape != Y.shape:
        raise ValueError("X and Y must have equal shape")
    return X, Y


def _u(u: float) -> float:
    if not np.isfinite(u) or not 0 <= u <= 1:
        raise ValueError("u must be in [0,1]")
    return float(u)


def _permutation(reference: ArrayLike | None, n: int) -> NDArray[np.int64]:
    if reference is None:
        return np.arange(n)
    p = np.asarray(reference)
    if p.shape != (n,) or not np.issubdtype(p.dtype, np.integer) or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError("reference must be a permutation of range(n)")
    return p.astype(np.int64, copy=False)


def score(X: ArrayLike, Y: ArrayLike, u: float, reference: ArrayLike | None = None) -> float:
    """Mean bilinear score of a specified empirical coupling (identity by default)."""
    X, Y = _data(X, Y)
    p, u = _permutation(reference, len(X)), _u(u)
    return float(np.mean(u * X[:, 0]*Y[p, 0] + (1-u) * X[:, 1]*Y[p, 1]))


def optimal_assignment(X: ArrayLike, Y: ArrayLike, u: float) -> NDArray[np.int64]:
    """Exact maximum-score equal-weight assignment, returned as row permutation.

    Rank-one endpoints use the rearrangement inequality (sorting), with the
    same exact optimal objective as a Hungarian solve. Interior points use
    scipy.optimize.linear_sum_assignment; no regularization is introduced.
    """
    X, Y = _data(X, Y)
    u = _u(u)
    if u in (0., 1.):
        coordinate = int(u)
        # u=0 weights coordinate 1; u=1 weights coordinate 0.
        coordinate = 1-coordinate
        p = np.empty(len(X), dtype=np.int64)
        p[np.argsort(X[:, coordinate], kind="stable")] = np.argsort(Y[:, coordinate], kind="stable")
        return p
    K = u * np.outer(X[:, 0], Y[:, 0]) + (1-u) * np.outer(X[:, 1], Y[:, 1])
    rows, columns = linear_sum_assignment(K, maximize=True)
    p = np.empty(len(X), dtype=np.int64)
    p[rows] = columns
    return p


def gap_curve(X: ArrayLike, Y: ArrayLike, us: ArrayLike,
              reference: ArrayLike | None = None) -> NDArray[np.float64]:
    """Exact sampled gap sup_perm E[x.T A y_perm] - E[x.T A y_ref].

    The default reference is the observed paired plan. For independently
    sampled marginals, pass reference=optimal_assignment(X,Y,.5) to study
    an empirical OT reference instead. Only negative floating-point noise
    within a scale-dependent tolerance is clipped.
    """
    X, Y = _data(X, Y)
    us = np.asarray(us, dtype=float)
    if us.ndim != 1 or not np.isfinite(us).all() or np.any((us < 0) | (us > 1)):
        raise ValueError("us must be a finite vector in [0,1]")
    p = _permutation(reference, len(X))
    K1 = np.outer(X[:, 0], Y[:, 0])
    K2 = np.outer(X[:, 1], Y[:, 1])
    obs1, obs2 = np.mean(X[:, 0]*Y[p, 0]), np.mean(X[:, 1]*Y[p, 1])
    out = np.empty(len(us))
    for j, u in enumerate(us):
        if u in (0., 1.):
            opt = score(X, Y, u, optimal_assignment(X, Y, u))
        else:
            K = u*K1 + (1-u)*K2
            rows, columns = linear_sum_assignment(K, maximize=True)
            opt = float(K[rows, columns].sum()/len(X))
        observed = u*obs1+(1-u)*obs2
        out[j] = opt-observed
        tolerance = 128*np.finfo(float).eps * max(1., abs(opt), abs(observed))
        if out[j] < -tolerance:
            raise ArithmeticError(f"assignment score below feasible reference by {out[j]}")
    return np.maximum(out, 0.)


def sampling_gap_curve(model: str, n: int, seed: int, us: ArrayLike,
                       delta: float = .1, scheme: str = "paired") -> NDArray:
    """Convenience comparison of paired and independently matched sampling."""
    if scheme == "paired":
        X, Y = sample_pairs(model, n, seed, delta)
        return gap_curve(X, Y, us)
    if scheme == "independent_ot":
        X, Y = sample_independent_marginals(model, n, seed, delta)
        return gap_curve(X, Y, us, reference=optimal_assignment(X, Y, .5))
    raise ValueError("scheme must be 'paired' or 'independent_ot'")

"""Validate the committed publication data, seeds, formulas and checksums."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'figures' / 'data'
manifest = json.loads((DATA / 'manifest.json').read_text())
assert not manifest['preview'], 'Preview data cannot stand in for article results'
for relative, expected in manifest['checksums'].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
for relative, expected in manifest['code_checksums'].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
worst_center = worst_oracle = worst_convexity = 0.
tasks = manifest['tasks']
assert len(tasks) == 56
for spec in tasks:
    delta = '' if spec['delta'] is None else f"_delta{spec['delta']:g}"
    path = DATA / f"{spec['model']}{delta}_n{spec['n']}_rep{spec['rep']}.npz"
    with np.load(path, allow_pickle=False) as z:
        X, Y, us, gaps = (z[key] for key in ('X','Y','u','gap'))
        rng = np.random.default_rng(spec['seed'])
        for _ in range(spec['rep'] + 1): expected_X = rng.standard_normal((spec['n'], 2))
        assert np.array_equal(X, expected_X), path.name
        if spec['model'] == 'cosine':
            expected_Y = X - spec['delta']*np.stack([np.sin(X[:,0])+np.sin(X.sum(axis=1)), np.sin(X[:,1])+np.sin(X.sum(axis=1))], axis=1)
        else:
            r = np.linalg.norm(X, axis=1)
            rho = np.sqrt(.8**2+(.85**2-.8**2)*(-np.expm1(-r*r/2)))
            expected_Y = X*(rho/r)[:,None]
        assert np.allclose(Y, expected_Y, rtol=1e-14, atol=1e-14), path.name
        assert np.array_equal(us, np.asarray(spec['grid']))
        assert np.isfinite(gaps).all() and gaps.min() >= 0
        center = gaps[np.isclose(us, .5, atol=1e-14, rtol=0)].max()
        assert center < 1e-11, path.name
        worst_center = max(worst_center, float(center))
        # Collapse floating-point near-duplicates for a stable convexity test,
        # without changing the stored or rendered original grid.
        indices = np.unique(np.round(us, 13), return_index=True)[1]
        slopes = np.diff(gaps[indices])/np.diff(us[indices])
        violation = max(0., -float(np.diff(slopes).min()))
        assert violation < 1e-10, (path.name, violation)
        worst_convexity = max(worst_convexity, violation)
        # Reconstruct one interior original-notebook objective independently.
        j = int(np.argmin(abs(us-.3))); u = us[j]
        K = u*np.outer(X[:,0],Y[:,0])+(1-u)*np.outer(X[:,1],Y[:,1])
        rows, cols = linear_sum_assignment(K, maximize=True)
        direct = (K[rows,cols].sum()-np.trace(K))/len(X)
        error = abs(float(gaps[j])-direct)
        assert error < 1e-11, (path.name, error)
        worst_oracle = max(worst_oracle, error)
with np.load(DATA / 'annulus_assignments.npz', allow_pickle=False) as z:
    # Figure 4 illustrates forward OT between independent empirical marginals;
    # Figure 3 retains paired sampling from the population transport plan.
    X, Y, us, permutations = (z[key] for key in ('X','Y','u','permutations'))
    spec = json.loads(str(z['spec_json']))
    assert X.shape == Y.shape == (1000,2)
    assert np.array_equal(us, [.5,.9,.1]) and permutations.shape == (3,1000)
    assert int(z['seed']) == spec['seed'] == 42
    assert spec['n'] == 1000 and spec['a'] == .8 and spec['b'] == .85
    assert spec['sampling'] == 'independent_empirical_marginals'
    assert spec['spec_version'] == 3
    assert spec['computation_sha256'] == hashlib.sha256((ROOT / 'src/iot_gap/core.py').read_bytes()).hexdigest()
    rng = np.random.default_rng(42)
    expected_X = rng.standard_normal((1000,2))
    radii = np.sqrt(rng.uniform(.8**2,.85**2,1000))
    angles = rng.uniform(0,2*np.pi,1000)
    expected_Y = np.column_stack((radii*np.cos(angles),radii*np.sin(angles)))
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_allclose(Y, expected_Y, rtol=1e-14, atol=1e-14)
    worst_arrow_objective = 0.
    for u, p in zip(us, permutations):
        np.testing.assert_array_equal(np.sort(p), np.arange(1000))
        K = u*np.outer(X[:,0],Y[:,0])+(1-u)*np.outer(X[:,1],Y[:,1])
        rows, cols = linear_sum_assignment(K, maximize=True)
        error = abs(float(K[np.arange(1000),p].sum()-K[rows,cols].sum()))/1000
        assert error < 1e-11, ('Figure 4', float(u), error)
        worst_arrow_objective = max(worst_arrow_objective, error)
result = {'passed':True, 'full_experiment_tasks':len(tasks), 'figure_assets_checked':12,
          'max_true_cost_gap':worst_center, 'max_direct_objective_error':worst_oracle,
          'max_convexity_slope_violation':worst_convexity,
          'all_manifest_checksums_match':True,
          'figure_4_sampling':'independent_empirical_marginals',
          'figure_4_max_direct_objective_error':worst_arrow_objective}
destination = ROOT / 'validation'
destination.mkdir(exist_ok=True)
(destination / 'data_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))


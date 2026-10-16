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
    assert z['X'].shape == (1000,2) and np.array_equal(z['u'], [.5,.9,.1])
    assert np.array_equal(z['X'], np.random.default_rng(1000).standard_normal((1000,2)))
    for p in z['permutations']: assert np.array_equal(np.sort(p), np.arange(1000))
    assert np.array_equal(z['permutations'][0], np.arange(1000))
result = {'passed':True, 'full_experiment_tasks':len(tasks), 'figure_assets_checked':12,
          'max_true_cost_gap':worst_center, 'max_direct_objective_error':worst_oracle,
          'max_convexity_slope_violation':worst_convexity,
          'all_manifest_checksums_match':True, 'figure_4_paired_identity_at_true_cost':True}
destination = ROOT / 'validation'
destination.mkdir(exist_ok=True)
(destination / 'data_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))


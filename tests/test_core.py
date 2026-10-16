"""Small exact mathematical checks; no costly publication regeneration."""
import itertools
import unittest

import numpy as np

from iot_gap import (annulus_map, cosine_map, gap_curve, optimal_assignment,
                     sample_independent_marginals, sample_pairs, score)


class ExactGapTests(unittest.TestCase):
    def test_assignment_and_gap_against_all_permutations(self):
        X = np.array([[1.,-2.],[.5,3.],[-1.,1.],[2.,.1]])
        Y = np.array([[.1,2.],[-1.,1.],[3.,-.2],[.5,-2.]])
        for u in [0., .13, .5, .91, 1.]:
            candidates = [score(X,Y,u,p) for p in itertools.permutations(range(len(X)))]
            p = optimal_assignment(X,Y,u)
            self.assertAlmostEqual(score(X,Y,u,p),max(candidates),places=13)
            expected = max(candidates)-score(X,Y,u)
            self.assertAlmostEqual(gap_curve(X,Y,[u])[0],expected,places=13)

    def test_sign_and_probability_normalization(self):
        X = np.array([[-1.,0.],[1.,0.]])
        Y = X[::-1].copy()
        # Mean observed score=-1, optimal=+1: gap=2, not n*gap.
        self.assertAlmostEqual(gap_curve(X,Y,[1.])[0],2.)
        self.assertAlmostEqual(gap_curve(X,Y,[1.],reference=np.array([1,0]))[0],0.)

    def test_convex_sampled_gap_slopes(self):
        X,Y = sample_pairs("cosine",16,123,delta=.3)
        us = np.linspace(0,1,31)
        gaps = gap_curve(X,Y,us)
        slopes = np.diff(gaps)/np.diff(us)
        self.assertGreaterEqual(np.diff(slopes).min(),-1e-11)

    def test_true_paired_plan_zero_at_center(self):
        for model in ["identity","cosine","annulus"]:
            X,Y = sample_pairs(model,30,4,delta=.32)
            self.assertLess(gap_curve(X,Y,[.5])[0],1e-12)

    def test_identity_nonidentifiability_on_psd_slice(self):
        X,Y = sample_pairs("identity",20,10)
        np.testing.assert_allclose(gap_curve(X,Y,np.linspace(0,1,21)),0,atol=1e-12)

    def test_independent_ot_reference_zero_at_center(self):
        X,Y = sample_independent_marginals("annulus",24,38)
        p=optimal_assignment(X,Y,.5)
        self.assertLess(gap_curve(X,Y,[.5],reference=p)[0],1e-12)
        self.assertGreater(np.max(np.abs(Y-annulus_map(X))),.1)

    def test_seed_is_local_and_repeatable(self):
        np.random.seed(18)
        before=np.random.get_state()
        X1,Y1=sample_pairs("cosine",8,19,.1)
        X2,Y2=sample_pairs("cosine",8,19,.3)
        after=np.random.get_state()
        np.testing.assert_array_equal(X1,X2)
        np.testing.assert_array_equal(before[1],after[1])
        self.assertFalse(np.array_equal(Y1,Y2))

    def test_annulus_radial_quantile_and_null_extension(self):
        radii=np.array([0.,1e-16,.01,.5,2.,9.])
        X=np.column_stack((radii,np.zeros_like(radii)))
        Y=annulus_map(X)
        rho=np.linalg.norm(Y,axis=1)
        q=(rho**2-.8**2)/(.85**2-.8**2)
        np.testing.assert_allclose(q,-np.expm1(-radii**2/2),atol=5e-15)
        self.assertTrue(np.isfinite(Y).all())
        self.assertGreaterEqual(rho.min(),.8-1e-15)
        self.assertLessEqual(rho.max(),.85+1e-15)

    def test_cosine_gradient_and_hessian_bound(self):
        X=np.random.default_rng(1).normal(size=(40,2))
        delta=.32
        step=1e-5
        def potential(X):
            return .5*np.sum(X**2,axis=1)+delta*(np.cos(X[:,0])+np.cos(X[:,1])+np.cos(X.sum(axis=1)))
        numerical=np.empty_like(X)
        for j in range(2):
            E=np.zeros_like(X);E[:,j]=step
            numerical[:,j]=(potential(X+E)-potential(X-E))/(2*step)
        np.testing.assert_allclose(cosine_map(X,delta),numerical,atol=1e-9)
        for x in X:
            s=np.cos(x.sum())
            h=np.eye(2)-delta*np.array([[np.cos(x[0])+s,s],[s,np.cos(x[1])+s]])
            self.assertGreaterEqual(np.linalg.eigvalsh(h).min(),1-3*delta-1e-14)

    def test_invalid_inputs_fail(self):
        X,Y=sample_pairs("identity",3,2)
        with self.assertRaises(ValueError):gap_curve(X,Y,[1.1])
        with self.assertRaises(ValueError):gap_curve(X,Y,[.5],reference=[0,0,2])
        with self.assertRaises(ValueError):annulus_map(X,a=1,b=.9)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Validate saved publication data and manifest without repeating large solves."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from scipy.optimize import linear_sum_assignment
from iot_gap import annulus_map,cosine_map


def main():
    data=ROOT/"figures/data"
    manifest=json.loads((data/"manifest.json").read_text())
    checks={"tasks":0,"reference_notebook_objectives":0,
            "max_reference_gap_difference":0.,"max_center_gap":0.,
            "max_radial_map_difference":0.,"minimum_loss":float("inf"),
            "maximum_convexity_violation":0.,"checksum_files":0}
    for name,expected in manifest["checksums"].items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=expected:raise AssertionError(f"Checksum mismatch: {name}")
        checks["checksum_files"]+=1
    for name,expected in manifest["code_checksums"].items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=expected:raise AssertionError(f"Code checksum mismatch: {name}; rerender to update manifest")
    for path in sorted(data.glob("*.npz")):
        if path.name=="annulus_assignments.npz":continue
        with np.load(path,allow_pickle=False) as z:
            X,Y,u,gap=z["X"],z["Y"],z["u"],z["gap"]
            spec=json.loads(str(z["spec_json"]))
        if spec["computation_sha256"]!=hashlib.sha256((ROOT/"src/iot_gap/core.py").read_bytes()).hexdigest():
            raise AssertionError(f"Numerical source checksum mismatch: {path}")
        rng=np.random.default_rng(spec["seed"])
        for _ in range(spec["rep"]+1):expected_X=rng.standard_normal((spec["n"],2))
        np.testing.assert_array_equal(X,expected_X)
        if spec["model"]=="cosine":
            s=np.sin(X.sum(axis=1))
            notebook_Y=X-spec["delta"]*np.stack([np.sin(X[:,0])+s,np.sin(X[:,1])+s],axis=1)
        else:
            r=np.maximum(np.linalg.norm(X,axis=1),1e-12)
            notebook_Y=X*(np.sqrt(.8**2+(1-np.exp(-r*r/2))*(.85**2-.8**2))/r)[:,None]
            checks["max_radial_map_difference"]=max(checks["max_radial_map_difference"],float(np.max(np.abs(Y-notebook_Y))))
        np.testing.assert_allclose(Y,notebook_Y,atol=5e-15,rtol=5e-15)
        center=gap[np.isclose(u,.5,rtol=0,atol=1e-14)]
        checks["max_center_gap"]=max(checks["max_center_gap"],float(center.max()))
        checks["minimum_loss"]=min(checks["minimum_loss"],float(gap.min()))
        # Exclude only adjacent binary representations separated by ~1e-16
        # from the slope diagnostic, not from computations or saved arrays.
        keep=np.r_[True,np.diff(u)>1e-10]
        slopes=np.diff(gap[keep])/np.diff(u[keep])
        violation=max(0.,float(-np.diff(slopes).min()))
        checks["maximum_convexity_violation"]=max(checks["maximum_convexity_violation"],violation)
        if violation>1e-9:raise AssertionError(f"Convexity violation: {path.name}")
        if spec["n"]<=100:
            K1=np.outer(X[:,0],notebook_Y[:,0]);K2=np.outer(X[:,1],notebook_Y[:,1])
            reference=[]
            for v in u:
                K=v*K1+(1-v)*K2
                rows,cols=linear_sum_assignment(K,maximize=True)
                reference.append(max((K[rows,cols].sum()-np.trace(K))/len(X),0.))
            error=float(np.max(np.abs(np.asarray(reference)-gap)))
            checks["max_reference_gap_difference"]=max(checks["max_reference_gap_difference"],error)
            checks["reference_notebook_objectives"]+=len(u)
            if error>1e-12:raise AssertionError(f"Original notebook discrepancy: {path.name}")
        checks["tasks"]+=1
    if checks["max_center_gap"]>1e-11:raise AssertionError("Nonzero true-parameter loss")
    with np.load(data/"annulus_assignments.npz",allow_pickle=False) as arrows:
        X,Y,u,permutations=(arrows[key] for key in ("X","Y","u","permutations"))
        arrow_spec=json.loads(str(arrows["spec_json"]))
        if X.shape!=(1000,2) or Y.shape!=(1000,2) or permutations.shape!=(3,1000):
            raise AssertionError("Unexpected Figure 4 sample or assignment shape")
        np.testing.assert_array_equal(u,[.5,.9,.1])
        if int(arrows["seed"])!=42 or arrow_spec["seed"]!=42:
            raise AssertionError("Unexpected Figure 4 seed")
        if arrow_spec["n"]!=1000 or arrow_spec["a"]!=.8 or arrow_spec["b"]!=.85:
            raise AssertionError("Unexpected Figure 4 model parameters")
        if arrow_spec["sampling"]!="independent_empirical_marginals" or arrow_spec["spec_version"]!=3:
            raise AssertionError("Figure 4 requires the independent empirical-marginal protocol")
        if arrow_spec["computation_sha256"]!=hashlib.sha256((ROOT/"src/iot_gap/core.py").read_bytes()).hexdigest():
            raise AssertionError("Figure 4 numerical source checksum mismatch")
        # Reconstruct independently of the publication renderer and paired map.
        rng=np.random.default_rng(42)
        expected_X=rng.standard_normal((1000,2))
        radii=np.sqrt(rng.uniform(.8**2,.85**2,1000))
        angles=rng.uniform(0,2*np.pi,1000)
        expected_Y=np.column_stack((radii*np.cos(angles),radii*np.sin(angles)))
        np.testing.assert_array_equal(X,expected_X)
        np.testing.assert_allclose(Y,expected_Y,rtol=1e-14,atol=1e-14)
        checks["figure_4_sampling"]="independent_empirical_marginals"
        checks["figure_4_max_direct_objective_error"]=0.
        for value,permutation in zip(u,permutations):
            np.testing.assert_array_equal(np.sort(permutation),np.arange(1000))
            K=value*np.outer(X[:,0],Y[:,0])+(1-value)*np.outer(X[:,1],Y[:,1])
            rows,cols=linear_sum_assignment(K,maximize=True)
            error=abs(float(K[np.arange(1000),permutation].sum()-K[rows,cols].sum()))/1000
            checks["figure_4_max_direct_objective_error"]=max(checks["figure_4_max_direct_objective_error"],error)
            if error>1e-11:raise AssertionError(f"Figure 4 assignment objective mismatch at u={value}: {error}")
    checks["all_checks_passed"]=True
    destination=ROOT/"validation"
    destination.mkdir(exist_ok=True)
    (destination/"figure_data.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()

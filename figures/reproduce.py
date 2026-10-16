#!/usr/bin/env python3
"""Reproduce the four manuscript figures with exact empirical assignments.

Run from any directory: python figures/reproduce.py --workers 4
Use --quick only for a clearly labelled PREVIEW in figures/preview/.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from iot_gap import annulus_map, cosine_map, gap_curve, optimal_assignment

SPEC_VERSION = 2
N_LIST = [50, 100, 1000, 2000]
DELTAS = [.01, .1, .3]
REPS = {50: 5, 100: 5, 1000: 1, 2000: 1}
COSINE_US = np.unique(np.r_[np.linspace(0, 1, 40), np.linspace(.4, .6, 11)])
ANNULUS_US = np.unique(np.r_[np.linspace(0, 1, 41), np.linspace(.4, .6, 21)])
COLORS = {50: "#0072B2", 100: "#E69F00", 1000: "#009E73", 2000: "#D55E00"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tasks(quick=False):
    core_hash=file_hash(ROOT/"src/iot_gap/core.py")
    ns = [50, 100] if quick else N_LIST
    for n in ns:
        reps = 1 if quick else REPS[n]
        for rep in range(reps):
            for delta in DELTAS:
                yield {"model": "cosine", "n": n, "seed": n, "rep": rep, "delta": delta,
                       "grid": (np.linspace(0, 1, 11) if quick else COSINE_US).tolist(),
                       "spec_version": SPEC_VERSION,"computation_sha256":core_hash}
        for rep in range(1 if quick else 5):
            yield {"model": "annulus", "n": n, "seed": n, "rep": rep, "delta": None,
                   "grid": (np.linspace(0, 1, 11) if quick else ANNULUS_US).tolist(),
                   "spec_version": SPEC_VERSION,"computation_sha256":core_hash}


def cache_name(spec):
    d = "" if spec["delta"] is None else f"_delta{spec['delta']:g}"
    return f"{spec['model']}{d}_n{spec['n']}_rep{spec['rep']}.npz"


def calculate(payload):
    spec, data_dir = payload
    path = Path(data_dir) / cache_name(spec)
    encoded = json.dumps(spec, sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["spec_json"]) == encoded:
                return path.name, 0., True
        raise RuntimeError(f"Cache specification mismatch: {path}. Remove that file to recompute.")
    rng = np.random.default_rng(spec["seed"])
    # Exactly the notebook RNG sequence: each repetition consumes one (n,2)
    # Gaussian array. Resetting and discarding here makes tasks restartable.
    for _ in range(spec["rep"]+1):
        X = rng.standard_normal((spec["n"], 2))
    Y = cosine_map(X, spec["delta"]) if spec["model"] == "cosine" else annulus_map(X)
    us = np.asarray(spec["grid"])
    start = time.perf_counter()
    loss = gap_curve(X, Y, us)
    center = np.flatnonzero(np.isclose(us, .5, rtol=0, atol=1e-14))
    if len(center) and np.max(loss[center]) > 1e-11:
        raise ArithmeticError("The paired plan must be optimal at the true parameter u=1/2")
    elapsed = time.perf_counter()-start
    # Endpoint sorting is exact by rearrangement. Compare Hungarian objectives
    # explicitly on every small task, and center is always solved by Hungarian.
    endpoint_error = 0.
    if spec["n"] <= 100:
        for u in (0., 1.):
            K = u*np.outer(X[:, 0],Y[:, 0])+(1-u)*np.outer(X[:, 1],Y[:, 1])
            r,c = linear_sum_assignment(K, maximize=True)
            p = optimal_assignment(X,Y,u)
            endpoint_error = max(endpoint_error, abs((K[r,c].sum()-K[np.arange(len(X)),p].sum())/len(X)))
        if endpoint_error > 1e-11:
            raise ArithmeticError("Sorted endpoint and assignment objective disagree")
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, X=X, Y=Y, u=us, gap=loss,
                        spec_json=np.asarray(encoded), elapsed_seconds=np.asarray(elapsed),
                        endpoint_objective_error=np.asarray(endpoint_error))
    tmp.replace(path)
    return path.name, elapsed, False


def style():
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 15,
        "xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "lines.linewidth": 1.5, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "iot-gap-publication-v1",
        "figure.facecolor": "white", "axes.facecolor": "white"})


def save_figure(fig, output, stem, quick):
    if quick:
        fig.suptitle("PREVIEW — reduced samples and grid; not a manuscript figure", fontsize=10, color="darkred")
    for extension in ("pdf", "svg", "png"):
        options = {"dpi": 220} if extension == "png" else {}
        if extension == "pdf": options["metadata"]={"CreationDate":None,"ModDate":None}
        if extension == "svg": options["metadata"]={"Date":None}
        fig.savefig(output / f"{stem}.{extension}", **options)
    plt.close(fig)


def load_group(specs, data, model, n, delta=None):
    group = [s for s in specs if s["model"] == model and s["n"] == n and s["delta"] == delta]
    arrays = []
    for s in group:
        with np.load(data / cache_name(s), allow_pickle=False) as z:
            if str(z["spec_json"])!=json.dumps(s,sort_keys=True):
                raise RuntimeError(f"Cache specification mismatch: {data/cache_name(s)}")
            u = z["u"].copy()
            arrays.append(z["gap"].copy())
    return u, np.asarray(arrays)


def render_cosine(specs, data, out, quick):
    ns = sorted({s["n"] for s in specs})
    fig, axes = plt.subplots(1,3,figsize=(12.0,3.6),sharex=True,layout="constrained")
    for ax, delta in zip(axes, DELTAS):
        for n in ns:
            us, values = load_group(specs,data,"cosine",n,delta)
            mean, sd = values.mean(axis=0), values.std(axis=0,ddof=0)
            ax.plot(us,mean,color=COLORS[n],label=f"$n={n}$")
            ax.fill_between(us,np.maximum(mean-sd,0),mean+sd,color=COLORS[n],alpha=.15,linewidth=0)
        ax.set(xlabel="$u$",title=rf"$\varepsilon={delta:g}$",xlim=(0,1),ylim=(0,None))
        ax.ticklabel_format(axis="y",style="sci",scilimits=(-2,2),useMathText=True)
        ax.axvline(.5,color=".5",linestyle=":",lw=.8,zorder=0)
        ax.grid(alpha=.16)
    axes[0].set_ylabel(r"Paired empirical gap $\mathcal{L}_{n,0}(A_u)$")
    axes[-1].legend(frameon=False,loc="upper center",ncol=2)
    save_figure(fig,out,"figure_2_cosine",quick)


def render_annulus(specs,data,out,quick):
    ns = sorted({s["n"] for s in specs})
    fig,axes = plt.subplots(1,len(ns),figsize=(12,3.4),sharex=True,sharey=True,layout="constrained")
    for ax,n in zip(np.atleast_1d(axes),ns):
        us,values = load_group(specs,data,"annulus",n)
        mean,sd = values.mean(axis=0), values.std(axis=0,ddof=0)
        ax.plot(us,mean,color=COLORS[n])
        ax.fill_between(us,np.maximum(mean-sd,0),mean+sd,color=COLORS[n],alpha=.18,linewidth=0)
        ax.set(xlabel="$u$",title=f"$n={n}$",xlim=(0,1),ylim=(0,.035))
        ax.set_xticks([0,.5,1],labels=["0","0.5","1"])
        ax.axvline(.5,color=".5",linestyle=":",lw=.8,zorder=0)
        ax.grid(alpha=.16)
    np.atleast_1d(axes)[0].set_ylabel(r"Paired empirical gap $\mathcal{L}_{n,0}(A_u)$")
    save_figure(fig,out,"figure_3_annulus",quick)


def render_maps(data,out,quick,render_only=False):
    n = 100 if quick else 1000
    X = np.random.default_rng(1000).standard_normal((n,2))
    Y = annulus_map(X)
    us = [.5,.9,.1]
    arrow_spec={"n":n,"seed":1000,"rep":0,"a":.8,"b":.85,"u":us,"sampling":"paired",
                "spec_version":SPEC_VERSION,"computation_sha256":file_hash(ROOT/"src/iot_gap/core.py")}
    encoded=json.dumps(arrow_spec,sort_keys=True)
    path=data/"annulus_assignments.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if "spec_json" not in z or str(z["spec_json"])!=encoded:
                raise RuntimeError(f"Assignment cache specification mismatch: {path}. Remove that file to recompute.")
            np.testing.assert_array_equal(z["X"],X);np.testing.assert_array_equal(z["Y"],Y)
            np.testing.assert_array_equal(z["u"],us)
            perms=z["permutations"].copy()
            for p in perms:np.testing.assert_array_equal(np.sort(p),np.arange(n))
    else:
        if render_only:raise FileNotFoundError(f"Missing cached assignments: {path}")
        perms = np.asarray([optimal_assignment(X,Y,u) for u in us])
        np.savez_compressed(path,X=X,Y=Y,u=np.asarray(us),permutations=perms,seed=1000,spec_json=np.asarray(encoded))
    fig,axes = plt.subplots(1,4,figsize=(12,3.3),sharex=True,sharey=True,layout="constrained")
    source_color = "#0072B2"; target_color="#D55E00"
    axes[0].scatter(X[:,0],X[:,1],s=3,c=source_color,alpha=.35,rasterized=False,label=r"$X_i\sim\mathcal{N}(0,I_2)$")
    axes[0].scatter(Y[:,0],Y[:,1],s=3,c=target_color,alpha=.6,rasterized=False,label=r"$Y_i=T(X_i)$")
    axes[0].set_title(rf"Samples ($n={n}$)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",
               ncol=2,frameon=False,fontsize=12,markerscale=2)
    for ax,u,p in zip(axes[1:],us,perms):
        # Draw every empirical assignment, not a subset or a continuous OT map.
        ax.quiver(X[:,0],X[:,1],Y[p,0]-X[:,0],Y[p,1]-X[:,1],angles="xy",scale_units="xy",scale=1,
            color=source_color,alpha=.16,width=.0018,headwidth=3,headlength=4)
        ax.scatter(Y[:,0],Y[:,1],s=2,c=target_color,alpha=.45)
        ax.set_title(rf"$u={u:g}$")
    limit=np.ceil(2*max(np.max(np.abs(X)),np.max(np.abs(Y)))*1.05)/2
    tick=float(np.floor(limit))
    for ax in axes:
        ax.set_aspect("equal"); ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel="$x_1$")
        ax.set_xticks([-tick,0,tick]);ax.set_yticks([-tick,0,tick]);ax.grid(alpha=.1)
    axes[0].set_ylabel("$x_2$")
    save_figure(fig,out,"figure_4_annulus_maps",quick)


def render_cycles(out,quick):
    # Faithful Python vector rendering of the supplied standalone TikZ source.
    source = ROOT/"figures"/"source"/"discrete_identifiability_cycles.tex"
    fig,axes = plt.subplots(1,3,figsize=(12,2.6),layout="constrained")
    panels=[(r"$-e_1$",r"$0$",r"$e_2$",r"$A_{12}=0$"),
            (r"$-e_2$",r"$0$",r"$e_1$",r"$A_{21}=0$"),
            (r"$e_1+e_2$",r"$e_1$",r"$e_2$",r"$A_{11}+A_{21}=A_{12}+A_{22}$")]
    for ax,(s,t0,t1,condition) in zip(axes,panels):
        xy=np.asarray([[0,1.4],[0,.3],[2.6,1.4],[2.6,.3]])
        for start,end,add in [(0,2,True),(1,3,True),(0,3,False),(1,2,False)]:
            c="#1e5f96" if add else "#af3c2d"
            ax.add_patch(FancyArrowPatch(xy[start],xy[end],arrowstyle="-|>",mutation_scale=12,
                linewidth=1.25,color=c,linestyle="-" if add else "--",shrinkA=5,shrinkB=5))
            frac=.5 if add else .27
            pos=(1-frac)*xy[start]+frac*xy[end]
            pos[1]+=(.1 if (start,end) in [(0,2),(1,2)] else -.13)
            ax.text(*pos,"$+t$" if add else "$-t$",ha="center",va="center",color=c,
                    bbox={"facecolor":"white","edgecolor":"none","pad":1},fontsize=13)
        ax.scatter(xy[:2,0],xy[:2,1],s=24,c="white",edgecolors="black",zorder=4)
        ax.scatter(xy[2:,0],xy[2:,1],s=24,c="white",edgecolors="black",marker="s",zorder=4)
        ax.text(0,1.8,"source",ha="center",fontsize=12);ax.text(2.6,1.8,"target",ha="center",fontsize=12)
        for point,text,offset,ha in [(xy[0],r"$0$",-.13,"right"),(xy[1],s,-.13,"right"),
                                     (xy[2],t0,.13,"left"),(xy[3],t1,.13,"left")]:
            ax.text(point[0]+offset,point[1],text,ha=ha,va="center",fontsize=14)
        ax.text(1.3,-.25,condition,ha="center",va="center",fontsize=14)
        ax.set(xlim=(-.8,3.35),ylim=(-.5,2));ax.axis("off")
    save_figure(fig,out,"figure_1_cycles",quick)


def write_tables(specs,data,out,quick):
    with (data/"gap_values.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["model","delta","n","seed","rep","u","gap"])
        for s in specs:
            with np.load(data/cache_name(s),allow_pickle=False) as z:
                for u,gap in zip(z["u"],z["gap"]):
                    writer.writerow([s["model"],s["delta"],s["n"],s["seed"],s["rep"],format(float(u),".17g"),format(float(gap),".17g")])
    versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
              "matplotlib":matplotlib.__version__,"platform":platform.platform()}
    manifest={"spec_version":SPEC_VERSION,"preview":quick,"software":versions,
      "objective":"max_permutation mean_i X_i^T diag(u,1-u) Y_perm(i) - mean_i X_i^T diag(u,1-u) Y_i",
      "solver":"SciPy linear_sum_assignment(maximize=True); exact sorted rank-one endpoints",
      "sampling":"paired X~N(0,I2), Y=T(X); NumPy default_rng(n); repetition consumes next (n,2) array",
      "cosine_common_random_numbers":"The identical X for fixed (n,rep) is reused for all three epsilon values.",
      "standard_deviation":"population ddof=0; mean ± sd bands clipped below at zero; no smoothing",
      "arrow_spec":{"n":100 if quick else 1000,"seed":1000,"rep":0,"a":.8,"b":.85,"u":[.5,.9,.1],"sampling":"paired","all_arrows":True},
      "tasks":specs,"checksums":{}}
    files=sorted(list(data.glob("*.npz"))+list(out.glob("figure_*.*"))+[data/"gap_values.csv"])
    for p in files: manifest["checksums"][str(p.relative_to(ROOT))]=file_hash(p)
    manifest["code_checksums"]={str(p.relative_to(ROOT)):file_hash(p) for p in [Path(__file__),ROOT/"src/iot_gap/core.py",ROOT/"figures/source/discrete_identifiability_cycles.tex"]}
    (data/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument("--quick",action="store_true",help="labelled preview only, separate data/output directory")
    parser.add_argument("--render-only",action="store_true",help="use the saved exact arrays without assignment solves")
    args=parser.parse_args()
    if args.workers < 1: parser.error("--workers must be positive")
    folder=ROOT/"figures"/("preview" if args.quick else "")
    data=folder/"data";out=folder/"generated"
    data.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    specs=list(tasks(args.quick))
    if not args.render_only:
        print(f"Exact experiment: {len(specs)} tasks, {args.workers} workers; cache at {data}",flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending=[executor.submit(calculate,(spec,str(data))) for spec in specs]
            for future in concurrent.futures.as_completed(pending):
                name,seconds,cached=future.result()
                print(f"{'cached' if cached else 'computed'} {name} ({seconds:.1f} s)",flush=True)
    missing=[cache_name(s) for s in specs if not (data/cache_name(s)).exists()]
    if missing: raise FileNotFoundError(f"Missing cached tasks: {missing}")
    style()
    render_cycles(out,args.quick);render_cosine(specs,data,out,args.quick)
    render_annulus(specs,data,out,args.quick);render_maps(data,out,args.quick,args.render_only)
    write_tables(specs,data,out,args.quick)
    print(f"Saved all four figures as vector PDF/SVG and PNG in {out}",flush=True)


if __name__ == "__main__":
    main()

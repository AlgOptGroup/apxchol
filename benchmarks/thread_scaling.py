#!/usr/bin/env python3
"""Thread-scaling sweep: run each parallel solver at 1/2/4/8/16 threads on a few
representative matrices, then emit speedup + parallel-efficiency charts (and the
t1 numbers). Cells go to results/scaling_cells/ (separate from the fair cells).
Run from repo root, ALONE: python3 benchmarks/thread_scaling.py
"""
import json, os, re, subprocess, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runner_common import (ROOT, sh, git_sha, boost_state, parse_csv,
                           PARAC_CPU_DRIVER as DRIVER, PARAC_REORD as REORD,
                           PARAC_LDLIB as LDLIB)

BIN = f"{ROOT}/benchmarks/build/benchmark"
DUMP = "/tmp/parac_fair_dump"
CELLS = f"{ROOT}/results/scaling_cells"
TOL = "1e-8"; THREADS = [1, 2, 4, 8, 16]; TIMEOUT = 900; REPS = 2

# (mid, family, margs, reg, is2d)
# reg=False everywhere: the current protocol runs the ORIGINAL singular Laplacians
# unshifted (multi-component Dirichlet pin, no --reg-rel), matching sweep_fair.
MATS = [
    ("grid_2000", "grids", "--graph grid --n 2000", False, True),
    ("grid3d_100", "grids", "--graph grid3d --n 100", False, False),
    ("iter0040", "ipm", f"--mtx {ROOT}/data/ipm/iter0040/matrix.mtx", False, False),
    ("ecology1", "suitesparse", f"--mtx {ROOT}/data/matrices/ecology1.mtx", False, False),
]
# cpp solvers: label -> (solver, config)
CPP = [("apxchol bg+tree", "apxchol_v1", "bg+tree[vec_pool]"),
       ("apxchol luby+tree", "apxchol_v1", "luby+tree[vec_pool]"),
       ("apxchol root+tree", "apxchol_v1", "root+tree[vec_pool]"),
       ("apxchol bk+tree", "apxchol_v1", "bk+tree[vec_pool]"),
       ("RCHOL", "rchol", ""), ("pRCHOL", "rchol_par", ""),
       ("BoomerAMG", "hypre_boomeramg", ""),
       ("AMGCL", "amgcl", "")]   # OMP-parallel (builtin backend) — belongs on the scaling chart
COLORS = {"apxchol bg+tree": "#0b5394",
          "apxchol luby+tree": "#073763",
          "apxchol root+tree": "#6fa8dc", "apxchol bk+tree": "#3d85c6",
          "RCHOL": "#d62728", "BoomerAMG": "#2ca02c",
          "AMGCL": "#8c564b", "ParAC": "#ff8c00"}

# sh/git_sha/boost_state/parse_csv come from runner_common — the previous local
# sh used the UNHARDENED subprocess.run (orphan-on-timeout bug); the common one
# kills the whole process group.
BOOST = boost_state()
PROV = {"note": "thread-scaling sweep", "git_sha": git_sha(), "boost": BOOST}
LOCKED = " — freq-locked 2.5 GHz (boost off)" if BOOST == "off" else ""

def emit(mid, family, lab, t, m, status):
    os.makedirs(CELLS, exist_ok=True)
    tag = re.sub(r'[^A-Za-z0-9]+', '_', lab)
    json.dump({"cell": {"matrix_id": mid, "family": family, "label": lab, "threads": t},
               "metrics": m or {}, "status": status, "provenance": PROV},
              open(f"{CELLS}/{mid}__{tag}__t{t}.json", "w"))

def done(mid, lab, t):
    tag = re.sub(r'[^A-Za-z0-9]+', '_', lab)
    p = f"{CELLS}/{mid}__{tag}__t{t}.json"
    return os.path.exists(p) and json.load(open(p)).get("status") in ("complete", "not_converged", "timeout", "failed")

def run_cpp(margs, solver, config, reg, t):
    cfg = f"--v1-configs '{config}'" if solver == "apxchol_v1" else ""
    regf = "--reg-rel 1e-6" if reg else ""
    cmd = (f"taskset -c 0-{t-1} {BIN} {margs} --solver {solver} {cfg} {regf} "
           f"--threads {t} --tol {TOL} --maxiter 500 --repeat {REPS} --csv")
    try: m = parse_csv(sh(cmd, timeout=TIMEOUT).stdout)
    except subprocess.TimeoutExpired: return "timeout", None
    if m is None: return "failed", None
    return ("complete" if m["rel_res"] <= float(TOL) * 10 else "not_converged"), m

def run_parac(mid, t):
    # reuse the cached AMD-reordered matrix (pure for grids/ipm, pin for SS;
    # 'reg' is the legacy eps*I-regularized cache, kept as a fallback)
    for tag in ("pure", "pin", "reg"):
        amd = f"{REORD}/{mid}-{tag}-amd.mtx"
        if os.path.exists(amd): break
    else:
        return "failed", None
    amds = 0.0
    tf = f"{amd}.time"
    if os.path.exists(tf): amds = float(open(tf).read())
    env = dict(os.environ, LD_LIBRARY_PATH=LDLIB, MKL_NUM_THREADS=str(t),
               OMP_PROC_BIND="close", KMP_AFFINITY="norespect")
    try:
        o = sh(f"taskset -c 0-{t-1} {DRIVER} {amd} {t} \"\"", timeout=TIMEOUT, env=env).stdout
    except subprocess.TimeoutExpired:
        return "timeout", None
    g = lambda p: (re.search(p, o).group(1) if re.search(p, o) else None)
    fac, sol, it, rr = (g(r"Factorization execution time:\s*([0-9.]+)"),
                        g(r"Solve time taken:\s*([0-9]+)"), g(r"Iterations:\s*([0-9]+)"),
                        g(r"relative residual:\s*([0-9.eE+-]+)"))
    if not (fac and sol and it): return "failed", None
    setup = float(fac) + amds; solve = float(sol) / 1000
    return ("complete" if float(rr) <= 1e-7 else "not_converged"), dict(
        total_s=setup + solve, setup_s=setup, solve_s=solve, iters=int(it), rel_res=float(rr))

def sweep():
    for mid, fam, margs, reg, is2d in MATS:
        print(f"[{mid}]", flush=True)
        soln = list(CPP) + [("ParAC", None, None)]
        for lab, solver, config in soln:
            for t in THREADS:
                if done(mid, lab, t):
                    continue
                if lab == "ParAC":
                    st, m = run_parac(mid, t)
                else:
                    st, m = run_cpp(margs, solver, config, reg, t)
                emit(mid, fam, lab, t, m, st)
                print(f"   {lab:16} t{t:<2} {st} total={m['total_s'] if m else '-'}", flush=True)

def charts(out=f"{ROOT}/benchmarks/latest"):
    recs = [json.load(open(f)) for f in glob.glob(f"{CELLS}/*.json")]
    mats = sorted({r["cell"]["matrix_id"] for r in recs})
    # only genuinely multi-threaded solvers belong on a thread-scaling chart.
    # AMGCL was removed from the comparison (old cells linger). RCHOL has a
    # SERIAL factorization, so its "speedup vs threads" is meaningless -- drop it.
    SERIAL = {"RCHOL"}
    KEEP = ({lab for lab, *_ in CPP} | {"ParAC"}) - SERIAL
    os.makedirs(f"{out}/figures", exist_ok=True)

    def fig_for(field, phase, kind, fname):
        # field = setup_s | solve_s ; kind = speedup (t1/tN) | efficiency (speedup/N).
        # Setup and solve scale very differently, so they get separate charts.
        fig, axes = plt.subplots(1, len(mats), figsize=(4.2 * len(mats), 4.2), squeeze=False)
        for j, mid in enumerate(mats):
            ax = axes[0][j]
            labs = sorted({r["cell"]["label"] for r in recs
                           if r["cell"]["matrix_id"] == mid and r["cell"]["label"] in KEEP})
            for lab in labs:
                pts = {r["cell"]["threads"]: r["metrics"].get(field)
                       for r in recs if r["cell"]["matrix_id"] == mid
                       and r["cell"]["label"] == lab and r["status"] == "complete"
                       and r["metrics"].get(field)}
                if 1 not in pts or len(pts) < 2: continue
                ts = sorted(pts); t1 = pts[1]
                sp = [t1 / pts[t] for t in ts]
                ys = sp if kind == "speedup" else [s / t for s, t in zip(sp, ts)]
                ax.plot(ts, ys, marker="o", label=lab, color=COLORS.get(lab, "#888"))
            ax.set_title(mid, fontsize=10); ax.set_xlabel("threads")
            ax.set_xscale("log", base=2); ax.set_xticks(THREADS); ax.set_xticklabels(THREADS)
            ax.set_ylabel(f"{phase} " + ("speedup (t1/tN)" if kind == "speedup"
                                         else "parallel efficiency"))
            ax.grid(True, alpha=0.3)
        # one global legend (union across panels) so no line is missing from it
        hl = {}
        for ax in axes[0]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                hl.setdefault(l, h)
        fig.legend(hl.values(), hl.keys(), loc="lower center", ncol=max(1, len(hl)), fontsize=8)
        fig.suptitle(f"{phase} {kind} vs threads (tol 1e-8){LOCKED}")
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.savefig(fname, dpi=130); plt.close(fig)

    # Only the speedup (t1/tN) charts are kept; parallel-efficiency charts were
    # dropped 2026-06 (redundant — speedup already shows the scaling story).
    for field, phase in (("setup_s", "Setup"), ("solve_s", "Solve")):
        fig_for(field, phase, "speedup", f"{out}/figures/threads_{phase.lower()}_speedup.png")
    # Remove any stale efficiency / combined-total charts from earlier runs.
    for old in ("threads_speedup.png", "threads_efficiency.png",
                "threads_setup_efficiency.png", "threads_solve_efficiency.png"):
        p = f"{out}/figures/{old}"
        if os.path.exists(p): os.remove(p)
    print(f"thread-scaling charts -> {out}/figures/threads_setup_speedup.png, threads_solve_speedup.png")

if __name__ == "__main__":
    sweep()
    charts()
    print("thread-scaling done")

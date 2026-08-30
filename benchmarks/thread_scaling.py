#!/usr/bin/env python3
"""Thread-scaling sweep: run each parallel solver at 1/2/4/8/16 threads on a few
representative matrices, then emit total/setup/solve speedup charts and a
portable CSV. Cells go to results/scaling_cells/ (separate from the fair cells).
Run from repo root, ALONE: python3 benchmarks/thread_scaling.py
"""
import argparse, csv, json, os, re, statistics, subprocess, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sweep_fair import UNKNOWN_TOOLCHAIN
import parac_runner as parac
import runner_common as rc
from runner_common import (margs_for, ROOT, sh, git_sha, boost_state, parse_csv,
                           parse_build_meta, binary_toolchain,
                           benchmark_openmp_env, benchmark_openmp_provenance,
                           taskset_prefix,
                           PARAC_CPU_DRIVER as DRIVER, PARAC_REORD as REORD,
                           PARAC_LDLIB as LDLIB)

BIN = os.environ.get("APXCHOL_BENCH_CPU_BIN", f"{ROOT}/benchmarks/build/benchmark")
DUMP = "/tmp/parac_fair_dump"
CELLS = os.environ.get("APXCHOL_SCALING_STORE", f"{ROOT}/results/scaling_cells")
TOL = "1e-8"; THREADS = [1, 2, 4, 8, 16]; TIMEOUT = 900; REPS = 3

# (mid, family, margs, reg, is2d)
# reg=False everywhere: the current protocol runs the ORIGINAL singular Laplacians
# unshifted (multi-component Dirichlet pin, no --reg-rel), matching sweep_fair.
MATS = [
    ("grid_2000", "grids", "--graph grid --n 2000", False, True),
    ("grid3d_100", "grids", "--graph grid3d --n 100", False, False),
    ("iter0040", "ipm", margs_for("iter0040"), False, False),
    ("ecology1", "suitesparse", margs_for("ecology1"), False, False),
]
# cpp solvers: label -> (solver, config)
CPP = [("apxchol bg+tree", "apxchol_v1", "bg+tree[vec_pool_aos]"),
       ("apxchol greedy+tree", "apxchol_v1", "greedy+tree[vec_pool]"),
       ("apxchol bk+tree", "apxchol_v1", "bk+tree[vec_pool]"),
       ("RCHOL", "rchol", ""), ("pRCHOL", "rchol_par", ""),
       ("BoomerAMG", "hypre_boomeramg", ""),
       ("AMGCL", "amgcl", "")]   # OMP-parallel (builtin backend) — belongs on the scaling chart
COLORS = {"apxchol bg+tree": "#0b5394",
          "apxchol greedy+tree": "#073763",
          "apxchol bk+tree": "#3d85c6",
          "RCHOL": "#d62728", "pRCHOL": "#ff9896", "BoomerAMG": "#2ca02c",
          "AMGCL": "#8c564b", "ParAC": "#ff8c00"}

# sh/git_sha/boost_state/parse_csv come from runner_common — the previous local
# sh used the UNHARDENED subprocess.run (orphan-on-timeout bug); the common one
# kills the whole process group.
BOOST = boost_state()
PROV = {"note": "thread-scaling sweep", "git_sha": git_sha(), "boost": BOOST}
LOCKED = " — freq-locked 2.5 GHz (boost off)" if BOOST == "off" else ""

# Toolchain per cell, and this sweep runs TWO binaries: ours (which reports its
# own BUILD_META) and ParAC's driver (read off its ELF). A thread-scaling curve
# compares a solver against itself across thread counts, so which compiler drew
# it is exactly the thing that must not be left to memory.
BUILD = {}
ONLY_SERIES = set()
ONLY_MATRICES = set()
RERUN_STATUSES = set()
_PARAC_PREPARED = {}

def _cell_tag(solver, config):
    return re.sub(r'[^A-Za-z0-9]+', '_', f"{solver}__{config or 'none'}").strip('_')


def emit(mid, family, lab, solver, config, t, m, status, prov=None):
    os.makedirs(CELLS, exist_ok=True)
    tag = _cell_tag(solver, config)
    json.dump({"cell": {"matrix_id": mid, "family": family, "label": lab,
                         "solver": solver, "config": config, "threads": t,
                         "device": "cpu"},
               "metrics": m or {}, "status": status, "provenance": {**PROV, **(prov or UNKNOWN_TOOLCHAIN)}},
              open(f"{CELLS}/{mid}__{tag}__t{t}.json", "w"))

def done(mid, solver, config, t):
    tag = _cell_tag(solver, config)
    p = f"{CELLS}/{mid}__{tag}__t{t}.json"
    if not os.path.exists(p):
        return False
    status = json.load(open(p)).get("status")
    return status not in RERUN_STATUSES and status in (
        "complete", "not_converged", "timeout", "failed", "oom", "n/a")

def run_cpp(margs, solver, config, reg, t):
    cfg = f"--v1-configs '{config}'" if solver == "apxchol_v1" else ""
    regf = "--reg-rel 1e-6" if reg else ""
    cmd = (f"{taskset_prefix(t)} {BIN} {margs} --solver {solver} {cfg} {regf} "
           f"--threads {t} --tol {TOL} --maxiter 500 --repeat {REPS} --csv")
    try: p = sh(cmd, timeout=TIMEOUT, env=benchmark_openmp_env(t))
    except subprocess.TimeoutExpired as e:
        BUILD.update(parse_build_meta(e.stderr))   # even a timeout names its toolchain
        return "timeout", None
    BUILD.update(parse_build_meta(p.stderr))       # what built the binary that just ran
    m = parse_csv(p.stdout)
    if m is None: return "failed", None
    # THE GRADING RULE (benchmarks/README.md): true relative residual <= exactly
    # tol, same for every solver, no grace factor. Kept in sync with rc.classify.
    return ("complete" if m["rel_res"] <= float(TOL) else "not_converged"), m

def _prepare_parac(mid):
    """Reuse the canonical ParAC input route, including component handling.

    The old scaling runner guessed obsolete `pure/pin/reg` cache names, skipped
    ParAC's adapter interval and tolerance calibration, and silently emitted
    failures.  This prepares exactly the same AMD-reordered operands as
    parac_runner.py and returns one entry per nontrivial connected component.
    """
    if mid in _PARAC_PREPARED:
        return _PARAC_PREPARED[mid]
    prepared = []
    if parac._uses_physics(mid):
        amd, reorder_s, _prep = parac._prep_amd(mid, "op", augment=True)
        if amd:
            prepared.append((amd, reorder_s, True))
    else:
        rank = 0
        while True:
            src, n_nodes, _n_components = parac._dump_component(mid, rank)
            if src is None or n_nodes < parac.COMP_THRESHOLD:
                break
            amd, reorder_s, _prep = parac._reorder_amd(mid, src, f"comp{rank}")
            if amd:
                prepared.append((amd, reorder_s, False))
            rank += 1
    _PARAC_PREPARED[mid] = prepared
    return prepared


def run_parac(mid, t):
    """Three measured ParAC runs through the canonical fair-runner path."""
    parac.THREADS = t
    parac.TIMEOUT_CPU = TIMEOUT
    try:
        operands = _prepare_parac(mid)
        if not operands:
            return "failed", None
        setup = solve = 0.0
        iters = 0
        rel_res = 0.0
        n = nnz = 0
        for amd, reorder_s, physics in operands:
            rel_tol = parac._calibrate_rel_tol(amd, physics, tau=float(TOL))
            runs = [parac._run_once_cpu(amd, physics, rel_tol=rel_tol)
                    for _ in range(REPS)]
            ok = [run for run in runs
                  if run["factor_setup"] and run["adapter"] and run["solve"]
                  and run["iters"] and run["rr"]]
            if len(ok) != REPS:
                return "failed", None
            med = lambda key: statistics.median(float(run[key]) for run in ok)
            setup += reorder_s + med("adapter") + med("factor_setup")
            solve += med("solve") / 1000.0
            iters = max(iters, int(statistics.median(int(run["iters"]) for run in ok)))
            rel_res = max(rel_res, med("rr"))
            n += int(ok[-1]["n"] or 0)
            nnz += int(ok[-1]["nnz"] or 0)
        total = setup + solve
        return ("complete" if rel_res <= float(TOL) else "not_converged"), dict(
            n=n, nnz=nnz, total_s=total, setup_s=setup, solve_s=solve,
            iters=iters, rel_res=rel_res)
    except subprocess.TimeoutExpired:
        return "timeout", None

def sweep():
    for mid, fam, margs, reg, is2d in MATS:
        if ONLY_MATRICES and mid not in ONLY_MATRICES:
            continue
        print(f"[{mid}]", flush=True)
        soln = list(CPP) + [("ParAC", "parac", "")]
        for lab, solver, config in soln:
            if ONLY_SERIES and lab not in ONLY_SERIES and solver not in ONLY_SERIES:
                continue
            for t in THREADS:
                if done(mid, solver, config, t):
                    continue
                if solver == "parac":
                    st, m = run_parac(mid, t)
                    prov = {**binary_toolchain(DRIVER),
                            **benchmark_openmp_provenance(t)}
                else:
                    st, m = run_cpp(margs, solver, config, reg, t)
                    prov = {**BUILD, **benchmark_openmp_provenance(t)}
                emit(mid, fam, lab, solver, config, t, m, st, prov)
                print(f"   {lab:16} t{t:<2} {st} total={m['total_s'] if m else '-'}", flush=True)


def validate_cells():
    solvers = list(CPP) + [("ParAC", "parac", "")]
    expected = {(mid, solver, config, threads)
                for mid, *_ in MATS for _label, solver, config in solvers
                for threads in THREADS}
    found = {}
    for filename in sorted(glob.glob(f"{CELLS}/*.json")):
        with open(filename) as handle:
            record = json.load(handle)
        cell = record.get("cell", {})
        key = (cell.get("matrix_id"), cell.get("solver"), cell.get("config", ""),
               cell.get("threads"))
        if key in found:
            raise RuntimeError(f"duplicate scaling cell {key}: {found[key]} and {filename}")
        found[key] = filename
        if record.get("status") not in (
                "complete", "not_converged", "timeout", "failed", "oom", "n/a"):
            raise RuntimeError(f"non-terminal scaling cell {filename}: {record.get('status')}")
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    if missing or extra:
        raise RuntimeError(
            f"scaling denominator mismatch: found {len(found)}/{len(expected)}, "
            f"missing={missing[:8]}, extra={extra[:8]}")
    print(f"thread-scaling denominator -> {len(found)}/{len(expected)} cells")


def charts(out=f"{ROOT}/benchmarks/latest"):
    recs = [json.load(open(f)) for f in glob.glob(f"{CELLS}/*.json")]
    mats = sorted({r["cell"]["matrix_id"] for r in recs})
    # Only genuinely multi-threaded solvers belong on a thread-scaling chart.
    # RCHOL has a serial factorization, so its "speedup vs threads" is
    # meaningless; pRCHOL is the parallel series retained from that family.
    SERIAL = {"RCHOL"}
    KEEP = ({lab for lab, *_ in CPP} | {"ParAC"}) - SERIAL
    os.makedirs(f"{out}/figures", exist_ok=True)

    def fig_for(field, phase, kind, fname):
        # Setup and solve scale very differently, so they get separate charts;
        # total is the user-facing single-RHS outcome.
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

    # Keep total, setup and solve speedups together.  Total is the user-facing
    # single-RHS outcome; the two phase charts explain why its curve bends.
    # Parallel-efficiency charts remain omitted because they duplicate the same
    # speedup data divided by the thread count.
    for field, phase in (("total_s", "Total"), ("setup_s", "Setup"),
                         ("solve_s", "Solve")):
        fig_for(field, phase, "speedup", f"{out}/figures/threads_{phase.lower()}_speedup.png")
    # Remove stale efficiency and ambiguously named legacy charts.
    for old in ("threads_speedup.png", "threads_efficiency.png",
                "threads_setup_efficiency.png", "threads_solve_efficiency.png"):
        p = f"{out}/figures/{old}"
        if os.path.exists(p): os.remove(p)
    print(f"thread-scaling charts -> {out}/figures/threads_total_speedup.png, "
          f"threads_setup_speedup.png, threads_solve_speedup.png")


def export_csv(path):
    """Portable extract for the exact cells behind the scaling figures."""
    fields = ("matrix", "family", "label", "solver", "config", "threads", "status", "setup_s",
              "solve_s", "total_s", "iters", "rel_res", "git_sha", "repeat",
              "compiler", "compiler_version", "openmp_runtime")
    rows = []
    for filename in sorted(glob.glob(f"{CELLS}/*.json")):
        with open(filename) as handle:
            record = json.load(handle)
        cell = record["cell"]
        metrics = record.get("metrics", {})
        provenance = record.get("provenance", {})
        rows.append({
            "matrix": cell["matrix_id"], "family": cell["family"],
            "label": cell["label"], "solver": cell.get("solver", ""),
            "config": cell.get("config", ""), "threads": cell["threads"],
            "status": record["status"],
            **{key: metrics.get(key, "")
               for key in ("setup_s", "solve_s", "total_s", "iters", "rel_res")},
            **{key: provenance.get(key, "")
               for key in ("git_sha", "repeat", "compiler", "compiler_version",
                            "openmp_runtime")},
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"thread-scaling data -> {path} ({len(rows)} cells)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default=CELLS,
                        help="isolated scaling-cell directory")
    parser.add_argument("--binary", default=BIN,
                        help="benchmark executable to run")
    parser.add_argument("--out", default=f"{ROOT}/benchmarks/latest")
    parser.add_argument("--repeat", type=int, default=REPS)
    parser.add_argument("--only-series", default="",
                        help="comma-separated display labels or solver ids to run")
    parser.add_argument("--only-matrices", default="",
                        help="comma-separated matrix ids to run; validation still covers all")
    parser.add_argument("--rerun-status", default="",
                        help="comma-separated terminal statuses to overwrite")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    CELLS = os.path.abspath(args.store)
    BIN = os.path.abspath(args.binary)
    REPS = args.repeat
    ONLY_SERIES = {value.strip() for value in args.only_series.split(",") if value.strip()}
    ONLY_MATRICES = {value.strip() for value in args.only_matrices.split(",") if value.strip()}
    known_matrices = {mid for mid, *_ in MATS}
    if not ONLY_MATRICES <= known_matrices:
        parser.error(f"unknown --only-matrices: {sorted(ONLY_MATRICES - known_matrices)}")
    RERUN_STATUSES = {value.strip() for value in args.rerun_status.split(",") if value.strip()}
    known_statuses = {"complete", "not_converged", "timeout", "failed", "oom", "n/a"}
    if not RERUN_STATUSES <= known_statuses:
        parser.error(f"unknown --rerun-status: {sorted(RERUN_STATUSES - known_statuses)}")
    rc.BIN["cpu"] = BIN
    PROV["repeat"] = REPS
    if not args.render_only:
        sweep()
    validate_cells()
    charts(args.out)
    export_csv(f"{args.out}/thread_scaling.csv")
    print("thread-scaling done")

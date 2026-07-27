#!/usr/bin/env python3
"""CMG (Koutis combinatorial multigrid) runner via the canonical MATLAB R2026a.

Runs the canonical MATLAB CMG (Koutis original) inside the mathworks/matlab-deps
container; the CMG MEX is recompiled for R2026a once via
benchmarks/cmg/Dockerfile.matlab + makeCMG (mx_*.mexa64).

Per matrix: dump the pure Laplacian, run bench_cmg.m (reg_rel=1e-6 — CMG is the lone
reg-rel solver, scored on L+eps*I, per the Protocol) in the container, parse the CSV,
emit a cell. CMG's hierarchy setup is O(n), so the giants are capped out. Resume-safe.

  python3 benchmarks/cmg_matlab_runner.py            # all chart matrices (grids/ipm/ss)
  python3 benchmarks/cmg_matlab_runner.py G3_circuit grid_2000   # a subset
"""
import os, subprocess, sys, time

import runner_common as rc
from runner_common import ROOT, sh

TOL = "1e-8"; MAXITER = "500"; SEED = "42"; REG = "1e-6"
THREADS = 16
CMG_MAX_N = int(os.environ.get("CMG_MAX_N", "30000000"))  # O(n) hierarchy, but MATLAB CMG
# is fast (grid_5000 @ 25M nodes runs fine) — covers every matrix in the registry. The
# old 5M cap was an overly-conservative leftover from the interpreter-bound Octave era.
TIMEOUT = int(os.environ.get("CMG_TIMEOUT_S", "1800"))
DUMP_DIR = "/tmp/cmg_mtx"
IMAGE = "apxchol-matlab-cmg:r2026a"
# Out-of-tree installs, mounted read-only into the container: set the environment
# variables below, or define the same names in a gitignored benchmarks/paths_local.py.
MATLAB = os.environ.get("APXCHOL_MATLAB_ROOT", "")        # a MATLAB R2026a install tree
CMG_SOLVER = os.environ.get("APXCHOL_CMG_SOLVER", "")     # Koutis' cmg-solver checkout

# Machine-local overrides (gitignored; assigns the path constants above by name).
try:
    from paths_local import *          # noqa: F401,F403
except ImportError:
    pass

PROV = {"boost": "on", "git_sha": rc.git_sha(),
        "note": f"CMG (Koutis) canonical MATLAB R2026a in matlab-deps container, "
                f"reg_rel={REG} (scored on L+eps*I), MEX-recompiled, tol 1e-8",
        "repeat": 1, "tier": "broad",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

TERMINAL = frozenset({"complete", "not_converged", "failed", "timeout", "n/a"})


def _dump(mid):
    os.makedirs(DUMP_DIR, exist_ok=True)
    p = f"{DUMP_DIR}/{mid}.mtx"
    if not os.path.exists(p):
        sh(f"{rc.BIN['cpu']} {rc.margs_for(mid)} --dump-mtx {p} --solver none", timeout=TIMEOUT)
    return p if os.path.exists(p) else None


def _run_matlab_cmg(mtx_basename):
    """Run bench_cmg.m on /cmg-mtx/<basename> inside the container; return stdout."""
    rc.require_path(MATLAB, "APXCHOL_MATLAB_ROOT", "MATLAB", "the MATLAB R2026a install")
    rc.require_path(CMG_SOLVER, "APXCHOL_CMG_SOLVER", "CMG_SOLVER", "the cmg-solver checkout")
    inner = (f"cd /bench && /opt/MATLAB/R2026a/bin/matlab -batch "
             f"\\\"bench_cmg('/cmg-mtx/{mtx_basename}', {TOL}, {MAXITER}, {SEED}, {REG})\\\"")
    cmd = (f"docker run --rm --network=host "
           f"-v {MATLAB}:/opt/MATLAB/R2026a:ro "
           f"-v {CMG_SOLVER}:/cmg-solver:ro "
           f"-v {DUMP_DIR}:/cmg-mtx:ro "
           f"-v {ROOT}/benchmarks/cmg:/bench:ro "
           f"-e HOME=/tmp -e CMG_ROOT=/cmg-solver "
           f"{IMAGE} bash -c \"{inner}\"")
    return sh(cmd, timeout=TIMEOUT).stdout


def run_one(mid):
    fam = rc.MATRICES[mid]["family"]
    if rc.cell_done(fam, mid, "cmg", "", THREADS, "cpu", terminal=TERMINAL):
        return "skip(done)"
    n = rc.MATRICES[mid].get("n", 0)
    if n and n > CMG_MAX_N:
        rc.emit_cell(fam, mid, "cmg", "", "n/a", {"n": n}, THREADS, "cpu",
                     {**PROV, "note": PROV["note"] + f" [skipped: n={n} > {CMG_MAX_N}]"})
        return f"skip(n={n}>cap)"
    try:
        src = _dump(mid)
        if not src:
            return "SKIP(dump)"
        out = _run_matlab_cmg(os.path.basename(src))
    except subprocess.TimeoutExpired:
        rc.emit_cell(fam, mid, "cmg", "", "timeout", {}, THREADS, "cpu", PROV)
        return "TIMEOUT"
    status, metrics = rc.classify(rc.parse_csv(out), TOL)
    rc.emit_cell(fam, mid, "cmg", "", status, metrics, THREADS, "cpu", PROV)
    return f"{status} it={metrics.get('iters') if metrics else '?'} rr={metrics.get('rel_res') if metrics else '?'}"


CHART_MATS = [m for m, meta in rc.MATRICES.items()
              if meta["family"] in ("grids", "ipm", "suitesparse")]

if __name__ == "__main__":
    mats = sys.argv[1:] or CHART_MATS
    for mid in mats:
        print(f"[cmg] {mid:18} {run_one(mid)}", flush=True)

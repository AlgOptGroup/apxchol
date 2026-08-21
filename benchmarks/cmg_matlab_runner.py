#!/usr/bin/env python3
"""CMG (Koutis combinatorial multigrid) runner via the canonical MATLAB R2026a.

Runs the canonical MATLAB CMG (Koutis original) inside the mathworks/matlab-deps
container; the CMG MEX is recompiled for R2026a once via
benchmarks/cmg/Dockerfile.matlab + makeCMG (mx_*.mexa64).

Per matrix: dump the operator the sweep's own solvers were given (--dump-mtx, the
one seam every external solver goes through), run bench_cmg.m in the container,
parse the CSV, emit a cell. CMG's hierarchy setup is O(n), so the giants are
capped out. Resume-safe.

The matrix reaches CMG the way its declared kind says it should:

  kind=graph     the dumped L, re-derived from |off-diagonal| (which for a graph
                 dump reproduces exactly the L that was written), shifted by
                 reg_rel=1e-6*mean|diag| — CMG is the lone reg-rel solver and is
                 scored on L+eps*I, per the Protocol.
  kind=operator  the dumped matrix read AS AN OPERATOR — diagonal included — and
                 solved unpinned and unshifted, so CMG is measured on the same
                 published operator as everything else in the cell. (The
                 Laplacian reading rebuilds the diagonal as the degree, which for
                 an SDDM operator silently discards its diagonal excess.)

sweep_fair.py runs this in-process, so a normal sweep fills the cmg cells; it can
also be driven on its own:

  python3 benchmarks/cmg_matlab_runner.py            # all chart matrices (grids/ipm/ss)
  python3 benchmarks/cmg_matlab_runner.py G3_circuit grid_2000   # a subset
"""
import os, re, subprocess, sys, time

import runner_common as rc
from runner_common import ROOT, sh

TOL = "1e-8"; MAXITER = "500"; SEED = "42"; REG = "1e-6"
THREADS = 16
CMG_MAX_N = int(os.environ.get("CMG_MAX_N", "30000000"))  # O(n) hierarchy, but MATLAB CMG
# is fast (grid_5000 @ 25M nodes runs fine) — covers every matrix in the registry. The
# old 5M cap was an overly-conservative leftover from the interpreter-bound Octave era.
TIMEOUT = int(os.environ.get("CMG_TIMEOUT_S", "1800"))
# Dumped .mtx cache, bind-mounted read-only into the container (so: absolute
# path). /tmp is a tmpfs on this box — RAM, with a per-user quota — and the
# giants dump multi-GB files, so point this at a disk-backed directory before
# sweeping them, exactly as APXCHOL_BENCH_DUMP_DIR does for sweep_fair.
DUMP_DIR = os.environ.get("APXCHOL_CMG_DUMP_DIR", "/tmp/cmg_mtx")
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
        "note": "CMG (Koutis) canonical MATLAB R2026a in matlab-deps container, "
                "MEX-recompiled, tol 1e-8",
        "repeat": 1, "tier": "broad",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

# CMG's compiled part is the MEX that MATLAB's `mex` built inside the container
# (benchmarks/cmg/Dockerfile.matlab), so the toolchain is read off that .mexa64
# rather than off this host's compiler — the container's gcc is not this
# machine's, and the file is the only thing that knows which one built it. The
# rest of CMG is MATLAB m-code run by the R2026a interpreter (see the note).
_MEX = "matlab/cmg/mex/mx_preconditioner_.mexa64"
_TOOLCHAIN = {}

def _toolchain():
    if not _TOOLCHAIN:
        _TOOLCHAIN.update(rc.binary_toolchain(os.path.join(CMG_SOLVER, _MEX)
                                              if CMG_SOLVER else ""))
    return _TOOLCHAIN

def _prov(as_operator):
    """PROV with the grounding this matrix actually got — the two paths are scored
    on DIFFERENT operators, so one note cannot describe both."""
    tail = ("published operator read as-is, solved unpinned and unshifted (reg_rel=0)"
            if as_operator else
            f"singular L, reg_rel={REG} (scored on L+eps*I)")
    return {**PROV, **_toolchain(), "note": f"{PROV['note']}, {tail}"}

TERMINAL = frozenset({"complete", "not_converged", "failed", "timeout", "n/a"})


def _dump(mid):
    os.makedirs(DUMP_DIR, exist_ok=True)
    p = f"{DUMP_DIR}/{mid}.mtx"
    if not os.path.exists(p):
        sh(f"{rc.BIN['cpu']} {rc.margs_for(mid)} --dump-mtx {p} --solver none", timeout=TIMEOUT)
    return p if os.path.exists(p) else None


def available():
    """(ok, reason) — is CMG runnable on this machine at all?

    Probed ONCE by the sweep, so a machine without MATLAB says so in one loud
    line instead of failing 27 times or, worse, leaving 27 silent gaps."""
    for val, what in ((MATLAB, "the MATLAB R2026a install tree "
                               "($APXCHOL_MATLAB_ROOT / paths_local.MATLAB)"),
                      (CMG_SOLVER, "the cmg-solver checkout "
                                   "($APXCHOL_CMG_SOLVER / paths_local.CMG_SOLVER)")):
        if not val:
            return False, f"{what} is not configured"
        if not os.path.exists(val):
            return False, f"{what} is not at {val}"
    try:
        cp = sh(f"docker image inspect {IMAGE}", timeout=120)
    except Exception as e:                                  # docker missing / not usable
        return False, f"docker is not usable here ({e})"
    if cp.returncode != 0:
        return False, (f"the container image {IMAGE} is missing — build it from "
                       f"benchmarks/cmg/Dockerfile.matlab")
    return True, ""


def _run_matlab_cmg(mtx_basename, as_operator):
    """Run bench_cmg.m on /cmg-mtx/<basename> inside the container; return
    (stdout, stderr).

    as_operator=1 for a kind=operator matrix: bench_cmg then reads the dump with
    its diagonal intact and solves it unpinned and unshifted, i.e. the same
    system every other solver in the cell got. as_operator=0 keeps the Laplacian
    reading + reg_rel shift, which is the singular-L path."""
    rc.require_path(MATLAB, "APXCHOL_MATLAB_ROOT", "MATLAB", "the MATLAB R2026a install")
    rc.require_path(CMG_SOLVER, "APXCHOL_CMG_SOLVER", "CMG_SOLVER", "the cmg-solver checkout")
    reg = "0" if as_operator else REG
    inner = (f"cd /bench && /opt/MATLAB/R2026a/bin/matlab -batch "
             f"\\\"bench_cmg('/cmg-mtx/{mtx_basename}', {TOL}, {MAXITER}, {SEED}, {reg}, "
             f"{1 if as_operator else 0})\\\"")
    cmd = (f"docker run --rm --network=host "
           f"-v {MATLAB}:/opt/MATLAB/R2026a:ro "
           f"-v {CMG_SOLVER}:/cmg-solver:ro "
           f"-v {DUMP_DIR}:/cmg-mtx:ro "
           f"-v {ROOT}/benchmarks/cmg:/bench:ro "
           f"-e HOME=/tmp -e CMG_ROOT=/cmg-solver "
           f"{IMAGE} bash -c \"{inner}\"")
    cp = sh(cmd, timeout=TIMEOUT)
    return cp.stdout, (cp.stderr or "")


def run_one(mid):
    fam = rc.MATRICES[mid]["family"]
    if rc.cell_done(fam, mid, "cmg", "", THREADS, "cpu", terminal=TERMINAL):
        return "skip(done)"
    n = rc.MATRICES[mid].get("n", 0)
    if n and n > CMG_MAX_N:
        rc.emit_cell(fam, mid, "cmg", "", "n/a", {"n": n}, THREADS, "cpu",
                     {**_prov(rc.class_of(mid) == "sddm"),
                      "note": PROV["note"] + f" [skipped: n={n} > {CMG_MAX_N}]"},
                     matrix_meta={"cmg_na_reason": f"n={n} exceeds CMG_MAX_N={CMG_MAX_N}, the "
                                                   f"runner's cap on CMG's O(n) hierarchy setup"})
        return f"skip(n={n}>cap)"
    as_operator = rc.class_of(mid) == "sddm"
    meta = {"cmg_input": (
        "the DUMPED operator, read with its diagonal intact and solved unpinned and "
        "unshifted (the same system every other solver in this cell got)" if as_operator else
        f"the DUMPED L, re-derived from |off-diagonal| (identical to the dump for a "
        f"kind=graph matrix) and shifted by reg_rel={REG}*mean|diag| — CMG is the lone "
        f"reg-rel solver and is scored on L+eps*I")}
    try:
        src = _dump(mid)
        if not src:
            return "SKIP(dump)"
        out, err = _run_matlab_cmg(os.path.basename(src), as_operator)
    except subprocess.TimeoutExpired:
        meta["cmg_na_reason"] = f"exceeded the {TIMEOUT}s per-cell wall cap"
        rc.emit_cell(fam, mid, "cmg", "", "timeout", {}, THREADS, "cpu",
                     _prov(as_operator), matrix_meta=meta)
        return "TIMEOUT"
    status, metrics = rc.classify(rc.parse_csv(out), TOL)
    if status == "n/a":
        # bench_cmg.m tags its refusals '[n/a] cmg on <name>: <reason>' on stderr.
        hit = re.search(r"^\[n/a\] cmg[^:]*:\s*(.+)$", err, re.M)
        meta["cmg_na_reason"] = (hit.group(1).strip() if hit else
                                 "CMG declined this matrix (no reason line captured)")
    rc.emit_cell(fam, mid, "cmg", "", status, metrics, THREADS, "cpu",
                 _prov(as_operator), matrix_meta=meta)
    note = f" [{meta['cmg_na_reason']}]" if status == "n/a" else ""
    return (f"{status} it={metrics.get('iters') if metrics else '?'} "
            f"rr={metrics.get('rel_res') if metrics else '?'}{note}")


CHART_MATS = [m for m, meta in rc.MATRICES.items()
              if meta["family"] in ("grids", "ipm", "suitesparse")]

if __name__ == "__main__":
    mats = sys.argv[1:] or CHART_MATS
    for mid in mats:
        print(f"[cmg] {mid:18} {run_one(mid)}", flush=True)

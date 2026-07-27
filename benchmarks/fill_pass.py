#!/usr/bin/env python3
"""One-time FILL measurement pass for the Cholesky-family solvers.

Records each solver's fill ratio on a CONSISTENT definition — fill =
2*offdiag(L) / offdiag(A) = 2*(factor_nnz - n) / (matrix_nnz - n) — so apxchol,
ParAC and RCHOL are comparable (RCHOL's stored `fillin` uses 2*Lnnz/Annz incl.
the diagonal, which is recomputed here from the off-diagonal counts). AMG solvers
(BoomerAMG/AMGCL) have NO triangular factor, so no comparable number — they are
intentionally absent from the fill chart.

  - apxchol: APXCHOL_REPORT_FILL=1 re-factors and prints "FILL ... Lnnz=.. offdiag=.. adj_nnz=.."
  - ParAC  : the GPU graph driver prints "factorization nnz", "laplacian nnz", "num cols"
  - RCHOL/pRCHOL: back-computed from the existing per-cell store (fillin, nnz, n)

Writes results/fill_cells/<mid>__<solver>.json. No PCG-solve timing is used, so
this is safe to run alongside other (timing) work. Run from repo root:
  python3 benchmarks/fill_pass.py
"""
import json, os, re, glob

from runner_common import (ROOT, CELLS, sh as _sh, parac_amd_mtx,
                           PARAC_GPU_DRIVER as GPU_DRIVER,
                           PARAC_GPU_DRIVER_PHYS as GPU_DRIVER_PHYS,
                           PARAC_CPU_DRIVER as CPU_DRIVER,
                           PARAC_SORTED as SORTED)
# The sort-cache tag rule lives with the writer (parac_runner); import it so the
# reader here can never drift from the name parac_runner actually writes.
from parac_runner import _dump_tag as parac_dump_tag

BIN = f"{ROOT}/benchmarks/build/benchmark"
# ParAC drivers (constants from runner_common). The CPU and GPU implementations
# produce DIFFERENT factors (different ordering -- AMD vs ParAC's random-nnz-sort
# -- AND different elimination code), so CPU/GPU fill genuinely differ and are
# charted as separate series. Each has a graph (Laplacian) and a physics (SDDM,
# trims the augmentation) mode.
OUT = f"{ROOT}/results/fill_cells"
REG = "1e-6"

# (matrix_id, family, apxchol-args, needs_reg, parac_sorted_mtx)
MATS = [
    ("grid_500", "grids", "--graph grid --n 500", False),
    ("grid_1000", "grids", "--graph grid --n 1000", False),
    ("grid_2000", "grids", "--graph grid --n 2000", False),
    ("grid_3000", "grids", "--graph grid --n 3000", False),
    ("grid3d_100", "grids", "--graph grid3d --n 100", False),
    ("grid3d_150", "grids", "--graph grid3d --n 150", False),
    ("parabolic_fem", "suitesparse", f"--mtx {ROOT}/data/matrices/parabolic_fem.mtx", True),
    ("apache2", "suitesparse", f"--mtx {ROOT}/data/matrices/apache2.mtx", True),
    ("ecology1", "suitesparse", f"--mtx {ROOT}/data/matrices/ecology1.mtx", True),
    ("G3_circuit", "suitesparse", f"--mtx {ROOT}/data/matrices/G3_circuit.mtx", True),
    ("thermal2", "suitesparse", f"--mtx {ROOT}/data/matrices/thermal2.mtx", True),
    ("com-Amazon", "suitesparse", f"--mtx {ROOT}/data/matrices/com-Amazon.mtx", True),
    ("coAuthorsDBLP", "suitesparse", f"--mtx {ROOT}/data/matrices/coAuthorsDBLP.mtx", True),
    ("kron_g500-logn16", "suitesparse", f"--mtx {ROOT}/data/matrices/kron_g500-logn16.mtx", True),
    ("iter0010", "ipm", f"--mtx {ROOT}/data/ipm/iter0010/matrix.mtx", False),
    ("iter0020", "ipm", f"--mtx {ROOT}/data/ipm/iter0020/matrix.mtx", False),
    ("iter0030", "ipm", f"--mtx {ROOT}/data/ipm/iter0030/matrix.mtx", False),
    ("iter0040", "ipm", f"--mtx {ROOT}/data/ipm/iter0040/matrix.mtx", False),
]


def sh(cmd, timeout=900, env=None):
    # runner_common.sh = hardened (process-group kill on timeout); the previous
    # local subprocess.run carried the orphan-on-timeout bug.
    return _sh(cmd, timeout=timeout, env=env)


def done(mid, solver):
    return os.path.exists(f"{OUT}/{mid}__{solver}.json")


def emit(mid, family, solver, n, adj_nnz, factor_offdiag):
    os.makedirs(OUT, exist_ok=True)
    fill = 2.0 * factor_offdiag / adj_nnz if adj_nnz else 0.0
    json.dump({"matrix_id": mid, "family": family, "solver": solver,
               "n": n, "adj_nnz": adj_nnz, "factor_offdiag": factor_offdiag,
               "fill": round(fill, 4)},
              open(f"{OUT}/{mid}__{solver}.json", "w"), indent=2)
    print(f"  {mid:16} {solver:14} fill={fill:.3f}  (offdiag(L)={factor_offdiag} offdiag(A)={adj_nnz})")
    return fill


AC_FILL_JL = f"{ROOT}/benchmarks/julia/ac_fill.jl"
DUMP = "/tmp/parac_fair_dump"

def ac_mtx(mid, family, args, reg):
    # AC reads a .mtx. grids: pure Laplacian dump; suitesparse: the data .mtx;
    # IPM: the SDDM .mtx (ac_fill.jl augments it to a Laplacian, mirroring
    # approxchol_sddm, so the fill is comparable to apxchol's SDDM factor).
    if "--graph" in args:
        p = f"{DUMP}/{mid}-pure.mtx"
        if not os.path.exists(p):
            sh(f"{BIN} {args} --dump-mtx {p} --solver none")
        return p if os.path.exists(p) else None
    m = re.search(r"--mtx (\S+)", args)
    return m.group(1) if m else None

def ac_fill(mid, family, args, reg):
    src = ac_mtx(mid, family, args, reg)
    if not src or not os.path.exists(AC_FILL_JL):
        return
    sddm = "sddm" if family == "ipm" else ""   # IPM matrices are SDDM (augment to Laplacian)
    for solver, variant in (("ac", "ac"), ("ac2", "ac2")):
        if done(mid, solver):
            continue
        o = sh(f"julia --project={ROOT}/benchmarks/julia {AC_FILL_JL} {src} {variant} {sddm}").stdout
        m = re.search(r"offdiagL=(\d+) offdiagA=(\d+)", o)
        if not m:
            print(f"  {mid:16} {solver:14} FAILED (ac_fill.jl)"); continue
        offL, adjA = int(m.group(1)), int(m.group(2))
        n = int(re.search(r"n=(\d+)", o).group(1))
        emit(mid, family, solver, n, adjA, offL)


# apxchol fill per IS-selector (bg/luby/root/bk): the elimination order differs, so the
# factor density does too -- charted as separate series next to the AC reference.
APX_SELECTORS = [("apxchol_bg", "bg+tree[vec_pool]"), ("apxchol_luby", "luby+tree[vec_pool]"),
                 ("apxchol_root", "root+tree[vec_pool]"), ("apxchol_bk", "bk+tree[vec_pool]")]

def apxchol_fill(mid, family, args, reg):
    regflag = f"--reg-rel {REG}" if reg else ""
    for solver_key, cfg in APX_SELECTORS:
        if done(mid, solver_key):
            continue
        env = dict(os.environ, APXCHOL_REPORT_FILL="1")
        cmd = (f"{BIN} {args} {regflag} --solver apxchol_v1 --v1-configs '{cfg}' "
               f"--threads 16 --tol 1e-8 --maxiter 1 --repeat 1 --csv")
        o = sh(cmd, env=env)
        m = re.search(r"FILL.*offdiag=(\d+)\s+adj_nnz=(\d+)", o.stderr)
        if not m:
            print(f"  {mid:16} {solver_key:16} FAILED (no FILL line)"); continue
        offdiag, adj = int(m.group(1)), int(m.group(2))
        n = 0
        for line in o.stdout.splitlines():           # n = adj-row count, from the CSV
            f = line.split(",")
            if len(f) > 2 and f[0] not in ("solver",):
                try: n = int(f[2])
                except: pass
        emit(mid, family, solver_key, n, adj, offdiag)


# AMD-reorder cache lookup -> runner_common.parac_amd_mtx (family-strict).
amd_mtx = parac_amd_mtx


def parac_gpu_fill(mid, family, solver_key, driver):
    """ParAC GPU fill: factor nnz from the GPU driver (graph or physics) on ParAC's
    own random-nnz-sorted input. offdiag(L)=fac-n, offdiag(A)=lap-n.
    parac_runner names that cache '{mid}-{tag}-nnz-sorted.mtx' (tag = pin/pure);
    the un-tagged name is the legacy cache from before the tag existed."""
    tag = parac_dump_tag(mid)
    src = next((p for p in (f"{SORTED}/{mid}-{tag}-nnz-sorted.mtx",
                            f"{SORTED}/{mid}-nnz-sorted.mtx") if os.path.exists(p)), None)
    if src is None or not os.path.exists(driver):
        print(f"  {mid:16} {solver_key:16} SKIP (no sorted mtx / driver)"); return
    o = sh(f"{driver} {src} 512 1 1e-8").stdout
    g = lambda p: (int(re.search(p, o).group(1)) if re.search(p, o) else None)
    fac, lap, n = g(r"factorization nnz:\s*(\d+)"), g(r"laplacian nnz:\s*(\d+)"), g(r"num cols:\s*(\d+)")
    if not (fac and lap and n):
        print(f"  {mid:16} {solver_key:16} FAILED (nnz parse)"); return
    emit(mid, family, solver_key, n, lap - n, fac - n)


def parac_cpu_fill(mid, family, solver_key, physics):
    """ParAC CPU fill: factor nnz from the CPU driver (5th arg '1' => physics, which
    trims the SDDM augmentation) on the AMD-reordered input. Graph mode reads the
    `factor nnz` / `number of nonzeros` lines; physics reads the `trimmed ...` ones."""
    amd = amd_mtx(mid, family)
    if not amd or not os.path.exists(CPU_DRIVER):
        print(f"  {mid:16} {solver_key:16} SKIP (no AMD mtx / driver)"); return
    arg5 = "1" if physics else ""
    o = sh(f'taskset -c 0-15 {CPU_DRIVER} {amd} 16 "" {arg5}'.strip()).stdout
    g = lambda p: (int(re.search(p, o).group(1)) if re.search(p, o) else None)
    n = g(r"number of nodes:\s*(\d+)")
    if physics:
        fac, lap = g(r"trimmed factor nnz:\s*(\d+)"), g(r"trimmed laplacian nnz:\s*(\d+)")
    else:
        fac, lap = g(r"factor nnz:\s*(\d+)"), g(r"number of nonzeros:\s*(\d+)")
    if not (fac and lap and n):
        print(f"  {mid:16} {solver_key:16} FAILED (nnz parse)"); return
    emit(mid, family, solver_key, n, lap - n, fac - n)


def rchol_from_cells(mid, family):
    # RCHOL stored fillin = 2*G.nnz()/A.nnz() (incl diagonal). Recover the
    # off-diagonal definition: G.nnz() = fillin*A.nnz()/2; offdiag(L)=G.nnz()-n.
    for solver in ("rchol", "rchol_par"):
        for f in glob.glob(f"{CELLS}/{family}/{mid}__{solver}__*.json"):
            d = json.load(open(f)); m = d.get("metrics", {})
            fillin, annz, n = m.get("fillin"), m.get("nnz"), m.get("n")
            if not (fillin and annz and n) or fillin == 0.0:
                continue
            gnnz = fillin * annz / 2.0
            emit(mid, family, solver, n, annz - n, gnnz - n)
            break


def main():
    for mid, family, args, reg in MATS:
        print(f"[{mid}]", flush=True)
        try: apxchol_fill(mid, family, args, reg)   # 4 IS-selectors (own per-key skip)
        except Exception as e: print(f"  apxchol error: {e}")
        # ParAC: graph + physics, each on CPU (AMD-reordered) and GPU (nnz-sorted).
        for sk, drv in (("parac_graph_gpu", GPU_DRIVER), ("parac_physics_gpu", GPU_DRIVER_PHYS)):
            if not done(mid, sk):
                try: parac_gpu_fill(mid, family, sk, drv)
                except Exception as e: print(f"  {sk} error: {e}")
        for sk, phys in (("parac_graph_cpu", False), ("parac_physics_cpu", True)):
            if not done(mid, sk):
                try: parac_cpu_fill(mid, family, sk, phys)
                except Exception as e: print(f"  {sk} error: {e}")
        if not done(mid, "rchol"):
            rchol_from_cells(mid, family)
        try: ac_fill(mid, family, args, reg)   # AC + AC2 (own per-variant skip)
        except Exception as e: print(f"  ac error: {e}")
    print(f"fill pass done -> {OUT}")


if __name__ == "__main__":
    main()

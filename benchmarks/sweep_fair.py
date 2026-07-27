#!/usr/bin/env python3
"""Single fair runner for ALL matrix families (grids + SuiteSparse + IPM), on
either device (--device cpu | gpu). Both write the SAME per-cell JSON store
(results/cells/<family>/<mid>__<solver>__<cfg>__t16__<device>.json) and are
RESUME-SAFE: a cell that already has a terminal status is skipped, so re-running
only fills empty cells. fair_charts.py / gpu_charts.py / combined_charts.py all
read this one store.

Singular-Laplacian families (grids, SuiteSparse) now run UNSHIFTED, on the
ORIGINAL singular L (no --reg-rel). The benchmark grounds singular Laplacians with
a symmetric multi-component Dirichlet pin (one pin per connected component) and
scores every solver's true residual against the original L, so multigrid reaches
1e-8 without the eps*I perturbation that --reg-rel used to add. (reg-rel was
masking real solver limitations -- e.g. AMGCL floors on the disconnected
as-Skitter -- by solving a better-conditioned L+eps*I.) IPM normal-equation
matrices are already strictly SDDM and likewise run unshifted. CMG is the lone
reg-rel solver and runs from its own script (benchmarks/cmg_matlab_runner.py),
not from here; everything in this sweep is reg-free.

  # CPU axis (default):
  systemd-inhibit --what=sleep:idle --mode=block python3 benchmarks/sweep_fair.py
  # GPU axis (needs the CUDA build; apxchol uses the GPU-resident PCG):
  python3 benchmarks/sweep_fair.py --device gpu

ParAC runs IN-PROCESS via parac_runner.py (graph+physics on both axes), emitting
into the same store. Shared harness (hardened sh, matrix registry, cell store,
CSV parse, VRAM sidecar) lives in runner_common.py.
"""
import argparse, json, os, subprocess, sys, time

import runner_common as rc
import parac_runner
from runner_common import ROOT, BIN, CELLS, GRIDS, SS, IPM, matrix_args, sh

DUMP = "/tmp/fair_dump"
TOL = "1e-8"
REPS = 3
THREADS = 16
# 30 min/cell: the >=1e8-nnz giants (com-Orkut apxchol ~170s/rep x3, BoomerAMG
# mega-hub setup, RCHOL fill blowup) can legitimately exceed the old 1200s.
TIMEOUT = 1800
TIMEOUT_MULT = 10      # competitor wall-clock cap = TIMEOUT_MULT x GPU-apxchol's total time
TIMEOUT_FLOOR = 60     # ...but never below this (so fast matrices still give solvers room)
# Per-benchmark host-memory cap (GB), CPU only. A runaway solver -- e.g. serial RCHOL's
# natural-order fill blowup on the 237M-nnz com-Orkut hit 110GB RSS and OOM-killed the
# desktop (2026-06-12) -- gets an address-space ceiling via `ulimit -v`, so it dies with
# bad_alloc (recorded as 'oom') instead of exhausting system RAM. Legit com-Orkut apxchol
# peaks ~62GB (root+tree[vec]); 100GB clears that comfortably and still leaves ~24GB
# headroom on the 124GB box. NOT applied on GPU (CUDA reserves large host VM; breaks runs).
MEM_CAP_GB = float(os.environ.get("APXCHOL_BENCH_MEM_CAP_GB", "100"))
DEVICE = "cpu"   # set in main() from --device
# CPU PCG caps iters at 500; the GPU axis allows more (cheap per-iter), matching
# the retired gpu_chart_sweep.sh (2000, or 1000 for IPM).
MAXITER = {"cpu": 500, "gpu": 2000, "gpu_ipm": 1000}

PROV = {"boost":"on","boost_expected":"on","git_sha":rc.git_sha(),
        "note":"FAIR run","repeat":REPS,
        "tier":"broad","timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}

def classify(m):
    return rc.classify(m, TOL)

def emit(family, mid, solver, config, status, metrics):
    return rc.emit_cell(family, mid, solver, config, status, metrics, THREADS, DEVICE, PROV)

def run_cpp(margs, solver, config, reg, family=None, boomeramg_cfg=None, timeout=None, ground=None):
    cfg = f"--v1-configs '{config}'" if solver=="apxchol_v1" else ""
    # AMGCL de-singularization series: ground=coarse charts the relaxed-coarse cell
    # (--decompose stays auto, so whole+coarse on connected / split+coarse on
    # disconnected). The headline AMGCL series passes no --ground (auto = pin).
    gflag = f"--ground {ground}" if ground else ""
    regflag = f"--reg-rel {reg}" if reg else ""   # IPM is native SDDM: no shift
    mi = MAXITER["gpu_ipm"] if (DEVICE=="gpu" and family=="ipm") else MAXITER[DEVICE]
    # env vars passed via `env VAR=...` so /usr/bin/time wraps the whole call.
    #   BOOMERAMG_CFG=cut : CoarsenCutFactor variant (hub->F-point; fixes mega-hub blowup).
    base_envs = []
    if boomeramg_cfg: base_envs.append(f"BOOMERAMG_CFG={boomeramg_cfg}")
    def _diag(returncode, stdout, stderr):
        # Persist the evidence of a non-complete run INTO the cell (the com-Orkut
        # BoomerAMG segfault was diagnosable only because systemd-coredump happened
        # to catch it -- the runner used to drop stdout/stderr on the floor).
        d = {"returncode": returncode}
        if stdout: d["stdout_tail"] = stdout[-4000:]
        if stderr: d["stderr_tail"] = stderr[-4000:]
        return d
    def _run(extra):
        envs = base_envs + extra
        envpfx = f"env {' '.join(envs)} " if envs else ""
        inner=(f"{envpfx}taskset -c 0-{THREADS-1} {BIN[DEVICE]} {margs} --solver {solver} {cfg} "
               f"{gflag} {regflag} --threads {THREADS} --tol {TOL} --maxiter {mi} --repeat {REPS} --csv")
        # /usr/bin/time -f '%M': peak RSS (kbytes) -> stderr 'APXRSS <kb>'; the max over
        # all reps (setup-phase peak dominates). solve_rss_mb (held during solve) comes
        # from the binary's CSV; peak - solve_rss = the setup transient.
        cmd=f"/usr/bin/time -f 'APXRSS %M' {inner}"
        # CPU only: rc.sh's mem_cap ceils the address space so a memory-bomb solver
        # dies with bad_alloc (-> 'oom') instead of OOM-killing the box.
        cap = MEM_CAP_GB if DEVICE == "cpu" else None
        with rc.VramSampler("benchmark", enabled=(DEVICE == "gpu")) as vram:
            try: p=sh(cmd, timeout=timeout or TIMEOUT, mem_cap_gb=cap)
            except subprocess.TimeoutExpired as e:
                return "timeout", _diag(None, e.output, e.stderr)
        st,m = classify(rc.parse_csv(p.stdout))
        if m is None:
            # No CSV row = crash. Distinguish OUT-OF-MEMORY (the giants: cuSPARSE SpSV
            # OOMs ~8.8GB on Orkut; CPU bad_alloc / OOM-killer SIGKILL=-9/137) from a
            # generic failure, so the charts can show "doesn't fit" apart from "broke".
            blob=((p.stdout or "")+(p.stderr or "")).lower()
            if (p.returncode in (-9,137) or any(s in blob for s in
                    ("out of memory","bad_alloc","cudaerrormemoryalloc","cuda_error_out_of_memory",
                     "cublas_status_alloc_failed","cusparse_status_alloc_failed","cudamalloc",
                     "alloc_failed","cannot allocate","cuda oom","gpu out of memory"))):
                st="oom"
            m = _diag(p.returncode, p.stdout, p.stderr)   # failed/oom cells keep the evidence
        if m is not None:
            for ln in p.stderr.splitlines():
                if ln.startswith("APXRSS "):
                    try: m["max_rss_mb"]=round(int(ln.split()[1])/1024.0,1)
                    except: pass
            pk = vram.peak_mb()                 # whole-run peak VRAM (GPU axis; sidecar)
            if pk: m["max_vram_mb"] = pk
        return st,m
    st,m = _run([])
    # apxchol-GPU: cuSPARSE's O(nnz) SpSV buffer OOMs the giants (com-Orkut). Retry with
    # our O(n) level-set backend, which fits 16GB -- so Orkut-GPU gets a real number
    # instead of OOM. (Tagged via APXCHOL_GPU_SPTRSV; the cell's config string is the
    # apxchol config either way, so the chart just sees a populated apxchol-GPU cell.)
    if st=="oom" and DEVICE=="gpu" and solver=="apxchol_v1":
        st,m = _run(["APXCHOL_GPU_SPTRSV=levelset"])
        # Provenance: this cell is the O(n) level-set fallback (cuSPARSE SpSV OOM'd),
        # NOT the default cuSPARSE backend -- recorded so the charts/notes can mark it.
        # The recorded setup/solve/total are the level-set run ALONE (a fresh invocation);
        # the failed cuSPARSE attempt's time is not included.
        if m is not None: m["gpu_sptrsv"]="levelset (cuSPARSE OOM fallback)"
    return st,m

def run_julia(mtx, solver):
    # AC/AC2 (Laplacians.jl) are Laplacian-native (handle the null-space internally),
    # so they read the PURE Laplacian mtx (eps negligible vs the regularized others).
    cmd=(f"taskset -c 0-{THREADS-1} julia --project=benchmarks/julia "
         f"benchmarks/julia/bench_laplacians.jl --mtx {mtx} --solver {solver} "
         f"--tol {TOL} --maxiter 500 --csv")
    try: out=sh(cmd, timeout=TIMEOUT).stdout
    except subprocess.TimeoutExpired: return "timeout", None
    return classify(rc.parse_csv(out))

def run_parac(mid):
    # In-process via parac_runner (graph+physics cells, resume-safe). Per-step
    # timeouts/mem-caps live inside parac_runner (PARAC_TIMEOUT_S / PARAC_GPU_
    # TIMEOUT_S / PARAC_MEM_CAP_GB); a failure there never aborts the sweep.
    label = "parac (GPU)" if DEVICE == "gpu" else "parac (CPU)"
    print(f"   {label:24} (graph+physics)...", flush=True)
    try:
        res = parac_runner.run_gpu(mid) if DEVICE == "gpu" else parac_runner.run_cpu(mid)
        print(f"   {label:24} {res}")
    except Exception as e:
        print(f"   {label:24} ERROR: {e}")

def grid_mtx(mid, kind, spec):
    os.makedirs(DUMP,exist_ok=True); p=f"{DUMP}/{mid}.mtx"
    if not os.path.exists(p):
        sh(f"{BIN[DEVICE]} {matrix_args(kind,spec)} --dump-mtx {p} --solver none", timeout=TIMEOUT)
    return p if os.path.exists(p) else None

# --- CPU axis solver sets ---
# Solver set: (solver, config)
APX = [("apxchol_v1","bg+tree[vec_pool]"),
       ("apxchol_v1","root+tree[vec_pool]"),       # rootset IS (ablation/thread-scaling only)
       ("apxchol_v1","bk+tree[vec_pool]"),         # Baumann-Kyng IS (ablation only)
       ("apxchol_v1","luby+tree[vec_pool]"),       # Luby IS (ablation only)
       # vec<vec> storage variants (ablation only) -- the dense-array incidence
       # backend vs the default vec_pool. Swept on every family in the SAME run as
       # vec_pool so the storage ablation is an honest same-session A/B (the old
       # [vec] cells were sha aeac9836 "capped first pass", cross-run-incomparable).
       ("apxchol_v1","bg+tree[vec]"),
       ("apxchol_v1","root+tree[vec]"),
       ("apxchol_v1","bk+tree[vec]"),
       ("apxchol_v1","luby+tree[vec]"),
       # Full selector x storage grid for the ablation HEATMAP: {bg,luby,root,bk} x
       # {fwd_star, bstr}. (vec / vec_pool already covered above for all four
       # selectors -> the 4-storage axis fwd_star/vec/bstr/vec_pool.) Shows the
       # backend progression forward_star (old linked-list default) -> vec
       # (SBO) -> bstr (bit-string) -> vec_pool (the default, drops fwd_star's per-edge
       # pointer chase). forward_star uses the bare-named base combo; bstr has
       # dedicated combos in benchmark.cpp (one per selector).
       ("apxchol_v1","bg+tree[fwd_star]"),
       ("apxchol_v1","luby+tree[fwd_star]"),
       ("apxchol_v1","root+tree[fwd_star]"),
       ("apxchol_v1","bk+tree[fwd_star]"),
       ("apxchol_v1","bg+tree[bstr]"),
       ("apxchol_v1","luby+tree[bstr]"),
       ("apxchol_v1","root+tree[bstr]"),
       ("apxchol_v1","bk+tree[bstr]")]
# The single headline apxchol config is bg+tree.
# CHOLMOD not included (direct solver; the suite compares preconditioned
# iterative methods).
# rchol_par = pRCHOL (parallel ND factorization) — beats serial RCHOL on 3D grids.
# AMGCL re-added: a distinct algebraic-MG competitor
# (smoothed-aggregation + spai0, vs BoomerAMG's classical AMG); converges to true
# 1e-8 under --reg-rel.
COMP = ["rchol","rchol_par","hypre_boomeramg","amgcl"]
# AC/AC2 = Laplacians.jl Julia reference of our method (read .mtx). Slow (julia).
JULIA = ["ac","ac2"]
RUN_PARAC = True     # ParAC runs as part of every sweep (skip via --no-parac)
PARAC_ONLY = False   # --parac-only: run ONLY ParAC (skip apxchol/competitors/AC)
# --no-cap: drop the 10x-GPU-apxchol competitor wall cap and RE-RUN cells currently
# stored as 'timeout' (clamped), so AMGCL/BoomerAMG get an HONEST uncapped wall time on
# the matrices where they previously hit the cap. NOCAP_TIMEOUT is still a sane outer
# ceiling (a true hang can't run forever); override via NOCAP_TIMEOUT_S.
NO_CAP = False
NOCAP_TIMEOUT = int(os.environ.get("NOCAP_TIMEOUT_S", str(4 * 3600)))

# --- GPU axis solver sets ---
# apxchol on GPU = the single bg+tree config via the GPU-resident PCG (run_cpp adds
# is 2D-structured-grid only (added per-matrix in do_matrix). amgcl_cuda's host-side
# AMG setup is now OpenMP-parallel (CMakeLists -Xcompiler=-fopenmp fix).
APX_GPU = [("apxchol_v1","bg+tree[vec_pool]"),
           ("apxchol_v1","luby+tree[vec_pool]"),   # luby/root produce shallow, deterministic
           ("apxchol_v1","root+tree[vec_pool]"),    # factors -> faster + stable GPU SpTRSV than
           ("apxchol_v1","bk+tree[vec_pool]")]      # bg's variable depth; bk is the deep worst case
COMP_GPU = ["hypre_boomeramg_gpu","amgcl_cuda"]

def cell_done(family, mid, solver, config):
    st = rc.cell_status(family, mid, solver, config, THREADS, DEVICE)
    # --no-cap: a previously CLAMPED 'timeout' cell is NOT terminal -- re-run it
    # uncapped for an honest wall time. complete/oom/failed stay terminal (no point).
    if NO_CAP and st == "timeout": return False
    return st in ("complete","not_converged","failed","timeout","oom")

def cell_metrics(family, mid, solver, config):
    """(status, metrics) from an existing cell, or (None, None)."""
    path = rc.cell_path(family, mid, solver, config, THREADS, DEVICE)
    try:
        c = json.load(open(path)); return c.get("status"), c.get("metrics")
    except: return None, None

def step(family, mid, solver, config, runner, label):
    if cell_done(family, mid, solver, config):
        print(f"   {label:24} (skip, done)")
        return cell_metrics(family, mid, solver, config)   # so resume can still read apxchol's time
    st,m=runner(); emit(family,mid,solver,config,st,m)
    print(f"   {label:24} {st} it={m.get('iters','-') if m else '-'} "
          f"total={m.get('total_s','-') if m else '-'}", flush=True)
    return st, m

def gpu_apx_total(family, mid):
    """Min total_s over the device=gpu apxchol cells for this matrix -- the reference
    for the 10x competitor cap. None if the GPU axis hasn't run this matrix yet (the
    caller then falls back to same-device apxchol). Reads the store directly so the CPU
    sweep can cap against the GPU number produced by an earlier `--device gpu` pass."""
    best = None
    for _, cfg in APX_GPU:
        path = rc.cell_path(family, mid, "apxchol_v1", cfg, 16, "gpu")
        try:
            c = json.load(open(path))
            if c.get("status") == "complete":
                t = c.get("metrics", {}).get("total_s")
                if t and (best is None or t < best):
                    best = t
        except Exception:
            pass
    return best

def do_matrix(mid, family, kind, spec, is2d, mtx_for_julia, n, reg):
    print(f"[{mid}] (n={n}) device={DEVICE}", flush=True)
    if PARAC_ONLY:                 # --parac-only: ParAC and nothing else
        run_parac(mid); return
    margs=matrix_args(kind,spec)
    # The competitor wall-clock cap is TIMEOUT_MULT x GPU-apxchol's total time -- the
    # fastest apxchol config is the universal reference, so "10x of GPU apxchol" is the
    # same bar on both axes (the charts clamp+mark anything slower at that cap). A
    # competitor exceeding it is recorded as status 'timeout' (a TERMINAL status,
    # distinct from blank = not-tested), so one slow solver on a giant doesn't stall the
    # sweep. Falls back to same-device apxchol when the GPU axis hasn't run this matrix
    # yet (run GPU before CPU so the cap is the intended GPU reference).
    def comp_timeout(t_apx_same):
        if NO_CAP:
            return NOCAP_TIMEOUT          # uncapped: only the sane outer ceiling applies
        ref = gpu_apx_total(family, mid) or t_apx_same
        return int(min(TIMEOUT, max(TIMEOUT_FLOOR, TIMEOUT_MULT * (ref or TIMEOUT/TIMEOUT_MULT))))
    if DEVICE == "gpu":
        t_apx = None
        for i,(solver,config) in enumerate(APX_GPU):
            _,m = step(family,mid,solver,config, lambda s=solver,c=config: run_cpp(margs,s,c,reg,family), config)
            if i==0 and m and m.get("total_s"): t_apx = m["total_s"]
        to = comp_timeout(t_apx)
        comp = list(COMP_GPU)
        for solver in comp:
            if solver == "hypre_boomeramg_gpu":
                # default (Hypre defaults) + CoarsenCutFactor 'cut' (hub->F-point),
                # charted as two GPU series like the CPU axis -- cut is the fix on the
                # mega-hub social giants, a no-op on grids/IPM.
                step(family,mid,solver,"", lambda s=solver: run_cpp(margs,s,"",reg,family,timeout=to), solver)
                step(family,mid,solver,"cut",
                     lambda s=solver: run_cpp(margs,s,"cut",reg,family,boomeramg_cfg="cut",timeout=to), solver+"/cut")
            else:
                step(family,mid,solver,"", lambda s=solver: run_cpp(margs,s,"",reg,family,timeout=to), solver)
        if RUN_PARAC:
            run_parac(mid)
        return  # no Julia on the GPU axis
    t_apx = None
    # com-Orkut is huge (~237M nnz) + int32-overflow-prone on CPU; the non-vec storage
    # backends (fwd_star/bstr) are almost never better than vec_pool yet cost a lot
    # of wall-time here, so keep only [vec] + [vec_pool] for it. "[vec]" does not match
    # "[vec_pool]"/"[bstr]" as a substring, so the gate is exact.
    apx_list = ([(s, c) for (s, c) in APX if "[vec_pool]" in c or "[vec]" in c]
                if mid == "com-Orkut" else APX)
    for i,(solver,config) in enumerate(apx_list):
        _,m = step(family,mid,solver,config, lambda s=solver,c=config: run_cpp(margs,s,c,reg,family), config)
        if i==0 and m and m.get("total_s"): t_apx = m["total_s"]
    to = comp_timeout(t_apx)
    for solver in COMP:
        if solver == "hypre_boomeramg":
            # Two series: classical default + CoarsenCutFactor 'cut' (hub->F-point).
            # The cut variant is a no-op on uniform-degree grids/IPM and the fix on
            # mega-hub social graphs (com-Youtube/as-Skitter), so charting both shows
            # the methodology point instead of letting default blow up the timeout.
            step(family,mid,solver,"", lambda s=solver: run_cpp(margs,s,"",reg,family,timeout=to), solver)
            step(family,mid,solver,"cut",
                 lambda s=solver: run_cpp(margs,s,"cut",reg,family,boomeramg_cfg="cut",timeout=to), solver+"/cut")
        else:
            step(family,mid,solver,"", lambda s=solver: run_cpp(margs,s,"",reg,family,timeout=to), solver)
    if mtx_for_julia:  # AC/AC2 read the .mtx directly (grids via dump, SS/IPM native)
        for s in JULIA:
            step(family,mid,s,"", lambda ss=s: run_julia(mtx_for_julia,ss), s)
    if RUN_PARAC:      # ParAC graph+physics (resume-safe; the runner skips done cells)
        run_parac(mid)
    # CMG is run separately by benchmarks/cmg_matlab_runner.py.

def main():
    global DEVICE, APX, JULIA, COMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu","gpu"], default="cpu",
                    help="cpu (build/) or gpu (build-cuda/, apxchol GPU-resident PCG)")
    ap.add_argument("--headline-only", action="store_true",
                    help="run only the 4 [vec_pool] IS-selectors (best-eliminator for the "
                         "comparison charts) + competitors; skip the vec/fwd_star/bstr "
                         "storage-ablation configs AND Julia AC/AC2. Use for new large "
                         "matrices that only feed the comparison charts, not the ablation.")
    ap.add_argument("--no-rchol", action="store_true",
                    help="drop RCHOL / pRCHOL from the competitor set. They blow up (high "
                         "fill, 489 iters / non-convergent) on the large social graphs, so "
                         "on those ParAC is the Cholesky-family competitor we keep.")
    ap.add_argument("--families", default="grids,suitesparse,ipm",
                    help="comma-separated subset of {grids,suitesparse,ipm} to run")
    ap.add_argument("--only", default="",
                    help="comma-separated matrix ids to run (subset of GRIDS/SS/IPM); "
                         "empty = all. Use to refresh smaller matrices first.")
    ap.add_argument("--no-julia", action="store_true",
                    help="skip the Julia AC/AC2 references (slow; unaffected "
                         "by the C++ de-singularization changes).")
    ap.add_argument("--no-parac", action="store_true",
                    help="skip ParAC (run in-process via parac_runner.py).")
    ap.add_argument("--parac-only", action="store_true",
                    help="run ONLY ParAC (skip apxchol / competitors / AC); for filling the "
                         "ParAC giant-matrix cells without touching the rest.")
    ap.add_argument("--no-cap", action="store_true",
                    help="drop the 10x-GPU-apxchol competitor wall cap AND re-run cells "
                         "currently stored as 'timeout' (clamped), so AMGCL/BoomerAMG get "
                         "an honest uncapped wall time. NOCAP_TIMEOUT_S (default 4h) is the "
                         "only outer ceiling. Pair with --only/--families to target matrices.")
    a = ap.parse_args()
    global RUN_PARAC, PARAC_ONLY, NO_CAP
    DEVICE = a.device
    fams = {f.strip() for f in a.families.split(",") if f.strip()}
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    want = lambda mid: (not only) or (mid in only)
    if a.no_julia: JULIA = []
    if a.no_parac: RUN_PARAC = False
    if a.parac_only: PARAC_ONLY = True
    if a.no_cap: NO_CAP = True
    if a.headline_only:
        APX = [("apxchol_v1","bg+tree[vec_pool]"), ("apxchol_v1","luby+tree[vec_pool]"),
               ("apxchol_v1","root+tree[vec_pool]"), ("apxchol_v1","bk+tree[vec_pool]")]
        JULIA = []   # AC/AC2 are too slow / fail on the large social graphs
    if a.no_rchol:
        COMP = [c for c in COMP if c not in ("rchol","rchol_par")]
    if DEVICE == "gpu" and not os.path.exists(BIN["gpu"]):
        sys.exit(f"GPU build not found: {BIN['gpu']} (configure with -DAPXCHOL_USE_CUDA=ON)")
    PROV["note"] = f"FAIR {DEVICE} run, singular L, multi-component Dirichlet pin (per-solver grounding)"
    print(f"=== fair sweep: device={DEVICE}, families={sorted(fams)}, "
          f"headline_only={a.headline_only}, store={CELLS} (resume-safe) ===", flush=True)
    # Resume-safe: cells with a terminal status are skipped (see cell_done).
    if "grids" in fams:
        for mid,family,kind,spec,is2d in GRIDS:
            if not want(mid): continue
            n = spec*spec if kind=="grid" else spec*spec*spec
            # grid_mtx dumps a (possibly multi-GB) .mtx only the Julia path needs;
            # skip the dump when AC/AC2 are not being run.
            gm = grid_mtx(mid,kind,spec) if JULIA else None
            do_matrix(mid,family,kind,spec,is2d,gm,n,None)   # singular L, multi-component Dirichlet pin
    if "suitesparse" in fams:
        for mid,path,n in SS:
            if not want(mid): continue
            do_matrix(mid,"suitesparse","mtx",f"{ROOT}/{path}",False,f"{ROOT}/{path}",n,None)  # singular L
    if "ipm" in fams:
        for mid,path,n in IPM:  # native SDDM -> unshifted (reg=None)
            if not want(mid): continue
            do_matrix(mid,"ipm","mtx",f"{ROOT}/{path}",False,f"{ROOT}/{path}",n,None)
    print("FAIR sweep done")

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Single fair runner for ALL matrix families (grids + SuiteSparse + IPM), on
either device (--device cpu | gpu). Both write the SAME per-cell JSON store
(results/cells/<family>/<mid>__<solver>__<cfg>__t16__<device>.json) and are
RESUME-SAFE: a cell that already has a terminal status is skipped, so re-running
only fills empty cells. fair_charts.py / gpu_charts.py / combined_charts.py all
read this one store.

Every matrix DECLARES how it is to be read (runner_common: kind=graph|operator),
and the binary requires that declaration on the command line — nothing here
guesses. kind=graph builds L = D - A from the file's |values| (the system a graph
file defines); kind=operator solves the published matrix as it stands, diagonal
included. Each matrix prints one interpretation line, and every cell records it
under matrix_meta. External solvers (ParAC, AC/AC2) are handed that same operator
through --dump-mtx rather than re-reading the registry file themselves.

Singular-Laplacian families (grids, SuiteSparse graphs) now run UNSHIFTED, on the
ORIGINAL singular L (no --reg-rel). The benchmark grounds singular Laplacians with
a symmetric multi-component Dirichlet pin (one pin per connected component) and
scores every solver's true residual against the original L, so multigrid reaches
1e-8 without the eps*I perturbation that --reg-rel used to add. (reg-rel was
masking real solver limitations -- e.g. AMGCL floors on the disconnected
as-Skitter -- by solving a better-conditioned L+eps*I.) IPM normal-equation
matrices are already strictly SDDM and likewise run unshifted. CMG remains the
lone reg-rel solver on the singular families (it is scored on L+eps*I there, and
on the published operator itself for kind=operator); everything else in this
sweep is reg-free.

EVERY solver in the comparison runs from here, so one sweep fills every cell:
apxchol + the C++ competitors in-process, ParAC via parac_runner.py, AC/AC2 via
benchmarks/julia/bench_laplacians.jl and CMG via cmg_matlab_runner.py (canonical
MATLAB CMG in the matlab-deps container). --no-julia / --no-cmg / --no-parac opt
out individually; --headline-only restricts the apxchol config list (AC/AC2 are
dropped only by an explicit --no-julia).
A solver that cannot take a given matrix emits an explicit `n/a` cell carrying
the reason, never a silent gap.

  # CPU axis (default):
  systemd-inhibit --what=sleep:idle --mode=block python3 benchmarks/sweep_fair.py
  # GPU axis (needs the CUDA build; apxchol uses the GPU-resident PCG):
  python3 benchmarks/sweep_fair.py --device gpu

ParAC runs IN-PROCESS via parac_runner.py (graph+physics on both axes), emitting
into the same store. Shared harness (hardened sh, matrix registry, cell store,
CSV parse, VRAM sidecar) lives in runner_common.py. Schema-2 timeout cells persist
the exact wall-clock cap as `timeout_cap_s`; the charting pipeline never reconstructs
that lower bound from later timings.
"""
import argparse, json, os, re, subprocess, sys, time

import runner_common as rc
import parac_runner
import cmg_matlab_runner
from runner_common import BIN, CELLS, matrix_args, sh

# Dumped .mtx cache (grid Laplacians + the exact operator handed to the external
# solvers). /tmp is a tmpfs on this box, i.e. RAM with a per-user quota, and the
# giants dump multi-GB files — point this at a disk-backed directory when
# sweeping them. See benchmarks/README.md.
DUMP = os.environ.get("APXCHOL_BENCH_DUMP_DIR", "/tmp/fair_dump")
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


def selected_matrices(families, only=()):
    """Yield canonical registry records selected for a sweep.

    Keep the sweep decoupled from the positional layout of GRIDS/SS/IPM.  In
    particular, file-backed registry entries gained a fifth ``cls`` field when
    grounding became explicit; consuming MATRICES avoids duplicating that tuple
    schema here.
    """
    only = set(only)
    for mid, matrix in rc.MATRICES.items():
        if matrix["family"] in families and (not only or mid in only):
            yield mid, matrix

# What the BINARY reported about the operator it assembled for each matrix
# (`MATRIX_META ...` on stderr): kind, n/nnz, and whether the assembled operator
# turned out to be a singular Laplacian (grounded) or full-rank (solved as-is).
# Recorded once per matrix and merged into every cell for it, including the ones
# produced by external solvers that never run the binary themselves — they are
# handed the same operator via --dump-mtx, so the record applies to them too.
OBSERVED = {}

# What the BINARY reported about ITSELF (`BUILD_META ...` on stderr): compiler,
# version, OpenMP runtime, arch flags. Filled from every run of it — including a
# run that timed out — and merged into the provenance of the cells it produced,
# so a cell names the toolchain its numbers came from. NOT applied to the AC/AC2
# cells: those never run this binary (see julia_toolchain).
BUILD = {}

# If the binary never reported a BUILD_META line, the cell must say so OUT LOUD rather
# than be written with no toolchain keys at all. A silent omission is indistinguishable
# from a cell predating this mechanism, and the case where it happens is exactly the one
# this exists to catch: a stale binary sitting in the build directory that nobody
# rebuilt. That happened in this repo on 2026-08-20.
UNKNOWN_TOOLCHAIN = {
    "compiler": "unknown",
    "toolchain_note": "binary emitted no BUILD_META line - stale or pre-BUILD_META build",
}
_warned_no_build_meta = False


def build_toolchain():
    """The binary's self-reported toolchain, or an explicit `unknown` record.

    Never returns {}: an empty merge would leave the cell silently unstamped."""
    global _warned_no_build_meta
    if BUILD:
        return BUILD
    if not _warned_no_build_meta:
        _warned_no_build_meta = True
        print("WARNING: the benchmark binary emitted no BUILD_META line; cells will be "
              "stamped compiler=unknown. Rebuild it before trusting this sweep.",
              file=sys.stderr, flush=True)
    return UNKNOWN_TOOLCHAIN

# AC/AC2 (Laplacians.jl) run under julia, not under any binary we build.
JULIA_SOLVERS = ("ac", "ac2")
_JULIA_TOOLCHAIN = {}

def julia_toolchain():
    """The toolchain behind an AC/AC2 cell, asked of the julia that runs them.

    There is no ahead-of-time compiler to record: Laplacians.jl is JIT-compiled
    per process, so what identifies the code generator is julia's version and its
    libLLVM, and the CPU that JIT targets — julia's default `--cpu-target native`
    tunes for the build machine exactly as our own -march=native builds do.
    approxChol.jl is pure julia (no ccall, no threading), so no OpenMP runtime is
    in its solve path. Probed once; unknown if julia cannot be run."""
    if not _JULIA_TOOLCHAIN:
        probe = ('println(string(VERSION)); println(Base.libllvm_version); '
                 'println(Base.JLOptions().cpu_target == C_NULL ? "native" : '
                 'unsafe_string(Base.JLOptions().cpu_target)); println(Sys.CPU_NAME)')
        ver = llvm = target = cpu = "unknown"
        try:
            out = sh(f"julia -e '{probe}'", timeout=300).stdout.split()
            if len(out) >= 4: ver, llvm, target, cpu = out[:4]
        except Exception:
            pass
        _JULIA_TOOLCHAIN.update({
            "compiler": "julia", "compiler_version": f"{ver} (libLLVM {llvm})",
            "openmp_runtime": "none (pure-julia solver path)",
            "arch_flags": f"cpu_target={target} ({cpu})"})
    return _JULIA_TOOLCHAIN

def emit(family, mid, solver, config, status, metrics, extra_meta=None,
         timeout_cap_s=None):
    """One cell. `extra_meta` is whatever the runner learned that the registry
    cannot say — notably WHY a cell is n/a, so an unsupported combination reads as
    unsupported instead of as a hole in the table.

    The toolchain fields go into PROVENANCE, and which ones depends on what ran:
    the binary's own BUILD_META for the in-process solvers, julia's for AC/AC2."""
    meta = dict(OBSERVED.get(mid) or {})
    if extra_meta:
        meta.update(extra_meta)
    prov = {**PROV, **(julia_toolchain() if solver in JULIA_SOLVERS else build_toolchain())}
    return rc.emit_cell(family, mid, solver, config, status, metrics, THREADS, DEVICE, prov,
                        matrix_meta=meta or None,
                        timeout_cap_s=timeout_cap_s if status == "timeout" else None)

def run_cpp(margs, solver, config, reg, family=None, boomeramg_cfg=None, timeout=None, ground=None,
            mid=None):
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
                # BUILD_META is the binary's FIRST stderr line, so even a run
                # killed at the wall cap identifies the toolchain behind the
                # 'timeout' cell it is about to write.
                BUILD.update(rc.parse_build_meta(e.stderr))
                return "timeout", _diag(None, e.output, e.stderr)
        BUILD.update(rc.parse_build_meta(p.stderr))   # what built the binary that just ran
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
        # How the binary read this matrix, straight from its own report.
        meta = rc.parse_matrix_meta(p.stderr)
        if meta and mid:
            OBSERVED[mid] = meta
        return st,m
    st,m = _run([])
    # (The "retry with APXCHOL_GPU_SPTRSV=levelset after an OOM" fallback that used
    # to sit here went with the level-set backend on 2026-08-20. It existed because
    # AUTO used to be cuSPARSE, whose O(nnz) SpSV analysis buffer OOMs the giants;
    # AUTO has been our O(n)-state dataflow kernel since 2026-08-18, so there is
    # nothing to fall back FROM.)
    return st,m

def run_julia(mtx, solver, cls):
    """AC/AC2 (Laplacians.jl) on the DUMPED operator, with its declared class.

    They used to be handed the raw registry .mtx and re-derive a Laplacian from
    it with their own copy of the adjacency reader, so their agreeing with the
    in-process solvers was a property of two code paths happening to match — not
    something the run demonstrated. `--operator` makes the Julia driver read the
    matrix that `--dump-mtx` wrote, i.e. the exact operator every in-process
    solver was given, and solve THAT: approxchol_lap on a singular Laplacian
    (recovering A = -offdiag(L)), approxchol_sddm on a full-rank operator.

    Laplacians.jl needs non-negative edge weights, so an operator with POSITIVE
    off-diagonals is one it genuinely cannot take (parabolic_fem, thermal2). The
    Julia driver says so on stderr and emits its n/a sentinel row (iters/rel_res
    = -1); we lift that sentence into the cell, so the n/a carries its reason
    instead of reading as a hole.

    `cls` is the registry's declared grounding class (rc.class_of), passed as
    --class: the Julia driver used to sniff singular-vs-SDDM off the dump with the
    same global row-sum ratio the C++ benchmark carried, which on the IPM family
    is unattainable (see benchmarks/julia/bench_laplacians.jl's operator_facts).
    It is the class of the DUMP, which is what the registry declares because the
    dump is unregularized (dump_mtx passes no --reg-rel).

    Needs the Julia project instantiated once:
      julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()'
    """
    cmd=(f"taskset -c 0-{THREADS-1} julia --project=benchmarks/julia "
         f"benchmarks/julia/bench_laplacians.jl --operator {mtx} --class {cls} "
         f"--solver {solver} --tol {TOL} --maxiter 500 --csv")
    try: cp = sh(cmd, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "timeout", None, {"na_reason": f"exceeded the {TIMEOUT}s per-cell wall cap"}
    st, m = classify(rc.parse_csv(cp.stdout))
    meta = {}
    if st == "n/a":
        hit = re.search(r"^\[n/a\][^:]*:\s*(.+)$", cp.stderr or "", re.M)
        meta["na_reason"] = (hit.group(1).strip() if hit else
                             "Laplacians.jl reported this operator unsupported "
                             "(no reason line captured)")
    elif st == "failed" and (cp.stderr or "").strip():
        meta["na_reason"] = cp.stderr.strip().splitlines()[-1][:500]
    return st, m, meta

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

_CMG_STATE = {"probed": False, "ok": False, "why": ""}

def run_cmg(mid):
    """CMG (canonical MATLAB CMG in the matlab-deps container), in-process via
    cmg_matlab_runner — resume-safe, and it emits its own cell.

    Availability (MATLAB tree + cmg-solver checkout + container image) is probed
    ONCE per sweep. On a machine without them the sweep says so in one line
    naming exactly what is missing, instead of failing 27 times."""
    if not _CMG_STATE["probed"]:
        _CMG_STATE["ok"], _CMG_STATE["why"] = cmg_matlab_runner.available()
        _CMG_STATE["probed"] = True
        if not _CMG_STATE["ok"]:
            print(f"   {'cmg':24} UNAVAILABLE — no cmg cells this run: {_CMG_STATE['why']}",
                  flush=True)
    if not _CMG_STATE["ok"]:
        return
    try:
        print(f"   {'cmg':24} {cmg_matlab_runner.run_one(mid)}", flush=True)
    except Exception as e:                       # never abort the sweep for one solver
        print(f"   {'cmg':24} ERROR: {e}", flush=True)

def dump_mtx(mid):
    """Write the EXACT operator the in-process solvers solve to a .mtx, cached.

    This is the one seam every external solver goes through, so "every solver
    received the same matrix" is demonstrated by construction rather than
    assumed: the binary assembles the operator per the registry's declared kind
    (L = D - A for a graph, the published matrix for an operator) and writes it
    out. Grids dump their generated Laplacian; files dump whichever of the two
    readings their kind selects.

    NOTE the dumps are large (the giants run to several GB) and DUMP defaults to
    a tmpfs — set APXCHOL_BENCH_DUMP_DIR to a disk-backed path before sweeping
    them. Returns None when the dump failed.
    """
    os.makedirs(DUMP, exist_ok=True)
    p = f"{DUMP}/{mid}.mtx"
    if not os.path.exists(p):
        sh(f"{BIN[DEVICE]} {rc.margs_for(mid)} --dump-mtx {p} --solver none", timeout=TIMEOUT)
    return p if os.path.exists(p) else None

# --- CPU axis solver sets ---
# Solver set: (solver, config)
APX_DEFAULT_CONFIG = rc.APXCHOL_DEFAULT_CONFIG
APX = [("apxchol_v1", APX_DEFAULT_CONFIG),
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
# CMG (canonical MATLAB CMG in the matlab-deps container) likewise runs as part of
# every CPU sweep -- it is a charted competitor, so leaving it to a separate
# command is how its column went empty. --no-cmg skips it; a machine without
# MATLAB/docker says so once and moves on. CPU only: it is serial MATLAB.
RUN_CMG = True
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
APX_GPU = [("apxchol_v1", APX_DEFAULT_CONFIG),
           ("apxchol_v1","luby+tree[vec_pool]"),   # luby/root produce shallow, deterministic
           ("apxchol_v1","root+tree[vec_pool]"),    # factors -> faster + stable GPU SpTRSV than
           ("apxchol_v1","bk+tree[vec_pool]")]      # bg's variable depth; bk is the deep worst case
COMP_GPU = ["hypre_boomeramg_gpu","amgcl_cuda"]

def cell_done(family, mid, solver, config):
    path = rc.cell_path(family, mid, solver, config, THREADS, DEVICE)
    try:
        with open(path) as handle:
            cell = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    st = cell.get("status")
    # --no-cap: a previously CLAMPED 'timeout' cell is NOT terminal -- re-run it
    # uncapped for an honest wall time. complete/oom/failed stay terminal (no point).
    if NO_CAP and st == "timeout": return False
    # Schema-1 timeout cells do not say which cap was actually used. They cannot
    # support an honest lower bound and are deliberately resumed, not trusted.
    if st == "timeout" and rc.timeout_cap(cell) is None:
        return False
    return st in ("complete","not_converged","failed","timeout","oom")

def cell_metrics(family, mid, solver, config):
    """(status, metrics) from an existing cell, or (None, None)."""
    path = rc.cell_path(family, mid, solver, config, THREADS, DEVICE)
    try:
        c = json.load(open(path)); return c.get("status"), c.get("metrics")
    except: return None, None

def step(family, mid, solver, config, runner, label, timeout_cap_s=TIMEOUT):
    """Run one cell. `runner` returns (status, metrics) or, when it has something
    to say about the cell, (status, metrics, extra_meta)."""
    if cell_done(family, mid, solver, config):
        print(f"   {label:24} (skip, done)")
        return cell_metrics(family, mid, solver, config)   # so resume can still read apxchol's time
    res = runner()
    st, m = res[0], res[1]
    emit(family, mid, solver, config, st, m, res[2] if len(res) > 2 else None,
         timeout_cap_s=timeout_cap_s)
    print(f"   {label:24} {st} it={m.get('iters','-') if m else '-'} "
          f"total={m.get('total_s','-') if m else '-'}", flush=True)
    return st, m

def gpu_apx_total(family, mid):
    """GPU time of the declared apxchol default used to derive competitor caps.

    This is intentionally one fixed configuration, not the fastest selector in the
    store. None means the CPU pass must fall back to the same declared default on
    its own device."""
    path = rc.cell_path(family, mid, "apxchol_v1", APX_DEFAULT_CONFIG, 16, "gpu")
    try:
        with open(path) as handle:
            c = json.load(handle)
        if c.get("status") == "complete":
            return c.get("metrics", {}).get("total_s")
    except (OSError, json.JSONDecodeError):
        pass
    return None

def do_matrix(mid, family, source, spec, is2d, n, reg):
    kind = rc.kind_of(mid)         # declared, never guessed; raises if missing
    print(f"[{mid}] (n={n}) device={DEVICE} kind={kind} -> "
          f"{rc.KIND_INTERPRETATION[kind]}", flush=True)
    if PARAC_ONLY:                 # --parac-only: ParAC and nothing else
        run_parac(mid); return
    margs=rc.margs_for(mid)
    # The competitor wall-clock cap is TIMEOUT_MULT x GPU-apxchol's total time -- the
    # declared apxchol default is the universal reference, so "10x of GPU apxchol" is the
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
        for solver,config in APX_GPU:
            _,m = step(family,mid,solver,config, lambda s=solver,c=config: run_cpp(margs,s,c,reg,family,mid=mid), config)
            if config == APX_DEFAULT_CONFIG and m and m.get("total_s"):
                t_apx = m["total_s"]
        to = comp_timeout(t_apx)
        comp = list(COMP_GPU)
        for solver in comp:
            if solver == "hypre_boomeramg_gpu":
                # default (Hypre defaults) + CoarsenCutFactor 'cut' (hub->F-point),
                # charted as two GPU series like the CPU axis -- cut is the fix on the
                # mega-hub social giants, a no-op on grids/IPM.
                step(family,mid,solver,"", lambda s=solver: run_cpp(margs,s,"",reg,family,timeout=to,mid=mid), solver,
                     timeout_cap_s=to)
                step(family,mid,solver,"cut",
                     lambda s=solver: run_cpp(margs,s,"cut",reg,family,boomeramg_cfg="cut",timeout=to,mid=mid), solver+"/cut",
                     timeout_cap_s=to)
            else:
                step(family,mid,solver,"", lambda s=solver: run_cpp(margs,s,"",reg,family,timeout=to,mid=mid), solver,
                     timeout_cap_s=to)
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
    for solver,config in apx_list:
        _,m = step(family,mid,solver,config, lambda s=solver,c=config: run_cpp(margs,s,c,reg,family,mid=mid), config)
        if config == APX_DEFAULT_CONFIG and m and m.get("total_s"):
            t_apx = m["total_s"]
    to = comp_timeout(t_apx)
    for solver in COMP:
        if solver == "hypre_boomeramg":
            # Two series: classical default + CoarsenCutFactor 'cut' (hub->F-point).
            # The cut variant is a no-op on uniform-degree grids/IPM and the fix on
            # mega-hub social graphs (com-Youtube/as-Skitter), so charting both shows
            # the methodology point instead of letting default blow up the timeout.
            step(family,mid,solver,"", lambda s=solver: run_cpp(margs,s,"",reg,family,timeout=to,mid=mid), solver,
                 timeout_cap_s=to)
            step(family,mid,solver,"cut",
                 lambda s=solver: run_cpp(margs,s,"cut",reg,family,boomeramg_cfg="cut",timeout=to,mid=mid), solver+"/cut",
                 timeout_cap_s=to)
        else:
            step(family,mid,solver,"", lambda s=solver: run_cpp(margs,s,"",reg,family,timeout=to,mid=mid), solver,
                 timeout_cap_s=to)
    # AC/AC2 read the DUMPED operator — the same seam every other external solver
    # goes through — so their cells demonstrably solve the same system as the
    # in-process ones instead of re-deriving one from the registry file.
    if JULIA and not all(cell_done(family, mid, s, "") for s in JULIA):
        dumped = dump_mtx(mid)
        if dumped:
            for s in JULIA:
                step(family,mid,s,"",
                     lambda ss=s: run_julia(dumped,ss,rc.class_of(mid)), s)
        else:
            print(f"   {'AC/AC2':24} SKIP (operator dump failed)")
    if RUN_PARAC:      # ParAC graph+physics (resume-safe; the runner skips done cells)
        run_parac(mid)
    if RUN_CMG:        # CMG in the MATLAB container (resume-safe; emits its own cell)
        run_cmg(mid)

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
    ap.add_argument("--no-cmg", action="store_true",
                    help="skip CMG (canonical MATLAB CMG in the matlab-deps container, run "
                         "in-process via cmg_matlab_runner.py). It is skipped automatically "
                         "on --device gpu (serial MATLAB) and on a machine where MATLAB, the "
                         "cmg-solver checkout or the container image is missing.")
    ap.add_argument("--parac-only", action="store_true",
                    help="run ONLY ParAC (skip apxchol / competitors / AC); for filling the "
                         "ParAC giant-matrix cells without touching the rest.")
    ap.add_argument("--no-cap", action="store_true",
                    help="drop the 10x-GPU-apxchol competitor wall cap AND re-run cells "
                         "currently stored as 'timeout' (clamped), so AMGCL/BoomerAMG get "
                         "an honest uncapped wall time. NOCAP_TIMEOUT_S (default 4h) is the "
                         "only outer ceiling. Pair with --only/--families to target matrices.")
    a = ap.parse_args()
    global RUN_PARAC, PARAC_ONLY, NO_CAP, RUN_CMG
    DEVICE = a.device
    fams = {f.strip() for f in a.families.split(",") if f.strip()}
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    if a.no_julia: JULIA = []
    if a.no_parac: RUN_PARAC = False
    if a.parac_only: PARAC_ONLY = True
    if a.no_cap: NO_CAP = True
    # CMG is serial MATLAB: it has no GPU axis, so its cells are device=cpu only.
    if a.no_cmg or DEVICE == "gpu": RUN_CMG = False
    if a.headline_only:
        APX = [("apxchol_v1","bg+tree[vec_pool]"), ("apxchol_v1","luby+tree[vec_pool]"),
               ("apxchol_v1","root+tree[vec_pool]"), ("apxchol_v1","bk+tree[vec_pool]")]
        # NOTE: --headline-only restricts the apxchol CONFIG list only. It no longer
        # disables AC/AC2: leaving their cells empty silently understates coverage.
        # Use --no-julia explicitly when you want the speed (they are slow, and hit
        # the cap or OOM on the social giants -- recorded as timeout/failed cells).
    if a.no_rchol:
        COMP = [c for c in COMP if c not in ("rchol","rchol_par")]
    if DEVICE == "gpu" and not os.path.exists(BIN["gpu"]):
        sys.exit(f"GPU build not found: {BIN['gpu']} (configure with -DAPXCHOL_USE_CUDA=ON)")
    PROV["note"] = f"FAIR {DEVICE} run, singular L, multi-component Dirichlet pin (per-solver grounding)"
    print(f"=== fair sweep: device={DEVICE}, families={sorted(fams)}, "
          f"headline_only={a.headline_only}, store={CELLS} (resume-safe) ===", flush=True)
    print(f"    external solvers: parac={'on' if RUN_PARAC else 'off'} "
          f"ac/ac2={'on' if JULIA else 'off'} cmg={'on' if RUN_CMG else 'off'}", flush=True)
    if not JULIA:
        print("    NOTE: AC/AC2 are OFF -> their cells stay empty. Fill them with "
              "`python3 benchmarks/sweep_fair.py` without --no-julia/--headline-only "
              "(needs: julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()').",
              flush=True)
    if not RUN_CMG and DEVICE == "cpu":
        print("    NOTE: CMG is OFF -> its cells stay empty. Fill them with "
              "`python3 benchmarks/cmg_matlab_runner.py`.", flush=True)
    # Resume-safe: cells with a terminal status are skipped (see cell_done).
    # MATRICES is the canonical, named registry view; do not unpack the backing
    # GRIDS/SS/IPM tuples here, because their positional schemas can evolve.
    for mid, matrix in selected_matrices(fams, only):
        do_matrix(mid, matrix["family"], matrix["source"], matrix["spec"],
                  matrix["is2d"], matrix["n"], None)
    print("FAIR sweep done")

if __name__=="__main__":
    main()

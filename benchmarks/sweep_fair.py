#!/usr/bin/env python3
"""Single fair runner for ALL matrix families (grids + SuiteSparse + IPM), on
any device (--device cpu | gpu | metal). All write the SAME per-cell JSON store
(results/cells/<family>/<mid>__<solver>__<cfg>__t16__<device>.json) and are
RESUME-SAFE: a terminal cell from this revision under the current contract is
skipped; any other cell is re-run and replaced only after its runner finishes
(runner_common.cell_is_current).

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
matrices are already strictly SDDM and likewise run unshifted.

The CPU sweep runs apxchol and the C++ competitors in-process, ParAC through
parac_runner.py, AC/AC2 through benchmarks/julia/bench_laplacians.jl, and
canonical MATLAB CMG through cmg_matlab_runner.py on the exported operator and
RHS. --no-julia, --no-parac and --no-cmg skip those adapters; CMG is also
skipped, with its reason printed once, when MATLAB or CMG is not configured.
--headline-only restricts the apxchol config list; AC/AC2 are dropped only by
an explicit --no-julia.
A solver that cannot take a given matrix emits an explicit `n/a` cell carrying
the reason, never a silent gap.

  # CPU axis (default):
  systemd-inhibit --what=sleep:idle --mode=block python3 benchmarks/sweep_fair.py
  # GPU axis (needs the CUDA build; apxchol uses the GPU-resident PCG):
  python3 benchmarks/sweep_fair.py --device gpu
  # Metal axis (macOS Metal build; apxchol only, no Metal competitors):
  python3 benchmarks/sweep_fair.py --device metal

ParAC runs IN-PROCESS via parac_runner.py (graph+physics on both axes), emitting
into the same store. Shared harness (hardened sh, matrix registry, cell store,
CSV parse, VRAM sidecar) lives in runner_common.py. Schema-2 timeout cells persist
the exact wall-clock cap as `timeout_cap_s`; the charting pipeline never reconstructs
that lower bound from later timings.
"""
import argparse, json, os, re, shlex, shutil, subprocess, sys, time

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
WARMUP = 0
THREADS = 16
# 30 min/cell: the >=1e8-nnz giants (com-Orkut apxchol ~170s/rep x3, BoomerAMG
# mega-hub setup, RCHOL fill blowup) can legitimately exceed the old 1200s.
TIMEOUT = 1800         # per-solve competitor cap when no apxchol reference exists
CAP_FACTOR = 20        # per-solve cap = CAP_FACTOR x the best same-device apxchol total
CAP_FLOOR_S = 30       # ...never below this, so noise on tiny matrices does not cut a run
CAP_STARTUP_S = 300    # per-process allowance for startup and input parsing, outside the cap
# Per-benchmark host-memory cap (GB), CPU only. A runaway solver -- e.g. serial RCHOL's
# natural-order fill blowup on the 237M-nnz com-Orkut hit 110GB RSS and OOM-killed the
# desktop (2026-06-12) -- gets an address-space ceiling via `ulimit -v`, so it dies with
# bad_alloc (recorded as 'oom') instead of exhausting system RAM. Legit com-Orkut apxchol
# historically peaked ~62GB (retired root+tree[vec]); 100GB clears that
# stress case comfortably and still leaves ~24GB
# headroom on the 124GB box. NOT applied on GPU (CUDA reserves large host VM; breaks runs).
MEM_CAP_GB = float(os.environ.get("APXCHOL_BENCH_MEM_CAP_GB", "100"))
DEVICE = "cpu"   # set in main() from --device
# CPU PCG caps iters at 500; the GPU axes allow more (cheap per-iter), matching
# the retired gpu_chart_sweep.sh (2000, or 1000 for IPM).
MAXITER = {"cpu": 500, "gpu": 2000, "metal": 2000, "gpu_ipm": 1000}
# The benchmark's --v1-backend for each device; a Metal build's gpu route is Metal.
V1_BACKEND = {"cpu": "cpu", "gpu": "gpu", "metal": "gpu"}

PROV = {"boost":"on","boost_expected":"on","git_sha":rc.git_sha(),
        "source_id": os.environ.get("APXCHOL_BENCH_SOURCE_ID", ""),
        "note":"FAIR run","repeat":REPS,
        "tier":"broad","timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}

def classify(m):
    return rc.classify(m, TOL)


def julia_preflight():
    """Return (ready, reason) for the AC/AC2 runtime before a sweep starts.

    Manifest.toml is intentionally gitignored, so an isolated worktree can have
    a fully populated depot yet no mapping from Project.toml to those installed
    package trees. That exact staging error otherwise turns every AC/AC2 cell
    into the same misleading numerical ``failed`` outcome.
    """
    project = os.path.join(rc.ROOT, "benchmarks", "julia")
    manifest = os.path.join(project, "Manifest.toml")
    if not os.path.isfile(manifest):
        return False, (f"missing {manifest}; instantiate this exact worktree with "
                       f"`julia --project={project} -e 'using Pkg; Pkg.instantiate()'`")
    julia = shutil.which("julia")
    if not julia:
        return False, "julia is not on PATH"
    cmd = (f"{shlex.quote(julia)} --startup-file=no --project={shlex.quote(project)} "
           f"-e {shlex.quote('using Laplacians')}")
    try:
        cp = sh(cmd, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "`using Laplacians` exceeded the 300s environment-preflight cap"
    if cp.returncode != 0:
        detail = " | ".join(
            line.strip() for line in (cp.stderr or cp.stdout or "").splitlines()[-8:]
        ) or "no output"
        return False, f"`using Laplacians` failed (rc={cp.returncode}): {detail}"
    return True, "Laplacians load succeeded"


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
    if solver in JULIA_SOLVERS:
        runtime = julia_toolchain()
    elif solver == "cmg":
        runtime = cmg_matlab_runner.provenance()
    else:
        runtime = {**rc.benchmark_openmp_provenance(THREADS), **build_toolchain()}
    prov = {**PROV, **runtime}
    return rc.emit_cell(family, mid, solver, config, status, metrics, THREADS, DEVICE, prov,
                        matrix_meta=meta or None,
                        timeout_cap_s=timeout_cap_s if status in ("timeout", "exceeded_cap") else None)

def device_init(stderr):
    """The process-wide CUDA or Metal initialization the binary timed outside
    every solver's setup, as cell fields."""
    out = {}
    for key, parse in (("cuda_init_s", rc.parse_cuda_init), ("metal_init_s", rc.parse_metal_init)):
        value = parse(stderr)
        if value is not None:
            out[key] = value
    return out

def run_cpp(margs, solver, config, reg, family=None, boomeramg_cfg=None, timeout=None, ground=None,
            mid=None):
    cfg = (f"--v1-configs '{config}' --v1-backend {V1_BACKEND[DEVICE]}"
           if solver=="apxchol_v1" else "")
    # AMGCL de-singularization series: ground=coarse charts the relaxed-coarse cell
    # (--decompose stays auto, so whole+coarse on connected / split+coarse on
    # disconnected). The headline AMGCL series passes no --ground (auto = pin).
    gflag = f"--ground {ground}" if ground else ""
    regflag = f"--reg-rel {reg}" if reg else ""   # IPM is native SDDM: no shift
    mi = MAXITER["gpu_ipm"] if (DEVICE!="cpu" and family=="ipm") else MAXITER[DEVICE]
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
        run_env = rc.benchmark_openmp_env(THREADS)
        for assignment in envs:
            key, value = assignment.split("=", 1)
            run_env[key] = value
        inner=(f"{rc.taskset_prefix(THREADS)} {BIN[DEVICE]} {margs} --solver {solver} {cfg} "
               f"{gflag} {regflag} --threads {THREADS} --tol {TOL} --maxiter {mi} --repeat {REPS} --warmup {WARMUP} --csv")
        # /usr/bin/time: peak RSS on stderr (rc.peak_rss_prefix); the max over
        # all reps (setup-phase peak dominates). solve_rss_mb (held during solve) comes
        # from the binary's CSV; peak - solve_rss = the setup transient.
        cmd=f"{rc.peak_rss_prefix()} {inner}"
        # CPU only: rc.sh's mem_cap ceils the address space so a memory-bomb solver
        # dies with bad_alloc (-> 'oom') instead of OOM-killing the box.
        cap = MEM_CAP_GB if DEVICE == "cpu" else None
        # Keep peak-memory diagnostics separate: nvidia-smi polling perturbs GH200 setup.
        try: p=sh(cmd, timeout=timeout or TIMEOUT, env=run_env, mem_cap_gb=cap)
        except subprocess.TimeoutExpired as e:
            # BUILD_META is the binary's FIRST stderr line, so even a run
            # killed at the wall cap identifies the toolchain behind the
            # 'timeout' cell it is about to write.
            BUILD.update(rc.parse_build_meta(e.stderr))
            diag = _diag(None, e.output, e.stderr)
            diag.update(device_init(e.stderr))
            diag.update(rc.parse_gpu_memory(e.stderr))
            return "timeout", diag
        BUILD.update(rc.parse_build_meta(p.stderr))   # what built the binary that just ran
        st,m = classify(rc.parse_csv(p.stdout))
        if st in {"complete", "not_converged"} and m.get("stop_contract") != "original-v1":
            st = "failed"
            m["stopping_failure"] = "binary lacks original-v1 stopping/timing contract"
        if solver == "apxchol_v1" and st in {"complete", "not_converged"}:
            route_error = rc.v1_route_error(m, p.stderr, DEVICE, REPS, WARMUP)
            if route_error:
                st = "failed"
                m["route_failure"] = route_error
        if m is None:
            # No CSV row = crash. Distinguish allocation failures across all
            # solvers, including third-party CUDA libraries, from other errors.
            blob=((p.stdout or "")+(p.stderr or "")).lower()
            if (p.returncode in (-9,137) or any(s in blob for s in
                    ("out of memory","bad_alloc","cudaerrormemoryalloc","cuda_error_out_of_memory",
                     "cublas_status_alloc_failed","cusparse_status_alloc_failed","cudamalloc",
                     "alloc_failed","cannot allocate","cuda oom","gpu out of memory"))):
                st="oom"
            m = _diag(p.returncode, p.stdout, p.stderr)   # failed/oom cells keep the evidence
        if m is not None:
            m["warmup_repeats"] = WARMUP
            m["repeat_receipts"] = [line for line in p.stderr.splitlines() if line.startswith("BENCH_REPEAT ")]
            m.update(device_init(p.stderr))
            m.update(rc.parse_gpu_memory(p.stderr))
            rss = rc.parse_peak_rss_mb(p.stderr)
            if rss is not None:
                m["max_rss_mb"] = rss
            m["gpu_memory_polling"] = False
        # How the binary read this matrix, straight from its own report.
        meta = rc.parse_matrix_meta(p.stderr)
        if meta and mid:
            OBSERVED[mid] = meta
        return st,m
    st,m = _run([])
    return st,m

def run_julia(mtx, solver, cls, timeout=TIMEOUT):
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
    project = os.path.join(rc.ROOT, "benchmarks", "julia")
    driver = os.path.join(project, "bench_laplacians.jl")
    cmd=(f"{rc.taskset_prefix(THREADS)} julia --project={shlex.quote(project)} "
         f"{shlex.quote(driver)} --operator {shlex.quote(mtx)} --class {cls} "
         f"--solver {solver} --tol {TOL} --maxiter 500 --csv")
    runs = []
    deadline = time.monotonic() + timeout
    for _ in range(REPS):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return "timeout", None, {
                "na_reason": f"the {REPS}-repetition cell exceeded its {timeout}s wall cap"
            }
        try:
            cp = sh(cmd, timeout=remaining)
        except subprocess.TimeoutExpired:
            return "timeout", None, {
                "na_reason": f"the {REPS}-repetition cell exceeded its {timeout}s wall cap"
            }
        st, metrics = classify(rc.parse_csv(cp.stdout))
        if st in {"complete", "not_converged"} and metrics.get("stop_contract") != "original-v1":
            st = "failed"
            metrics["stopping_failure"] = "Julia driver lacks original-v1 stopping/timing contract"
        meta = {}
        if st == "n/a":
            hit = re.search(r"^\[n/a\][^:]*:\s*(.+)$", cp.stderr or "", re.M)
            meta["na_reason"] = (hit.group(1).strip() if hit else
                                 "Laplacians.jl reported this operator unsupported "
                                 "(no reason line captured)")
            return st, metrics, meta
        if st == "failed":
            if (cp.stderr or "").strip():
                meta["na_reason"] = cp.stderr.strip().splitlines()[-1][:500]
            return st, metrics, meta
        runs.append((st, metrics, meta))

    # Match the C++ benchmark's full-run median: keep one internally consistent
    # record rather than taking a field-wise median that never occurred.
    runs.sort(key=lambda run: run[1]["total_s"])
    return runs[len(runs) // 2]

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
        if DEVICE == "gpu":
            stamped = parac_runner.record_gpu_failure(
                mid, "failed", f"unhandled ParAC runner error: {type(e).__name__}: {e}")
            print(f"   {label:24} {stamped}")

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
    return rc.dump_matrix(mid, DUMP, BIN[DEVICE], THREADS, TIMEOUT)

def dump_rhs(mid):
    """Write the RHS the in-process solvers solve, cached next to dump_mtx."""
    return rc.dump_matrix(mid, DUMP, BIN[DEVICE], THREADS, TIMEOUT, rhs=True)

# --- CPU axis solver sets ---
# Solver set: (solver, config)
APX_DEFAULT_CONFIG = rc.APXCHOL_DEFAULT_CONFIG
APX = [("apxchol_v1", APX_DEFAULT_CONFIG),
       ("apxchol_v1", "bk+tree[vec_pool_aos]"),
       ("apxchol_v1", "greedy+tree[vec_pool_aos]")]
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
# Canonical MATLAB CMG on the shared A and b (CPU only; skip via --no-cmg).
RUN_CMG = True
# --no-cap: drop the 10x-GPU-apxchol competitor wall cap and RE-RUN cells currently
# stored as 'timeout' (clamped), so AMGCL/BoomerAMG get an HONEST uncapped wall time on
# the matrices where they previously hit the cap. NOCAP_TIMEOUT is still a sane outer
# ceiling (a true hang can't run forever); override via NOCAP_TIMEOUT_S.
NO_CAP = False
NOCAP_TIMEOUT = int(os.environ.get("NOCAP_TIMEOUT_S", str(4 * 3600)))

# --- GPU axis solver sets ---
# The GPU route owns both setup and solve. Host-only storage/selector ablations
# remain on the CPU axis; historical mixed-route cells retain their own sources.
APX_GPU = [("apxchol_v1", APX_DEFAULT_CONFIG)]
COMP_GPU = ["hypre_boomeramg_gpu","amgcl_cuda"]

# --- Metal axis: apxchol only (no competitor has a Metal backend) ---
APX_METAL = [("apxchol_v1", APX_DEFAULT_CONFIG)]


def planned_cell_count(matrices, device):
    """Return the exact cell denominator for the selected fair-sweep axis."""
    matrices = list(matrices)
    if PARAC_ONLY:
        return 2 * len(matrices)
    if device == "metal":
        return len(APX_METAL) * len(matrices)
    if device == "gpu":
        per_matrix = len(APX_GPU) + len(COMP_GPU)
        per_matrix += int("hypre_boomeramg_gpu" in COMP_GPU)  # cut variant
        per_matrix += 2 if RUN_PARAC else 0
        return per_matrix * len(matrices)
    if device != "cpu":
        raise ValueError(f"unknown device: {device}")
    total = 0
    for mid, _matrix in matrices:
        total += len(APX) + len(COMP)
        total += int("hypre_boomeramg" in COMP)  # cut variant
        total += len(JULIA)
        total += 2 if RUN_PARAC else 0
        total += 1 if RUN_CMG else 0
    return total

def cell_done(family, mid, solver, config):
    return rc.cell_done(
        family, mid, solver, config, THREADS, DEVICE,
        terminal=rc.DEFAULT_TERMINAL, rerun_timeouts=NO_CAP,
    )


def cell_metrics(family, mid, solver, config):
    """(status, metrics) from an existing cell, or (None, None)."""
    cell = rc.stored_cell(family, mid, solver, config, THREADS, DEVICE)
    return ((cell.get("status"), cell.get("metrics"))
            if cell is not None else (None, None))

def step(family, mid, solver, config, runner, label, timeout_cap_s=TIMEOUT):
    """Run one cell. `runner` returns (status, metrics) or, when it has something
    to say about the cell, (status, metrics, extra_meta). Existing stale data is
    left untouched until that return, then replaced by ``emit`` below."""
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

def solve_cap(apx_totals):
    """Per-solve competitor cap from this device's complete apxchol totals.

    The capped quantity is a run's setup + solve as its solver reports them;
    process startup, input parsing and CUDA/Julia initialization are excluded.
    Without a reference the cap is TIMEOUT."""
    totals = [t for t in apx_totals if isinstance(t, (int, float)) and t > 0]
    return max(CAP_FACTOR * min(totals), CAP_FLOOR_S) if totals else float(TIMEOUT)


def _repeat_totals(m):
    """Every measured repetition's setup + solve total, warmups included."""
    totals = [m["total_s"]] if isinstance(m.get("total_s"), (int, float)) else []
    for line in m.get("repeat_receipts") or []:
        hit = re.search(r"\btotal_s=([0-9.eE+-]+)", line)
        if hit:
            totals.append(float(hit.group(1)))
    return totals


def capped(result, cap):
    """Report a run over the per-solve cap as `exceeded_cap`, never as a time.

    A process killed at its cap-derived deadline is also exceeded_cap. The
    measured timings stay in `capped_timings` as raw evidence only."""
    st, m = result[0], dict(result[1] or {})
    extra = result[2] if len(result) > 2 else None
    over = st == "timeout" or (st in {"complete", "not_converged"}
                               and any(t > cap for t in _repeat_totals(m)))
    if over:
        timing = {k: m.pop(k) for k in ("setup_s", "solve_s", "total_s", "us_per_nnz") if k in m}
        if timing:
            m["capped_timings"] = timing
        st = "exceeded_cap"
    m.update(cap_s=cap, cap_factor=CAP_FACTOR, cap_floor_s=CAP_FLOOR_S)
    return (st, m, extra) if extra is not None else (st, m)

def do_matrix(mid, family, source, spec, is2d, n, reg):
    kind = rc.kind_of(mid)         # declared, never guessed; raises if missing
    print(f"[{mid}] (n={n}) device={DEVICE} kind={kind} -> "
          f"{rc.KIND_INTERPRETATION[kind]}", flush=True)
    if PARAC_ONLY:                 # --parac-only: ParAC and nothing else
        run_parac(mid); return
    margs=rc.margs_for(mid)
    # Competitor caps are relative to apxchol on the same matrix and device: a run
    # whose setup + solve exceeds CAP_FACTOR x the best apxchol total is recorded
    # as exceeded_cap (a terminal status), never as a time. Each process is killed
    # once all its repetitions could have used the cap plus a startup allowance.
    def process_timeout(cap, runs):
        return int(runs * cap + CAP_STARTUP_S) if not NO_CAP else NOCAP_TIMEOUT
    if DEVICE == "metal":
        for solver,config in APX_METAL:
            step(family,mid,solver,config, lambda s=solver,c=config: run_cpp(margs,s,c,reg,family,mid=mid), config)
        return
    if DEVICE == "gpu":
        apx_totals = []
        for solver,config in APX_GPU:
            st,m = step(family,mid,solver,config, lambda s=solver,c=config: run_cpp(margs,s,c,reg,family,mid=mid), config)
            if st == "complete" and m:
                apx_totals.append(m.get("total_s"))
        cap = solve_cap(apx_totals)
        to = process_timeout(cap, WARMUP + REPS)
        comp = list(COMP_GPU)
        for solver in comp:
            if solver == "hypre_boomeramg_gpu":
                # default (Hypre defaults) + CoarsenCutFactor 'cut' (hub->F-point),
                # charted as two GPU series like the CPU axis -- cut is the fix on the
                # mega-hub social giants, a no-op on grids/IPM.
                step(family,mid,solver,"", lambda s=solver: capped(run_cpp(margs,s,"",reg,family,timeout=to,mid=mid), cap), solver,
                     timeout_cap_s=cap)
                step(family,mid,solver,"cut",
                     lambda s=solver: capped(run_cpp(margs,s,"cut",reg,family,boomeramg_cfg="cut",timeout=to,mid=mid), cap), solver+"/cut",
                     timeout_cap_s=cap)
            else:
                step(family,mid,solver,"", lambda s=solver: capped(run_cpp(margs,s,"",reg,family,timeout=to,mid=mid), cap), solver,
                     timeout_cap_s=cap)
        if RUN_PARAC:
            run_parac(mid)
        return  # no Julia on the GPU axis
    apx_totals = []
    for solver,config in APX:
        st,m = step(family,mid,solver,config, lambda s=solver,c=config: run_cpp(margs,s,c,reg,family,mid=mid), config)
        if st == "complete" and m:
            apx_totals.append(m.get("total_s"))
    cap = solve_cap(apx_totals)
    to = process_timeout(cap, WARMUP + REPS)
    for solver in COMP:
        if solver == "hypre_boomeramg":
            # Two series: classical default + CoarsenCutFactor 'cut' (hub->F-point).
            # The cut variant is a no-op on uniform-degree grids/IPM and the fix on
            # mega-hub social graphs (com-Youtube/as-Skitter), so charting both shows
            # the methodology point instead of letting default blow up the timeout.
            step(family,mid,solver,"", lambda s=solver: capped(run_cpp(margs,s,"",reg,family,timeout=to,mid=mid), cap), solver,
                 timeout_cap_s=cap)
            step(family,mid,solver,"cut",
                 lambda s=solver: capped(run_cpp(margs,s,"cut",reg,family,boomeramg_cfg="cut",timeout=to,mid=mid), cap), solver+"/cut",
                 timeout_cap_s=cap)
        else:
            step(family,mid,solver,"", lambda s=solver: capped(run_cpp(margs,s,"",reg,family,timeout=to,mid=mid), cap), solver,
                 timeout_cap_s=cap)
    # AC/AC2 and CMG read the DUMPED operator — the same seam every other external
    # solver goes through — so their cells demonstrably solve the same system as
    # the in-process ones instead of re-deriving one from the registry file.
    need_julia = JULIA and not all(cell_done(family, mid, s, "") for s in JULIA)
    need_cmg = RUN_CMG and not cell_done(family, mid, "cmg", "")
    dumped = dump_mtx(mid) if need_julia or need_cmg else None
    if need_julia:
        if dumped:
            for s in JULIA:
                step(family,mid,s,"",
                     lambda ss=s: capped(run_julia(
                         dumped, ss, rc.class_of(mid),
                         timeout=process_timeout(cap + CAP_STARTUP_S, REPS)), cap),
                     s, timeout_cap_s=cap)
        else:
            print(f"   {'AC/AC2':24} SKIP (operator dump failed)")
    if RUN_PARAC:      # ParAC graph+physics (resume-safe; the runner skips done cells)
        run_parac(mid)
    if need_cmg:
        # CMG also needs the RHS: it solves the exported A and b in one MATLAB
        # process (1 warmup + REPS retained repetitions).
        rhs = dump_rhs(mid) if dumped else None
        if rhs:
            step(family,mid,"cmg","",
                 lambda: capped(cmg_matlab_runner.run(
                     dumped, rhs, timeout=process_timeout(cap, 1 + REPS),
                     repeats=REPS, tol=TOL, maxiter=MAXITER["cpu"]), cap),
                 "cmg", timeout_cap_s=cap)
        else:
            print(f"   {'cmg':24} SKIP (operator or RHS dump failed)")
    elif RUN_CMG:
        print(f"   {'cmg':24} (skip, done)")

def main():
    global DEVICE, APX, JULIA, COMP, THREADS, REPS, CELLS, WARMUP
    ap = argparse.ArgumentParser(
        description="Run or resume the fair benchmark sweep; reusable terminal cells "
                    "skip, while stale terminal cells are replaced after their rerun.")
    ap.add_argument("--device", choices=list(rc.DEVICES), default="cpu",
                    help="cpu (build/), gpu (build-cuda/, apxchol GPU-resident PCG) "
                         "or metal (build-metal/, apxchol only)")
    ap.add_argument("--threads", type=int, default=THREADS,
                    help="threads and recorded cell thread count (default: 16)")
    ap.add_argument("--repeat", type=int, default=REPS,
                    help="repetitions inside each benchmark cell (default: 3)")
    ap.add_argument("--warmup", type=int, default=WARMUP, help="explicit C++ solver warmups before retained repeats")
    ap.add_argument("--store", default=CELLS,
                    help="cell-store directory (default: results/cells)")
    ap.add_argument("--no-rchol", action="store_true",
                    help="drop RCHOL / pRCHOL from the competitor set. They blow up (high "
                         "fill, 489 iters / non-convergent) on the large social graphs, so "
                         "on those ParAC is the Cholesky-family competitor we keep.")
    ap.add_argument("--families", default="grids,suitesparse,ipm",
                    help="comma-separated families to run (including manifest families)")
    ap.add_argument("--matrix-manifest", help="hash-verified additional matrix registry JSON")
    ap.add_argument("--only", default="",
                    help="comma-separated registered matrix ids to run; "
                         "empty = all. Use to refresh smaller matrices first.")
    ap.add_argument("--no-julia", action="store_true",
                    help="skip the Julia AC/AC2 references (slow; unaffected "
                         "by the C++ de-singularization changes).")
    ap.add_argument("--no-parac", action="store_true",
                    help="skip ParAC (run in-process via parac_runner.py).")
    ap.add_argument("--no-cmg", action="store_true",
                    help="skip canonical MATLAB CMG (needs APXCHOL_MATLAB_ROOT and "
                         "APXCHOL_CMG_SOLVER; see benchmarks/README.md).")
    ap.add_argument("--parac-only", action="store_true",
                    help="run ONLY ParAC (skip apxchol / competitors / AC); for filling the "
                         "ParAC giant-matrix cells without touching the rest.")
    ap.add_argument("--no-cap", action="store_true",
                    help="drop the 10x-GPU-apxchol competitor wall cap AND re-run cells "
                         "currently stored as 'timeout' (clamped), so AMGCL/BoomerAMG get "
                         "an honest uncapped wall time. NOCAP_TIMEOUT_S (default 4h) is the "
                         "only outer ceiling. Pair with --only/--families to target matrices.")
    a = ap.parse_args()
    if a.matrix_manifest:
        rc.load_matrix_manifest(a.matrix_manifest)
    global RUN_PARAC, PARAC_ONLY, NO_CAP, RUN_CMG
    DEVICE = a.device
    if a.threads < 1 or a.repeat < 1:
        ap.error("--threads and --repeat must be positive")
    if a.warmup < 0: ap.error("--warmup must be nonnegative")
    WARMUP = a.warmup
    PROV["warmup"] = WARMUP
    THREADS, REPS, CELLS = a.threads, a.repeat, os.path.abspath(a.store)
    rc.CELLS = CELLS
    parac_runner.THREADS = THREADS
    parac_runner.REPS = REPS
    parac_runner.PROV_CPU["repeat"] = REPS
    parac_runner.PROV_GPU["repeat"] = REPS
    PROV["repeat"] = REPS
    fams = {f.strip() for f in a.families.split(",") if f.strip()}
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    if a.no_julia: JULIA = []
    if a.no_parac: RUN_PARAC = False
    if a.parac_only: PARAC_ONLY = True
    if a.no_cap: NO_CAP = True
    # CMG is serial MATLAB: it has no GPU axis, so its cells are device=cpu only.
    if a.no_cmg or DEVICE != "cpu": RUN_CMG = False
    if DEVICE == "metal":
        if a.parac_only:
            ap.error("--parac-only has no Metal axis")
        RUN_PARAC = False   # ParAC has no Metal driver
    if RUN_CMG:
        cmg_ok, cmg_reason = cmg_matlab_runner.available()
        if not cmg_ok:
            print(f"    NOTE: CMG is unavailable, no CMG cells are planned: {cmg_reason}",
                  flush=True)
            RUN_CMG = False
    if DEVICE == "cpu" and JULIA:
        julia_ok, julia_reason = julia_preflight()
        if not julia_ok:
            sys.exit(f"AC/AC2 environment preflight failed: {julia_reason}. "
                     "Use --no-julia only if omitting those planned cells is intentional.")
    if a.no_rchol:
        COMP = [c for c in COMP if c not in ("rchol","rchol_par")]
    if DEVICE == "gpu" and not os.path.exists(BIN["gpu"]):
        sys.exit(f"GPU build not found: {BIN['gpu']} (configure with -DAPXCHOL_DEVICE=CUDA)")
    if DEVICE == "metal" and not os.path.exists(BIN["metal"]):
        sys.exit(f"Metal build not found: {BIN['metal']} (configure with -DAPXCHOL_DEVICE=METAL)")
    PROV["note"] = f"FAIR {DEVICE} run, singular L, multi-component Dirichlet pin (per-solver grounding)"
    print(f"=== fair sweep: device={DEVICE}, families={sorted(fams)}, "
          f"store={CELLS} "
          f"(stale-aware resume) ===", flush=True)
    julia_on = bool(JULIA) and DEVICE == "cpu"   # Julia has only a CPU axis
    print(f"    external solvers: parac={'on' if RUN_PARAC else 'off'} "
          f"ac/ac2={'on' if julia_on else 'off'} cmg={'on' if RUN_CMG else 'off'}", flush=True)
    if DEVICE == "cpu" and not JULIA:
        print("    NOTE: AC/AC2 are OFF -> their cells stay empty. Fill them with "
              "`python3 benchmarks/sweep_fair.py` without --no-julia/--headline-only "
              "(needs: julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()').",
              flush=True)
    if a.no_cmg and DEVICE == "cpu":
        print("    NOTE: CMG is OFF; no CMG cells are planned.", flush=True)
    # Resume-safe: reusable terminal cells skip; stale terminal cells rerun while
    # their old JSON stays in place until the replacement runner has finished.
    # MATRICES is the canonical, named registry view; do not unpack the backing
    # GRIDS/SS/IPM tuples here, because their positional schemas can evolve.
    selected = list(selected_matrices(fams, only))
    print(f"    planned denominator: {planned_cell_count(selected, DEVICE)} cells "
          f"across {len(selected)} matrices", flush=True)
    for mid, matrix in selected:
        do_matrix(mid, matrix["family"], matrix["source"], matrix["spec"],
                  matrix["is2d"], matrix["n"], None)
    print("FAIR sweep done")

if __name__=="__main__":
    main()

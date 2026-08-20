#!/usr/bin/env python3
"""Shared harness for the benchmark runners.

One canonical copy of the pieces that were duplicated (with drift) across
sweep_fair.py / parac_runner.py / thread_scaling.py / fill_pass.py /
level_stats.py:

  - sh()           hardened subprocess harness: process-group kill on timeout
                   (the orphan-on-timeout bug fix, 2026-06-11), optional env and
                   per-call `ulimit -v` memory cap.
  - matrix registry  GRIDS/SS/IPM + matrix_args()/margs_for() — the single list
                   the per-runner MATS were drifting copies of.
  - cell store     cell_path/emit_cell/cell_status/cell_done with the terminal
                   status set as a PARAMETER (the per-runner sets deliberately
                   differ: parac_runner retries failed/timeout, sweep_fair doesn't).
  - parse_csv/classify  the benchmark-binary CSV row -> metrics dict + status.
  - VramSampler    nvidia-smi peak-VRAM sidecar thread (GPU axis), parameterized
                   by the compute-process name to match.
  - ParAC constants + the AMD-reorder cache path helper.

All runners (sweep_fair, thread_scaling, fill_pass, level_stats,
selector_levels) import these helpers rather than carrying their own copies.
"""
import json, os, re, signal, subprocess, threading, time
from pathlib import Path

# Repo root, derived from this file's location (benchmarks/runner_common.py).
ROOT = str(Path(__file__).resolve().parents[1])
BIN = {"cpu": f"{ROOT}/benchmarks/build/benchmark",
       "gpu": f"{ROOT}/benchmarks/build-cuda/benchmark"}
CELLS = f"{ROOT}/results/cells"


def require_path(value, env_var, const, what, must_exist=True):
    """Return the configured path for an external tool, or raise a clear error.

    External checkouts (the ParAC drivers, MATLAB, ...) live outside this repo and
    differ per machine, so they are configured through environment variables or a
    gitignored benchmarks/paths_local.py — never hardcoded here.
    """
    if not value:
        raise FileNotFoundError(
            f"{what} is not configured: set ${env_var}, or define {const} in "
            f"benchmarks/paths_local.py (gitignored — see benchmarks/README.md).")
    if must_exist and not os.path.exists(value):
        raise FileNotFoundError(
            f"{what} not found at {value} (from ${env_var} / paths_local.{const}).")
    return value


# ── hardened shell harness ──────────────────────────────────────────────────────
def sh(cmd, timeout=1800, env=None, mem_cap_gb=None):
    """Run `cmd` through the shell in its OWN SESSION and SIGKILL the entire
    process group on timeout.

    CRITICAL: subprocess.run(shell=True, timeout=...) only SIGKILLs the direct
    child (the /bin/sh wrapper) — NOT its grandchildren. The real worker is a
    grandchild (sh -> /usr/bin/time -> taskset -> benchmark -> N OpenMP threads),
    so a timed-out solver kept running, oversubscribing the cores for every
    subsequent cell (the fake-competitor-regression cascade of 2026-06-11).

    mem_cap_gb > 0 prepends `ulimit -v` so a runaway solver dies with bad_alloc
    instead of OOM-killing the desktop (CPU only — CUDA reserves large host VM).
    Raises subprocess.TimeoutExpired (with captured output) on timeout.
    """
    if mem_cap_gb and mem_cap_gb > 0:
        cmd = f"ulimit -v {int(mem_cap_gb * 1024 * 1024)}; {cmd}"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError: pass
        try: out, err = p.communicate(timeout=10)
        except subprocess.TimeoutExpired: out, err = "", ""
        raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


# ── provenance ──────────────────────────────────────────────────────────────────
def git_sha():
    return sh(f"git -C {ROOT} rev-parse HEAD", timeout=30).stdout.strip()

def boost_state():
    try:
        return "off" if open("/sys/devices/system/cpu/cpufreq/boost").read().strip() == "0" else "on"
    except Exception:
        return "unknown"

def make_prov(note, **extra):
    p = {"note": note, "git_sha": git_sha(), "boost": boost_state(),
         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p.update(extra)
    return p


# ── matrix registry ─────────────────────────────────────────────────────────────
# Every entry DECLARES its kind. Two things can live in a .mtx file and they
# define two different linear systems:
#
#   kind="graph"     the file is an adjacency / pattern matrix. The system it
#                    defines is the graph Laplacian L = D - A, assembled from
#                    |value| (unit weights for a `pattern` file). That IS the
#                    system a graph file defines, so building it is not a
#                    deviation from the published data.
#   kind="operator"  the file is an already-assembled Laplacian / SDDM operator.
#                    The system it defines is the matrix itself, and we solve it
#                    exactly as published, diagonal included.
#
# The declaration is EXPLICIT and never auto-detected. A heuristic cannot tell
# the two apart: kron_g500-logn16 stores integer values that are edge weights,
# apache2 stores values that are an assembled operator, and both are "a
# symmetric valued .mtx". Letting the benchmark guess would let it silently
# change which problem the whole suite reports on. Anything undeclared raises
# (see kind_of) rather than falling back to a default.
#
# `source` is a separate axis: where the matrix comes from (a generated grid vs
# a file), not how its values are to be read.
#
# Kinds below are grounded in the files themselves (census 2026-08-20, stored
# diagonal vs sum |off-diagonal| per row):
#   operator — a stored, strictly positive diagonal on every row:
#     ecology1       max rel. deviation 0        (published as an exact Laplacian)
#     parabolic_fem  6.4e-6                      (Laplacian to rounding)
#     apache2        10.4% worst row             (SDDM)
#     thermal2       143% worst row, mean 4.1e-4 (SDDM)
#     G3_circuit     16876% worst row, mean 161% (SDDM, large diagonal excess)
#     iter00x0       IPM normal equations, positive diagonal on every row
#   graph — no usable diagonal at all (`pattern` files store none; kron_g500
#     stores one on 327 of 65536 rows, i.e. self-loops, not an operator).
# ParAC's own benchmark splits its matrices exactly this way ("physics matrices:
# used as-is" vs "graph matrices: build Laplacian from adjacency"), and four of
# our five operator files are on its physics list.
#
# (mid, family, source, spec, is2d) for generated grids; (mid, path, n, kind) for .mtx.
GRIDS = [("grid_500", "grids", "grid", 500, True), ("grid_1000", "grids", "grid", 1000, True),
         ("grid_2000", "grids", "grid", 2000, True), ("grid_3000", "grids", "grid", 3000, True),
         ("grid3d_100", "grids", "grid3d", 100, False), ("grid3d_150", "grids", "grid3d", 150, False),
         ("grid3d_200", "grids", "grid3d", 200, False),   # ~5.6e7 nnz
         ("grid_4000", "grids", "grid", 4000, True),      # ~8.0e7 nnz
         ("grid3d_250", "grids", "grid3d", 250, False),   # ~1.1e8 nnz
         ("grid_5000", "grids", "grid", 5000, True)]      # ~1.25e8 nnz
SS = [("parabolic_fem", "data/matrices/parabolic_fem.mtx", 525825, "operator"),
      ("apache2", "data/matrices/apache2.mtx", 715176, "operator"),
      ("ecology1", "data/matrices/ecology1.mtx", 1000000, "operator"),
      ("G3_circuit", "data/matrices/G3_circuit.mtx", 1585478, "operator"),
      ("thermal2", "data/matrices/thermal2.mtx", 1228045, "operator"),
      ("com-Amazon", "data/matrices/com-Amazon.mtx", 334863, "graph"),
      ("coAuthorsDBLP", "data/matrices/coAuthorsDBLP.mtx", 299067, "graph"),
      ("kron_g500-logn16", "data/matrices/kron_g500-logn16.mtx", 65536, "graph"),
      ("com-Youtube", "data/matrices/com-Youtube.mtx", 1134890, "graph"),
      ("coPapersDBLP", "data/matrices/coPapersDBLP.mtx", 540486, "graph"),
      ("as-Skitter", "data/matrices/as-Skitter.mtx", 1696415, "graph"),
      ("com-LiveJournal", "data/matrices/com-LiveJournal.mtx", 3997962, "graph"),
      ("com-Orkut", "data/matrices/com-Orkut.mtx", 3072441, "graph")]
# IPM normal-equation matrices: assembled A D A^T operators with their own
# diagonal, so they are solved as published like any other operator file.
IPM = [("iter0010", "data/ipm/iter0010/matrix.mtx", 524288, "operator"),
       ("iter0020", "data/ipm/iter0020/matrix.mtx", 524288, "operator"),
       ("iter0030", "data/ipm/iter0030/matrix.mtx", 524288, "operator"),
       ("iter0040", "data/ipm/iter0040/matrix.mtx", 524288, "operator")]

KINDS = ("graph", "operator")

def matrix_args(source, spec, kind="graph"):
    """The benchmark binary's matrix-selection argv for a registry entry.

    A generated grid is a graph by construction and takes no --kind; a file
    always carries its declared --kind, which the binary REQUIRES.
    """
    if source == "grid":   return f"--graph grid --n {spec}"
    if source == "grid3d": return f"--graph grid3d --n {spec}"
    if kind not in KINDS:
        raise ValueError(f"matrix_args: kind must be one of {KINDS}, got {kind!r}")
    return f"--mtx {spec} --kind {kind}"

# id -> {family, source, kind, spec(absolute for mtx), is2d, n}
MATRICES = {}
for mid, fam, source, spec, is2d in GRIDS:
    MATRICES[mid] = dict(family=fam, source=source, kind="graph", spec=spec, is2d=is2d,
                         n=spec * spec if source == "grid" else spec ** 3)
for mid, path, n, kind in SS:
    MATRICES[mid] = dict(family="suitesparse", source="mtx", kind=kind,
                         spec=f"{ROOT}/{path}", is2d=False, n=n)
for mid, path, n, kind in IPM:
    MATRICES[mid] = dict(family="ipm", source="mtx", kind=kind,
                         spec=f"{ROOT}/{path}", is2d=False, n=n)

def kind_of(mid):
    """The declared kind of `mid`. Raises rather than guessing — an undeclared
    or mis-declared matrix must stop the run, not silently pick a reading."""
    try:
        m = MATRICES[mid]
    except KeyError:
        raise KeyError(
            f"{mid!r} is not in the matrix registry (benchmarks/runner_common.py). "
            f"Add it with an explicit kind: one of {KINDS}.") from None
    kind = m.get("kind")
    if kind not in KINDS:
        raise ValueError(
            f"{mid!r} has no declared kind (got {kind!r}). Every registry entry must "
            f"declare kind={'|'.join(KINDS)}; there is deliberately no default.")
    return kind

def margs_for(mid):
    m = MATRICES[mid]
    return matrix_args(m["source"], m["spec"], kind_of(mid))

# The one-line human description of each reading, for the runners' output and
# for the README table. Keep in step with benchmark.cpp's `interpretation`.
KIND_INTERPRETATION = {
    "graph":    "L = D - A assembled from |values| (unit weights for a pattern file)",
    "operator": "solved as published (assembled operator, diagonal as stored)",
}

def matrix_meta_for(mid):
    """The interpretation record stored in every cell's `matrix_meta`, so a
    result can always be read back with the reading that produced it."""
    m = MATRICES[mid]
    kind = kind_of(mid)
    meta = {"kind": kind, "family": m["family"], "source": m["source"],
            "interpretation": KIND_INTERPRETATION[kind]}
    if m["source"] == "mtx":
        meta["path"] = os.path.relpath(m["spec"], ROOT)
    else:
        meta["generator"] = f"{m['source']}(n={m['spec']})"
        # A generated graph has no file, so the file-reading half of the graph
        # wording does not apply to it.
        meta["interpretation"] = "L = D - A assembled from the generator's edge weights"
    return meta

# ── chart axis labels ───────────────────────────────────────────────────────────
def mat_label(mid):
    """The name a chart may show for `mid`.

    A chart must never present a matrix we TRANSFORMED under its bare file name.
    A kind=graph FILE is an adjacency matrix: what gets benchmarked is the
    Laplacian assembled from it, not the matrix as it sits on disk, so the tick
    says so. A kind=operator file IS solved as published and its bare name is
    accurate; a generated grid has no published file to deviate from.
    """
    try:
        m = MATRICES[mid]
        if m["source"] == "mtx" and kind_of(mid) == "graph":
            return f"{mid}\n(L=D−A)"
    except (KeyError, ValueError):
        pass
    return mid


def mat_labels(mats):
    return [mat_label(m) for m in mats]


_META_RE = re.compile(r'^MATRIX_META\s+(.*)$', re.M)

def parse_matrix_meta(stderr):
    """Lift the binary's `MATRIX_META ...` line out of stderr into a dict.

    The binary reports how it actually read the matrix; the registry says how it
    was asked to. Storing the binary's line means a cell records what was solved,
    not what we intended to solve. Returns {} when the line is absent (external
    solvers that never run the binary)."""
    m = _META_RE.search(stderr or "")
    if not m:
        return {}
    out = {}
    for key, qval, val in re.findall(r'(\w+)=(?:"([^"]*)"|(\S+))', m.group(1)):
        v = qval if qval else val
        if key in ("n", "nnz", "laplacian", "pos_offdiag"):
            try: v = int(v)
            except ValueError: pass
        elif key == "pos_offdiag_mass":
            try: v = float(v)
            except ValueError: pass
        out[key] = v
    if "laplacian" in out:
        out["laplacian"] = bool(out["laplacian"])
    return out


# ── benchmark-binary CSV row -> metrics dict ────────────────────────────────────
def parse_csv(out):
    rows = [l for l in out.splitlines() if l and not l.startswith("solver,") and "," in l]
    if not rows: return None
    f = rows[-1].split(",")
    if len(f) < 11: return None
    try:
        d = dict(n=int(f[2]), nnz=int(f[3]), setup_s=float(f[4]), solve_s=float(f[5]),
                 total_s=float(f[6]), iters=int(f[7]), rel_res=float(f[8]),
                 fillin=float(f[9]), us_per_nnz=float(f[10]))
        if len(f) >= 12:                     # solve_rss_mb (host VmRSS held during solve)
            try: d["solve_rss_mb"] = float(f[11])
            except ValueError: pass
        # NOTE: the binary also emits solve_vram_mb (col 12), but per-process solve-phase
        # VRAM can't be cleanly isolated (whole-device cudaMemGetInfo is desktop+context
        # contaminated and doesn't span the separate amgcl TU), so we DON'T store it. The
        # reliable VRAM metric is peak per-process from the nvidia-smi sidecar (max_vram_mb).
        return d
    except (ValueError, IndexError):
        return None

def classify(m, tol):
    """(status, metrics). n/a = the C++ side's unsupported-combo sentinel
    (iters/rel_res = -1), distinct from not_converged (ran, missed tol) and
    failed (no CSV row / crash)."""
    if m is None: return "failed", None
    if m["rel_res"] < 0: return "n/a", m
    return ("complete" if m["rel_res"] <= float(tol) * 10 else "not_converged"), m


# ── per-cell result store (schema 1) ────────────────────────────────────────────
DEFAULT_TERMINAL = frozenset({"complete", "not_converged", "failed", "timeout", "oom"})

def cell_path(family, mid, solver, config, threads, device):
    cfgtag = re.sub(r'[^A-Za-z0-9]+', '_', config) if config else "none"
    return f"{CELLS}/{family}/{mid}__{solver}__{cfgtag}__t{threads}__{device}.json"

def emit_cell(family, mid, solver, config, status, metrics, threads, device, prov,
              matrix_meta=None):
    """Write one cell. `matrix_meta` records HOW the matrix was interpreted —
    the registry's declaration, plus whatever the binary reported about the
    operator it actually assembled (parse_matrix_meta). It is filled from the
    registry when the caller passes nothing, so no cell can be written without
    saying which system it solved."""
    path = cell_path(family, mid, solver, config, threads, device)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(matrix_meta_for(mid)) if mid in MATRICES else {}
    if matrix_meta:
        meta.update(matrix_meta)
    cell = {"cell": {"config": config, "device": device, "family": family, "matrix_id": mid,
                     "solver": solver, "threads": threads}, "matrix_meta": meta,
            "metrics": metrics or {}, "provenance": prov, "schema": 1, "status": status}
    json.dump(cell, open(path, "w"), indent=2)
    return path

def cell_status(family, mid, solver, config, threads, device):
    path = cell_path(family, mid, solver, config, threads, device)
    if not os.path.exists(path): return None
    try: return json.load(open(path)).get("status")
    except (json.JSONDecodeError, OSError): return None

def cell_done(family, mid, solver, config, threads, device, terminal=DEFAULT_TERMINAL):
    return cell_status(family, mid, solver, config, threads, device) in terminal


# ── peak-VRAM sidecar (GPU axis) ────────────────────────────────────────────────
class VramSampler:
    """Background thread polling `nvidia-smi --query-compute-apps` for the device
    memory of processes whose name contains `proc_match`, keeping the max -> the
    whole-run PEAK VRAM (device analog of /usr/bin/time -%M peak RSS; the solver's
    own cudaMemGetInfo emits the exact solve-phase VRAM). One heavy task runs at a
    time, so the matched processes belong to this solver. enabled=False = no-op
    (CPU axis). accounting.mode is Disabled on this box; live polling is the route.
    """
    def __init__(self, proc_match, enabled=True, period=0.05):
        self.proc_match = proc_match; self.enabled = enabled; self.period = period
        self.peak = 0.0; self._stop = threading.Event(); self._t = None

    def _query(self):
        try:
            out = subprocess.run(
                "nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return None
        total = 0.0; seen = False
        for ln in out.splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) < 2: continue
            if self.proc_match in parts[0]:
                try: total += float(parts[1]); seen = True
                except ValueError: pass
        return total if seen else 0.0

    def _loop(self):
        while not self._stop.is_set():
            v = self._query()
            if v and v > self.peak: self.peak = v
            self._stop.wait(self.period)

    def __enter__(self):
        if self.enabled:
            self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t: self._t.join(timeout=1.0)

    def peak_mb(self):
        return round(self.peak, 1) if (self.enabled and self.peak > 0) else None


# ── ParAC (external checkout) constants ─────────────────────────────────────────
# The CPU driver and its AMD-reorder cache live in an out-of-tree ParAC checkout:
# set the environment variables below, or define the same names in a gitignored
# benchmarks/paths_local.py. Unset is fine unless a ParAC cell is actually run.
PARAC_CPU_DRIVER = os.environ.get("APXCHOL_PARAC_DRIVER", "")   # 5th arg "1" => physics/SDDM mode
# AMD-reorder cache. Point this OUTSIDE the ParAC checkout: it is our derived data
# (tens of GB), and that checkout is meant to stay at upstream plus the one patch
# in benchmarks/patches/parac/.
PARAC_REORD = os.environ.get("APXCHOL_PARAC_REORDER_DIR", "")
# ParAC's OWN input producer, cpu_implementation/write_graph.jl, which the runner
# calls (READ-ONLY, from their checkout) instead of reimplementing its
# preprocessing — that preprocessing is charged to ParAC's setup time, so it has
# to be their code. Empty => parac_runner derives it from PARAC_CPU_DRIVER's
# checkout (<checkout>/experiment/driver -> <checkout>/cpu_implementation/
# write_graph.jl); set this only when the two do not sit in the same tree.
PARAC_WRITE_GRAPH = os.environ.get("APXCHOL_PARAC_WRITE_GRAPH", "")
# Runtime library path the CPU driver needs (its MKL/compiler runtime), if any.
PARAC_LDLIB = os.environ.get("APXCHOL_PARAC_LDLIB", os.environ.get("LD_LIBRARY_PATH", ""))
PARAC_SORTED = "/tmp/parac_gpu_sorted"                       # GPU input: random-nnz-sort cache
PARAC_GPU_DRIVER = f"{ROOT}/benchmarks/build-cuda/gpu_rchol_gpu_driver"
PARAC_GPU_DRIVER_PHYS = f"{ROOT}/benchmarks/build-cuda/gpu_rchol_gpu_driver_physics"

# Machine-local overrides (gitignored; assigns the constants above by name).
try:
    from paths_local import *          # noqa: F401,F403
except ImportError:
    pass

def parac_amd_mtx(mid, family=None):
    """The cached AMD-reordered .mtx parac_runner feeds the CPU driver. Tag =
    'op' for a kind=operator matrix (the published operator, ParAC's own physics
    input), 'pin' for the Dirichlet-pinned graph-derived SuiteSparse Laplacians
    ('reg' is the legacy eps*I-regularized cache, kept as a fallback), 'pure' for
    grids. With a family, only that family's tags are accepted (a fill/timing
    comparison must not silently switch operators); family=None tries all.
    """
    if not PARAC_REORD:               # ParAC not configured on this machine
        return None
    if mid in MATRICES and kind_of(mid) == "operator":
        tags = ("op",)
    elif family is not None:
        tags = ("pin", "reg") if family == "suitesparse" else ("pure",)
    else:
        tags = ("op", "pure", "pin", "reg")
    for tag in tags:
        for cand in (f"{mid}-{tag}-amd.mtx", f"{mid}-amd.mtx"):
            p = f"{PARAC_REORD}/{cand}"
            if os.path.exists(p):
                return p
    return None

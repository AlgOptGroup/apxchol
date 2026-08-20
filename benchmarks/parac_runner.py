#!/usr/bin/env python3
"""ParAC runner — importable module, called in-process by sweep_fair.py.

Consolidates the former standalone parac_fair.py (CPU) / parac_gpu.py (GPU)
on top of runner_common. Same cells, same measurement semantics:

kind=operator matrices (the published SuiteSparse operators + the IPM normal
equations) take ParAC's OWN published-operator route: the matrix is dumped
untouched and run through physics mode, exactly as ParAC's own
experiment/physics_test_amd.sh benchmarks apache2 / G3_circuit / parabolic_fem.
Their graph-mode cell is n/a — graph mode grounds a Laplacian. Everything below
describes the kind=graph route.

CPU (`parac` / `parac_physics`, device=cpu):
  dump (grids pure Laplacian; SuiteSparse graphs Dirichlet-pinned —
  one symmetric pin per connected component, a full-rank SDDM whose residual
  against the pinned matrix equals the residual against the original singular L,
  so it's a FAIR replacement for the old eps*I reg-rel that ParAC needed to not
  diverge on the heterogeneous pure ones) -> julia AMD reorder (cached
  with a .time sidecar) -> the patched CPU driver, graph (is_graph=1) and
  physics (is_graph=0, SDDM) modes, REPS medians. setup = AMD + host prep
  (etree/ftree/summary) + elimination kernel; peak host RSS via /usr/bin/time.

GPU (`parac_graph` / `parac_physics`, device=gpu):
  dump -> ParAC's own random-nnz-sort (julia wrapper; the random step is
  ESSENTIAL — deterministic degree-sort makes the level-set SpTRSV ~1000x
  slower) -> the two CUDA drivers (driver.cu / driver_physics.cu), REPS
  medians. setup = sort + etree/ftree/summary + kernel + CSR conversion +
  SpSV analysis (ALL pre-solve work, apples-to-apples with apxchol); peak
  VRAM via the nvidia-smi sidecar. The legacy flat-CSV append is gone (the
  charts read only the per-cell store since 2dcf3a2).

Resume semantics differ BY DESIGN: CPU treats failed/timeout as terminal;
GPU retries them (transient CUDA hiccups). Unlike the old standalone runners,
a dump/reorder timeout no longer crashes the pass: CPU emits 'timeout' cells
for both modes (delete the cells to force a retry), GPU just skips (its
terminal set retries anyway).
"""
import os, re, statistics, subprocess, time

import runner_common as rc
from runner_common import ROOT, sh

TOL = "1e-8"
REPS = 3
THREADS = 16
# Physics is split per connected component (each connected -> its single-node trim grounds
# it -> fair). Run ParAC on every component with >= this many nodes; below it the component
# is a singleton/pair speck solved trivially (negligible setup/solve). Tunable.
COMP_THRESHOLD = int(os.environ.get("PARAC_COMP_THRESHOLD", "1000"))
BLOCKS = 512                  # GPU driver block count
# Per-step (dump/reorder/driver) wall caps. The >=1e8-nnz giants need more
# than 1200s for the AMD reorder step -> override per run.
TIMEOUT_CPU = int(os.environ.get("PARAC_TIMEOUT_S", "1200"))
TIMEOUT_GPU = int(os.environ.get("PARAC_GPU_TIMEOUT_S", "600"))
# CPU host-memory cap (GB): the giants' reorder/driver can balloon; die with
# bad_alloc instead of OOM-killing the box. GPU OOM is VRAM -> no host cap.
MEM_CAP_GB = float(os.environ.get("PARAC_MEM_CAP_GB", "100"))

# ParAC's own AMD-reorder script, from the out-of-tree ParAC checkout: set
# $APXCHOL_PARAC_REORDER_JL, or define REORDER_JL in benchmarks/paths_local.py.
REORDER_JL = os.environ.get("APXCHOL_PARAC_REORDER_JL", "")
NNZ_SORT_JL = f"{ROOT}/benchmarks/parac_nnz_sort.jl"
DUMP_CPU = "/tmp/parac_fair_dump"     # keep the historical cache dirs (warm caches)
DUMP_GPU = "/tmp/parac_gpu_dump"

# Machine-local overrides (gitignored; assigns the path constants above by name).
try:
    from paths_local import *          # noqa: F401,F403
except ImportError:
    pass

TERMINAL_CPU = frozenset({"complete", "not_converged", "failed", "timeout"})
TERMINAL_GPU = frozenset({"complete", "not_converged"})   # failed/timeout retry

PROV_CPU = {"boost": "on", "boost_expected": "on", "git_sha": rc.git_sha(),
            "note": "ParAC FAIR vs ORIGINAL L (no reg-rel): GRAPH reads the per-component "
                    "Dirichlet-pinned P; PHYSICS reads the giant component's pure L "
                    "(single-node trim grounding, Hypre-style split on disconnected). Both "
                    "use a per-component-consistent pin-zeroed RHS in the patched driver, so "
                    "the residual is scored vs the original singular L (verified ~1e-12). "
                    "AMD-reordered, true 1e-8, MKL serial, 16 cores",
            "repeat": REPS, "tier": "broad",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
PROV_GPU = {"source": "parac_runner.py", "git_sha": PROV_CPU["git_sha"]}

_g = lambda pat, o: (re.search(pat, o).group(1) if re.search(pat, o) else None)


# ── shared: regularized/pure dump ───────────────────────────────────────────────
def _dump_tag(mid):
    """The dump variant for this matrix.

    kind=operator -> 'op': the PUBLISHED operator, dumped untouched. This is
    ParAC's own physics input — its benchmark scripts feed exactly these files
    (parabolic_fem, ecology2, apache2, G3_circuit) to `driver <mtx> <threads> ""
    1`, i.e. physics mode, and its data README says physics matrices are "used
    as-is". Handing it our derived Laplacian instead bypassed that entry point.

    kind=graph -> the Laplacian we build from the file, either Dirichlet-pinned
    ('pin', SuiteSparse graphs) or pure ('pure', grids).

    Knowable without dumping (family + registry kind), which lets the GPU path
    tag its cache before the dump.
    """
    if rc.kind_of(mid) == "operator":
        return "op"
    return "pin" if rc.MATRICES[mid]["family"] == "suitesparse" else "pure"


def _dump_flag(tag):
    """The --dump-mtx modifier for a dump tag. 'op' and 'pure' dump L as built;
    only 'pin' asks for the Dirichlet-pinned variant."""
    return "--pin-dump" if tag == "pin" else ""


def _dump(mid, dump_dir, bin_path, timeout, mem_cap_gb=None):
    """Dump the matrix the drivers read. SuiteSparse -> Dirichlet-pinned ('pin':
    one symmetric pin per connected component, full-rank SDDM whose residual
    against the pinned matrix equals the residual against the original singular L
    — fair, unlike the old eps*I reg-rel); grids + native-SDDM IPM -> pure.
    Returns (path|None, tag). mem_cap_gb: host cap for the CPU axis only — the
    CUDA-build binary breaks under ulimit -v (CUDA reserves large host VM), so
    the GPU axis passes None."""
    tag = _dump_tag(mid)
    os.makedirs(dump_dir, exist_ok=True)
    p = f"{dump_dir}/{mid}-{tag}.mtx"
    if not os.path.exists(p):
        pinflag = _dump_flag(tag)
        sh(f"{bin_path} {rc.margs_for(mid)} {pinflag} --dump-mtx {p} --solver none",
           timeout=timeout, mem_cap_gb=mem_cap_gb)
    return (p if os.path.exists(p) else None), tag


# ── CPU axis ────────────────────────────────────────────────────────────────────
def _reorder_amd(mid, src, tag):
    """julia AMD reorder, cached; the .time sidecar keeps the measured reorder
    seconds available for cached hits."""
    rc.require_path(rc.PARAC_REORD, "APXCHOL_PARAC_REORDER_DIR", "PARAC_REORD",
                    "the ParAC AMD-reorder cache directory", must_exist=False)
    rc.require_path(REORDER_JL, "APXCHOL_PARAC_REORDER_JL", "REORDER_JL",
                    "ParAC's reorder_amd.jl")
    os.makedirs(rc.PARAC_REORD, exist_ok=True)
    amd = f"{rc.PARAC_REORD}/{mid}-{tag}-amd.mtx"
    log = f"/tmp/parac_fair_reord_{mid}_{tag}.log"
    tfile = f"{amd}.time"
    if not os.path.exists(amd):
        sh(f"julia {REORDER_JL} {src} {amd} > {log} 2>&1",
           timeout=TIMEOUT_CPU, mem_cap_gb=MEM_CAP_GB)
        secs = 0.0
        if os.path.exists(log):
            m = re.search(r"amd time:\s*([0-9.]+)", open(log).read())
            secs = float(m.group(1)) if m else 0.0
        open(tfile, "w").write(str(secs))
    secs = float(open(tfile).read()) if os.path.exists(tfile) else 0.0
    return (amd if os.path.exists(amd) else None), secs


def _run_once_cpu(amd, physics):
    rc.require_path(rc.PARAC_CPU_DRIVER, "APXCHOL_PARAC_DRIVER", "PARAC_CPU_DRIVER",
                    "the ParAC CPU driver binary")
    env = dict(os.environ, LD_LIBRARY_PATH=rc.PARAC_LDLIB, MKL_NUM_THREADS="1",
               OMP_PROC_BIND="close", KMP_AFFINITY="norespect")
    # 4 args -> is_graph=1 (graph/Laplacian); a 5th arg -> is_graph=0 (physics/
    # SDDM: handles the diagonal excess). /usr/bin/time -f 'APXRSS %M' -> peak
    # host RSS on stderr, the ParAC analog of the C++ solvers' max_rss_mb (the
    # driver is one self-contained binary, so whole-driver peak is what we chart).
    arg5 = "1" if physics else ""
    cp = sh(f"/usr/bin/time -f 'APXRSS %M' taskset -c 0-{THREADS-1} "
            f"{rc.PARAC_CPU_DRIVER} {amd} {THREADS} \"\" {arg5}".strip(),
            timeout=TIMEOUT_CPU, env=env, mem_cap_gb=MEM_CAP_GB)
    o = cp.stdout
    rss_mb = None
    for ln in (cp.stderr or "").splitlines():
        if ln.startswith("APXRSS "):
            try: rss_mb = round(int(ln.split()[1]) / 1024.0, 1)
            except ValueError: pass
    # "Factorization execution time" wraps ONLY the elimination kernel; etree/
    # ftree/summary host prep are printed separately (seconds) and count too.
    return dict(factor=_g(r"Factorization execution time:\s*([0-9.]+)", o),
                solve=_g(r"Solve time taken:\s*([0-9]+)", o),
                etree=_g(r"build etree:\s*([0-9.eE+-]+)", o),
                ftree=_g(r"factorization tree:\s*([0-9.eE+-]+)", o),
                summary=_g(r"generate summary:\s*([0-9.eE+-]+)", o),
                iters=_g(r"Iterations:\s*([0-9]+)", o),
                rr=_g(r"relative residual:\s*([0-9.eE+-]+)", o),
                n=_g(r"number of nodes:\s*([0-9]+)", o),
                nnz=_g(r"number of nonzeros:\s*([0-9]+)", o),
                rss_mb=rss_mb)


def _measure_cpu(family, mid, amd, amds, physics, solver, extra_meta=None):
    """REPS runs of one driver mode (graph or physics), one cell."""
    if rc.cell_done(family, mid, solver, "", THREADS, "cpu", terminal=TERMINAL_CPU):
        return "skip(done)"
    try:
        runs = [_run_once_cpu(amd, physics) for _ in range(REPS)]
    except subprocess.TimeoutExpired:
        rc.emit_cell(family, mid, solver, "", "timeout", {}, THREADS, "cpu", PROV_CPU,
                     matrix_meta=extra_meta)
        return "TIMEOUT"
    ok = [r for r in runs if r["factor"] and r["solve"] and r["iters"]]
    if not ok:
        rc.emit_cell(family, mid, solver, "", "failed", {}, THREADS, "cpu", PROV_CPU,
                     matrix_meta=extra_meta)
        return "FAILED"
    med = lambda key: statistics.median(float(r[key] or 0) for r in ok)
    factor = med("factor"); etree = med("etree"); ftree = med("ftree"); summ = med("summary")
    solve = statistics.median(float(r["solve"]) / 1000 for r in ok)
    iters = int(statistics.median(int(r["iters"]) for r in ok))
    rr = statistics.median(float(r["rr"]) for r in ok if r["rr"])
    prep = etree + ftree + summ
    n = int(ok[-1]["n"]); nnz = int(ok[-1]["nnz"])
    setup = factor + prep + amds   # setup = reorder + full pre-solve factorization
    total = setup + solve
    status = "complete" if rr <= 1e-7 else "not_converged"
    metrics = {"n": n, "nnz": nnz, "setup_s": round(setup, 6), "solve_s": round(solve, 6),
               "total_s": round(total, 6), "iters": iters, "rel_res": rr, "fillin": 0.0,
               "us_per_nnz": round(total / nnz * 1e6, 4), "amd_reorder_s": round(amds, 6),
               "factor_s": round(factor, 6), "prep_s": round(prep, 6)}
    rss = [float(r["rss_mb"]) for r in ok if r.get("rss_mb")]
    if rss: metrics["max_rss_mb"] = round(max(rss), 1)   # peak host RSS over reps
    rc.emit_cell(family, mid, solver, "", status, metrics, THREADS, "cpu", PROV_CPU,
                 matrix_meta=extra_meta)
    return f"{status} it={iters} solve={solve:.3f}"


def _prep_amd(mid, tag, dump_flag):
    """Dump one input variant (dump_flag) + AMD-reorder it, cached by tag (the dump only
    feeds the reorder; a warm {mid}-{tag}-amd.mtx + .time sidecar skips the re-dump).
    Returns (amd_path|None, reorder_seconds)."""
    amd = f"{rc.PARAC_REORD}/{mid}-{tag}-amd.mtx"
    if os.path.exists(amd) and os.path.exists(amd + ".time"):
        return amd, float(open(amd + ".time").read())
    os.makedirs(DUMP_CPU, exist_ok=True)
    src = f"{DUMP_CPU}/{mid}-{tag}.mtx"
    if not os.path.exists(src):
        flag = (dump_flag + " ") if dump_flag else ""
        sh(f"{rc.BIN['cpu']} {rc.margs_for(mid)} {flag}--dump-mtx {src} --solver none",
           timeout=TIMEOUT_CPU, mem_cap_gb=MEM_CAP_GB)
    if not os.path.exists(src):
        return None, 0.0
    return _reorder_amd(mid, src, tag)


def _dump_component(mid, rank):
    """Dump the rank-th largest connected component's PURE Laplacian (relabeled
    0..cn-1) via --giant-dump --comp-rank. Returns (src_path|None, n_nodes, n_comps):
    a None path means rank >= n_comps ('nothing to dump'). Cached, with a .meta
    sidecar holding (n_nodes n_comps) so warm hits skip the re-dump. rank 0 on a
    connected matrix is the full pure L."""
    os.makedirs(DUMP_CPU, exist_ok=True)
    src = f"{DUMP_CPU}/{mid}-comp{rank}.mtx"
    meta = src + ".meta"
    if os.path.exists(src) and os.path.exists(meta):
        n_nodes, n_comps = (int(x) for x in open(meta).read().split())
        return src, n_nodes, n_comps
    cp = sh(f"{rc.BIN['cpu']} {rc.margs_for(mid)} --giant-dump --comp-rank {rank} "
            f"--dump-mtx {src} --solver none", timeout=TIMEOUT_CPU, mem_cap_gb=MEM_CAP_GB)
    # stderr: "[component-dump] rank R / K components: N / M nodes" (group1=K, group2=N).
    # "nothing to dump" (rank >= K) doesn't match -> (None, 0, 0).
    m = re.search(r"rank\s+\d+\s*/\s*(\d+)\s+components:\s*(\d+)\s*/", cp.stderr or "")
    if not m:
        return None, 0, 0
    n_comps, n_nodes = int(m.group(1)), int(m.group(2))
    if os.path.exists(src):
        open(meta, "w").write(f"{n_nodes} {n_comps}")
    return (src if os.path.exists(src) else None), n_nodes, n_comps


def _measure_cpu_physics_split(family, mid):
    """ParAC physics over the LITERAL per-component split: dump every connected
    component with >= COMP_THRESHOLD nodes (descending size), AMD-reorder it, run
    REPS times, and recombine into one parac_physics cell. setup/solve are SUMMED
    (the components are solved sequentially), iters/rel_res are the worst (MAX) over
    components. Components below the threshold are singleton/pair specks with
    negligible cost (skipped). Each component is connected, so the driver's
    single-node trim grounds it and the residual is scored vs the original pure L."""
    if rc.cell_done(family, mid, "parac_physics", "", THREADS, "cpu", terminal=TERMINAL_CPU):
        return "skip(done)"
    setup = solve = factor_tot = prep_tot = amds_tot = 0.0
    iters = 0; rr = 0.0; n_tot = nnz_tot = 0; rss_peak = 0.0
    n_solved = 0; n_comps_total = None; rank = 0
    try:
        while True:
            src, n_nodes, n_comps = _dump_component(mid, rank)
            if n_comps_total is None and n_comps:
                n_comps_total = n_comps
            if src is None:                  # rank >= n_comps: no more components
                break
            if n_nodes < COMP_THRESHOLD:     # sorted descending -> the rest are smaller too
                break
            amd, amds = _reorder_amd(mid, src, f"comp{rank}")
            if not amd:
                rank += 1
                continue
            runs = [_run_once_cpu(amd, True) for _ in range(REPS)]
            ok = [r for r in runs if r["factor"] and r["solve"] and r["iters"]]
            if not ok:
                rc.emit_cell(family, mid, "parac_physics", "", "failed", {},
                             THREADS, "cpu", PROV_CPU)
                return "FAILED"
            med = lambda key: statistics.median(float(r[key] or 0) for r in ok)
            factor = med("factor"); prep = med("etree") + med("ftree") + med("summary")
            c_solve = statistics.median(float(r["solve"]) / 1000 for r in ok)
            c_iters = int(statistics.median(int(r["iters"]) for r in ok))
            c_rr = statistics.median(float(r["rr"]) for r in ok if r["rr"])
            factor_tot += factor; prep_tot += prep; amds_tot += amds
            setup += factor + prep + amds; solve += c_solve
            iters = max(iters, c_iters); rr = max(rr, c_rr)
            n_tot += int(ok[-1]["n"]); nnz_tot += int(ok[-1]["nnz"])
            rss = [float(r["rss_mb"]) for r in ok if r.get("rss_mb")]
            if rss: rss_peak = max(rss_peak, max(rss))
            n_solved += 1; rank += 1
    except subprocess.TimeoutExpired:
        rc.emit_cell(family, mid, "parac_physics", "", "timeout", {}, THREADS, "cpu", PROV_CPU)
        return "TIMEOUT"
    if n_solved == 0 or nnz_tot == 0:
        rc.emit_cell(family, mid, "parac_physics", "", "failed", {}, THREADS, "cpu", PROV_CPU)
        return "FAILED"
    total = setup + solve
    status = "complete" if rr <= 1e-7 else "not_converged"
    metrics = {"n": n_tot, "nnz": nnz_tot, "setup_s": round(setup, 6),
               "solve_s": round(solve, 6), "total_s": round(total, 6), "iters": iters,
               "rel_res": rr, "fillin": 0.0, "us_per_nnz": round(total / nnz_tot * 1e6, 4),
               "amd_reorder_s": round(amds_tot, 6), "factor_s": round(factor_tot, 6),
               "prep_s": round(prep_tot, 6), "n_components_solved": n_solved,
               "n_components_total": n_comps_total or n_solved}
    if rss_peak: metrics["max_rss_mb"] = round(rss_peak, 1)
    rc.emit_cell(family, mid, "parac_physics", "", status, metrics, THREADS, "cpu", PROV_CPU)
    return f"{status} it={iters} solve={solve:.3f} comps={n_solved}/{n_comps_total}"


def _run_cpu_operator(mid, family):
    """ParAC CPU pass for a kind=operator matrix: its OWN published-operator path.

    ParAC benchmarks published operators through `driver <mtx> <threads> "" 1` —
    physics mode — on the matrix as downloaded (its experiment/physics_test_amd.sh
    runs exactly parabolic_fem / ecology2 / apache2 / G3_circuit that way, and its
    data README says physics matrices are "used as-is"). Its reader auto-detects
    that the file already carries a diagonal and keeps it, rather than rebuilding
    one from the off-diagonals. So: dump the published operator untouched,
    AMD-reorder it, run physics.

    The GRAPH-mode cell is n/a here. Graph mode grounds by pinning isolated nodes
    and building a per-component mean-zero RHS — a Laplacian construction, which
    a published full-rank operator has no need of and is not defined by.

    Physics mode grounds by trimming the LAST row and column; that is ParAC's own
    protocol for these matrices and is recorded in the cell so the reading is not
    lost.
    """
    rc.emit_cell(family, mid, "parac", "", "n/a", {}, THREADS, "cpu", PROV_CPU,
                 matrix_meta={"parac_mode": "n/a",
                              "parac_na_reason":
                                  "ParAC graph mode grounds a Laplacian (isolated-node pins "
                                  "+ per-component mean-zero RHS); this matrix is a published "
                                  "operator, which goes through ParAC's physics path instead"})
    try:
        amd, amds = _prep_amd(mid, "op", "")     # the published operator, untouched
    except subprocess.TimeoutExpired:
        if not rc.cell_done(family, mid, "parac_physics", "", THREADS, "cpu", terminal=TERMINAL_CPU):
            rc.emit_cell(family, mid, "parac_physics", "", "timeout", {}, THREADS, "cpu", PROV_CPU)
        return "graph[n/a] physics[TIMEOUT(dump/reorder)]"
    if not amd:
        return "graph[n/a] physics[SKIP(dump/reorder)]"
    p = _measure_cpu(family, mid, amd, amds, True, "parac_physics",
                     extra_meta={"parac_mode": "physics",
                                 "parac_input": "the published operator, AMD-reordered",
                                 "parac_grounding": "physics mode trims the last row/column"})
    return f"graph[n/a] physics[{p}]"


def run_cpu(mid):
    """ParAC CPU pass for one matrix: graph + physics cells. Returns a status string.

    kind=operator goes through _run_cpu_operator (ParAC's own physics entry point on
    the published matrix). For kind=graph:

    GRAPH reads the per-component-Dirichlet-pinned P (--pin-dump for SuiteSparse, pure
    for grids) — fair on connected AND disconnected (one pin per component). PHYSICS
    is the LITERAL per-component split: each connected component >= COMP_THRESHOLD nodes
    is dumped as a PURE Laplacian, solved with the single-node trim grounding (which is
    valid only on a connected operator), and recombined (sum setup/solve, max iters) —
    so the solution is scored vs the ORIGINAL L. rank 0 = the full pure L on a connected
    matrix; on disconnected matrices the small specks below the threshold are negligible.
    Both built on the consistent pin-zeroed RHS inside the patched driver."""
    family = rc.MATRICES[mid]["family"]
    if rc.kind_of(mid) == "operator":
        return _run_cpu_operator(mid, family)
    g_tag = _dump_tag(mid)
    g_flag = _dump_flag(g_tag)
    try:
        g_amd, g_amds = _prep_amd(mid, g_tag, g_flag)            # graph: per-component pin
        if not g_amd:
            g = "SKIP(graph dump/reorder)"
        else:
            g = _measure_cpu(family, mid, g_amd, g_amds, False, "parac")   # is_graph=1
    except subprocess.TimeoutExpired:
        if not rc.cell_done(family, mid, "parac", "", THREADS, "cpu", terminal=TERMINAL_CPU):
            rc.emit_cell(family, mid, "parac", "", "timeout", {}, THREADS, "cpu", PROV_CPU)
        g = "TIMEOUT(graph dump/reorder)"
    p = _measure_cpu_physics_split(family, mid)                  # is_graph=0, per-component split
    return f"graph[{g}] physics[{p}]"


# ── GPU axis ────────────────────────────────────────────────────────────────────
def _nnz_sort(mid, src, tag):
    """ParAC's OWN nnz-sort (random permutation then per-column-nnz sort, from
    write_graph.jl) via parac_nnz_sort.jl; prints SORT COMPUTE time (excl. disk
    I/O), same convention as reorder_amd.jl. Returns (path|None, seconds). The
    tag is in the cache name so a pin-derived sort never aliases an old reg one."""
    os.makedirs(rc.PARAC_SORTED, exist_ok=True)
    out = f"{rc.PARAC_SORTED}/{mid}-{tag}-nnz-sorted.mtx"
    tfile = f"{out}.time"
    if os.path.exists(out) and os.path.exists(tfile):
        return out, float(open(tfile).read())
    o = sh(f"julia {NNZ_SORT_JL} {src} {out}", timeout=TIMEOUT_GPU).stdout
    m = re.search(r"sort time:\s*([0-9.eE+-]+)", o)
    secs = float(m.group(1)) if m else 0.0
    if os.path.exists(out):
        open(tfile, "w").write(str(secs))
    return (out if os.path.exists(out) else None), secs


def _run_once_gpu(driver, mtx, tol):
    with rc.VramSampler("gpu_rchol") as vram:   # matches both driver binaries
        o = sh(f"{driver} {mtx} {BLOCKS} 1 {tol}", timeout=TIMEOUT_GPU).stdout
    # Pre-solve setup is NOT just the GPU kernel: host prep (etree/ftree/summary,
    # seconds) + CSR conversion + cuSPARSE SpSV analysis (ms) all precede the
    # first PCG iteration; counting only the kernel under-reported setup ~17x.
    return dict(etree=_g(r"build etree:\s*([0-9.eE+-]+)", o),
                ftree=_g(r"factorization tree:\s*([0-9.eE+-]+)", o),
                summary=_g(r"generate summary:\s*([0-9.eE+-]+)", o),
                factor=_g(r"Kernel execution time:\s*([0-9.]+)", o),       # ms
                conv=_g(r"total conversion time:\s*([0-9.]+)", o),         # ms
                spsv=_g(r"solve preprocess time:\s*([0-9.]+)", o),         # ms
                solve=_g(r"pcg time:\s*([0-9.]+)", o),                     # ms
                iters=_g(r"final iteration:\s*([0-9]+)", o),
                rr=_g(r"normalized diff norm:\s*([0-9.eE+-]+)", o),
                n=_g(r"num cols:\s*([0-9]+)", o),
                nnz=_g(r"laplacian nnz:\s*([0-9]+)", o),
                vram_mb=vram.peak_mb())


def run_gpu(mid, tol=TOL):
    """ParAC GPU pass for one matrix: parac_graph + parac_physics cells."""
    family = rc.MATRICES[mid]["family"]
    tag = _dump_tag(mid)
    try:
        # Warm nnz-sort cache (+.time) -> the dump is unneeded; skip it.
        sorted_cached = f"{rc.PARAC_SORTED}/{mid}-{tag}-nnz-sorted.mtx"
        if os.path.exists(sorted_cached) and os.path.exists(sorted_cached + ".time"):
            sorted_mtx, sort_s = sorted_cached, float(open(sorted_cached + ".time").read())
        else:
            src, _tag = _dump(mid, DUMP_GPU, rc.BIN["gpu"], TIMEOUT_GPU)
            if not src: return "SKIP(dump)"
            sorted_mtx, sort_s = _nnz_sort(mid, src, tag)
            if not sorted_mtx: return "SKIP(nnz-sort)"
    except subprocess.TimeoutExpired:
        return "TIMEOUT(dump/sort)"   # no cell: GPU terminal set retries anyway
    results = []
    is_operator = rc.kind_of(mid) == "operator"
    for solver_key, driver in (("parac_graph", rc.PARAC_GPU_DRIVER),
                               ("parac_physics", rc.PARAC_GPU_DRIVER_PHYS)):
        if is_operator and solver_key == "parac_graph":
            # Same reasoning as the CPU axis: graph mode grounds a Laplacian, and
            # a published operator goes through ParAC's physics path instead.
            rc.emit_cell(family, mid, solver_key, "", "n/a", {}, THREADS, "gpu", PROV_GPU,
                         matrix_meta={"parac_mode": "n/a",
                                      "parac_na_reason":
                                          "ParAC graph mode grounds a Laplacian; this matrix "
                                          "is a published operator and runs physics mode"})
            results.append(f"{solver_key}[n/a]"); continue
        if rc.cell_done(family, mid, solver_key, "", THREADS, "gpu", terminal=TERMINAL_GPU):
            results.append(f"{solver_key}[skip(done)]"); continue
        if not os.path.exists(driver):
            results.append(f"{solver_key}[SKIP(driver missing)]"); continue
        try:
            runs = [_run_once_gpu(driver, sorted_mtx, tol) for _ in range(REPS)]
        except subprocess.TimeoutExpired:
            results.append(f"{solver_key}[TIMEOUT]"); continue   # retried next run
        ok = [r for r in runs if r["factor"] and r["solve"] and r["iters"]]
        if not ok:
            rc.emit_cell(family, mid, solver_key, "", "failed", None, THREADS, "gpu", PROV_GPU)
            results.append(f"{solver_key}[FAILED]"); continue
        med = lambda key, scale=1.0: statistics.median(float(r[key] or 0) * scale for r in ok)
        etree = med("etree"); ftree = med("ftree"); summ = med("summary")
        factor = med("factor", 1/1000); conv = med("conv", 1/1000)
        spsv = med("spsv", 1/1000); solve = med("solve", 1/1000)
        iters = int(statistics.median(int(r["iters"]) for r in ok))
        rr = statistics.median(float(r["rr"]) for r in ok if r["rr"])
        # setup = EVERYTHING before the first PCG iteration (only disk I/O excluded).
        setup = sort_s + etree + ftree + summ + factor + conv + spsv
        total = setup + solve
        status = "complete" if rr <= float(tol) * 10 else "not_converged"
        metrics = {"n": int(ok[0]["n"]) if ok[0]["n"] else None,
                   "nnz": int(ok[0]["nnz"]) if ok[0]["nnz"] else None,
                   "setup_s": setup, "solve_s": solve, "total_s": total,
                   "iters": iters, "rel_res": rr}
        vram = [float(r["vram_mb"]) for r in ok if r.get("vram_mb")]
        if vram: metrics["max_vram_mb"] = round(max(vram), 1)   # peak VRAM over reps
        rc.emit_cell(family, mid, solver_key, "", status, metrics, THREADS, "gpu", PROV_GPU)
        results.append(f"{solver_key}[{status} it={iters} solve={solve:.3f}]")
    return " ".join(results)


def main():
    """Standalone CLI (replaces parac_fair.py / parac_gpu.py):
       python3 benchmarks/parac_runner.py [--device cpu|gpu] [--only id1,id2]"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--only", default="", help="comma-separated matrix ids; empty = all")
    a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    run = run_gpu if a.device == "gpu" else run_cpu
    for mid in rc.MATRICES:
        if only and mid not in only: continue
        print(f"[{mid}] {run(mid)}", flush=True)
    print(f"ParAC {a.device} done")


if __name__ == "__main__":
    main()

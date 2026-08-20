#!/usr/bin/env python3
"""ParAC runner — importable module, called in-process by sweep_fair.py.

Consolidates the former standalone parac_fair.py (CPU) / parac_gpu.py (GPU)
on top of runner_common.

Each matrix goes through exactly ONE ParAC mode: the one ParAC documents for it.
The other mode's cell is n/a, with the reason recorded.

PREPROCESSING IS THEIRS. Every input ParAC reads is prepared by ParAC's OWN
cpu_implementation/write_graph.jl, invoked out of their checkout through the thin
dispatcher benchmarks/parac_produce_upstream.jl. We charge that preprocessing to
ParAC's setup time (setup = factor + prep + amds), so it has to be their code
doing their work — not our reimplementation of it. Outputs land in OUR cache
directories under THEIR naming convention (`<prefix>.mtx` in, `<prefix>-amd.mtx`
/ `<prefix>-nnz-sorted.mtx` out), so their checkout is never written to.
benchmarks/parac_reorder_amd.jl and parac_nnz_sort.jl remain as the FALLBACK for
an input their producer rejects; a cell prepared by the fallback records that,
and the reason, in matrix_meta.parac_prep.

kind=graph -> GRAPH mode (`driver <mtx> <threads> ""`).
  Dump the PURE L = D - A, one connected component at a time (--giant-dump
  --comp-rank R; rank 0 IS the whole matrix when it is connected), hand it to
  their `graph_produce(prefix, "amd")`, and let ParAC generate its own zero-sum
  RHS. That RHS is consistent for a connected singular Laplacian, so ParAC solves
  the very L the benchmark reports on and its printed residual is against that L.
  We do NOT hand it a Dirichlet-pinned matrix: its RHS generator knows nothing
  about our pin, and the residual against the original L then floors around 1e-3
  no matter the tolerance (measured; see benchmarks/patches/parac/README.md). Its
  physics cell is n/a — physics mode trims a row, which on a pure Laplacian
  deletes a real vertex.

  graph_produce strips the diagonal, forces the off-diagonals negative and
  REBUILDS the diagonal as -colsum, i.e. re-derives the pure Laplacian. On our
  component dumps — which already ARE pure Laplacians of a connected component —
  that rebuild is a no-op: verified byte-identical to a permutation-only reorder
  on com-Amazon (unweighted) and grid_1000 (weighted).

kind=operator -> PHYSICS mode (`driver <mtx> <threads> "" 1`).
  Dump the operator as PUBLISHED and hand it to their `physics_produce(prefix,
  "amd")`: permute, then append the ground row/column, so the appended node is
  last. Physics mode trims exactly that node, so what it solves is the published
  operator itself. The augmentation is not optional — run physics mode on an
  un-augmented operator and the trim deletes a real degree of freedom (measured:
  apache2 scores 3.1e-3 against the published matrix while ParAC prints 8.8e-9;
  G3_circuit does not converge at all). Its graph cell is n/a — graph mode
  grounds a Laplacian, which a full-rank operator is not.

  physics_produce refuses (`@assert false`, "not diagonally dominant") an input
  whose TOTAL entry sum is below -1e-9. All nine kind=operator matrices clear it
  with room: apache2 +2.20e4, ecology1 +2.05e-9, G3_circuit +6.91e8,
  parabolic_fem +2.00, thermal2 +2.01e3, iter0010..0040 +5.24e-1.

TOLERANCE. ParAC's stopping test compares the residual NORM against
sqrt(rel_tol): an ABSOLUTE test, and on the recurrence residual, which runs
optimistic by a matrix-dependent factor. Rather than patch the test we calibrate
it from one probe run using ParAC's own two printed numbers (see
_calibrate_rel_tol) and pass the result through PARAC_REL_TOL. The achieved true
relative residual is ParAC's own `relative residual:` line and lands just under
TOL; both it and the calibrated tolerance are recorded in the cell.

CPU (`parac` / `parac_physics`, device=cpu):
  dump -> their write_graph.jl producer, method "amd" (cached with .time/.prep
  sidecars) -> calibrate -> REPS runs of the driver. setup = AMD + host prep
  (etree/ftree/summary) + elimination kernel; peak host RSS via /usr/bin/time.

GPU (`parac_graph` / `parac_physics`, device=gpu):
  dump -> their write_graph.jl producer, method "nnz-sort" (their random
  permutation THEN degree sort; the random step is ESSENTIAL — a deterministic
  degree-sort makes the level-set SpTRSV ~1000x slower, and physics_produce
  appends the ground node after it for an operator) -> the two CUDA drivers
  (driver.cu / driver_physics.cu), REPS medians. Those take the tolerance on argv
  already, so the same calibration applies with no patch.
  setup = sort + etree/ftree/summary + kernel + CSR conversion + SpSV analysis
  (ALL pre-solve work, apples-to-apples with apxchol); peak VRAM via the
  nvidia-smi sidecar.

The ParAC checkout itself is upstream 44ef39d plus ONE patch
(benchmarks/patches/parac/0001-*.patch), applied by benchmarks/parac_build.sh.

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
MAX_ITER = 2000               # ParAC's default is 1000; patch 0001 lets us raise it
PROBE_REL_TOL = 1e-7          # ParAC's own default; the CPU calibration probe runs at it
PROBE_TOL_GPU = 1e-7          # the CUDA drivers' tolerance is already argv[4]
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

DUMP_CPU = "/tmp/parac_fair_dump"     # keep the historical cache dirs (warm caches)
DUMP_GPU = "/tmp/parac_gpu_dump"

# Machine-local overrides (gitignored; assigns the path constants above by name).
try:
    from paths_local import *          # noqa: F401,F403
except ImportError:
    pass

# THEIR preprocessing, run from THEIR checkout. parac_produce_upstream.jl is a
# dispatcher in OUR repo that includes ParAC's cpu_implementation/write_graph.jl
# and calls physics_produce / graph_produce on it; every line of preprocessing is
# theirs. It runs under benchmarks/julia (the project that carries AMD, Metis and
# Laplacians — instantiate with
#   julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()').
PRODUCE_JL = f"{ROOT}/benchmarks/parac_produce_upstream.jl"
JULIA_PROJECT = f"{ROOT}/benchmarks/julia"

# OUR reimplementations, kept as the FALLBACK for an input their producer rejects.
# These used to live untracked inside the ParAC checkout; they are our harness's
# business, not a modification of ParAC, so they are set AFTER paths_local — a
# machine-local file must not be able to point them back at a copy in someone's
# ParAC tree. Drop any REORDER_JL line from paths_local.py.
REORDER_JL = f"{ROOT}/benchmarks/parac_reorder_amd.jl"
NNZ_SORT_JL = f"{ROOT}/benchmarks/parac_nnz_sort.jl"

TERMINAL_CPU = frozenset({"complete", "not_converged", "failed", "timeout"})
TERMINAL_GPU = frozenset({"complete", "not_converged"})   # failed/timeout retry

PROV_CPU = {"boost": "on", "boost_expected": "on", "git_sha": rc.git_sha(),
            "note": "ParAC at upstream 44ef39d + benchmarks/patches/parac/0001 (thread-count "
                    "gate + configurable tolerance; no numerics touched). kind=graph -> GRAPH "
                    "mode on the PURE L per connected component, ParAC's own zero-sum RHS; "
                    "kind=operator -> PHYSICS mode on the PUBLISHED operator augmented ParAC's "
                    "own way (permute, then append the ground row/column the trim removes). "
                    "Input prepared by ParAC's OWN cpu_implementation/write_graph.jl "
                    "(graph_produce / physics_produce, method 'amd'), not by a reimplementation. "
                    "Tolerance calibrated from a probe run so ParAC's own printed "
                    "relative residual lands under 1e-8. AMD-reordered, MKL serial, 16 cores",
            "repeat": REPS, "tier": "broad",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
PROV_GPU = {"source": "parac_runner.py", "git_sha": PROV_CPU["git_sha"]}

_g = lambda pat, o: (re.search(pat, o).group(1) if re.search(pat, o) else None)


# ── shared: regularized/pure dump ───────────────────────────────────────────────
def _dump_tag(mid):
    """The dump variant for this matrix.

    kind=operator -> 'op': the PUBLISHED operator, dumped untouched, which then
    gets ParAC's own ground-node augmentation at the reorder step (--augment).
    kind=graph -> 'pure': L = D - A as built, NOT Dirichlet-pinned. ParAC's graph
    mode grounds a singular Laplacian itself, by generating a zero-sum RHS; a pin
    it does not know about is what used to floor its residual against the
    original L at ~1e-3.

    Knowable without dumping (registry kind), which lets the GPU path tag its
    cache before the dump.
    """
    return "op" if rc.kind_of(mid) == "operator" else "pure"


def _augments(mid):
    """True when the reorder step must append ParAC's ground row/column. Only a
    kind=operator matrix needs it: it is not a Laplacian, and physics mode's trim
    is what takes the appended node back off."""
    return rc.kind_of(mid) == "operator"


def _dump(mid, dump_dir, bin_path, timeout, mem_cap_gb=None):
    """Dump the matrix the drivers read: the published operator for kind=operator,
    the pure L = D - A for kind=graph. Returns (path|None, tag). mem_cap_gb: host
    cap for the CPU axis only — the CUDA-build binary breaks under ulimit -v
    (CUDA reserves large host VM), so the GPU axis passes None."""
    tag = _dump_tag(mid)
    os.makedirs(dump_dir, exist_ok=True)
    p = f"{dump_dir}/{mid}-{tag}.mtx"
    if not os.path.exists(p):
        sh(f"{bin_path} {rc.margs_for(mid)} --dump-mtx {p} --solver none",
           timeout=timeout, mem_cap_gb=mem_cap_gb)
    return (p if os.path.exists(p) else None), tag


# ── ParAC's OWN preprocessing (their write_graph.jl) ────────────────────────────
# "amd time:" / "sort time:" is what THEIR producer prints; both scripts print it
# the same way, so the reorder seconds we charge are the ones their code measured.
_PREP_TIME_RE = re.compile(r"(?:amd|sort|random|nd) time:\s*([0-9.eE+-]+)")
_PREP_SUFFIX = {"amd": "-amd.mtx", "nnz-sort": "-nnz-sorted.mtx"}
# A cache entry written before the .prep sidecar existed came from our
# reimplementation. Say so rather than claiming provenance we cannot check —
# delete the cached *-amd.mtx to have their producer rebuild (and stamp) it.
_PREP_UNKNOWN = ("unrecorded: this cache entry predates the .prep sidecar, so it was "
                 "written by benchmarks/parac_reorder_amd.jl / parac_nnz_sort.jl. Those "
                 "were byte-identical to ParAC's own producer on every matrix compared "
                 "(see benchmarks/patches/parac/README.md), but this particular file was "
                 "not checked; delete it to have their producer rebuild it")


def _write_graph_jl():
    """ParAC's own cpu_implementation/write_graph.jl.

    Configured explicitly (APXCHOL_PARAC_WRITE_GRAPH / paths_local.PARAC_WRITE_GRAPH)
    or, by default, derived from the CPU driver's checkout — <checkout>/experiment/
    driver sits next to <checkout>/cpu_implementation/write_graph.jl."""
    if rc.PARAC_WRITE_GRAPH:
        return rc.PARAC_WRITE_GRAPH
    if rc.PARAC_CPU_DRIVER:
        return os.path.join(os.path.dirname(os.path.dirname(rc.PARAC_CPU_DRIVER)),
                            "cpu_implementation", "write_graph.jl")
    return ""


def _produce_upstream(prefix, src, mode, method, timeout, mem_cap_gb=None):
    """Run ParAC's OWN producer (write_graph.jl `<mode>_produce`) on `src`.

    Their producer takes a path PREFIX: it reads `<prefix>.mtx` and writes
    `<prefix>-amd.mtx` / `<prefix>-nnz-sorted.mtx`. We honour that convention
    inside OUR cache directory and point `<prefix>.mtx` at the dump with a
    symlink, so nothing is ever written into their checkout and the cache file
    names are unchanged.

    Returns (out_path|None, seconds, error|None): error is a short string when
    their producer could not take this input, which is the caller's cue to fall
    back to our reimplementation AND to record why.
    """
    wg = _write_graph_jl()
    if not wg or not os.path.exists(wg):
        return None, 0.0, (f"ParAC write_graph.jl not found at {wg or '<unset>'} "
                           f"(set $APXCHOL_PARAC_WRITE_GRAPH / paths_local.PARAC_WRITE_GRAPH)")
    out = prefix + _PREP_SUFFIX[method]
    link = prefix + ".mtx"
    log = f"/tmp/parac_produce_{os.path.basename(prefix)}_{method}.log"
    if os.path.exists(link) and not os.path.islink(link):
        return None, 0.0, f"{link} exists and is not our symlink; refusing to overwrite it"
    try:
        if os.path.islink(link):
            os.remove(link)
        os.symlink(os.path.abspath(src), link)
        cp = sh(f"julia --project={JULIA_PROJECT} {PRODUCE_JL} {wg} {prefix} {mode} {method} "
                f"> {log} 2>&1", timeout=timeout, mem_cap_gb=mem_cap_gb)
    finally:
        if os.path.islink(link):
            os.remove(link)
    text = open(log).read() if os.path.exists(log) else ""
    if cp.returncode != 0 or not os.path.exists(out):
        tail = " | ".join(l.strip() for l in text.strip().splitlines()[-4:]) or "no output"
        return None, 0.0, f"{mode}_produce(\"{method}\") rc={cp.returncode}: {tail}"
    m = _PREP_TIME_RE.search(text)
    return out, (float(m.group(1)) if m else 0.0), None


# ── CPU axis ────────────────────────────────────────────────────────────────────
def _reorder_amd(mid, src, tag, augment=False):
    """AMD reorder via ParAC's OWN write_graph.jl producer, cached. Returns
    (path|None, seconds, prep_provenance); the .time and .prep sidecars keep both
    available on a warm cache hit.

    augment=True selects `physics_produce`, which permutes and then appends
    ParAC's ground row/column, so the appended node is last and physics mode's
    trim removes exactly it. Otherwise `graph_produce`, which re-derives the pure
    Laplacian (a no-op on our component dumps, which already are one). The tag
    carries '-aug' so an augmented input can never be served from a cache entry
    built without it.

    If their producer rejects the input — `physics_produce` asserts on a globally
    diagonally-deficient matrix — we fall back to benchmarks/parac_reorder_amd.jl
    and return the reason, which the caller records in the cell.
    """
    rc.require_path(rc.PARAC_REORD, "APXCHOL_PARAC_REORDER_DIR", "PARAC_REORD",
                    "the ParAC AMD-reorder cache directory", must_exist=False)
    os.makedirs(rc.PARAC_REORD, exist_ok=True)
    if augment:
        tag = f"{tag}-aug"
    prefix = f"{rc.PARAC_REORD}/{mid}-{tag}"
    amd = prefix + "-amd.mtx"
    tfile, pfile = amd + ".time", amd + ".prep"
    if not os.path.exists(amd):
        mode = "physics" if augment else "graph"
        out, secs, err = _produce_upstream(prefix, src, mode, "amd",
                                           TIMEOUT_CPU, MEM_CAP_GB)
        if out:
            prep = f"ParAC write_graph.jl {mode}_produce(path, \"amd\"), upstream and unmodified"
        else:
            print(f"   [parac] upstream {mode}_produce refused {mid}: {err}\n"
                  f"   [parac] falling back to benchmarks/parac_reorder_amd.jl", flush=True)
            secs = _reorder_amd_ours(mid, src, amd, tag, augment)
            prep = f"benchmarks/parac_reorder_amd.jl (FALLBACK — upstream refused: {err})"
        open(tfile, "w").write(str(secs))
        open(pfile, "w").write(prep)
    secs = float(open(tfile).read()) if os.path.exists(tfile) else 0.0
    prep = open(pfile).read() if os.path.exists(pfile) else _PREP_UNKNOWN
    return (amd if os.path.exists(amd) else None), secs, prep


def _reorder_amd_ours(mid, src, amd, tag, augment):
    """FALLBACK reorder with our own parac_reorder_amd.jl (see _reorder_amd).
    Runs on julia's default environment, so it stays usable even when the
    benchmarks/julia project their producer needs is not instantiated."""
    log = f"/tmp/parac_fair_reord_{mid}_{tag}.log"
    flag = " --augment" if augment else ""
    sh(f"julia {REORDER_JL} {src} {amd}{flag} > {log} 2>&1",
       timeout=TIMEOUT_CPU, mem_cap_gb=MEM_CAP_GB)
    if os.path.exists(log):
        m = _PREP_TIME_RE.search(open(log).read())
        if m:
            return float(m.group(1))
    return 0.0


def _run_once_cpu(amd, physics, rel_tol=None):
    rc.require_path(rc.PARAC_CPU_DRIVER, "APXCHOL_PARAC_DRIVER", "PARAC_CPU_DRIVER",
                    "the ParAC CPU driver binary")
    env = dict(os.environ, LD_LIBRARY_PATH=rc.PARAC_LDLIB, MKL_NUM_THREADS="1",
               OMP_PROC_BIND="close", KMP_AFFINITY="norespect")
    if rel_tol is not None:
        # patch 0001 forwards these to example_pcg_solver's existing parameters.
        env["PARAC_REL_TOL"] = repr(float(rel_tol))
        env["PARAC_MAX_ITER"] = str(MAX_ITER)
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
                # the RECURRENCE residual its stopping test actually looked at,
                # needed to calibrate the tolerance (see _calibrate_rel_tol)
                recur=_g(r"Final residual norm:\s*([0-9.eE+-]+)", o),
                # "number of nodes/nonzeros" is what the driver READ, which in
                # physics mode includes the appended ground node. Report the
                # operator it actually SOLVED: the trimmed size.
                n=(int(_g(r"number of nodes:\s*([0-9]+)", o) or 0) - (1 if physics else 0)
                   or None),
                nnz=(_g(r"trimmed laplacian nnz:\s*([0-9]+)", o)
                     or _g(r"number of nonzeros:\s*([0-9]+)", o)),
                rss_mb=rss_mb)


def _calibrate_rel_tol(amd, physics, tau=float(TOL)):
    """The rel_tol that makes ParAC's own stopping test stop at true residual tau.

    Its test is `sqrt(dpar[4]) > sqrt(dpar[0])`: the RECURRENCE residual norm
    against sqrt(rel_tol) — absolute, and optimistic by a matrix-dependent factor
    (measured 1.8x on com-Amazon, 45x on apache2, 5x on G3_circuit). Rather than
    edit the test, probe once and rescale, using only the two numbers ParAC
    prints: recur_0 = "Final residual norm" (what the test compared) and
    R_0 = "relative residual" (the true ||Ax-b||/||b|| it achieved). The two move
    together, so

        rel_tol = ( tau * recur_0 / R_0 ) ** 2

    lands the true residual just under tau. Returns None if the probe did not
    produce both numbers, in which case the caller runs at the driver default.
    """
    p = _run_once_cpu(amd, physics, rel_tol=PROBE_REL_TOL)
    if not (p["recur"] and p["rr"]):
        return None
    recur0, r0 = float(p["recur"]), float(p["rr"])
    if not (recur0 > 0.0 and r0 > 0.0):
        return None
    return (tau * recur0 / r0) ** 2


def _measure_cpu(family, mid, amd, amds, physics, solver, extra_meta=None):
    """REPS runs of one driver mode (graph or physics), one cell. The probe run
    that calibrates the tolerance is NOT timed and NOT one of the REPS."""
    if rc.cell_done(family, mid, solver, "", THREADS, "cpu", terminal=TERMINAL_CPU):
        return "skip(done)"
    try:
        rel_tol = _calibrate_rel_tol(amd, physics)
        runs = [_run_once_cpu(amd, physics, rel_tol=rel_tol) for _ in range(REPS)]
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
    # rel_res is ParAC's own ||Ax-b||/||b|| against the operator it solved, which
    # the input construction makes the operator we report on. Judge it at the
    # tolerance we asked for, not at a looser one.
    status = "complete" if rr <= float(TOL) else "not_converged"
    metrics = {"n": n, "nnz": nnz, "setup_s": round(setup, 6), "solve_s": round(solve, 6),
               "total_s": round(total, 6), "iters": iters, "rel_res": rr, "fillin": 0.0,
               "us_per_nnz": round(total / nnz * 1e6, 4), "amd_reorder_s": round(amds, 6),
               "factor_s": round(factor, 6), "prep_s": round(prep, 6)}
    if rel_tol is not None:
        metrics["parac_rel_tol"] = rel_tol      # the calibrated value we passed
    rss = [float(r["rss_mb"]) for r in ok if r.get("rss_mb")]
    if rss: metrics["max_rss_mb"] = round(max(rss), 1)   # peak host RSS over reps
    rc.emit_cell(family, mid, solver, "", status, metrics, THREADS, "cpu", PROV_CPU,
                 matrix_meta=extra_meta)
    return f"{status} it={iters} solve={solve:.3f}"


def _prep_amd(mid, tag, augment=False):
    """Dump one input variant + AMD-reorder it, cached by tag (the dump only feeds
    the reorder; a warm {mid}-{tag}[-aug]-amd.mtx + .time sidecar skips the
    re-dump). Returns (amd_path|None, reorder_seconds, prep_provenance)."""
    cache_tag = f"{tag}-aug" if augment else tag
    amd = f"{rc.PARAC_REORD}/{mid}-{cache_tag}-amd.mtx"
    if os.path.exists(amd) and os.path.exists(amd + ".time"):
        prep = (open(amd + ".prep").read() if os.path.exists(amd + ".prep")
                else _PREP_UNKNOWN)
        return amd, float(open(amd + ".time").read()), prep
    os.makedirs(DUMP_CPU, exist_ok=True)
    src = f"{DUMP_CPU}/{mid}-{tag}.mtx"
    if not os.path.exists(src):
        sh(f"{rc.BIN['cpu']} {rc.margs_for(mid)} --dump-mtx {src} --solver none",
           timeout=TIMEOUT_CPU, mem_cap_gb=MEM_CAP_GB)
    if not os.path.exists(src):
        return None, 0.0, ""
    return _reorder_amd(mid, src, tag, augment=augment)


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


def _measure_cpu_graph_split(family, mid):
    """ParAC GRAPH mode over the connected-component split, into one `parac` cell.

    ParAC generates a globally zero-sum RHS, which is consistent for a connected
    singular Laplacian but NOT for a disconnected one (solvability needs one
    constraint per component). So each component with >= COMP_THRESHOLD nodes is
    dumped as its own PURE Laplacian (descending size), AMD-reordered, calibrated
    and run REPS times. rank 0 IS the whole matrix when it is connected, which is
    the common case; below the threshold the components are singleton/pair specks
    with negligible cost. setup/solve are SUMMED (the components are solved
    sequentially), iters/rel_res are the worst (MAX) over components."""
    if rc.cell_done(family, mid, "parac", "", THREADS, "cpu", terminal=TERMINAL_CPU):
        return "skip(done)"
    setup = solve = factor_tot = prep_tot = amds_tot = 0.0
    iters = 0; rr = 0.0; n_tot = nnz_tot = 0; rss_peak = 0.0
    n_solved = 0; n_comps_total = None; rank = 0; tol_used = None; preps = []
    try:
        while True:
            src, n_nodes, n_comps = _dump_component(mid, rank)
            if n_comps_total is None and n_comps:
                n_comps_total = n_comps
            if src is None:                  # rank >= n_comps: no more components
                break
            if n_nodes < COMP_THRESHOLD:     # sorted descending -> the rest are smaller too
                break
            amd, amds, prep_prov = _reorder_amd(mid, src, f"comp{rank}")
            if not amd:
                rank += 1
                continue
            if prep_prov not in preps:
                preps.append(prep_prov)
            rel_tol = _calibrate_rel_tol(amd, False)
            if rel_tol is not None:
                tol_used = rel_tol if tol_used is None else max(tol_used, rel_tol)
            runs = [_run_once_cpu(amd, False, rel_tol=rel_tol) for _ in range(REPS)]
            ok = [r for r in runs if r["factor"] and r["solve"] and r["iters"]]
            if not ok:
                rc.emit_cell(family, mid, "parac", "", "failed", {},
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
        rc.emit_cell(family, mid, "parac", "", "timeout", {}, THREADS, "cpu", PROV_CPU)
        return "TIMEOUT"
    if n_solved == 0 or nnz_tot == 0:
        rc.emit_cell(family, mid, "parac", "", "failed", {}, THREADS, "cpu", PROV_CPU)
        return "FAILED"
    total = setup + solve
    status = "complete" if rr <= float(TOL) else "not_converged"
    metrics = {"n": n_tot, "nnz": nnz_tot, "setup_s": round(setup, 6),
               "solve_s": round(solve, 6), "total_s": round(total, 6), "iters": iters,
               "rel_res": rr, "fillin": 0.0, "us_per_nnz": round(total / nnz_tot * 1e6, 4),
               "amd_reorder_s": round(amds_tot, 6), "factor_s": round(factor_tot, 6),
               "prep_s": round(prep_tot, 6), "n_components_solved": n_solved,
               "n_components_total": n_comps_total or n_solved}
    if tol_used is not None: metrics["parac_rel_tol"] = tol_used
    if rss_peak: metrics["max_rss_mb"] = round(rss_peak, 1)
    rc.emit_cell(family, mid, "parac", "", status, metrics, THREADS, "cpu", PROV_CPU,
                 matrix_meta={"parac_mode": "graph",
                              "parac_input": "the PURE L = D - A, per connected "
                                             "component, AMD-reordered",
                              "parac_prep": "; ".join(preps),
                              "parac_grounding": "graph mode: ParAC's own zero-sum RHS "
                                                 "on the singular Laplacian (no pin)"})
    return f"{status} it={iters} solve={solve:.3f} comps={n_solved}/{n_comps_total}"


GRAPH_NA = ("ParAC graph mode grounds a singular Laplacian by generating a zero-sum "
            "RHS for it; this matrix is a published full-rank operator, which has no "
            "null space to ground and goes through ParAC's physics path instead")
PHYSICS_NA = ("ParAC physics mode grounds by trimming the last row/column — the ground "
              "node its own augmentation appended to a non-Laplacian. This matrix IS a "
              "Laplacian, so there is nothing appended to trim and the trim would delete "
              "a real vertex; it goes through ParAC's graph path instead")


def _run_cpu_operator(mid, family):
    """ParAC CPU pass for a kind=operator matrix: its OWN physics route.

    ParAC benchmarks published operators through `driver <mtx> <threads> "" 1` —
    physics mode — and its own producer for that input is write_graph.jl's
    `physics_produce`: permute, THEN append a ground row/column that turns the
    SDDM operator into a Laplacian. Physics mode's `remove_last_row_and_column`
    takes exactly that node back off, so what it solves is the published operator.
    We run THEIR physics_produce, out of their checkout (see _produce_upstream).

    (The claim that its data README says physics matrices are "used as-is" does
    not apply here: that sentence is about the inputs generated for the Hypre and
    AMGX baselines. ParAC's own input is the augmented one.)

    Skipping the augmentation is not a smaller deviation, it is a wrong answer:
    the trim then deletes a real degree of freedom. Measured against the published
    operator, apache2 scores 3.1e-3 while ParAC prints 8.8e-9, and G3_circuit does
    not converge at all.
    """
    rc.emit_cell(family, mid, "parac", "", "n/a", {}, THREADS, "cpu", PROV_CPU,
                 matrix_meta={"parac_mode": "n/a", "parac_na_reason": GRAPH_NA})
    try:
        amd, amds, prep_prov = _prep_amd(mid, "op", augment=True)
    except subprocess.TimeoutExpired:
        if not rc.cell_done(family, mid, "parac_physics", "", THREADS, "cpu", terminal=TERMINAL_CPU):
            rc.emit_cell(family, mid, "parac_physics", "", "timeout", {}, THREADS, "cpu", PROV_CPU)
        return "graph[n/a] physics[TIMEOUT(dump/reorder)]"
    if not amd:
        return "graph[n/a] physics[SKIP(dump/reorder)]"
    p = _measure_cpu(family, mid, amd, amds, True, "parac_physics",
                     extra_meta={"parac_mode": "physics",
                                 "parac_input": "the PUBLISHED operator, AMD-reordered then "
                                                "augmented with ParAC's ground row/column",
                                 "parac_prep": prep_prov,
                                 "parac_grounding": "physics mode trims the appended ground "
                                                    "node, leaving the published operator"})
    return f"graph[n/a] physics[{p}]"


def run_cpu(mid):
    """ParAC CPU pass for one matrix: one real cell and one n/a. Returns a status.

    kind=operator -> physics mode on the augmented published operator
    (_run_cpu_operator); its graph cell is n/a.
    kind=graph -> graph mode on the PURE L, per connected component
    (_measure_cpu_graph_split); its physics cell is n/a, because a pure Laplacian
    has no appended ground node for the trim to remove.
    """
    family = rc.MATRICES[mid]["family"]
    if rc.kind_of(mid) == "operator":
        return _run_cpu_operator(mid, family)
    rc.emit_cell(family, mid, "parac_physics", "", "n/a", {}, THREADS, "cpu", PROV_CPU,
                 matrix_meta={"parac_mode": "n/a", "parac_na_reason": PHYSICS_NA})
    g = _measure_cpu_graph_split(family, mid)
    return f"graph[{g}] physics[n/a]"


# ── GPU axis ────────────────────────────────────────────────────────────────────
def _nnz_sort(mid, src, tag, augment=False):
    """ParAC's OWN nnz-sort — write_graph.jl method "nnz-sort": a random
    permutation, THEN a per-column-nnz sort — run out of their checkout, the same
    `<mode>_produce` call the CPU axis makes with method "amd". Its printed "sort
    time" (compute only, excluding the reindex + mmwrite I/O) is the reorder cost
    we charge. Returns (path|None, seconds, prep_provenance).

    augment=True selects physics_produce, which appends ParAC's ground row/column
    after the permutation; driver_physics.cu's trim then removes it — the GPU
    counterpart of the CPU physics route. The tag carries '-aug' so the two can
    never alias. On rejection we fall back to benchmarks/parac_nnz_sort.jl and
    say so, exactly as _reorder_amd does."""
    os.makedirs(rc.PARAC_SORTED, exist_ok=True)
    if augment:
        tag = f"{tag}-aug"
    prefix = f"{rc.PARAC_SORTED}/{mid}-{tag}"
    out = prefix + "-nnz-sorted.mtx"
    tfile, pfile = out + ".time", out + ".prep"
    if os.path.exists(out) and os.path.exists(tfile):
        prep = (open(pfile).read() if os.path.exists(pfile)
                else _PREP_UNKNOWN)
        return out, float(open(tfile).read()), prep
    mode = "physics" if augment else "graph"
    produced, secs, err = _produce_upstream(prefix, src, mode, "nnz-sort", TIMEOUT_GPU)
    if produced:
        prep = f"ParAC write_graph.jl {mode}_produce(path, \"nnz-sort\"), upstream and unmodified"
    else:
        print(f"   [parac] upstream {mode}_produce refused {mid}: {err}\n"
              f"   [parac] falling back to benchmarks/parac_nnz_sort.jl", flush=True)
        flag = " --augment" if augment else ""
        o = sh(f"julia {NNZ_SORT_JL} {src} {out}{flag}", timeout=TIMEOUT_GPU).stdout
        m = _PREP_TIME_RE.search(o)
        secs = float(m.group(1)) if m else 0.0
        prep = f"benchmarks/parac_nnz_sort.jl (FALLBACK — upstream refused: {err})"
    if os.path.exists(out):
        open(tfile, "w").write(str(secs))
        open(pfile, "w").write(prep)
    return (out if os.path.exists(out) else None), secs, prep


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


def _calibrate_tol_gpu(driver, mtx, tau=float(TOL)):
    """The TOL to hand the CUDA driver so its own test stops at true residual tau.

    Its test is on the PRECONDITIONED relative residual sqrt(<r,M^-1 r>)/sqrt(
    <r0,M^-1 r0>), which is optimistic; but it is already RELATIVE and already a
    CLI argument (argv[4]), and the driver already prints the true
    ||Ax-b||/||b|| as `normalized diff norm`. So one probe rescales it, no patch:

        TOL = tau * TOL_0 / R_0

    Returns None if the probe did not print a residual (caller falls back to TOL).
    """
    p = _run_once_gpu(driver, mtx, PROBE_TOL_GPU)
    if not p["rr"]:
        return None
    r0 = float(p["rr"])
    return (tau * PROBE_TOL_GPU / r0) if r0 > 0.0 else None


def run_gpu(mid, tol=TOL):
    """ParAC GPU pass for one matrix: one real cell and one n/a, the same split as
    the CPU axis — kind=graph runs driver.cu on the pure L, kind=operator runs
    driver_physics.cu on the augmented published operator."""
    family = rc.MATRICES[mid]["family"]
    tag = _dump_tag(mid)
    is_operator = rc.kind_of(mid) == "operator"
    aug = _augments(mid)
    try:
        # Warm nnz-sort cache (+.time) -> the dump is unneeded; skip it.
        cache_tag = f"{tag}-aug" if aug else tag
        sorted_cached = f"{rc.PARAC_SORTED}/{mid}-{cache_tag}-nnz-sorted.mtx"
        if os.path.exists(sorted_cached) and os.path.exists(sorted_cached + ".time"):
            sorted_mtx = sorted_cached
            sort_s = float(open(sorted_cached + ".time").read())
            prep_prov = (open(sorted_cached + ".prep").read()
                         if os.path.exists(sorted_cached + ".prep")
                         else _PREP_UNKNOWN)
        else:
            src, _tag = _dump(mid, DUMP_GPU, rc.BIN["gpu"], TIMEOUT_GPU)
            if not src: return "SKIP(dump)"
            sorted_mtx, sort_s, prep_prov = _nnz_sort(mid, src, tag, augment=aug)
            if not sorted_mtx: return "SKIP(nnz-sort)"
    except subprocess.TimeoutExpired:
        return "TIMEOUT(dump/sort)"   # no cell: GPU terminal set retries anyway
    results = []
    for solver_key, driver in (("parac_graph", rc.PARAC_GPU_DRIVER),
                               ("parac_physics", rc.PARAC_GPU_DRIVER_PHYS)):
        skip_reason = (GRAPH_NA if (is_operator and solver_key == "parac_graph")
                       else PHYSICS_NA if (not is_operator and solver_key == "parac_physics")
                       else None)
        if skip_reason:
            rc.emit_cell(family, mid, solver_key, "", "n/a", {}, THREADS, "gpu", PROV_GPU,
                         matrix_meta={"parac_mode": "n/a", "parac_na_reason": skip_reason})
            results.append(f"{solver_key}[n/a]"); continue
        if rc.cell_done(family, mid, solver_key, "", THREADS, "gpu", terminal=TERMINAL_GPU):
            results.append(f"{solver_key}[skip(done)]"); continue
        if not os.path.exists(driver):
            results.append(f"{solver_key}[SKIP(driver missing)]"); continue
        try:
            cal = _calibrate_tol_gpu(driver, sorted_mtx) or float(tol)
            runs = [_run_once_gpu(driver, sorted_mtx, cal) for _ in range(REPS)]
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
        # rr is the driver's own `normalized diff norm` = ||Ax-b||/||b||. Judge it
        # at the tolerance we report at, not a decade looser.
        status = "complete" if rr <= float(tol) else "not_converged"
        metrics = {"n": int(ok[0]["n"]) if ok[0]["n"] else None,
                   "nnz": int(ok[0]["nnz"]) if ok[0]["nnz"] else None,
                   "setup_s": setup, "solve_s": solve, "total_s": total,
                   "iters": iters, "rel_res": rr, "parac_tol": cal}
        vram = [float(r["vram_mb"]) for r in ok if r.get("vram_mb")]
        if vram: metrics["max_vram_mb"] = round(max(vram), 1)   # peak VRAM over reps
        rc.emit_cell(family, mid, solver_key, "", status, metrics, THREADS, "gpu", PROV_GPU,
                     matrix_meta={"parac_prep": prep_prov})
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

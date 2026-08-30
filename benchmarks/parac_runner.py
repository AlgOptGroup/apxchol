#!/usr/bin/env python3
"""ParAC runner — importable module, called in-process by sweep_fair.py.

Consolidates the former standalone parac_fair.py (CPU) / parac_gpu.py (GPU)
on top of runner_common.

Each matrix goes through exactly ONE ParAC mode: the one ParAC documents for it.
The other mode's cell is n/a, with the reason recorded.

PREPROCESSING IS THEIRS. Every input ParAC reads is prepared by ParAC's OWN
cpu_implementation/write_graph.jl, invoked out of their checkout through the thin
dispatcher benchmarks/parac_produce_upstream.jl. We charge that preprocessing to
ParAC's setup time, so it has to be their code
doing their work — not our reimplementation of it. Outputs land in OUR cache
directories under THEIR naming convention (`<prefix>.mtx` in, `<prefix>-amd.mtx`
/ `<prefix>-nnz-sorted.mtx` out), so their checkout is never written to.
benchmarks/parac_reorder_amd.jl and parac_nnz_sort.jl remain as the FALLBACK for
an input their producer rejects; a cell prepared by the fallback records that,
and the reason, in matrix_meta.parac_prep.

class=laplacian -> GRAPH mode (`driver <mtx> <threads> ""`).
  For a connected file-backed input, hand the original file directly to
  `graph_produce`: that producer itself constructs the PURE L = D - A, so an
  intermediate rewrite would be redundant. A lightweight --component-info pass
  checks the one-null-vector precondition without serializing another matrix.
  For generated or disconnected inputs, dump the PURE L = D - A one connected
  component at a time (--giant-dump --comp-rank R), then hand each component to
  their `graph_produce(prefix, "amd")`, and let ParAC generate its own zero-sum
  RHS. That RHS is consistent for a connected singular Laplacian, so ParAC solves
  the very L the benchmark reports on and its printed residual is against that L.
  We do NOT hand it a Dirichlet-pinned matrix: its RHS generator knows nothing
  about our pin, and the residual against the original L then floors around 1e-3
  no matter the tolerance (measured; see benchmarks/patches/parac/README.md). Its
  physics cell is n/a — physics mode trims a row, which on a pure Laplacian
  deletes a real vertex. This includes both Laplacians assembled from
  `kind=graph` inputs and a published Laplacian operator such as `ecology1`.

  graph_produce strips the diagonal, forces the off-diagonals negative and
  REBUILDS the diagonal as -colsum, i.e. re-derives the pure Laplacian. On our
  component dumps — which already ARE pure Laplacians of a connected component —
  that rebuild is a no-op: verified byte-identical to a permutation-only reorder
  on com-Amazon (unweighted) and grid_1000 (weighted).

class=sddm -> PHYSICS mode (`driver <mtx> <threads> "" 1`).
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

CPU TOLERANCE. ParAC's CPU stopping test compares the residual NORM against
sqrt(rel_tol): an ABSOLUTE test, and on the recurrence residual, which runs
optimistic by a matrix-dependent factor. Rather than patch the test we calibrate
it from one probe run using ParAC's own two printed numbers (see
_calibrate_rel_tol) and pass the result through PARAC_REL_TOL. The achieved true
relative residual is ParAC's own `relative residual:` line and lands just under
TOL; both it and the calibrated tolerance are recorded in the cell.

CPU (`parac` / `parac_physics`, device=cpu):
  dump -> their write_graph.jl producer, method "amd" (cached with versioned
  timing/provenance sidecars) -> calibrate -> REPS runs of the driver. setup =
  complete producer call + complete post-parse adapter interval + complete factor interval; one
  real median-total repetition supplies all reported fields. One logical-cell
  deadline covers every stage and component; peak host RSS via /usr/bin/time.

GPU (`parac_graph` / `parac_physics`, device=gpu):
  dump -> their write_graph.jl producer, method "nnz-sort" (their random
  permutation THEN degree sort; the random step is ESSENTIAL — a deterministic
  degree-sort makes the level-set SpTRSV ~1000x slower, and physics_produce
  appends the ground node after it for an operator) -> the two CUDA drivers
  (driver.cu / driver_physics.cu), REPS medians. Those take the tolerance on argv
  already. Patch 0003 makes their inconsistent first/subsequent stopping tests one
  standard relative recurrence-residual test; one probe still calibrates recurrence
  residual to the independently printed true residual.
  setup = complete producer + complete post-parse adapter/factor/solver-setup intervals;
  solve includes RHS work, PCG, and returning x to host; peak VRAM via the
  nvidia-smi sidecar. Patch 0004 reports the once-per-process CUDA context
  initialization separately as cuda_init_s, matching the shared C++ driver.
  The graph row is n/a on disconnected inputs: the upstream GPU driver has no
  component-wise RHS route, while one global zero-sum constraint is insufficient.

The ParAC checkout itself is upstream 44ef39d plus the four benchmark-only
patches under benchmarks/patches/parac/; CMake applies the stack automatically,
and benchmarks/parac_build.sh verifies it for an external checkout.

Resume semantics differ BY DESIGN: CPU treats failed/timeout as terminal;
GPU retries them (transient CUDA hiccups). Unlike the old standalone runners,
a dump/reorder timeout no longer crashes the pass. Both axes emit terminal cells
for the campaign audit; GPU keeps failed/timeout outside TERMINAL_GPU so a later
resume still retries the transient preparation.
"""
import os, re, subprocess, time

import runner_common as rc
from runner_common import ROOT, sh

TOL = "1e-8"
REPS = 3
THREADS = 16
MAX_ITER = 2000               # ParAC's default is 1000; patch 0001 lets us raise it
PROBE_REL_TOL = 1e-7          # ParAC's own default; the CPU calibration probe runs at it
PROBE_TOL_GPU = 1e-7          # the CUDA drivers' tolerance is already argv[4]
BLOCKS = 512                  # GPU driver block count
# Per-step (dump/reorder/driver) wall caps. The >=1e8-nnz giants need more
# than 1200s for the AMD reorder step -> override per run.
TIMEOUT_CPU = int(os.environ.get("PARAC_TIMEOUT_S", "1200"))
TIMEOUT_GPU = int(os.environ.get("PARAC_GPU_TIMEOUT_S", "600"))
# CPU host-memory cap (GB): the giants' reorder/driver can balloon; die with
# bad_alloc instead of OOM-killing the box. GPU OOM is VRAM -> no host cap.
MEM_CAP_GB = float(os.environ.get("PARAC_MEM_CAP_GB", "100"))

DUMP_CPU = rc.external_path(
    "APXCHOL_PARAC_CPU_DUMP_DIR", "PARAC_DUMP_CPU", "/tmp/parac_fair_dump")
DUMP_GPU = rc.external_path(
    "APXCHOL_PARAC_GPU_DUMP_DIR", "PARAC_DUMP_GPU", "/tmp/parac_gpu_dump")

# THEIR preprocessing, run from THEIR checkout. parac_produce_upstream.jl is a
# dispatcher in OUR repo that includes ParAC's cpu_implementation/write_graph.jl
# and calls physics_produce / graph_produce on it; every line of preprocessing is
# theirs. It runs under benchmarks/julia (the project that carries AMD, Metis and
# Laplacians — instantiate with
#   julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()').
PRODUCE_JL = f"{ROOT}/benchmarks/parac_produce_upstream.jl"
JULIA_PROJECT = rc.external_path(
    "APXCHOL_PARAC_JULIA_PROJECT", "PARAC_JULIA_PROJECT",
    f"{ROOT}/benchmarks/julia")

# OUR reimplementations, kept as the FALLBACK for an input their producer rejects.
# These used to live untracked inside the ParAC checkout; they are our harness's
# business, not a modification of ParAC, so they are set AFTER paths_local — a
# machine-local file must not be able to point them back at a copy in someone's
# ParAC tree. Drop any REORDER_JL line from paths_local.py.
REORDER_JL = f"{ROOT}/benchmarks/parac_reorder_amd.jl"
NNZ_SORT_JL = f"{ROOT}/benchmarks/parac_nnz_sort.jl"

TERMINAL_CPU = frozenset({"complete", "not_converged", "failed", "timeout"})
TERMINAL_GPU = frozenset({"complete", "not_converged"})   # failed/timeout retry


def _representative_run(runs, elapsed):
    """Return (zero-based index, run) for the median-total real repetition."""
    if not runs:
        raise ValueError("cannot select a representative run from an empty sequence")
    ranked = sorted(enumerate(runs), key=lambda item: (elapsed(item[1]), item[0]))
    return ranked[len(ranked) // 2]


def _cell_remaining(deadline, cap, label):
    """Remaining seconds in one logical cell, or its exact-cap timeout."""
    if deadline is None:
        return cap
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise subprocess.TimeoutExpired(label, cap)
    return remaining


def _cpu_cell_remaining(deadline):
    return _cell_remaining(deadline, TIMEOUT_CPU, "ParAC CPU cell")


def _gpu_cell_remaining(deadline):
    return _cell_remaining(deadline, TIMEOUT_GPU, "ParAC GPU cell")

PROV_CPU = {"boost": "on", "boost_expected": "on", "git_sha": rc.git_sha(),
            "source_id": os.environ.get("APXCHOL_BENCH_SOURCE_ID", ""),
            "note": "ParAC at upstream 44ef39d + benchmarks/patches/parac/0001-0002 "
                    "(thread-count gate, configurable tolerance, and complete setup timing; "
                    "no numerics touched). kind=graph -> GRAPH "
                    "mode on the PURE L per connected component, ParAC's own zero-sum RHS; "
                    "kind=operator -> PHYSICS mode on the PUBLISHED operator augmented ParAC's "
                    "own way (permute, then append the ground row/column the trim removes). "
                    "Input prepared by ParAC's OWN cpu_implementation/write_graph.jl "
                    "(graph_produce / physics_produce, method 'amd'), not by a reimplementation. "
                    "Tolerance calibrated from a probe run so ParAC's own printed "
                    "relative residual lands under 1e-8. AMD-reordered, MKL serial",
            "repeat": REPS, "tier": "broad",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # The toolchain behind these numbers, read off the ParAC CPU driver
            # ITSELF (ELF .comment + DT_NEEDED) rather than copied from
            # benchmarks/parac_build.sh. That script hardcodes `g++`, which names
            # no version and cannot tell us whether the driver in place today even
            # came from it — the same reason our own binary self-reports BUILD_META.
            # These describe the timed driver; the AMD-reorder prep charged to its
            # setup is julia, as the note above says.
            **rc.binary_toolchain(rc.PARAC_CPU_DRIVER)}
# The GPU cells' toolchain is added per driver in run_gpu (two drivers, and both
# are OURS to build — build-cuda/gpu_rchol_gpu_driver{,_physics}).
PROV_GPU = {"source": "parac_runner.py", "git_sha": PROV_CPU["git_sha"],
            "source_id": PROV_CPU["source_id"],
            "repeat": REPS, "tier": "broad",
            "note": "ParAC upstream 44ef39d + benchmark-only patches 0003-0004: consistent "
                    "relative recurrence-residual stopping test plus complete "
                    "post-parse adapter, factor setup, solver setup, and per-RHS timing; "
                    "the per-RHS interval includes returning x to host; process-wide "
                    "CUDA initialization is reported separately as cuda_init_s."}


def _cpu_provenance():
    """CPU-cell provenance resolved after the campaign sets THREADS/REPS."""
    return {**PROV_CPU, **rc.benchmark_openmp_provenance(THREADS),
            "repeat": REPS}

_g = lambda pat, o: (re.search(pat, o).group(1) if re.search(pat, o) else None)


# ── shared: regularized/pure dump ───────────────────────────────────────────────
def _uses_physics(mid):
    """Whether ParAC must use its SDDM/physics route for this matrix.

    Routing follows the operator class, not the file kind. A published singular
    Laplacian such as ecology1 belongs on graph mode just like an assembled graph
    Laplacian; only a full-rank SDDM needs ground-node augmentation followed by
    the physics driver's matching trim.
    """
    return rc.class_of(mid) == "sddm"


def _dump_tag(mid):
    """The dump variant for this matrix.

    class=sddm -> 'op': the PUBLISHED operator, dumped untouched, which then gets
    ParAC's own ground-node augmentation at the reorder step (--augment).
    class=laplacian -> 'pure': the published or assembled singular Laplacian,
    NOT Dirichlet-pinned. ParAC's graph mode grounds it by generating a zero-sum
    RHS; a pin it does not know about is what used to floor its residual against
    the original L at ~1e-3.

    Knowable without dumping (registry class), which lets the GPU path tag its
    cache before the dump.
    """
    return "op" if _uses_physics(mid) else "pure"


def _augments(mid):
    """True when the reorder step must append ParAC's ground row/column. Only a
    FULL-RANK operator needs it, and physics mode's trim is what takes the appended
    node back off. Keyed on the declared CLASS, not on kind: ecology1 is
    kind=operator and class=laplacian — already singular, so appending a ground row
    would be wrong. kind says how to READ the file, class says whether the system it
    defines is singular; only the second one answers this question."""
    return _uses_physics(mid)


_DUMP_TIME_RE = re.compile(r"APX dump setup time:\s*([0-9.eE+-]+)")
_COMPONENT_DISCOVERY_RE = re.compile(
    r"APX component discovery time:\s*([0-9.eE+-]+)")
_COMPONENT_SERIALIZATION_RE = re.compile(
    r"APX component serialization time:\s*([0-9.eE+-]+)")
_DUMP_TIMING_SCHEMA = "solver-input-dump-v2"
_COMPONENT_INFO_RE = re.compile(
    r"\[component-info\]\s+(\d+)\s+components:\s+largest\s+(\d+)\s*/")
_COMPONENT_INFO_SCHEMA = "component-info-v1"


def _dump_cache_valid(path, require_meta=False):
    required = [path, path + ".dump-time", path + ".dump-schema"]
    if require_meta:
        required.append(path + ".meta")
    if not all(os.path.exists(item) for item in required):
        return False
    with open(path + ".dump-schema") as handle:
        return handle.read().strip() == _DUMP_TIMING_SCHEMA


def _stamp_dump_cache(path, seconds):
    with open(path + ".dump-time", "w") as handle:
        handle.write(str(seconds))
    with open(path + ".dump-schema", "w") as handle:
        handle.write(_DUMP_TIMING_SCHEMA)


def _dump_cache_seconds(path):
    with open(path + ".dump-time") as handle:
        return float(handle.read())


def _invalidate_dump_cache(path):
    for cached in (path, path + ".meta", path + ".dump-time", path + ".dump-schema"):
        try:
            os.remove(cached)
        except FileNotFoundError:
            pass


def _dump(mid, dump_dir, bin_path, timeout, mem_cap_gb=None):
    """Dump the matrix the drivers read: the published operator for kind=operator,
    the pure L = D - A for kind=graph. Returns (path|None, tag, adapter_seconds).
    Common matrix parsing/assembly stays outside the printed adapter timer; component
    extraction and Matrix Market serialization are charged. mem_cap_gb is for the
    CPU axis only — CUDA reserves large host VM, so the GPU axis passes None."""
    tag = _dump_tag(mid)
    os.makedirs(dump_dir, exist_ok=True)
    p = f"{dump_dir}/{mid}-{tag}.mtx"
    if _dump_cache_valid(p):
        return p, tag, _dump_cache_seconds(p)
    _invalidate_dump_cache(p)
    cp = sh(f"{bin_path} {rc.margs_for(mid)} --dump-mtx {p} --solver none",
            timeout=timeout, env=rc.benchmark_openmp_env(THREADS),
            mem_cap_gb=mem_cap_gb)
    match = _DUMP_TIME_RE.search(cp.stderr or "")
    if not (os.path.exists(p) and match):
        _invalidate_dump_cache(p)
        return None, tag, 0.0
    seconds = float(match.group(1))
    _stamp_dump_cache(p, seconds)
    return p, tag, seconds


def _native_mtx(mid):
    """Return the published file when ParAC's own producer can read it directly."""
    matrix = rc.MATRICES[mid]
    return matrix["spec"] if matrix["source"] == "mtx" else None


def _file_identity(path):
    """Cheap cache identity for a concrete Matrix Market input.

    Derived matrices can be hundreds of gigabytes, so a cache lookup must not
    hash the whole file.  Canonical path, byte size and nanosecond mtime catch
    route changes and ordinary replacements while keeping validation O(1).
    """
    stat = os.stat(path)
    return f"{os.path.realpath(path)}\n{stat.st_size}\n{stat.st_mtime_ns}"


def _component_info(mid, dump_dir, bin_path, timeout, mem_cap_gb=None):
    """Return (component_count, largest_size, discovery_seconds), cached.

    This intentionally performs no Matrix Market serialization.  The check is
    required only to decide whether ParAC's one-zero-sum-RHS graph driver may
    consume a file directly or whether the CPU route must split it.
    """
    os.makedirs(dump_dir, exist_ok=True)
    base = f"{dump_dir}/{mid}.components"
    meta, schema, input_id = base + ".meta", base + ".schema", base + ".input"
    expected_input = _file_identity(_native_mtx(mid))
    if os.path.exists(meta) and os.path.exists(schema) and os.path.exists(input_id):
        with open(schema) as handle:
            valid = handle.read().strip() == _COMPONENT_INFO_SCHEMA
        with open(input_id) as handle:
            valid = valid and handle.read() == expected_input
        if valid:
            with open(meta) as handle:
                count, largest, seconds = handle.read().split()
            return int(count), int(largest), float(seconds)
    for path in (meta, schema, input_id):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    cp = sh(f"{bin_path} {rc.margs_for(mid)} --component-info --solver none",
            timeout=timeout, env=rc.benchmark_openmp_env(THREADS),
            mem_cap_gb=mem_cap_gb)
    info = _COMPONENT_INFO_RE.search(cp.stderr or "")
    discovery = _COMPONENT_DISCOVERY_RE.search(cp.stderr or "")
    if cp.returncode != 0 or not (info and discovery):
        return 0, 0, 0.0
    count, largest = int(info.group(1)), int(info.group(2))
    seconds = float(discovery.group(1))
    with open(meta, "w") as handle:
        handle.write(f"{count} {largest} {seconds}")
    with open(input_id, "w") as handle:
        handle.write(expected_input)
    with open(schema, "w") as handle:
        handle.write(_COMPONENT_INFO_SCHEMA)
    return count, largest, seconds


# ── ParAC's OWN preprocessing (their write_graph.jl) ────────────────────────────
# Their native "amd time:" / "sort time:" covers only the ordering kernel. Our
# dispatcher brackets the complete producer call after Julia/package loading so
# setup also includes permutation materialization, augmentation and file output.
_PREP_TOTAL_RE = re.compile(r"APX complete preprocessing time:\s*([0-9.eE+-]+)")
_PREP_SUFFIX = {"amd": "-amd.mtx", "nnz-sort": "-nnz-sorted.mtx"}
_PREP_TIMING_SCHEMA = "complete-producer-no-jit-v4-input-route"
# A cache entry written before the .prep sidecar existed came from our
# reimplementation. Say so rather than claiming provenance we cannot check —
# delete the cached *-amd.mtx to have their producer rebuild (and stamp) it.
_PREP_UNKNOWN = ("unrecorded: this cache entry predates the .prep sidecar, so it was "
                 "written by benchmarks/parac_reorder_amd.jl / parac_nnz_sort.jl. Those "
                 "were byte-identical to ParAC's own producer on every matrix compared "
                 "(see benchmarks/patches/parac/README.md), but this particular file was "
                 "not checked; delete it to have their producer rebuild it")


def _prep_cache_valid(path, src=None):
    """True only for a cache whose time is the complete producer interval.

    Historical ``.time`` files used the producer's narrow AMD/sort-kernel print.
    The schema rejects that boundary; the optional input sidecar additionally
    prevents a native-file result from aliasing a component-dump result.
    """
    schema = path + ".timing-schema"
    required = [path, path + ".time", path + ".prep", schema]
    if src is not None:
        required.append(path + ".input")
    if not all(os.path.exists(item) for item in required):
        return False
    with open(schema) as handle:
        valid = handle.read().strip() == _PREP_TIMING_SCHEMA
    if valid and src is not None:
        with open(path + ".input") as handle:
            valid = handle.read() == _file_identity(src)
    return valid


def _stamp_prep_cache(path, seconds, provenance, src=None):
    with open(path + ".time", "w") as handle:
        handle.write(str(seconds))
    with open(path + ".prep", "w") as handle:
        handle.write(provenance)
    if src is not None:
        with open(path + ".input", "w") as handle:
            handle.write(_file_identity(src))
    # Commit marker last: a partial record is never accepted as a warm cache.
    with open(path + ".timing-schema", "w") as handle:
        handle.write(_PREP_TIMING_SCHEMA)


def _invalidate_prep_cache(path):
    """Remove one derived cache entry whose timing contract is obsolete."""
    for cached in (path, path + ".time", path + ".prep", path + ".timing-schema",
                   path + ".input"):
        try:
            os.remove(cached)
        except FileNotFoundError:
            pass


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
    inside OUR cache directory and point `<prefix>.mtx` at the selected input with a
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
    if os.path.exists(log):
        with open(log) as handle:
            text = handle.read()
    else:
        text = ""
    if cp.returncode != 0 or not os.path.exists(out):
        tail = " | ".join(l.strip() for l in text.strip().splitlines()[-4:]) or "no output"
        return None, 0.0, f"{mode}_produce(\"{method}\") rc={cp.returncode}: {tail}"
    m = _PREP_TOTAL_RE.search(text)
    if not m:
        return None, 0.0, (f"{mode}_produce(\"{method}\") omitted the complete "
                           "preprocessing timer; refusing the narrow ordering-kernel time")
    return out, float(m.group(1)), None


# ── CPU axis ────────────────────────────────────────────────────────────────────
def _reorder_amd(mid, src, tag, augment=False, deadline=None):
    """AMD reorder via ParAC's OWN write_graph.jl producer, cached. Returns
    (path|None, seconds, prep_provenance); versioned sidecars keep both available
    on a warm cache hit and invalidate the historical narrow-kernel timer.

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
    if not _prep_cache_valid(amd, src):
        # The matrix and its timing are one cache record. Keeping an old matrix
        # while replacing only its narrow timer would claim work we did not run;
        # drop the derived record and rebuild it as one schema-stamped record.
        _invalidate_prep_cache(amd)
        mode = "physics" if augment else "graph"
        out, secs, err = _produce_upstream(
            prefix, src, mode, "amd", _cpu_cell_remaining(deadline), MEM_CAP_GB)
        if out:
            prep = f"ParAC write_graph.jl {mode}_produce(path, \"amd\"), upstream and unmodified"
        else:
            print(f"   [parac] upstream {mode}_produce refused {mid}: {err}\n"
                  f"   [parac] falling back to benchmarks/parac_reorder_amd.jl", flush=True)
            secs = _reorder_amd_ours(mid, src, amd, tag, augment, deadline=deadline)
            prep = f"benchmarks/parac_reorder_amd.jl (FALLBACK — upstream refused: {err})"
        if os.path.exists(amd):
            _stamp_prep_cache(amd, secs, prep, src)
    if os.path.exists(tfile):
        with open(tfile) as handle:
            secs = float(handle.read())
    else:
        secs = 0.0
    if os.path.exists(pfile):
        with open(pfile) as handle:
            prep = handle.read()
    else:
        prep = _PREP_UNKNOWN
    return (amd if os.path.exists(amd) else None), secs, prep


def _reorder_amd_ours(mid, src, amd, tag, augment, deadline=None):
    """FALLBACK reorder with our own parac_reorder_amd.jl (see _reorder_amd).
    Runs on julia's default environment, so it stays usable even when the
    benchmarks/julia project their producer needs is not instantiated."""
    log = f"/tmp/parac_fair_reord_{mid}_{tag}.log"
    flag = " --augment" if augment else ""
    started = time.monotonic()
    sh(f"julia {REORDER_JL} {src} {amd}{flag} > {log} 2>&1",
       timeout=_cpu_cell_remaining(deadline), mem_cap_gb=MEM_CAP_GB)
    wall = time.monotonic() - started
    if os.path.exists(log):
        m = _PREP_TOTAL_RE.search(open(log).read())
        if m:
            return float(m.group(1))
    return wall


def _run_once_cpu(amd, physics, rel_tol=None, deadline=None):
    rc.require_path(rc.PARAC_CPU_DRIVER, "APXCHOL_PARAC_DRIVER", "PARAC_CPU_DRIVER",
                    "the ParAC CPU driver binary")
    env = rc.benchmark_openmp_env(
        THREADS,
        dict(os.environ, LD_LIBRARY_PATH=rc.PARAC_LDLIB, MKL_NUM_THREADS="1"))
    if rel_tol is not None:
        # patch 0001 forwards these to example_pcg_solver's existing parameters.
        env["PARAC_REL_TOL"] = repr(float(rel_tol))
        env["PARAC_MAX_ITER"] = str(MAX_ITER)
    # 4 args -> is_graph=1 (graph/Laplacian); a 5th arg -> is_graph=0 (physics/
    # SDDM: handles the diagonal excess). /usr/bin/time -f 'APXRSS %M' -> peak
    # host RSS on stderr, the ParAC analog of the C++ solvers' max_rss_mb (the
    # driver is one self-contained binary, so whole-driver peak is what we chart).
    arg5 = "1" if physics else ""
    cp = sh(f"/usr/bin/time -f 'APXRSS %M' {rc.taskset_prefix(THREADS)} "
            f"{rc.PARAC_CPU_DRIVER} {amd} {THREADS} \"\" {arg5}".strip(),
            timeout=_cpu_cell_remaining(deadline), env=env, mem_cap_gb=MEM_CAP_GB)
    o = cp.stdout
    rss_mb = None
    for ln in (cp.stderr or "").splitlines():
        if ln.startswith("APXRSS "):
            try: rss_mb = round(int(ln.split()[1]) / 1024.0, 1)
            except ValueError: pass
    # Patch 0002 adds two complete, non-overlapping setup intervals around work
    # performed by ParAC itself.  `adapter` starts after MatrixMarket parsing and
    # includes triplet normalization/CSC construction plus the etree schedule;
    # `factor_setup` includes factor data structures, elimination, and final CSR.
    # The narrower upstream timers remain diagnostics but must not be summed into
    # setup (doing so omits the work around them).
    return dict(factor=_g(r"Factorization execution time:\s*([0-9.]+)", o),
                factor_setup=_g(r"APX factor setup time:\s*([0-9.eE+-]+)", o),
                adapter=_g(r"APX adapter preprocessing time:\s*([0-9.eE+-]+)", o),
                solve=_g(r"Solve time taken:\s*([0-9]+)", o),
                etree=_g(r"build etree:\s*([0-9.eE+-]+)", o),
                ftree=_g(r"factorization tree:\s*([0-9.eE+-]+)", o),
                summary=_g(r"generate summary:\s*([0-9.eE+-]+)", o),
                iters=_g(r"Iterations:\s*([0-9]+)", o),
                rr=_g(r"relative residual:\s*([0-9.eE+-]+)", o),
                rhs_norm=_g(r"rhs norm:\s*([0-9.eE+-]+)", o),
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


def _calibrate_rel_tol(amd, physics, tau=float(TOL), deadline=None):
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
    p = _run_once_cpu(amd, physics, rel_tol=PROBE_REL_TOL, deadline=deadline)
    if not (p["recur"] and p["rr"]):
        return None
    recur0, r0 = float(p["recur"]), float(p["rr"])
    if not (recur0 > 0.0 and r0 > 0.0):
        return None
    return (tau * recur0 / r0) ** 2


def _measure_cpu(family, mid, amd, amds, physics, solver, extra_meta=None,
                 deadline=None, dump_s=0.0):
    """REPS runs of one driver mode (graph or physics), one cell. The probe run
    that calibrates the tolerance is NOT timed and NOT one of the REPS."""
    if rc.cell_done(family, mid, solver, "", THREADS, "cpu", terminal=TERMINAL_CPU):
        return "skip(done)"
    try:
        rel_tol = _calibrate_rel_tol(amd, physics, deadline=deadline)
        runs = [_run_once_cpu(amd, physics, rel_tol=rel_tol, deadline=deadline)
                for _ in range(REPS)]
    except subprocess.TimeoutExpired:
        rc.emit_cell(family, mid, solver, "", "timeout", {}, THREADS, "cpu", _cpu_provenance(),
                     matrix_meta=extra_meta, timeout_cap_s=TIMEOUT_CPU)
        return "TIMEOUT"
    ok = [r for r in runs if r["factor_setup"] and r["adapter"] and
          r["solve"] and r["iters"] and r["rr"] and r["n"] and r["nnz"]]
    if len(ok) != REPS:
        rc.emit_cell(family, mid, solver, "", "failed", {}, THREADS, "cpu", _cpu_provenance(),
                     matrix_meta=extra_meta)
        return "FAILED"
    rep_index, chosen = _representative_run(
        ok, lambda r: (float(r["adapter"]) + float(r["factor_setup"])
                       + float(r["solve"]) / 1000))
    factor = float(chosen["factor"] or 0)
    factor_setup = float(chosen["factor_setup"])
    adapter = float(chosen["adapter"])
    solve = float(chosen["solve"]) / 1000
    iters = int(chosen["iters"])
    rr = float(chosen["rr"])
    n = int(chosen["n"]); nnz = int(chosen["nnz"])
    setup = dump_s + amds + adapter + factor_setup
    total = setup + solve
    # rel_res is ParAC's own ||Ax-b||/||b|| against the operator it solved, which
    # the input construction makes the operator we report on. THE GRADING RULE
    # (benchmarks/README.md): exactly TOL, the same mark every other solver gets.
    # ParAC's optimistic absolute test is handled by CALIBRATING the tolerance we
    # pass it (_calibrate_rel_tol), never by relaxing this comparison.
    status = "complete" if rr <= float(TOL) else "not_converged"
    metrics = {"n": n, "nnz": nnz, "setup_s": round(setup, 6), "solve_s": round(solve, 6),
               "total_s": round(total, 6), "iters": iters, "rel_res": rr, "fillin": 0.0,
               "us_per_nnz": round(total / nnz * 1e6, 4),
               "input_dump_s": round(dump_s, 6),
               "amd_reorder_s": round(amds, 6),
               "adapter_setup_s": round(dump_s + amds + adapter, 6),
               "native_setup_s": round(factor_setup, 6),
               "factor_kernel_s": round(factor, 6),
               "representative_repeat": rep_index + 1}
    if rel_tol is not None:
        metrics["parac_rel_tol"] = rel_tol      # the calibrated value we passed
    rss = [float(r["rss_mb"]) for r in ok if r.get("rss_mb")]
    if rss: metrics["max_rss_mb"] = round(max(rss), 1)   # peak host RSS over reps
    rc.emit_cell(family, mid, solver, "", status, metrics, THREADS, "cpu", _cpu_provenance(),
                 matrix_meta=extra_meta)
    return f"{status} it={iters} solve={solve:.3f}"


def _prep_amd(mid, tag, augment=False, deadline=None):
    """Select the native input or dump a generated one, then AMD-reorder it.

    Returns (amd_path|None, reorder_seconds, prep_provenance, dump_seconds).
    The dump adapter and ParAC producer are distinct setup intervals, even on a
    warm cache, because a deployment must perform both once for a new matrix.
    """
    src = _native_mtx(mid)
    dump_s = 0.0
    source_tag = f"native-{tag}" if src is not None else tag
    if src is None:
        src, _dump_tag_value, dump_s = _dump(
            mid, DUMP_CPU, rc.BIN["cpu"], _cpu_cell_remaining(deadline), MEM_CAP_GB)
        if not src:
            return None, 0.0, "", dump_s
    amd, reorder_s, prep = _reorder_amd(
        mid, src, source_tag, augment=augment, deadline=deadline)
    return amd, reorder_s, prep, dump_s


def _dump_component(mid, rank, deadline=None, timeout_cap=TIMEOUT_CPU,
                    bin_path=None, mem_cap_gb=MEM_CAP_GB, dump_dir=None,
                    discovery_charge_s=None):
    """Dump the rank-th largest connected component's PURE Laplacian (relabeled
    0..cn-1) via --giant-dump --comp-rank. Returns
    (src_path|None, n_nodes, n_comps, adapter_seconds): a None path means rank >=
    n_comps. Versioned sidecars retain both metadata and the mandatory component
    extraction/serialization setup cost. rank 0 on a connected matrix is the
    full pure L."""
    target_dir = dump_dir or DUMP_CPU
    os.makedirs(target_dir, exist_ok=True)
    src = f"{target_dir}/{mid}-comp{rank}.mtx"
    meta = src + ".meta"
    if _dump_cache_valid(src, require_meta=True):
        with open(meta) as handle:
            n_nodes, n_comps = (int(x) for x in handle.read().split())
        return src, n_nodes, n_comps, _dump_cache_seconds(src)
    _invalidate_dump_cache(src)
    binary = bin_path or rc.BIN["cpu"]
    cp = sh(f"{binary} {rc.margs_for(mid)} --giant-dump --comp-rank {rank} "
            f"--dump-mtx {src} --solver none",
            timeout=_cell_remaining(deadline, timeout_cap, "ParAC component preparation"),
            env=rc.benchmark_openmp_env(THREADS), mem_cap_gb=mem_cap_gb)
    # stderr: "[component-dump] rank R / K components: N / M nodes" (group1=K, group2=N).
    # "nothing to dump" (rank >= K) doesn't match -> (None, 0, 0).
    m = re.search(r"rank\s+\d+\s*/\s*(\d+)\s+components:\s*(\d+)\s*/", cp.stderr or "")
    discovery = _COMPONENT_DISCOVERY_RE.search(cp.stderr or "")
    serialization = _COMPONENT_SERIALIZATION_RE.search(cp.stderr or "")
    if not (m and discovery and serialization and os.path.exists(src)):
        _invalidate_dump_cache(src)
        return None, 0, 0, 0.0
    n_comps, n_nodes = int(m.group(1)), int(m.group(2))
    with open(meta, "w") as handle:
        handle.write(f"{n_nodes} {n_comps}")
    # Component discovery/sorting is shared setup for the split and belongs in
    # rank 0 exactly once. Every rank still pays its own extraction/serialization
    # adapter. The subprocess currently rediscovers components for later ranks,
    # but that is a harness implementation inefficiency, not solver-required work.
    seconds = float(serialization.group(1))
    if rank == 0:
        seconds += (float(discovery.group(1)) if discovery_charge_s is None
                    else float(discovery_charge_s))
    _stamp_dump_cache(src, seconds)
    return src, n_nodes, n_comps, seconds


def _measure_cpu_graph_split(family, mid, deadline=None):
    """ParAC GRAPH mode over the connected-component split, into one `parac` cell.

    ParAC generates a globally zero-sum RHS, which is consistent for a connected
    singular Laplacian but NOT for a disconnected one (solvability needs one
    constraint per component). So every non-singleton component is dumped as its
    own PURE Laplacian (descending size), AMD-reordered, calibrated and run REPS
    times. Singleton Laplacian blocks have b=0 and x=0 exactly and require no
    solve. Repetition i is aggregated across every component before selecting one
    median-total repetition. The global residual is weighted exactly as
    sqrt(sum ||r_c||^2) / sqrt(sum ||b_c||^2)."""
    if rc.cell_done(family, mid, "parac", "", THREADS, "cpu", terminal=TERMINAL_CPU):
        return "skip(done)"
    reps = [dict(adapter=0.0, factor_setup=0.0, factor=0.0, solve=0.0,
                 iters=0, residual_sq=0.0, rhs_sq=0.0) for _ in range(REPS)]
    dump_tot = amds_tot = 0.0
    nnz_tot = 0; rss_peak = 0.0
    n_solved = 0; n_comps_total = None; rank = 0; tol_used = None; preps = []
    try:
        native = _native_mtx(mid)
        direct_connected = False
        discovery_s = None
        largest = 0
        if native is not None:
            n_comps_total, largest, discovery_s = _component_info(
                mid, DUMP_CPU, rc.BIN["cpu"], _cpu_cell_remaining(deadline),
                MEM_CAP_GB)
            if n_comps_total <= 0:
                rc.emit_cell(
                    family, mid, "parac", "", "failed", {}, THREADS, "cpu",
                    _cpu_provenance(),
                    matrix_meta={"parac_mode": "graph",
                                 "parac_prep_failure":
                                     "component-info probe produced no result"})
                return "FAILED"
            direct_connected = n_comps_total == 1
        while n_comps_total is None or rank < n_comps_total:
            if direct_connected and rank == 0:
                src, n_nodes, n_comps, dump_s = (
                    native, largest, 1, float(discovery_s))
            else:
                src, n_nodes, n_comps, dump_s = _dump_component(
                    mid, rank, deadline=deadline,
                    discovery_charge_s=(discovery_s if rank == 0 else None))
            if n_comps_total is None and n_comps:
                n_comps_total = n_comps
            if src is None:                  # rank >= n_comps: no more components
                break
            # The adapter already found/extracted/serialized this component.
            # Charge that real work even when the descending size order tells us
            # this and all later components are singleton zero blocks.
            dump_tot += dump_s
            if n_nodes < 2:                  # singleton blocks have b=x=0 exactly
                break
            source_tag = f"{'native-' if direct_connected else ''}comp{rank}"
            amd, amds, prep_prov = _reorder_amd(
                mid, src, source_tag, deadline=deadline)
            if not amd:
                rc.emit_cell(
                    family, mid, "parac", "", "failed", {}, THREADS, "cpu",
                    _cpu_provenance(),
                    matrix_meta={"parac_mode": "graph",
                                 "parac_prep_failure":
                                     f"AMD preprocessing failed for component {rank}",
                                 "parac_prep": prep_prov})
                return "FAILED"
            if prep_prov not in preps:
                preps.append(prep_prov)
            rel_tol = _calibrate_rel_tol(amd, False, deadline=deadline)
            if rel_tol is not None:
                tol_used = rel_tol if tol_used is None else max(tol_used, rel_tol)
            runs = [_run_once_cpu(amd, False, rel_tol=rel_tol, deadline=deadline)
                    for _ in range(REPS)]
            ok = [r for r in runs if r["factor_setup"] and r["adapter"] and
                  r["solve"] and r["iters"] and r["rr"] and r["rhs_norm"]]
            if len(ok) != REPS:
                rc.emit_cell(family, mid, "parac", "", "failed", {},
                             THREADS, "cpu", _cpu_provenance())
                return "FAILED"
            for rep, run in zip(reps, runs):
                rhs_norm = float(run["rhs_norm"])
                abs_residual = float(run["rr"]) * rhs_norm
                rep["adapter"] += float(run["adapter"])
                rep["factor_setup"] += float(run["factor_setup"])
                rep["factor"] += float(run["factor"] or 0)
                rep["solve"] += float(run["solve"]) / 1000
                rep["iters"] = max(rep["iters"], int(run["iters"]))
                rep["residual_sq"] += abs_residual * abs_residual
                rep["rhs_sq"] += rhs_norm * rhs_norm
            amds_tot += amds
            nnz_tot += int(runs[0]["nnz"])
            rss = [float(r["rss_mb"]) for r in ok if r.get("rss_mb")]
            if rss: rss_peak = max(rss_peak, max(rss))
            n_solved += 1; rank += 1
    except subprocess.TimeoutExpired:
        rc.emit_cell(family, mid, "parac", "", "timeout", {}, THREADS, "cpu", _cpu_provenance(),
                     timeout_cap_s=TIMEOUT_CPU)
        return "TIMEOUT"
    if n_solved == 0 or nnz_tot == 0:
        rc.emit_cell(family, mid, "parac", "", "failed", {}, THREADS, "cpu", _cpu_provenance())
        return "FAILED"
    rep_index, chosen = _representative_run(
        reps, lambda rep: (dump_tot + amds_tot + rep["adapter"]
                           + rep["factor_setup"] + rep["solve"]))
    setup = dump_tot + amds_tot + chosen["adapter"] + chosen["factor_setup"]
    solve = chosen["solve"]
    total = setup + solve
    if chosen["rhs_sq"] <= 0.0:
        rc.emit_cell(family, mid, "parac", "", "failed", {},
                     THREADS, "cpu", _cpu_provenance())
        return "FAILED"
    rr = (chosen["residual_sq"] / chosen["rhs_sq"]) ** 0.5
    iters = chosen["iters"]
    status = "complete" if rr <= float(TOL) else "not_converged"
    # Singleton zero blocks contribute vertices but no stored entries or work.
    n_report = int(rc.MATRICES[mid]["n"])
    metrics = {"n": n_report, "nnz": nnz_tot, "setup_s": round(setup, 6),
               "solve_s": round(solve, 6), "total_s": round(total, 6), "iters": iters,
               "rel_res": rr, "fillin": 0.0, "us_per_nnz": round(total / nnz_tot * 1e6, 4),
               "input_dump_s": round(dump_tot, 6),
               "amd_reorder_s": round(amds_tot, 6),
               "adapter_setup_s": round(dump_tot + amds_tot + chosen["adapter"], 6),
               "native_setup_s": round(chosen["factor_setup"], 6),
               "factor_kernel_s": round(chosen["factor"], 6),
               "rhs_norm": chosen["rhs_sq"] ** 0.5,
               "representative_repeat": rep_index + 1,
               "n_components_solved": n_solved,
               "n_components_total": n_comps_total or n_solved}
    if tol_used is not None: metrics["parac_rel_tol"] = tol_used
    if rss_peak: metrics["max_rss_mb"] = round(rss_peak, 1)
    rc.emit_cell(family, mid, "parac", "", status, metrics, THREADS, "cpu", _cpu_provenance(),
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


def _run_cpu_operator(mid, family, deadline=None):
    """ParAC CPU pass for a class=sddm matrix: its OWN physics route.

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
    rc.emit_cell(family, mid, "parac", "", "n/a", {}, THREADS, "cpu", _cpu_provenance(),
                 matrix_meta={"parac_mode": "n/a", "parac_na_reason": GRAPH_NA})
    try:
        amd, amds, prep_prov, dump_s = _prep_amd(
            mid, "op", augment=True, deadline=deadline)
    except subprocess.TimeoutExpired:
        if not rc.cell_done(family, mid, "parac_physics", "", THREADS, "cpu", terminal=TERMINAL_CPU):
            rc.emit_cell(family, mid, "parac_physics", "", "timeout", {}, THREADS, "cpu", _cpu_provenance(),
                         timeout_cap_s=TIMEOUT_CPU)
        return "graph[n/a] physics[TIMEOUT(dump/reorder)]"
    if not amd:
        rc.emit_cell(
            family, mid, "parac_physics", "", "failed", {}, THREADS, "cpu",
            _cpu_provenance(),
            matrix_meta={"parac_mode": "physics",
                         "parac_prep_failure": "dump/reorder produced no matrix",
                         "parac_prep": prep_prov})
        return "graph[n/a] physics[FAILED(dump/reorder)]"
    p = _measure_cpu(family, mid, amd, amds, True, "parac_physics",
                     extra_meta={"parac_mode": "physics",
                                 "parac_input": "the PUBLISHED operator, AMD-reordered then "
                                                "augmented with ParAC's ground row/column",
                                 "parac_prep": prep_prov,
                                 "parac_grounding": "physics mode trims the appended ground "
                                                    "node, leaving the published operator"},
                     deadline=deadline, dump_s=dump_s)
    return f"graph[n/a] physics[{p}]"


def run_cpu(mid):
    """ParAC CPU pass for one matrix: one real cell and one n/a. Returns a status.

    class=sddm -> physics mode on the augmented published operator
    (_run_cpu_operator); its graph cell is n/a.
    class=laplacian -> graph mode on the PURE L, per connected component
    (_measure_cpu_graph_split); its physics cell is n/a, because a singular
    Laplacian has no appended ground node for the trim to remove.
    """
    family = rc.MATRICES[mid]["family"]
    deadline = time.monotonic() + TIMEOUT_CPU
    if _uses_physics(mid):
        return _run_cpu_operator(mid, family, deadline=deadline)
    rc.emit_cell(family, mid, "parac_physics", "", "n/a", {}, THREADS, "cpu", _cpu_provenance(),
                 matrix_meta={"parac_mode": "n/a", "parac_na_reason": PHYSICS_NA})
    g = _measure_cpu_graph_split(family, mid, deadline=deadline)
    return f"graph[{g}] physics[n/a]"


# ── GPU axis ────────────────────────────────────────────────────────────────────
def _nnz_sort(mid, src, tag, augment=False, deadline=None):
    """ParAC's OWN nnz-sort — write_graph.jl method "nnz-sort": a random
    permutation, THEN a per-column-nnz sort — run out of their checkout, the same
    `<mode>_produce` call the CPU axis makes with method "amd". We charge the
    complete producer call, not its narrow sort-kernel diagnostic. Returns
    (path|None, seconds, prep_provenance).

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
    if _prep_cache_valid(out, src):
        with open(tfile) as handle:
            secs = float(handle.read())
        if os.path.exists(pfile):
            with open(pfile) as handle:
                prep = handle.read()
        else:
            prep = _PREP_UNKNOWN
        return out, secs, prep
    _invalidate_prep_cache(out)
    mode = "physics" if augment else "graph"
    timeout = (_gpu_cell_remaining(deadline) if deadline is not None else TIMEOUT_GPU)
    produced, secs, err = _produce_upstream(prefix, src, mode, "nnz-sort", timeout)
    if produced:
        prep = f"ParAC write_graph.jl {mode}_produce(path, \"nnz-sort\"), upstream and unmodified"
    else:
        print(f"   [parac] upstream {mode}_produce refused {mid}: {err}\n"
              f"   [parac] falling back to benchmarks/parac_nnz_sort.jl", flush=True)
        flag = " --augment" if augment else ""
        timeout = (_gpu_cell_remaining(deadline) if deadline is not None else TIMEOUT_GPU)
        started = time.monotonic()
        cp = sh(f"julia {NNZ_SORT_JL} {src} {out}{flag}", timeout=timeout)
        wall = time.monotonic() - started
        o = cp.stdout
        m = _PREP_TOTAL_RE.search(o)
        secs = float(m.group(1)) if m else wall
        if cp.returncode == 0 and os.path.exists(out):
            prep = f"benchmarks/parac_nnz_sort.jl (FALLBACK — upstream refused: {err})"
        else:
            tail = " | ".join(
                line.strip() for line in (cp.stderr or cp.stdout or "").splitlines()[-4:]
            ) or "no output"
            prep = (f"benchmarks/parac_nnz_sort.jl FAILED (rc={cp.returncode}, "
                    f"output={'present' if os.path.exists(out) else 'missing'}): {tail}; "
                    f"upstream refused: {err}")
            return None, secs, prep
    if os.path.exists(out):
        _stamp_prep_cache(out, secs, prep, src)
    return (out if os.path.exists(out) else None), secs, prep


def _run_once_gpu(driver, mtx, tol, deadline=None):
    timeout = (_gpu_cell_remaining(deadline) if deadline is not None else TIMEOUT_GPU)
    with rc.VramSampler("gpu_rchol") as vram:   # matches both driver binaries
        o = sh(f"{driver} {mtx} {BLOCKS} 1 {tol}", timeout=timeout).stdout
    # Keep the original narrow timers as diagnostics alongside patch 0003's
    # complete, non-overlapping phases.
    return dict(etree=_g(r"build etree:\s*([0-9.eE+-]+)", o),
                ftree=_g(r"factorization tree:\s*([0-9.eE+-]+)", o),
                summary=_g(r"generate summary:\s*([0-9.eE+-]+)", o),
                cuda_init=_g(r"APX CUDA init time:\s*([0-9.eE+-]+)", o),
                adapter=_g(r"APX adapter preprocessing time:\s*([0-9.eE+-]+)", o),
                factor_setup=_g(r"APX GPU factor setup time:\s*([0-9.eE+-]+)", o),
                solver_setup=_g(r"APX GPU solver setup time:\s*([0-9.eE+-]+)", o),
                solve_total=_g(r"APX GPU solve phase time:\s*([0-9.eE+-]+)", o),
                factor=_g(r"Kernel execution time:\s*([0-9.]+)", o),       # ms
                conv=_g(r"total conversion time:\s*([0-9.]+)", o),         # ms
                spsv=_g(r"solve preprocess time:\s*([0-9.]+)", o),         # ms
                solve=_g(r"pcg time:\s*([0-9.]+)", o),                     # ms
                iters=_g(r"final iteration:\s*([0-9]+)", o),
                rr=_g(r"normalized diff norm:\s*([0-9.eE+-]+)", o),
                n=_g(r"num cols:\s*([0-9]+)", o),
                nnz=_g(r"laplacian nnz:\s*([0-9]+)", o),
                vram_mb=vram.peak_mb())


def _calibrate_tol_gpu(driver, mtx, tau=float(TOL), deadline=None):
    """The TOL to hand the CUDA driver so its own test stops at true residual tau.

    Patch 0003 makes the upstream first-iteration and loop tests consistently use
    the relative recurrence residual ||r||/||r0||. It remains optimistic relative
    to the independently printed true ||Ax-b||/||b||, so one probe rescales the
    existing CLI tolerance (argv[4]):

        TOL = tau * TOL_0 / R_0

    Returns None if the probe did not print a residual (caller falls back to TOL).
    """
    p = _run_once_gpu(driver, mtx, PROBE_TOL_GPU, deadline=deadline)
    if not p["rr"]:
        return None
    r0 = float(p["rr"])
    # A calibration may compensate for optimistic recurrence drift by tightening
    # the native tolerance. It must never relax the common 1e-8 request merely
    # because one randomized probe happened to look pessimistic.
    return min(tau, tau * PROBE_TOL_GPU / r0) if r0 > 0.0 else None


def _gpu_modes(mid):
    """The two published GPU series and whether each can take this matrix."""
    uses_physics = _uses_physics(mid)
    for solver_key, driver in (("parac_graph", rc.PARAC_GPU_DRIVER),
                               ("parac_physics", rc.PARAC_GPU_DRIVER_PHYS)):
        skip_reason = (GRAPH_NA if (uses_physics and solver_key == "parac_graph")
                       else PHYSICS_NA if (not uses_physics and solver_key == "parac_physics")
                       else None)
        # Read the toolchain off the driver intended for this series. This stays
        # useful provenance even when shared preprocessing fails before launch.
        prov = {**PROV_GPU, **rc.binary_toolchain(driver)}
        yield solver_key, driver, prov, skip_reason


def record_gpu_failure(mid, status, reason, prep=None):
    """Stamp both GPU series after a shared preparation failure.

    Exactly one mode applies to a matrix; it receives ``failed`` or ``timeout``.
    The incompatible mode still receives its required explicit ``n/a`` cell.
    Existing successful cells are never overwritten. Failed/timeouts remain
    retryable because TERMINAL_GPU deliberately excludes them.
    """
    if status not in ("failed", "timeout"):
        raise ValueError(f"GPU preparation outcome must be failed/timeout, got {status!r}")
    family = rc.MATRICES[mid]["family"]
    results = []
    for solver_key, _driver, prov, skip_reason in _gpu_modes(mid):
        if skip_reason:
            rc.emit_cell(family, mid, solver_key, "", "n/a", {}, THREADS, "gpu", prov,
                         matrix_meta={"parac_mode": "n/a",
                                      "parac_na_reason": skip_reason})
            results.append(f"{solver_key}[n/a]")
            continue
        if rc.cell_done(family, mid, solver_key, "", THREADS, "gpu",
                        terminal=TERMINAL_GPU):
            results.append(f"{solver_key}[skip(done)]")
            continue
        meta = {"parac_mode": "physics" if solver_key == "parac_physics" else "graph",
                "parac_prep_failure": reason}
        if prep:
            meta["parac_prep"] = prep
        kwargs = {"timeout_cap_s": TIMEOUT_GPU} if status == "timeout" else {}
        rc.emit_cell(family, mid, solver_key, "", status, {}, THREADS, "gpu", prov,
                     matrix_meta=meta, **kwargs)
        results.append(f"{solver_key}[{status.upper()}(prep)]")
    return " ".join(results)


def record_gpu_disconnected_na(mid, n_components):
    """ParAC's GPU driver has one global zero-sum RHS, not one per component."""
    family = rc.MATRICES[mid]["family"]
    reason = (f"ParAC GPU graph mode generates one globally zero-sum RHS, which is "
              f"not component-wise compatible for this {n_components}-component "
              "Laplacian; the CPU route solves every non-singleton component")
    results = []
    for solver_key, _driver, prov, skip_reason in _gpu_modes(mid):
        why = skip_reason or reason
        rc.emit_cell(family, mid, solver_key, "", "n/a", {}, THREADS, "gpu", prov,
                     matrix_meta={"parac_mode": "n/a", "parac_na_reason": why,
                                  "n_components": n_components})
        results.append(f"{solver_key}[n/a]")
    return " ".join(results)


def run_gpu(mid, tol=TOL):
    """ParAC GPU pass for one matrix: one real cell and one n/a, the same split as
    the CPU axis — kind=graph runs driver.cu on the pure L, kind=operator runs
    driver_physics.cu on the augmented published operator."""
    family = rc.MATRICES[mid]["family"]
    tag = _dump_tag(mid)
    aug = _augments(mid)
    # TIMEOUT_GPU is one logical cell's wall-clock cap, including mandatory
    # preprocessing, the calibration probe and all REPS timed executions.  It is
    # deliberately not renewed for each subprocess: doing that made one cell cost
    # up to (REPS + 1) times the advertised timeout and defeat Slurm resume jobs.
    deadline = time.monotonic() + TIMEOUT_GPU
    src = None
    dump_s = 0.0
    native_source = False
    try:
        # The graph driver generates one globally zero-sum RHS. That is valid
        # only for a connected Laplacian; unlike CPU ParAC, the current GPU
        # driver has no component-wise route. File-backed inputs go straight to
        # ParAC's producer after a no-serialization component-info probe.
        if not _uses_physics(mid):
            src = _native_mtx(mid)
            if src is not None:
                native_source = True
                n_components, _n_nodes, dump_s = _component_info(
                    mid, DUMP_GPU, rc.BIN["gpu"], _gpu_cell_remaining(deadline),
                    mem_cap_gb=None)
            else:
                src, _n_nodes, n_components, dump_s = _dump_component(
                    mid, 0, deadline=deadline, timeout_cap=TIMEOUT_GPU,
                    bin_path=rc.BIN["gpu"], mem_cap_gb=None, dump_dir=DUMP_GPU)
            if n_components > 1:
                return record_gpu_disconnected_na(mid, n_components)
            if native_source and n_components <= 0:
                return record_gpu_failure(
                    mid, "failed", "ParAC component-info probe produced no result")
            if not src or n_components <= 0:
                return record_gpu_failure(
                    mid, "failed", "ParAC component preparation produced no matrix")
        else:
            # physics_produce accepts the published operator directly and performs
            # the mandatory permutation/augmentation itself. Rewriting the same
            # Matrix Market input first would be benchmark-only work.
            src = _native_mtx(mid)
            dump_s = 0.0
            native_source = src is not None
            if src is None:
                src, _tag, dump_s = _dump(
                    mid, DUMP_GPU, rc.BIN["gpu"], _gpu_cell_remaining(deadline),
                    mem_cap_gb=None)
            if not src:
                return record_gpu_failure(mid, "failed",
                                          "GPU operator dump produced no matrix")
        input_tag = f"native-{tag}" if native_source else tag
        cache_tag = f"{input_tag}-aug" if aug else input_tag
        sorted_cached = f"{rc.PARAC_SORTED}/{mid}-{cache_tag}-nnz-sorted.mtx"
        if _prep_cache_valid(sorted_cached, src):
            sorted_mtx = sorted_cached
            with open(sorted_cached + ".time") as handle:
                sort_s = float(handle.read())
            if os.path.exists(sorted_cached + ".prep"):
                with open(sorted_cached + ".prep") as handle:
                    prep_prov = handle.read()
            else:
                prep_prov = _PREP_UNKNOWN
        else:
            sorted_mtx, sort_s, prep_prov = _nnz_sort(
                mid, src, input_tag, augment=aug, deadline=deadline)
            if not sorted_mtx:
                return record_gpu_failure(mid, "failed",
                                          "ParAC nnz-sort produced no matrix",
                                          prep=prep_prov)
    except subprocess.TimeoutExpired as exc:
        return record_gpu_failure(
            mid, "timeout",
            f"ParAC GPU dump/sort exceeded its {exc.timeout}s wall cap")
    results = []
    for solver_key, driver, prov, skip_reason in _gpu_modes(mid):
        if skip_reason:
            rc.emit_cell(family, mid, solver_key, "", "n/a", {}, THREADS, "gpu", prov,
                         matrix_meta={"parac_mode": "n/a", "parac_na_reason": skip_reason})
            results.append(f"{solver_key}[n/a]"); continue
        if rc.cell_done(family, mid, solver_key, "", THREADS, "gpu", terminal=TERMINAL_GPU):
            results.append(f"{solver_key}[skip(done)]"); continue
        if not os.path.exists(driver):
            rc.emit_cell(
                family, mid, solver_key, "", "failed", {}, THREADS, "gpu", prov,
                matrix_meta={"parac_mode": "physics" if solver_key == "parac_physics" else "graph",
                             "parac_prep_failure": f"ParAC GPU driver missing: {driver}"})
            results.append(f"{solver_key}[FAILED(driver missing)]"); continue
        try:
            cal = _calibrate_tol_gpu(driver, sorted_mtx, deadline=deadline) or float(tol)
            runs = [_run_once_gpu(driver, sorted_mtx, cal, deadline=deadline)
                    for _ in range(REPS)]
        except subprocess.TimeoutExpired:
            rc.emit_cell(family, mid, solver_key, "", "timeout", {}, THREADS, "gpu", prov,
                         matrix_meta={"parac_prep": prep_prov},
                         timeout_cap_s=TIMEOUT_GPU)
            results.append(f"{solver_key}[TIMEOUT]"); continue
        ok = [r for r in runs if r["cuda_init"] and r["adapter"]
              and r["factor_setup"] and r["solver_setup"]
              and r["solve_total"] and r["iters"] and r["rr"]
              and r["n"] and r["nnz"]]
        if len(ok) != REPS:
            rc.emit_cell(family, mid, solver_key, "", "failed", None, THREADS, "gpu", prov)
            results.append(f"{solver_key}[FAILED]"); continue
        rep_index, chosen = _representative_run(
            ok, lambda r: (float(r["adapter"]) + float(r["factor_setup"])
                           + float(r["solver_setup"]) + float(r["solve_total"])))
        val = lambda key, scale=1.0: float(chosen[key] or 0) * scale
        adapter = val("adapter"); factor_setup = val("factor_setup")
        solver_setup = val("solver_setup"); solve = val("solve_total")
        factor = val("factor", 1/1000); conv = val("conv", 1/1000)
        spsv = val("spsv", 1/1000); pcg = val("solve", 1/1000)
        iters = int(chosen["iters"])
        rr = float(chosen["rr"])
        # Complete non-overlapping intervals from ParAC's own code. The narrower
        # kernel/conversion/SpSV/PCG timers below are diagnostics only.
        setup = dump_s + sort_s + adapter + factor_setup + solver_setup
        total = setup + solve
        # rr is the driver's own `normalized diff norm` = ||Ax-b||/||b||. Judge it
        # at the tolerance we report at, not a decade looser.
        status = "complete" if rr <= float(tol) else "not_converged"
        metrics = {"n": int(chosen["n"]) if chosen["n"] else None,
                   "nnz": int(chosen["nnz"]) if chosen["nnz"] else None,
                   "setup_s": setup, "solve_s": solve, "total_s": total,
                   "iters": iters, "rel_res": rr, "parac_tol": cal,
                   "input_dump_s": dump_s,
                   "adapter_setup_s": dump_s + sort_s + adapter,
                   "native_setup_s": factor_setup + solver_setup,
                   "factor_kernel_s": factor, "factor_conversion_s": conv,
                   "spsv_analysis_s": spsv, "pcg_kernel_s": pcg,
                   "representative_repeat": rep_index + 1}
        if chosen.get("cuda_init"):
            metrics["cuda_init_s"] = float(chosen["cuda_init"])
        vram = [float(r["vram_mb"]) for r in ok if r.get("vram_mb")]
        if vram: metrics["max_vram_mb"] = round(max(vram), 1)   # peak VRAM over reps
        rc.emit_cell(family, mid, solver_key, "", status, metrics, THREADS, "gpu", prov,
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

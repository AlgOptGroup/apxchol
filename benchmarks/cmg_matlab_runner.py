#!/usr/bin/env python3
"""Canonical MATLAB CMG on the sweep's shared operator A and RHS b.

sweep_fair.py exports A and b with the benchmark binary and calls run(). One
MATLAB process runs cmg/bench_cmg_common.m: one warmup and the retained
repetitions, each building the hierarchy and solving with MATLAB pcg on
one computational thread. Disconnected Laplacians get one grounded coordinate
per component (cmg/component_gauge.h); every solution is graded by
||A x - b|| / ||b|| on the original A and b.

MATLAB runs from a local install, or inside the container from
cmg/Dockerfile.matlab when APXCHOL_CMG_IMAGE names its image. Either way
APXCHOL_MATLAB_ROOT is the MATLAB install and APXCHOL_CMG_SOLVER a CMG
checkout whose MEX files were built with makeCMG for that MATLAB.
"""
import hashlib, json, os, shlex, shutil, subprocess, tempfile

import runner_common as rc
from runner_common import ROOT, sh

MATLAB = rc.external_path("APXCHOL_MATLAB_ROOT", "MATLAB")
CMG_SOLVER = rc.external_path("APXCHOL_CMG_SOLVER", "CMG_SOLVER")
IMAGE = os.environ.get("APXCHOL_CMG_IMAGE", "")
BENCH = os.path.join(ROOT, "benchmarks", "cmg")
_MEX = os.path.join("matlab", "cmg", "mex", "mx_preconditioner_.mexa64")
_GAUGE_SOURCES = ("common_component_gauge_mex.cpp", "component_gauge.h")
NOTE = ("canonical MATLAB CMG (upstream m-code, makeCMG MEX) on the shared A and b; "
        "one MATLAB computational thread")


def available():
    """(ready, reason) for the configured MATLAB, CMG checkout and container."""
    if not MATLAB or not os.path.isfile(os.path.join(MATLAB, "bin", "matlab")):
        return False, "set APXCHOL_MATLAB_ROOT to a MATLAB install (bin/matlab)"
    if not CMG_SOLVER or not os.path.isfile(os.path.join(CMG_SOLVER, _MEX)):
        return False, ("set APXCHOL_CMG_SOLVER to a CMG checkout built with makeCMG "
                       f"(missing {_MEX})")
    if IMAGE:
        if not shutil.which("docker"):
            return False, "APXCHOL_CMG_IMAGE is set but docker is not on PATH"
        if sh(f"docker image inspect {shlex.quote(IMAGE)}", timeout=60).returncode != 0:
            return False, f"docker image {IMAGE} not found; build cmg/Dockerfile.matlab"
    return True, ""


def provenance():
    """Cell provenance: the MEX toolchain, the runtime and the CPU used."""
    return {**rc.binary_toolchain(os.path.join(CMG_SOLVER, _MEX) if CMG_SOLVER else ""),
            "cmg_note": NOTE, "cmg_runtime": f"container {IMAGE}" if IMAGE else "local MATLAB",
            "cmg_cpuset": rc.affinity_spec(1) or "unpinned"}


def gauge_dir(directory):
    """Build directory for the gauge MEX, keyed by its sources and runtime."""
    digest = hashlib.sha256()
    for name in _GAUGE_SOURCES:
        with open(os.path.join(BENCH, name), "rb") as handle:
            digest.update(handle.read())
    mode = "container" if IMAGE else "local"
    return os.path.join(directory, f"cmg-gauge-{mode}-{digest.hexdigest()[:12]}")


def command(operator, rhs, work, gauge, repeats, tol, maxiter, timeout):
    """Shell command for one cell. Host paths are mounted at the same paths in
    the container, so the MATLAB expression is the same in both modes."""
    paths = (operator, rhs, work, gauge, BENCH, CMG_SOLVER, MATLAB)
    if any("'" in p for p in paths):
        raise ValueError("CMG paths must not contain single quotes")
    expr = (f"addpath('{BENCH}'); g='{gauge}'; "
            f"if ~isfile(fullfile(g,['common_component_gauge_mex.' mexext])), "
            f"mex('-R2018a','-outdir',g,fullfile('{BENCH}','common_component_gauge_mex.cpp')); end; "
            f"addpath(g); bench_cmg_common('{operator}','{rhs}','{work}',"
            f"{repeats},{tol},{maxiter},1)")
    # The deadline is enforced next to MATLAB: killing a docker client does not
    # stop the container it started.
    inner = ["timeout", "-k", "10", str(int(timeout)),
             os.path.join(MATLAB, "bin", "matlab"), "-singleCompThread", "-batch", expr]
    cpus = rc.affinity_spec(1)
    if not IMAGE:
        prefix = ["env", f"CMG_ROOT={CMG_SOLVER}"] + (["taskset", "-c", cpus] if cpus else [])
        return shlex.join(prefix + inner)
    data = os.path.dirname(os.path.abspath(operator))
    docker = ["docker", "run", "--rm", "--network=host", "--user", f"{os.getuid()}:{os.getgid()}",
              "-e", "HOME=/tmp", "-e", f"CMG_ROOT={CMG_SOLVER}"]
    if cpus:
        docker += ["--cpuset-cpus", cpus]
    for path, mode in ((MATLAB, "ro"), (CMG_SOLVER, "ro"), (BENCH, "ro"), (data, "rw")):
        docker += ["-v", f"{path}:{path}:{mode}"]
    return shlex.join(docker + [IMAGE] + inner)


def read_records(work, repeats):
    """The repetition records bench_cmg_common.m wrote, in order, up to the
    first missing one."""
    records = []
    for index in range(repeats + 1):
        path = os.path.join(work, f"rep{index}.json")
        if not os.path.isfile(path):
            break
        with open(path) as handle:
            records.append(json.load(handle))
    return records


_METRICS = ("setup_s", "solve_s", "total_s", "iters", "rel_res", "n", "nnz",
            "adaptation_s", "components", "grounded_components", "matlab_version")


def summarize(records, returncode, stderr, repeats, tol):
    """(status, metrics, extra_meta) for one cell from its repetition records.

    bench_cmg_common.m stops at the first repetition that is not complete, so
    the last record decides n/a, failed and not_converged. A complete cell needs
    every repetition and a clean MATLAB exit. Every repetition's total, warmup
    included, goes into repeat_receipts for the sweep's per-solve cap."""
    receipts = [f"BENCH_REPEAT phase={'retained' if r.get('retained') else 'warmup'} "
                f"index={r.get('repetition')} total_s={r['total_s']}"
                for r in records if isinstance(r.get("total_s"), (int, float))]
    if returncode in (124, 137):
        return "timeout", {"repeat_receipts": receipts}, {
            "na_reason": f"MATLAB exceeded its process deadline (rc={returncode})"}
    last = records[-1] if records else {}
    status = last.get("status")
    if status == "n/a":
        return "n/a", {k: last[k] for k in ("n", "nnz") if k in last}, {
            "na_reason": last.get("reason", "CMG declined this operator")}
    if status not in ("complete", "not_converged") or (
            status == "complete" and (returncode != 0 or len(records) != repeats + 1)):
        detail = (last.get("error")
                  or ((stderr or "").strip().splitlines() or ["no output"])[-1][:500])
        return "failed", {"returncode": returncode, "repeat_receipts": receipts}, {
            "na_reason": f"CMG run failed: {detail}"}
    retained = [r for r in records if r.get("retained")]
    chosen = (sorted(retained, key=lambda r: r["total_s"])[len(retained) // 2]
              if status == "complete" else last)
    metrics = {k: chosen[k] for k in _METRICS if k in chosen}
    metrics.update(
        stop_contract="original-v1", effective_threads=1, warmup_repeats=1,
        retained_repeats=len(retained), representative_repeat=chosen.get("repetition"),
        pcg_flag=chosen.get("pcg_flag"), repeat_receipts=receipts)
    if status != "complete":
        return "not_converged", metrics, None
    metrics["max_repeat_rel_res"] = max(r["rel_res"] for r in retained)
    status, metrics = rc.classify(metrics, tol)
    return status, metrics, None


def run(operator, rhs, *, timeout, repeats, tol, maxiter):
    """Run one cell on the exported A (operator) and b (rhs)."""
    directory = os.path.dirname(os.path.abspath(operator))
    gauge = gauge_dir(directory)
    os.makedirs(gauge, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cmg-", dir=directory) as work:
        cmd = command(os.path.abspath(operator), os.path.abspath(rhs), work, gauge,
                      repeats, tol, maxiter, timeout)
        try:
            cp = sh(cmd, timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            return "timeout", {"repeat_receipts": []}, {
                "na_reason": f"MATLAB exceeded its {int(timeout)}s process deadline"}
        try:
            records = read_records(work, repeats)
        except (OSError, ValueError) as error:
            return "failed", {"returncode": cp.returncode}, {
                "na_reason": f"unreadable CMG repetition record: {error}"}
    return summarize(records, cp.returncode, cp.stderr, repeats, tol)

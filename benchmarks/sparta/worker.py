#!/usr/bin/env python3
"""Optional pinned SPARTA Metal worker; imports SPARTA only in this process."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import resource
import time

import numpy as np
from scipy import io, sparse
from scipy.sparse.csgraph import connected_components

REVISION = "7d82cdca257e1136c49e0cf84fe0cfc01c38ec4b"
CONTRACT = "sparta-metal-original-correction-v3"


class Unavailable(RuntimeError):
    pass


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def runtime(sparta):
    if platform.system() != "Darwin" or not sparta.gpu_available():
        raise Unavailable("SPARTA requires an available native Metal device")
    dist = importlib.metadata.distribution("sparta-solver")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != REVISION:
        raise Unavailable("install SPARTA from the pinned Git revision in benchmarks/README.md")
    import sparta._sparta as extension
    return {"sparta_revision": REVISION, "sparta_contract": CONTRACT,
            "sparta_version": dist.version, "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": importlib.metadata.version("scipy"),
            "sparta_binary_sha256": digest(extension.__file__)}


def ground(A, matrix_class):
    """Use the exported class or an exact constant-null component gauge."""
    if matrix_class not in ("laplacian", "sddm"):
        raise ValueError("missing canonical exported matrix class")
    if A.shape[0] != A.shape[1] or not np.isfinite(A.data).all():
        raise ValueError("expected a finite square operator")
    if (A != A.T).nnz:
        raise Unavailable("SPARTA benchmark requires a symmetric operator")
    count, labels = connected_components(A, directed=False)
    if matrix_class == "laplacian":
        # Match scan_operator's declaration check, not a new rank tolerance.
        # A graph's formed diagonal includes binary64 accumulation roundoff.
        # prepare_components trusts that declared Laplacian and pins every block.
        column_sums = np.zeros(A.shape[0])
        for col in range(A.shape[0]):
            for value in A.data[A.indptr[col]:A.indptr[col + 1]]:
                column_sums[col] += value
        slack = 1e-10 * np.maximum(np.abs(A.diagonal()), 1.0)
        if np.any(np.abs(column_sums) > slack):
            raise Unavailable("declared Laplacian contradicts the exported operator")
        singular = np.ones(count, dtype=bool)
    else:
        # Exact zero column sums prove a constant null vector for a symmetric
        # component, regardless of entry signs. This does not assert PSD or
        # nullity one; factorization and original-system verification still apply.
        singular = np.ones(count, dtype=bool)
        for col in range(A.shape[0]):
            component = labels[col]
            if singular[component] and math.fsum(A.data[A.indptr[col]:A.indptr[col + 1]]) != 0:
                singular[component] = False
    first = np.full(count, A.shape[0])
    np.minimum.at(first, labels, np.arange(A.shape[0]))
    pins = first[singular]
    keep = np.ones(A.shape[0]); keep[pins] = 0
    D = sparse.diags(keep)
    G = (D @ A @ D + sparse.diags(1 - keep)).tocsc()
    G.eliminate_zeros(); G.sort_indices()
    return G, pins, count


def checked_solve(F, A, b, pins, tol, budget):
    """Public solve(new_rhs) reuses factors; work counts factor applications."""
    start = time.perf_counter()
    x = np.zeros_like(b)
    residual = b.copy()
    norm = np.linalg.norm(b) or 1.0
    used = 0; history = []; check_seconds = 0.0
    # Pinned upstream default has a 50-correction ceiling. Reserve each call.
    reservation = min(51, budget)
    while budget and used + reservation <= budget and len(history) < 8:
        rhs = residual.copy(); rhs[pins] = 0
        delta = F.solve(rhs)
        reports = F.diagnostics()["refinement_reports"]
        if not reports and reservation == 1:
            reports = [{"corrections": 0, "status": "refinement_disabled"}]
        if len(reports) != 1:
            raise RuntimeError("SPARTA did not report native refinement work")
        report = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                  for k, v in reports[0].items()}
        work = 1 + int(report["corrections"])
        if not 1 <= work <= reservation:
            raise RuntimeError("SPARTA exceeded the reserved factor-application budget")
        used += work
        if not np.isfinite(delta).all() or "nonfinite" in str(report["status"]).lower():
            raise FloatingPointError("SPARTA numerical breakdown")
        x += delta
        check_start = time.perf_counter()
        residual = b - A @ x
        rel = float(np.linalg.norm(residual) / norm)
        check_seconds += time.perf_counter() - check_start
        if not math.isfinite(rel):
            raise FloatingPointError("nonfinite original residual")
        history.append({"original_rel_res": rel, "factor_applications": work,
                        "native_refinement": report})
        if rel <= tol:
            break
    if not history:
        check_start = time.perf_counter()
        rel = float(np.linalg.norm(b - A @ x) / norm)
        check_seconds += time.perf_counter() - check_start
    return {"solve_s": time.perf_counter() - start, "rel_res": rel,
            "factor_applications": used, "solve_passes": len(history),
            "stop_check_s": check_seconds, "passes": history}


def trial(sparta, A, b, tol, budget, threads, matrix_class):
    start = time.perf_counter()
    G, pins, components = ground(A, matrix_class)
    F = sparta.analyze(G, force_cpu=False, hybrid=False, threads=threads,
                       refinement_rtol=tol,
                       refinement_steps=min(50, max(0, budget - 1))).factorize()
    try:
        diagnostics = F.diagnostics()
        if diagnostics["backend"] != "metal":
            raise Unavailable("SPARTA selected a non-Metal route")
        setup = time.perf_counter() - start
        result = checked_solve(F, A, b, pins, tol, budget)
        result.update(setup_s=setup, total_s=setup + result["solve_s"],
                      components=int(components), grounded_components=len(pins),
                      n=A.shape[0], nnz=A.nnz, stop_contract="original-v1",
                      execution_route="metal", matrix_class=matrix_class, work_unit="factor_application", factor_application_budget=budget,
                      sparta_options=diagnostics["options"], pins=pins.tolist(),
                      host_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576,
                      status="complete" if result["rel_res"] <= tol else "not_converged")
        return result
    finally:
        F.destroy()


def write_record(output, index, record):
    target = output / f"rep{index}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, allow_nan=False))
    temporary.replace(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("operator"); p.add_argument("rhs"); p.add_argument("output")
    p.add_argument("--class", dest="matrix_class", choices=("laplacian", "sddm"), required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--maxiter", type=int, required=True)
    p.add_argument("--tol", type=float, required=True)
    a = p.parse_args()
    if a.repeat < 1 or a.warmup < 0 or a.threads < 1 or a.maxiter < 0 or not 0 < a.tol < 1:
        p.error("invalid repetition, thread, budget or tolerance")
    output = Path(a.output)
    index = 0
    try:
        import sparta
        provenance = runtime(sparta)
        init_start = time.perf_counter()
        tiny = sparta.analyze(sparse.eye(2, format="csc"), force_cpu=False,
                              hybrid=False, refinement_rtol=a.tol).factorize()
        try:
            if tiny.diagnostics()["backend"] != "metal":
                raise Unavailable("SPARTA selected a non-Metal route")
            tiny.solve(np.ones(2))
        finally:
            tiny.destroy()
        init_seconds = time.perf_counter() - init_start
        A = sparse.csc_matrix(io.mmread(a.operator), dtype=np.float64)
        A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
        raw_b = io.mmread(a.rhs)
        b = np.asarray(raw_b.toarray() if sparse.issparse(raw_b) else raw_b, dtype=np.float64).reshape(-1)
        if len(b) != A.shape[0] or not np.isfinite(b).all():
            raise ValueError("invalid shared RHS")
        provenance.update(operator_sha256=digest(a.operator),
                          rhs_sha256=digest(a.rhs))
        for index in range(a.warmup + a.repeat):
            record = trial(sparta, A, b, a.tol, a.maxiter, a.threads, a.matrix_class)
            record.update(repetition=index, retained=index >= a.warmup, provenance=provenance,
                          sparta_init_s=init_seconds)
            write_record(output, index, record)
            if record["status"] != "complete":
                break
    except Exception as error:
        write_record(output, index,
            {"status": "n/a" if isinstance(error, (Unavailable, ImportError)) else "failed",
             "error": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

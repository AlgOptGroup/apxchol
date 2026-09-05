#!/usr/bin/env python3
"""Private native CMG packed port, using the existing schema-2 benchmark cells.

No generated source is stored here. Unlike the canonical MATLAB legacy runner,
this series solves the original unshifted operator with a compatible b=A*g.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import runner_common as rc

TOL = 1e-8
BINARY = rc.external_path("APXCHOL_CMG_NATIVE_BIN", "CMG_NATIVE_BIN")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def parse_driver(stdout, stderr):
    lines = [line for line in stdout.splitlines() if line.startswith("CMG ")]
    if len(lines) != 1:
        return None
    raw = dict(item.split("=", 1) for item in lines[0].split()[1:])
    ints = {"n", "nnz", "hierarchy_valid", "levels", "setup_flag", "pcg_flag", "iter", "converged", "setup_calls"}
    values = {key: int(value) if key in ints else float(value) for key, value in raw.items()}
    required = {"n", "nnz", "hierarchy_valid", "pcg_flag", "iter", "converged", "setup_flag", "setup_calls", "setup_s", "solve_s", "total_s", "true_relres", "adaptation_s"}
    if not required.issubset(values):
        raise ValueError("incomplete native CMG output")
    for key in ("setup_s", "solve_s", "total_s", "adaptation_s"):
        if not math.isfinite(values[key]) or values[key] < 0:
            raise ValueError("invalid native CMG timing")
    if abs(values["total_s"] - values["setup_s"] - values["solve_s"]) > 1e-9 * max(1, values["total_s"]):
        raise ValueError("inconsistent setup/solve boundary")
    rss = re.search(r"CMGRSS\s+(\d+)", stderr)
    values["max_rss_mb"] = int(rss.group(1))/1024 if rss else None
    values["iters"] = values.pop("iter")
    values["rel_res"] = values["true_relres"]
    return values


def classify(values, returncode, independent_residual=None):
    if values is None:
        return "oom" if returncode in (137, -9) else "failed"
    if values["setup_flag"] == -1 or (values["setup_flag"] == 2 and not values["hierarchy_valid"]):
        return "n/a"
    residuals = [values["rel_res"]]
    if independent_residual is not None:
        residuals.append(independent_residual)
    if (returncode == 0 and values["converged"] == 1 and values["hierarchy_valid"] == 1
            and values["pcg_flag"] == 0 and values["setup_calls"] == 1
            and all(math.isfinite(r) and 0 <= r <= TOL for r in residuals)):
        return "complete"
    return "not_converged" if values["hierarchy_valid"] else "failed"


def prepare(record, output):
    # Imports are deferred: sweep availability does not require SciPy.
    import numpy as np
    import scipy.io
    import scipy.sparse as sp
    source = Path(record["path"])
    actual_hash = digest(source)
    if record.get("sha256") and actual_hash != record["sha256"]:
        raise ValueError(f"input SHA-256 differs: {record['id']}")
    A = scipy.io.mmread(source).tocsc().astype(np.float64)
    A.sum_duplicates()
    if A.shape[0] != A.shape[1] or not np.isfinite(A.data).all():
        raise ValueError("matrix must be square and finite")
    if record["mode"] == "graph":
        A.setdiag(0)
        A.eliminate_zeros()
        A.data = np.abs(A.data)
        A = (sp.diags(np.asarray(A.sum(axis=1)).ravel(), format="csc") - A).tocsc()
    asym = A-A.T
    if asym.nnz and np.max(np.abs(asym.data)) > 1e-12 * max(1, np.max(np.abs(A.data))):
        raise ValueError("matrix is not symmetric")
    # b is in range(A), including all disconnected Laplacian components.
    g = np.random.Generator(np.random.PCG64(42)).standard_normal(A.shape[0])
    b = A @ g
    norm = np.linalg.norm(b)
    if norm:
        b /= norm
    operator = output / "operator.mtx"
    rhs = output / "rhs.mtx"
    scipy.io.mmwrite(operator, A, symmetry="general", precision=17)
    scipy.io.mmwrite(rhs, b.reshape(-1, 1), precision=17)
    metadata = {"source_path": str(source), "source_sha256": actual_hash,
                "operator_sha256": digest(operator), "rhs_sha256": digest(rhs),
                "n": A.shape[0], "nnz": A.nnz, "mode": record["mode"],
                "regularization": 0, "grounding": "none; original operator",
                "rhs": "A*g normalized; NumPy PCG64(42) normal; common preparation excluded"}
    save(output / "input.json", metadata)
    return A, b, operator, rhs, metadata


def run_prepared(record, prepared, *, binary, threads, output, repetitions=3, timeout=240):
    import numpy as np
    import scipy.io
    A, b, operator, rhs, metadata = prepared
    metadata = {**metadata, "timeout_scope": "logical_cell"}
    env = rc.benchmark_openmp_env(threads)
    env.update({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    output.mkdir()
    cpus = rc.affinity_spec(threads)
    deadline = time.monotonic() + timeout
    runs = []
    cell_status = "complete"
    # One untimed calibration/warmup followed by retained independent processes.
    for rep in range(repetitions+1):
        prefix = output / ("warmup" if rep == 0 else f"rep{rep}")
        solution = prefix.with_suffix(".solution.mtx")
        cmd = shlex.join(["/usr/bin/time", "-f", "CMGRSS %M", "taskset", "-c", cpus,
                          str(binary), str(operator), str(rhs), str(TOL), "500", str(solution)])
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            cell_status = "timeout"
            break
        try:
            cp = rc.sh(cmd, timeout=remaining, env=env, mem_cap_gb=180)
            prefix.with_suffix(".stdout").write_text(cp.stdout)
            prefix.with_suffix(".stderr").write_text(cp.stderr)
            values = parse_driver(cp.stdout, cp.stderr)
            independent = None
            if solution.is_file():
                x = np.asarray(scipy.io.mmread(solution)).reshape(-1)
                if x.shape != b.shape:
                    raise ValueError("returned solution shape differs")
                independent = float(np.linalg.norm(A @ x-b)/np.linalg.norm(b))
            status = classify(values, cp.returncode, independent)
            if values is not None and independent is None:
                status = "failed"
            run = {"repetition": rep, "retained": rep > 0, "status": status,
                   "returncode": cp.returncode, "metrics": values,
                   "independent_true_relres": independent,
                   "solution_sha256": digest(solution) if solution.is_file() else None}
            save(prefix.with_suffix(".json"), run)
            runs.append(run)
            if status != "complete":
                cell_status = status
                break
        except subprocess.TimeoutExpired as error:
            prefix.with_suffix(".stdout").write_text(error.output or "")
            prefix.with_suffix(".stderr").write_text(error.stderr or "")
            runs.append({"repetition": rep, "retained": rep > 0, "status": "timeout"})
            cell_status = "timeout"
            break
    retained = [run for run in runs if run.get("retained") and run.get("metrics")]
    if time.monotonic() > deadline:
        cell_status = "timeout"
    representative = sorted(retained, key=lambda run: run["metrics"]["total_s"])[len(retained)//2] if retained else None
    metrics = dict(representative["metrics"]) if representative else {}
    if representative:
        metrics["representative_repeat"] = representative["repetition"]
        metrics["max_rss_mb"] = max(run["metrics"]["max_rss_mb"] or 0 for run in retained)
    provenance = {"git_sha": os.environ.get("CMG_SOURCE_COMMIT") or rc.git_sha(),
                  "driver_sha256": digest(binary), "repeat": repetitions,
                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "affinity": cpus, "effective_threads": 1,
                  "implementation": "MATLAB Coder packed port; serial generated core",
                  "canonical_exception": "shallow_water1 canonical H{0} setup error; port has valid terminal hierarchy",
                  "timing": "CSC adaptation and hierarchy setup included; packed PCG includes hierarchy destruction",
                  "all_retained_pass": cell_status == "complete" and len(retained) == repetitions}
    rc.emit_cell(record["family"], record["id"], "cmg_packed", "original-operator", cell_status,
                 metrics, threads, "cpu", provenance, matrix_meta=metadata,
                 timeout_cap_s=timeout if cell_status == "timeout" else None)
    result = {"matrix_id": record["id"], "threads": threads, "status": cell_status,
              "runs": runs, "provenance": provenance, "matrix_meta": metadata}
    save(output / "cell.json", result)
    return result


def run_one(mid, threads=16):
    family = rc.MATRICES[mid]["family"]
    if rc.cell_done(family, mid, "cmg_packed", "original-operator", threads, "cpu"):
        return "skip(done)"
    # Reuse the canonical external-operator dump seam, including generated grids.
    import cmg_matlab_runner
    dumped = cmg_matlab_runner._dump(mid)
    if not dumped:
        raise RuntimeError("common benchmark operator dump failed")
    # The common seam has already assembled every input, including graph L.
    # Preserve its diagonal exactly instead of interpreting that dump again.
    record = {"id": mid, "path": dumped, "family": rc.MATRICES[mid]["family"],
              "mode": "physics"}
    root = Path(rc.external_path("APXCHOL_CMG_NATIVE_RESULTS", "CMG_NATIVE_RESULTS", f"{rc.ROOT}/results/cmg-native"))
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{mid}-t{threads}-{time.time_ns()}"
    out.mkdir()
    prepared = prepare(record, out)
    return run_prepared(record, prepared, binary=BINARY, threads=threads, output=out/"run")["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[72,16])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240, help="complete cell cap, including warmup and retained runs")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())["matrices"]
    if len(records) != 14 or len({row["id"] for row in records}) != 14:
        raise ValueError("expected exactly 14 unique planned matrices")
    if args.smoke:
        records = [row for row in records if row.get("smoke")]
    if args.repetitions < 1 or args.timeout <= 0 or not 0 <= args.shard < args.shards:
        raise ValueError("invalid repetition, timeout or shard count")
    planned = [(row["id"], threads) for row in records for threads in args.threads]
    assigned = set(planned[args.shard::args.shards])
    args.output.mkdir(parents=True, exist_ok=False)
    rc.CELLS = str(args.output/"cells")
    save(args.output/"plan.json", {"matrices": records, "threads": args.threads,
                                  "repetitions": args.repetitions, "cell_timeout_s": args.timeout,
                                  "full_expected_cells": len(planned),
                                  "expected_cells": len(assigned), "assigned": sorted(assigned)})
    results = []
    for record in records:
        assigned_threads = [threads for threads in args.threads if (record["id"], threads) in assigned]
        if not assigned_threads:
            continue
        input_root = args.output/record["id"]
        input_root.mkdir()
        try:
            prepared = prepare(record, input_root)
            for threads in assigned_threads:
                result = run_prepared(record, prepared, binary=args.binary, threads=threads,
                                      output=input_root/f"t{threads}", repetitions=args.repetitions,
                                      timeout=args.timeout)
                results.append(result)
                save(args.output/"results.json", results)
                print(f"CMG_CELL {record['id']} T={threads} {result['status']}", flush=True)
        except Exception as error:
            for threads in assigned_threads:
                if any(row['matrix_id']==record['id'] and row['threads']==threads for row in results):
                    continue
                row = {"matrix_id": record["id"], "threads": threads, "status": "failed", "error": str(error)}
                results.append(row)
                rc.emit_cell(record["family"], record["id"], "cmg_packed", "original-operator", "failed", {}, threads, "cpu", {"error":str(error)})
                print(f"CMG_CELL {record['id']} T={threads} failed: {error}", flush=True)
            save(args.output/"results.json", results)
    save(args.output/"COMPLETE.json", {"checked": len(results), "expected": len(assigned),
        "status_counts": {status: sum(row["status"]==status for row in results) for status in sorted({row["status"] for row in results})}})


if __name__ == "__main__":
    main()

"""Small, reproducible Python API measurements for the documentation site.

Run after building/installing this checkout. Independent residual checks are
outside the timers; the solver's own exit verification remains inside them.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import cg

ROOT = Path(__file__).resolve().parents[1]
SIDES = (192, 512)
SEEDS = (811, 823, 839)
RTOL = 1e-8
NOISE_LIMIT = 0.10
METRICS = ("setup", "first_scalar", "setup_plus_one", "first_batch4", "batch4", "sequential4")


def grid(side):
    """Dirichlet five-point operator: four on the diagonal, minus one on edges."""
    line = sparse.diags((-np.ones(side - 1), 2 * np.ones(side), -np.ones(side - 1)),
                        (-1, 0, 1), format="csc")
    A = (sparse.kronsum(line, line, format="csc")).astype(np.float64)
    A.sort_indices()
    return A


class NumericalFailure(ValueError):
    pass


def validate(A, B, result, operation="apxchol"):
    relative = np.linalg.norm(A @ result.x - B, axis=0) / np.linalg.norm(B, axis=0)
    values = np.atleast_1d(relative)
    if not np.isfinite(values).all() or not (values <= RTOL).all() or not np.all(result.converged):
        print(f"{operation}: residual={relative}, converged={result.converged}", file=sys.stderr)
        raise NumericalFailure(operation)
    return values.tolist()


def timed(call):
    start = time.perf_counter()
    value = call()
    return value, time.perf_counter() - start


def owner(side, route, threads, seed, reverse):
    import apxchol
    A = grid(side)
    B = np.asfortranarray(A @ np.random.default_rng(seed).standard_normal((A.shape[0], 4)))
    residuals = []
    reference_result = {}

    def reference():
        (x, info), duration = timed(lambda: cg(A, B[:, 0], rtol=RTOL, atol=0, maxiter=2000))
        try:
            value, = validate(A, B[:, 0], SimpleNamespace(x=x, converged=(info == 0)), "scipy_cg")
            reference_result.update(status="passed", residual=value)
            return duration
        except NumericalFailure:
            reference_result.update(status="nonconverged", scipy_info=int(info))
            return None

    reference_seconds = reference() if not reverse else None
    solver, setup = timed(lambda: apxchol.factorize(
        A, backend=route, num_threads=threads, seed=seed,
        factor_storage_dtype=np.float32, keep_factor=False))
    if solver.backend != route:
        raise ValueError(f"requested {route}, selected {solver.backend}")

    def solve(rhs):
        result, duration = timed(lambda: solver.solve(rhs, rtol=RTOL, maxiter=2000))
        residuals.extend(validate(A, rhs, result))
        return duration

    first = solve(B[:, 0])
    first_batch = solve(B)
    def sequential():
        results, duration = timed(lambda: [solver.solve(B[:, column], rtol=RTOL, maxiter=2000)
                                          for column in range(4)])
        for column, result in enumerate(results):
            residuals.extend(validate(A, B[:, column], result))
        return duration

    calls = {"batch4": lambda: solve(B), "sequential4": sequential}
    timings = {key: calls[key]() for key in (reversed(calls) if reverse else calls)}
    timings.update(setup=setup, first_scalar=first, setup_plus_one=setup + first,
                   first_batch4=first_batch, scipy_cg=reference() if reverse else reference_seconds)
    return {"status": "passed", "seconds": timings, "rhs_checks": len(residuals),
            "max_residual": max(residuals), "reference_cg": reference_result, "rows": A.shape[0], "nnz": A.nnz,
            "backend": solver.backend, "apxchol_version": apxchol.__version__,
            "binary_sha256": hashlib.sha256(Path(apxchol._apxchol.__file__).read_bytes()).hexdigest()}


def control_spread(a, b):
    if not all(np.isfinite(x) and x > 0 for x in (a, b)):
        raise ValueError("timings must be finite and positive")
    return abs(a - b) / min(a, b)


def summarize(records):
    """Keep noisy observations, but never label their ratios as accepted."""
    if len(records) != 2 * len(SEEDS) or any(r["status"] != "passed" for r in records):
        return {"status": "failed", "accepted_pairs": 0}
    reference_ok = all(r.get("reference_cg", {}).get("status") == "passed" for r in records)
    metrics = METRICS + (("scipy_cg",) if reference_ok else ())
    spreads = [{metric: control_spread(a["seconds"][metric], b["seconds"][metric])
                for metric in metrics} for a, b in zip(records[::2], records[1::2])]
    clean = sum(max(pair.values()) <= NOISE_LIMIT for pair in spreads)
    return {"status": "accepted" if clean == len(SEEDS) else "noisy",
            "accepted_pairs": clean, "control_spread": spreads,
            "reference_status": "passed" if reference_ok else "nonconverged",
            "median_seconds": {metric: float(np.median([r["seconds"][metric] for r in records]))
                               for metric in metrics},
            "batch_speedup": float(np.median([r["seconds"]["sequential4"] / r["seconds"]["batch4"]
                                               for r in records]))}


def hardware(metal):
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        physical = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True))
    else:
        fields = [dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
                  for block in Path("/proc/cpuinfo").read_text().split("\n\n") if block.strip()]
        physical = len({(next((v.strip() for k, v in part.items() if k.strip() == "physical id"), "0"),
                         next((v.strip() for k, v in part.items() if k.strip() == "core id"), str(index)))
                        for index, part in enumerate(fields)})
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), platform.processor())
    data = {"os": f"{platform.system()} {platform.release()}", "architecture": platform.machine(),
            "cpu": cpu, "physical_cores": physical, "logical_cpus": os.cpu_count(),
            "available_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
            "runner_image": os.getenv("ImageVersion"),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    if metal:
        # Query the device selected by the solver, including on headless runners.
        query = ("import Metal; import Foundation; "
                 "guard let device = MTLCreateSystemDefaultDevice() else { exit(1) }; "
                 "print(device.name)")
        name = subprocess.check_output(
            ["xcrun", "swift", "-framework", "CoreGraphics", "-e", query],
            text=True, timeout=30).strip()
        if not name:
            raise ValueError("missing Metal device name")
        data["gpus"] = [name]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metal", action="store_true", help="also require the explicit Metal route")
    parser.add_argument("--owner", nargs=5, metavar=("SIDE", "ROUTE", "THREADS", "SEED", "REVERSE"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.owner:
        side, route, threads, seed, reverse = args.owner
        try:
            record = owner(int(side), route, int(threads), int(seed), bool(int(reverse)))
        except NumericalFailure as error:
            record = {"status": "nonconverged", "operation": str(error)}
        except (RuntimeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            record = {"status": "failed", "error": type(error).__name__}
        print(json.dumps(record, allow_nan=False))
        return
    configurations = [("CPU 1 worker", "cpu", 1), ("CPU up to 2 workers", "cpu", 2)]
    if args.metal:
        if sys.platform != "darwin":
            parser.error("--metal needs a macOS Metal build")
        configurations.append(("Metal (2 host workers)", "gpu", 2))
    data = {"schema": 1, "kind": "hosted" if os.getenv("GITHUB_ACTIONS") == "true" else "local", "date": datetime.now(timezone.utc).isoformat(),
            "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=ROOT)),
            "hardware": hardware(args.metal), "tolerance": RTOL, "noise_limit": NOISE_LIMIT,
            "build_settings": {key: os.getenv(key) for key in ("CC", "CXX", "CMAKE_ARGS")},
            "thread_environment": {key: os.getenv(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "planned_owners": len(SIDES) * len(configurations) * len(SEEDS) * 2,
            "planned_rhs_checks": len(SIDES) * len(configurations) * len(SEEDS) * 2 * 13,
            "cases": []}
    if os.getenv("GITHUB_RUN_ID"):
        data["run_url"] = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    started = time.monotonic()
    for side in SIDES:
        cases = [{"fixture": f"Grid {side}×{side}", "side": side, "configuration": name,
                  "route": route, "threads": threads, "records": []} for name, route, threads in configurations]
        for pair, seed in enumerate(SEEDS):
            for case in (reversed(cases) if pair % 2 else cases):
                for control in range(2):
                    record = {"seed": seed, "control": bool(control)}
                    remaining = 240 - (time.monotonic() - started)
                    if remaining <= 0:
                        record.update(status="not_run", reason="campaign_limit")
                        case["records"].append(record)
                        continue
                    try:
                        command = [sys.executable, str(Path(__file__).resolve()), str(args.output), "--owner",
                                   str(side), case["route"], str(case["threads"]), str(seed), str(pair % 2)]
                        run = subprocess.run(command, capture_output=True, text=True, timeout=min(45, remaining), check=True)
                        record.update(json.loads(run.stdout))
                        if record["status"] != "passed":
                            print(run.stderr, file=sys.stderr)
                    except (subprocess.SubprocessError, TimeoutError, ValueError) as error:
                        # Raw stderr may contain local paths. CI retains it; published JSON does not.
                        print(f"{case['fixture']} / {case['configuration']}: {error}", file=sys.stderr)
                        if isinstance(error, subprocess.CalledProcessError):
                            print(error.stderr, file=sys.stderr)
                        record.update(status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed",
                                      error=type(error).__name__)
                    case["records"].append(record)
        for case in cases:
            case["summary"] = summarize(case["records"])
        data["cases"].extend(cases)
    records = [r for case in data["cases"] for r in case["records"]]
    data["passed_owners"] = sum(r["status"] == "passed" for r in records)
    data["rhs_checks"] = sum(r.get("rhs_checks", 0) for r in records)
    data["reference_passes"] = sum(r.get("reference_cg", {}).get("status") == "passed" for r in records)
    data["wall_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if data["passed_owners"] != data["planned_owners"] or data["rhs_checks"] != data["planned_rhs_checks"]:
        raise SystemExit("incomplete numerical coverage; see report")


if __name__ == "__main__":
    main()

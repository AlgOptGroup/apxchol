"""Bounded same-runner base/head comparison using one frozen Python harness.

The report is regression evidence for these fixtures, not a benchmark ranking.
Noisy controls are inconclusive; a signal is not a statistical significance test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from docs_benchmark import RTOL, SIDES, control_spread, grid, hardware, timed, validate

ROUNDS = 3
SEED = 811
THREADS = 2
NOISE_LIMIT = 0.10
SLOWDOWN = 1.20
METRICS = ("setup", "solve", "total")
CAMPAIGN_SECONDS = 240
OWNER_SECONDS = 30


def order(round_index):
    return ("base", "head", "head", "base") if round_index % 2 == 0 else ("head", "base", "base", "head")


def digest(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        result.update(np.ascontiguousarray(array).tobytes())
    return result.hexdigest()


def owner(side, route):
    import apxchol

    def factor(A):
        solver = apxchol.factorize(A, backend=route, num_threads=THREADS, seed=SEED,
                                  factor_storage_dtype=np.float32, keep_factor=False)
        if solver.backend != route:
            raise ValueError(f"requested {route}, selected {solver.backend}")
        return solver

    # Common tiny warmup excludes process/device startup from both revisions.
    warm = grid(16)
    rhs = warm @ np.random.default_rng(SEED).standard_normal(warm.shape[0])
    warm_solver = factor(warm)
    warm_residual, = validate(warm, rhs, warm_solver.solve(rhs, rtol=RTOL, maxiter=2000))
    del warm_solver

    A = grid(side)
    b = A @ np.random.default_rng(SEED).standard_normal(A.shape[0])
    solver, setup = timed(lambda: factor(A))
    result, solve = timed(lambda: solver.solve(b, rtol=RTOL, maxiter=2000))
    residual, = validate(A, b, result)
    return {"status": "complete", "seconds": {"setup": setup, "solve": solve, "total": setup + solve},
            "backend": solver.backend, "residual": residual, "warmup_residual": warm_residual,
            "rows": A.shape[0], "nnz": A.nnz,
            "input_sha256": digest(A.indptr, A.indices, A.data, b),
            "binary_sha256": hashlib.sha256(Path(apxchol._apxchol.__file__).read_bytes()).hexdigest(),
            "apxchol_version": apxchol.__version__}


def summarize(records):
    if (len(records) != ROUNDS * 4 or any(r["status"] != "complete" for r in records)
            or any([r.get("revision") for r in records[4 * i:4 * i + 4]] != list(order(i))
                   for i in range(ROUNDS))):
        return {"status": "incomplete", "metrics": {}}
    if len({r["input_sha256"] for r in records}) != 1:
        return {"status": "invalid", "reason": "input identity mismatch", "metrics": {}}
    if any(len({r["binary_sha256"] for r in records if r["revision"] == revision}) != 1
           for revision in ("base", "head")):
        return {"status": "invalid", "reason": "binary changed during campaign", "metrics": {}}
    metrics = {}
    for metric in METRICS:
        ratios, spreads = [], []
        for index in range(ROUNDS):
            quartet = records[4 * index:4 * index + 4]
            values = {rev: [r["seconds"][metric] for r in quartet if r["revision"] == rev]
                      for rev in ("base", "head")}
            spreads.append(max(control_spread(*values[rev]) for rev in values))
            ratios.append(statistics.median(values["head"]) / statistics.median(values["base"]))
        status = ("inconclusive" if max(spreads) > NOISE_LIMIT else
                  "slowdown_signal" if min(ratios) >= SLOWDOWN else "no_signal")
        metrics[metric] = {"status": status, "head_over_base": ratios, "control_spread": spreads,
                           "median_seconds": {rev: statistics.median(
                               r["seconds"][metric] for r in records if r["revision"] == rev)
                                              for rev in ("base", "head")}}
    statuses = {metric["status"] for metric in metrics.values()}
    status = ("slowdown_signal" if "slowdown_signal" in statuses else
              "inconclusive" if "inconclusive" in statuses else "no_signal")
    return {"status": status, "metrics": metrics}


def markdown(report):
    lines = [f"## PR performance: {report['status'].replace('_', ' ').upper()}", "",
             f"Base `{report['base_revision']}`; head `{report['head_revision']}`.", "",
             "| Fixture | Phase | Base (s) | Head (s) | Head/base | Result |",
             "|---|---|---:|---:|---:|---|"]
    for case in report["cases"]:
        summary = case["summary"]
        for metric, values in summary["metrics"].items():
            seconds = values["median_seconds"]
            lines.append(f"| Grid {case['side']} | {metric} | {seconds['base']:.4g} | {seconds['head']:.4g} | "
                         f"{statistics.median(values['head_over_base']):.3f} | {values['status'].replace('_', ' ')} |")
        if not summary["metrics"]:
            lines.append(f"| Grid {case['side']} | — | — | — | — | {summary['status']} |")
    lines += ["", f"Completed {report['completed_owners']}/{report['planned_owners']} owners "
              f"and {report['rhs_checks']}/{report['planned_rhs_checks']} retained original-system residual checks. "
              f"Timing campaign: {report['wall_seconds']:.1f} s (builds excluded).",
              "", "Three alternating ABBA/BAAB rounds; identical inputs, seed, settings and two host workers. "
              "A phase signals only when every round is at least 20% slower and both revisions' "
              "repeat controls stay within 10%. Noisy phases are inconclusive, not performance passes. "
              "This conservative rule is not a statistical significance test; inspect the retained JSON and raw logs."]
    return "\n".join(lines) + "\n"


def campaign(args):
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS=str(THREADS), OMP_DYNAMIC="FALSE",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    # Linux workers inherit one fixed CPU set. macOS has no sched_setaffinity API.
    cpus = None
    if hasattr(os, "sched_setaffinity"):
        cpus = sorted(os.sched_getaffinity(0))[:THREADS]
        if len(cpus) < THREADS:
            raise ValueError("two available CPUs are required")
        os.sched_setaffinity(0, cpus)
        environment.update(OMP_PLACES=",".join(f"{{{cpu}}}" for cpu in cpus), OMP_PROC_BIND="close")
        environment.pop("GOMP_CPU_AFFINITY", None)
        environment.pop("KMP_AFFINITY", None)
    report = {"schema": 1, "date": datetime.now(timezone.utc).isoformat(),
              "base_revision": args.base_revision, "head_revision": args.head_revision,
              "hardware": hardware(args.route == "gpu"), "route": args.route,
              "threads": THREADS, "cpu_affinity": cpus, "seed": SEED, "rtol": RTOL,
              "factor_storage": "float32", "keep_factor": False, "maxiter": 2000,
              "warmup": {"side": 16, "owners_per_process": 1},
              "noise_limit": NOISE_LIMIT, "slowdown_ratio": SLOWDOWN,
              "harness_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("pr_regression.py", "docs_benchmark.py")},
              "build_settings": {key: os.getenv(key) for key in ("CC", "CXX", "CMAKE_ARGS")},
              "planned_cases": len(SIDES), "planned_owners": len(SIDES) * ROUNDS * 4,
              "planned_rhs_checks": len(SIDES) * ROUNDS * 4, "cases": []}
    started = time.monotonic()
    for side in SIDES:
        records = []
        for index in range(ROUNDS):
            for position, revision in enumerate(order(index)):
                record = {"revision": revision, "round": index, "position": position}
                remaining = CAMPAIGN_SECONDS - (time.monotonic() - started)
                if remaining <= 0:
                    record.update(status="not_run", reason="campaign deadline")
                else:
                    command = [str(getattr(args, revision + "_python")), str(Path(__file__).resolve()),
                               "--owner", str(side), args.route]
                    stem = args.output / f"grid{side}-{index}-{position}-{revision}"
                    try:
                        run = subprocess.run(command, capture_output=True, text=True, env=environment,
                                             timeout=min(OWNER_SECONDS, remaining), check=True)
                        stem.with_suffix(".stdout").write_text(run.stdout)
                        stem.with_suffix(".stderr").write_text(run.stderr)
                        record.update(json.loads(run.stdout))
                    except (subprocess.SubprocessError, ValueError, OSError) as error:
                        record.update(status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed",
                                      error=str(error))
                        if isinstance(error, subprocess.SubprocessError):
                            for suffix in ("stdout", "stderr"):
                                value = getattr(error, suffix, "") or ""
                                stem.with_suffix("." + suffix).write_text(value.decode(errors="replace") if isinstance(value, bytes) else value)
                records.append(record)
        report["cases"].append({"side": side, "records": records, "summary": summarize(records)})
        # Preserve partial coverage even when a later owner fails.
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    statuses = {case["summary"]["status"] for case in report["cases"]}
    report["status"] = next((status for status in ("invalid", "incomplete", "slowdown_signal", "inconclusive")
                             if status in statuses), "no_signal")
    report["completed_owners"] = sum(r["status"] == "complete" for c in report["cases"] for r in c["records"])
    report["rhs_checks"] = report["completed_owners"]
    report["wall_seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = markdown(report)
    (args.output / "summary.md").write_text(summary)
    print(summary)
    if report["status"] == "inconclusive":
        print("::warning::Performance comparison is INCONCLUSIVE because repeat controls exceeded the noise gate.")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as handle:
            handle.write(summary)
    return 1 if report["status"] in ("invalid", "incomplete", "slowdown_signal") else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", nargs=2, metavar=("SIDE", "ROUTE"), help=argparse.SUPPRESS)
    parser.add_argument("--base-python", type=Path)
    parser.add_argument("--head-python", type=Path)
    parser.add_argument("--base-revision")
    parser.add_argument("--head-revision")
    parser.add_argument("--route", choices=("cpu", "gpu"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.owner:
        print(json.dumps(owner(int(args.owner[0]), args.owner[1]), allow_nan=False))
        return 0
    if any(getattr(args, name) is None for name in
           ("base_python", "head_python", "base_revision", "head_revision", "route", "output")):
        parser.error("both Python environments, exact revisions, route and output are required")
    return campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())

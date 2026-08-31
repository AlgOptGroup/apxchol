#!/usr/bin/env python3
"""Run/merge the bounded local two-tree sparsifier Daint screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


MATRICES = (
    ("iter0010", "ipm/iter0010/matrix.mtx"),
    ("iter0020", "ipm/iter0020/matrix.mtx"),
    ("iter0040", "ipm/iter0040/matrix.mtx"),
    ("grid_500", "matrices/grid_500.mtx"),
)
ARMS = (
    ("baseline-before", None),
    ("q0.15", 0.15),
    ("q0.20", 0.20),
    ("q0.25", 0.25),
    ("baseline-after", None),
)
SOLVE_RE = re.compile(
    r"solve_result setup_ms=(?P<setup>[0-9.]+) "
    r"pcg_ms=(?P<pcg>[0-9.]+) total_ms=(?P<total>[0-9.]+) "
    r"iterations=(?P<iterations>[0-9]+) residual=(?P<residual>[^ ]+)"
)
STORED_RE = re.compile(r"sptrsv storage[^\n]*stored_nnz=(?P<stored>[0-9]+)")
FACTOR_RE = re.compile(r"\[apxchol\] nnz\(L\) = (?P<factor>[0-9]+)")
WORK_RE = re.compile(
    r"research clique work: pivots=(?P<pivots>[0-9]+) "
    r"raw_neighbors=(?P<raw>[0-9]+) "
    r"unique_neighbors=(?P<unique>[0-9]+) "
    r"emitted_edges=(?P<emitted>[0-9]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.strip()


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_output(text: str) -> dict[str, object]:
    solve = SOLVE_RE.search(text)
    stored = STORED_RE.search(text)
    factor = FACTOR_RE.search(text)
    work = WORK_RE.search(text)
    if solve is None or stored is None or factor is None or work is None:
        raise RuntimeError("missing solve, factor, storage, or clique-work output")
    return {
        "setup_ms": float(solve.group("setup")),
        "pcg_ms": float(solve.group("pcg")),
        "total_ms": float(solve.group("total")),
        "iterations": int(solve.group("iterations")),
        "residual": float(solve.group("residual")),
        "residual_text": solve.group("residual"),
        "factor_nnz": int(factor.group("factor")),
        "stored_nnz": int(stored.group("stored")),
        "pivots": int(work.group("pivots")),
        "raw_neighbors": int(work.group("raw")),
        "unique_neighbors": int(work.group("unique")),
        "emitted_edges": int(work.group("emitted")),
    }


def clean_environment(threads: int, keep: float | None, runner) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if (key.startswith("APXCHOL_") or key.startswith("OMP_") or
                key.startswith("KMP_") or key == "GOMP_CPU_AFFINITY"):
            env.pop(key)
    env.update({
        "APXCHOL_RESEARCH_LOCAL_TREE_STATS": "1",
        "APXCHOL_SPTRSV_FP16": "0",
        "APXCHOL_FACTOR_DROP": "1e-4",
        "APXCHOL_VERBOSE": "1",
        "APXCHOL_DUMP_NNZ": "1",
    })
    if keep is not None:
        env["APXCHOL_RESEARCH_LOCAL_TREE_KEEP"] = str(keep)
    return runner.benchmark_openmp_env(threads, base=env)


def source_metadata(source: Path, binary: Path) -> dict[str, object]:
    return {
        "hostname": socket.getfqdn(),
        "head": command_text(["git", "rev-parse", "HEAD"], source),
        "tracked_status": command_text(
            ["git", "status", "--short", "--untracked-files=no"], source),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "clang": command_text(["clang++", "--version"]).splitlines()[0],
    }


def run_rank(args: argparse.Namespace) -> int:
    rank = int(os.environ["SLURM_PROCID"])
    tasks = int(os.environ["SLURM_NTASKS"])
    if tasks != 4 or rank not in range(4):
        raise RuntimeError(f"expected four ranks, got rank={rank} tasks={tasks}")

    source = args.source.resolve()
    binary = args.binary.resolve()
    output = args.output.resolve()
    matrix_name, relative = MATRICES[rank]
    matrix = args.data_root.resolve() / relative
    if not binary.is_file() or not matrix.is_file():
        raise RuntimeError(f"missing binary or matrix: {binary}, {matrix}")
    rank_dir = output / f"rank-{rank}"
    if rank_dir.exists():
        raise RuntimeError(f"refusing to overwrite {rank_dir}")
    raw_dir = rank_dir / "raw"
    raw_dir.mkdir(parents=True)

    sys.path.insert(0, str(source / "benchmarks"))
    import runner_common as runner

    metadata = {
        **source_metadata(source, binary),
        "rank": rank,
        "tasks": tasks,
        "matrix": matrix_name,
        "matrix_path": str(matrix),
        "threads": args.threads,
        "seed": args.seed,
        "arms": [{"name": name, "keep": keep} for name, keep in ARMS],
        "affinity": runner.benchmark_openmp_provenance(args.threads),
        "allocation_affinity": sorted(os.sched_getaffinity(0)),
    }
    (rank_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    records: list[dict[str, object]] = []
    cpu_set = runner.affinity_spec(args.threads)
    for arm, keep in ARMS:
        raw_path = raw_dir / f"{matrix_name}-{arm}.log"
        command = [
            "taskset", "-c", cpu_set, str(binary), str(matrix),
            "--solve", "--graph-storage", "vec_pool_aos",
            "--is", "block_greedy", "--seed", str(args.seed),
        ]
        started = time.time()
        completed = subprocess.run(
            command, env=clean_environment(args.threads, keep, runner),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
        raw_path.write_text(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"{arm} failed rc={completed.returncode}: {raw_path}")
        parsed = parse_output(completed.stdout)
        if not math.isfinite(float(parsed["residual"])) or \
                float(parsed["residual"]) > 1e-8:
            raise RuntimeError(f"{arm} did not converge: {parsed['residual_text']}")
        record = {
            "rank": rank,
            "matrix": matrix_name,
            "threads": args.threads,
            "seed": args.seed,
            "arm": arm,
            "keep": keep,
            "elapsed_wall_s": time.time() - started,
            "raw": str(raw_path.relative_to(output)),
            "raw_sha256": sha256(raw_path),
            **parsed,
        }
        records.append(record)
        with (rank_dir / "records.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    baselines = [row for row in records if row["keep"] is None]
    signatures = {
        (row["factor_nnz"], row["stored_nnz"], row["iterations"],
         row["residual_text"], row["raw_neighbors"],
         row["unique_neighbors"], row["emitted_edges"])
        for row in baselines
    }
    if len(signatures) != 1:
        raise RuntimeError(f"baseline did not reproduce exactly: {signatures}")
    summary = {"checked_records": 5, "planned_records": 5,
               "baseline_exact": True, "converged_records": 5}
    (rank_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"rank {rank} {matrix_name}: checked 5/5; baseline exact; all converged")
    return 0


def merge(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    records: list[dict[str, object]] = []
    for rank in range(4):
        rank_dir = output / f"rank-{rank}"
        summary = json.loads((rank_dir / "summary.json").read_text())
        if summary != {"baseline_exact": True, "checked_records": 5,
                       "converged_records": 5, "planned_records": 5}:
            raise RuntimeError(f"rank {rank} summary mismatch: {summary}")
        records.extend(json.loads(line) for line in
                       (rank_dir / "records.jsonl").read_text().splitlines())
    if len(records) != 20:
        raise RuntimeError(f"checked {len(records)}/20 records")

    metrics = ("setup_ms", "pcg_ms", "total_ms", "factor_nnz",
               "stored_nnz", "iterations", "raw_neighbors",
               "unique_neighbors", "emitted_edges")
    candidates = ("q0.15", "q0.20", "q0.25")
    ratios = {candidate: {metric: [] for metric in metrics}
              for candidate in candidates}
    per_matrix: list[dict[str, object]] = []
    for matrix, _ in MATRICES:
        arms = {row["arm"]: row for row in records if row["matrix"] == matrix}
        before, after = arms["baseline-before"], arms["baseline-after"]
        for candidate in candidates:
            row = arms[candidate]
            candidate_ratios = {}
            for metric in metrics:
                base = math.sqrt(float(before[metric]) * float(after[metric]))
                value = float(row[metric]) / base
                ratios[candidate][metric].append(value)
                candidate_ratios[metric] = value
            per_matrix.append({
                "matrix": matrix,
                "candidate": candidate,
                "ratios": candidate_ratios,
                "iteration_delta": int(row["iterations"]) -
                                   int(before["iterations"]),
            })

    arm_order = [name for name, _ in ARMS]
    with (output / "records.jsonl").open("w") as handle:
        for row in sorted(records, key=lambda item: (
                [name for name, _ in MATRICES].index(item["matrix"]),
                arm_order.index(item["arm"]))):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "checked_records": 20,
        "planned_records": 20,
        "baseline_exact_cells": 4,
        "converged_records": 20,
        "per_matrix": per_matrix,
        "geomean_candidate_over_bracket": {
            candidate: {metric: geometric_mean(values)
                        for metric, values in candidate_metrics.items()}
            for candidate, candidate_metrics in ratios.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("checked 20/20 records; baseline exact 4/4; converged 20/20")
    print(json.dumps(summary["geomean_candidate_over_bracket"], sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.threads < 2 or args.threads > 72:
        parser.error("threads must be in [2,72]")
    return merge(args) if args.merge else run_rank(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

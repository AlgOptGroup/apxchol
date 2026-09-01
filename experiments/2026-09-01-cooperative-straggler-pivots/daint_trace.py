#!/usr/bin/env python3
"""Collect or merge structural elimination-straggler traces on Daint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


MATRICES = (
    ("com-Orkut", "matrices/com-Orkut.mtx"),
    ("com-LiveJournal", "matrices/com-LiveJournal.mtx"),
    ("grid_2000", "matrices/grid_2000.mtx"),
    ("iter0040", "ipm/iter0040/matrix.mtx"),
    ("com-Amazon", "matrices/com-Amazon.mtx"),
)
THREADS = (36, 72)
SEED = 42
PIVOT_RE = re.compile(r"^\[pivot-probe\]", re.MULTILINE)
FACTOR_RE = re.compile(r"rounds=(?P<rounds>[0-9]+)[^\n]*nnz\(L\)=(?P<nnz>[0-9]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cases() -> list[tuple[str, str, int]]:
    return [(name, relative, threads)
            for name, relative in MATRICES for threads in THREADS]


def clean_environment(threads: int, runner) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if (key.startswith("APXCHOL_") or key.startswith("OMP_") or
                key.startswith("KMP_") or key == "GOMP_CPU_AFFINITY"):
            env.pop(key)
    env.update({
        "APXCHOL_ROUND_TRACE": "1",
        "APXCHOL_SPTRSV_FP16": "0",
        "APXCHOL_FACTOR_DROP": "1e-4",
    })
    return runner.benchmark_openmp_env(threads, base=env)


def run_rank(args: argparse.Namespace) -> int:
    rank = int(os.environ["SLURM_PROCID"])
    tasks = int(os.environ["SLURM_NTASKS"])
    if tasks != 4 or rank not in range(4):
        raise RuntimeError(f"expected four ranks, got rank={rank} tasks={tasks}")

    source = args.source.resolve()
    binary = args.binary.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if not binary.is_file() or sha256(binary) != args.expected_binary_sha256:
        raise RuntimeError("baseline binary missing or SHA-256 mismatch")

    sys.path.insert(0, str(source / "benchmarks"))
    import runner_common as runner

    completed = 0
    for matrix_name, relative, threads in cases()[rank::tasks]:
        matrix = data_root / relative
        if not matrix.is_file():
            raise RuntimeError(f"missing matrix: {matrix}")
        record_dir = output / matrix_name / f"t{threads}"
        record_dir.mkdir(parents=True, exist_ok=False)
        log_path = record_dir / "run.log"
        command = [
            "taskset", "-c", runner.affinity_spec(threads),
            str(binary), str(matrix), "--profile",
            "--graph-storage", "vec_pool_aos",
            "--is", "block_greedy", "--seed", str(SEED),
        ]
        result = subprocess.run(
            command, env=clean_environment(threads, runner), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
        log_path.write_text(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"{matrix_name}/t{threads} failed rc={result.returncode}")
        factor = FACTOR_RE.search(result.stdout)
        pivot_records = len(PIVOT_RE.findall(result.stdout))
        if factor is None or pivot_records == 0:
            raise RuntimeError(
                f"{matrix_name}/t{threads}: missing factor or pivot trace")
        metadata = {
            "hostname": socket.getfqdn(),
            "rank": rank,
            "matrix": matrix_name,
            "matrix_path": str(matrix),
            "threads": threads,
            "seed": SEED,
            "binary": str(binary),
            "binary_sha256": args.expected_binary_sha256,
            "source_head": subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True, text=True, stdout=subprocess.PIPE,
            ).stdout.strip(),
            "rounds": int(factor.group("rounds")),
            "factor_nnz": int(factor.group("nnz")),
            "pivot_records": pivot_records,
            "log_sha256": sha256(log_path),
            "allocation_affinity": sorted(os.sched_getaffinity(0)),
        }
        (record_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        completed += 1
    print(f"rank {rank}: checked {completed}/{len(cases()[rank::tasks])} traces")
    return 0


def merge(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    paths = sorted(output.glob("*/t*/run.log"))
    if len(paths) != len(cases()):
        raise RuntimeError(f"expected {len(cases())} logs, found {len(paths)}")
    for path in paths:
        metadata = json.loads((path.parent / "metadata.json").read_text())
        if sha256(path) != metadata["log_sha256"]:
            raise RuntimeError(f"log hash mismatch: {path}")
        if metadata["binary_sha256"] != args.expected_binary_sha256:
            raise RuntimeError(f"binary provenance mismatch: {path}")

    analyzer = source / "benchmarks/dev/analyze_elimination_stragglers.py"
    summary = subprocess.run(
        [sys.executable, str(analyzer), str(output),
         "--expect-records", str(len(cases())),
         "--csv", str(output / "summary.csv")],
        check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    (output / "summary.txt").write_text(summary)
    print(summary, end="")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(merge(parsed) if parsed.merge else run_rank(parsed))


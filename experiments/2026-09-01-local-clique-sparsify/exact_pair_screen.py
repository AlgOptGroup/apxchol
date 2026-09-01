#!/usr/bin/env python3
"""Reusable repeated Daint bracket for output-identical q=.25 changes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import daint_screen as common
import fast_buffer_screen as buffer
import fast_union_screen as union


MATRICES = common.MATRICES
REPETITIONS = tuple(range(12))
ARMS = (
    ("control-before", "control"),
    ("candidate", "candidate"),
    ("control-after", "control"),
)
METRICS = buffer.METRICS


def run_rank(args: argparse.Namespace) -> int:
    rank = int(os.environ["SLURM_PROCID"])
    tasks = int(os.environ["SLURM_NTASKS"])
    if tasks != 4 or rank not in range(4):
        raise RuntimeError(f"expected four ranks, got rank={rank} tasks={tasks}")
    binaries = {
        "control": args.control_binary.resolve(),
        "candidate": args.candidate_binary.resolve(),
    }
    for label, binary in binaries.items():
        if not binary.is_file():
            raise RuntimeError(f"missing {label} binary: {binary}")
    matrix_name, relative = MATRICES[rank]
    matrix = args.data_root.resolve() / relative
    if not matrix.is_file():
        raise RuntimeError(f"missing matrix: {matrix}")
    rank_dir = args.output.resolve() / f"rank-{rank}"
    if rank_dir.exists():
        raise RuntimeError(f"refusing to overwrite {rank_dir}")
    raw_dir = rank_dir / "raw"
    raw_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.candidate_source.resolve() / "benchmarks"))
    import runner_common as runner
    cpu_set = runner.affinity_spec(args.threads)

    metadata = {
        "hostname": socket.getfqdn(), "rank": rank, "tasks": tasks,
        "matrix": {"name": matrix_name, "path": str(matrix)},
        "threads": args.threads, "seed": buffer.SEED,
        "repetitions": REPETITIONS,
        "arms": [{"name": name, "binary": binary} for name, binary in ARMS],
        "control": buffer.source_record(
            args.control_source, args.control_binary),
        "candidate": buffer.source_record(
            args.candidate_source, args.candidate_binary),
        "clang": common.command_text(["clang++", "--version"]).splitlines()[0],
        "affinity": runner.benchmark_openmp_provenance(args.threads),
        "allocation_affinity": sorted(os.sched_getaffinity(0)),
    }
    (rank_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    for label in ("control", "candidate"):
        _, text = buffer.run_one(
            binaries[label], matrix, args.threads, cpu_set, runner, args.timeout)
        (raw_dir / f"warm-{label}.log").write_text(text)

    for repetition in REPETITIONS:
        cell: list[dict[str, object]] = []
        for arm, binary_label in ARMS:
            started = time.time()
            parsed, text = buffer.run_one(
                binaries[binary_label], matrix, args.threads, cpu_set,
                runner, args.timeout)
            raw_path = raw_dir / f"rep{repetition}-{arm}.log"
            raw_path.write_text(text)
            row = {
                "rank": rank, "matrix": matrix_name, "threads": args.threads,
                "seed": buffer.SEED, "repetition": repetition, "arm": arm,
                "binary": binary_label, "elapsed_wall_s": time.time() - started,
                "raw": str(raw_path.relative_to(args.output.resolve())),
                "raw_sha256": common.sha256(raw_path), **parsed,
            }
            cell.append(row)
            with (rank_dir / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if len({union.signature(row) for row in cell}) != 1:
            raise RuntimeError(
                f"{matrix_name}/rep{repetition}: output mismatch")

    expected = len(REPETITIONS) * len(ARMS)
    summary = {
        "checked_records": expected, "planned_records": expected,
        "exact_repetitions": len(REPETITIONS),
        "converged_records": expected,
    }
    (rank_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"rank {rank}: checked {expected}/{expected}; "
          f"exact {len(REPETITIONS)}/{len(REPETITIONS)}")
    return 0


def merge(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    expected_sources = {
        "control": buffer.source_record(
            args.control_source, args.control_binary),
        "candidate": buffer.source_record(
            args.candidate_source, args.candidate_binary),
    }
    records: list[dict[str, object]] = []
    for rank in range(4):
        rank_dir = output / f"rank-{rank}"
        metadata = json.loads((rank_dir / "metadata.json").read_text())
        for label, expected in expected_sources.items():
            if metadata[label] != expected:
                raise RuntimeError(f"rank {rank} {label} metadata mismatch")
        summary = json.loads((rank_dir / "summary.json").read_text())
        if summary != {"checked_records": 36, "converged_records": 36,
                       "exact_repetitions": 12, "planned_records": 36}:
            raise RuntimeError(f"rank {rank} summary mismatch: {summary}")
        for line in (rank_dir / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            raw = output / row["raw"]
            if not raw.is_file() or common.sha256(raw) != row["raw_sha256"]:
                raise RuntimeError(f"rank {rank} raw mismatch: {row}")
            records.append(row)
    if len(records) != 144:
        raise RuntimeError(f"checked {len(records)}/144 records")

    ratios = {
        "candidate_over_control": {metric: [] for metric in METRICS},
        "control_right_over_left": {metric: [] for metric in METRICS},
    }
    per_repetition: list[dict[str, object]] = []
    for matrix, _ in MATRICES:
        for repetition in REPETITIONS:
            arms = {str(row["arm"]): row for row in records
                    if row["matrix"] == matrix and
                    row["repetition"] == repetition}
            values: dict[str, dict[str, float]] = {
                comparison: {} for comparison in ratios
            }
            for metric in METRICS:
                base = math.sqrt(
                    float(arms["control-before"][metric]) *
                    float(arms["control-after"][metric]))
                candidate = float(arms["candidate"][metric]) / base
                null = float(arms["control-after"][metric]) / \
                       float(arms["control-before"][metric])
                ratios["candidate_over_control"][metric].append(candidate)
                ratios["control_right_over_left"][metric].append(null)
                values["candidate_over_control"][metric] = candidate
                values["control_right_over_left"][metric] = null
            per_repetition.append({"matrix": matrix, "repetition": repetition,
                                   "ratios": values})

    aggregate = {
        comparison: {metric: common.geometric_mean(values)
                     for metric, values in metrics.items()}
        for comparison, metrics in ratios.items()
    }
    arm_order = [name for name, _ in ARMS]
    with (output / "records.jsonl").open("w") as handle:
        for row in sorted(records, key=lambda item: (
                [name for name, _ in MATRICES].index(item["matrix"]),
                item["repetition"], arm_order.index(item["arm"]))):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "checked_records": 144, "planned_records": 144,
        "exact_repetitions": 48, "converged_records": 144,
        "geomean_ratios": aggregate, "per_repetition": per_repetition,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("checked 144/144; exact 48/48; converged 144/144")
    print(json.dumps(aggregate, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    for label in ("control", "candidate"):
        parser.add_argument(f"--{label}-source", required=True, type=Path)
        parser.add_argument(f"--{label}-binary", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=72)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    return merge(args) if args.merge else run_rank(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, subprocess.TimeoutExpired,
            ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

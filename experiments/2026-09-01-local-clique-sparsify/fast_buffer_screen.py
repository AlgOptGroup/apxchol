#!/usr/bin/env python3
"""Repeated exact bracket for amortized deferred-edge reservation on Daint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import daint_screen as common
import fast_union_screen as union


MATRICES = common.MATRICES
SEED = 42
REPETITIONS = tuple(range(8))
ARMS = (
    ("old-before", "old"),
    ("grouped-before", "grouped"),
    ("amortized", "amortized"),
    ("grouped-after", "grouped"),
    ("old-after", "old"),
)
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(?P<rss>[0-9]+)")
METRICS = union.METRICS + ("max_rss_kb",)


def source_record(source: Path, binary: Path) -> dict[str, object]:
    return {
        "source": str(source.resolve()),
        "head": common.command_text(["git", "rev-parse", "HEAD"], source),
        "status": common.command_text(
            ["git", "status", "--short", "--untracked-files=no"], source),
        "binary": str(binary.resolve()),
        "binary_sha256": common.sha256(binary.resolve()),
    }


def run_one(binary: Path, matrix: Path, threads: int, cpu_set: str,
            runner, timeout: int) -> tuple[dict[str, object], str]:
    command = [
        "/usr/bin/time", "-v", "taskset", "-c", cpu_set, str(binary),
        str(matrix), "--solve", "--graph-storage", "vec_pool_aos",
        "--is", "block_greedy", "--seed", str(SEED),
    ]
    completed = subprocess.run(
        command,
        env=common.clean_environment(threads, 0.25, 1e-3, runner),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"binary failed rc={completed.returncode}")
    parsed = common.parse_output(completed.stdout)
    rss = RSS_RE.search(completed.stdout)
    if rss is None:
        raise RuntimeError("missing maximum RSS from /usr/bin/time")
    parsed["max_rss_kb"] = int(rss.group("rss"))
    residual = float(parsed["residual"])
    if not math.isfinite(residual) or residual > 1e-8:
        raise RuntimeError(f"did not converge: {parsed['residual_text']}")
    return parsed, completed.stdout


def run_rank(args: argparse.Namespace) -> int:
    rank = int(os.environ["SLURM_PROCID"])
    tasks = int(os.environ["SLURM_NTASKS"])
    if tasks != 4 or rank not in range(4):
        raise RuntimeError(f"expected four ranks, got rank={rank} tasks={tasks}")
    binaries = {
        "old": args.old_binary.resolve(),
        "grouped": args.grouped_binary.resolve(),
        "amortized": args.amortized_binary.resolve(),
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
    sys.path.insert(0, str(args.amortized_source.resolve() / "benchmarks"))
    import runner_common as runner
    cpu_set = runner.affinity_spec(args.threads)

    metadata = {
        "hostname": socket.getfqdn(), "rank": rank, "tasks": tasks,
        "matrix": {"name": matrix_name, "path": str(matrix)},
        "threads": args.threads, "seed": SEED,
        "repetitions": REPETITIONS,
        "arms": [{"name": name, "binary": binary} for name, binary in ARMS],
        "old": source_record(args.old_source, args.old_binary),
        "grouped": source_record(args.grouped_source, args.grouped_binary),
        "amortized": source_record(
            args.amortized_source, args.amortized_binary),
        "clang": common.command_text(["clang++", "--version"]).splitlines()[0],
        "affinity": runner.benchmark_openmp_provenance(args.threads),
        "allocation_affinity": sorted(os.sched_getaffinity(0)),
    }
    (rank_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    # Untimed process/page-cache warm-up for every executable.
    for label in ("old", "grouped", "amortized"):
        _, text = run_one(
            binaries[label], matrix, args.threads, cpu_set, runner, args.timeout)
        (raw_dir / f"warm-{label}.log").write_text(text)

    records: list[dict[str, object]] = []
    for repetition in REPETITIONS:
        cell: list[dict[str, object]] = []
        for arm, binary_label in ARMS:
            started = time.time()
            parsed, text = run_one(
                binaries[binary_label], matrix, args.threads, cpu_set,
                runner, args.timeout)
            raw_path = raw_dir / f"rep{repetition}-{arm}.log"
            raw_path.write_text(text)
            row = {
                "rank": rank, "matrix": matrix_name, "threads": args.threads,
                "seed": SEED, "repetition": repetition, "arm": arm,
                "binary": binary_label, "elapsed_wall_s": time.time() - started,
                "raw": str(raw_path.relative_to(args.output.resolve())),
                "raw_sha256": common.sha256(raw_path), **parsed,
            }
            records.append(row)
            cell.append(row)
            with (rank_dir / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if len({union.signature(row) for row in cell}) != 1:
            raise RuntimeError(
                f"{matrix_name}/rep{repetition}: factor output mismatch")

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


def geomean(values: list[float]) -> float:
    return common.geometric_mean(values)


def merge(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    expected_sources = {
        "old": source_record(args.old_source, args.old_binary),
        "grouped": source_record(args.grouped_source, args.grouped_binary),
        "amortized": source_record(
            args.amortized_source, args.amortized_binary),
    }
    records: list[dict[str, object]] = []
    for rank in range(4):
        rank_dir = output / f"rank-{rank}"
        metadata = json.loads((rank_dir / "metadata.json").read_text())
        for label, expected in expected_sources.items():
            if metadata[label] != expected:
                raise RuntimeError(f"rank {rank} {label} metadata mismatch")
        summary = json.loads((rank_dir / "summary.json").read_text())
        if summary != {"checked_records": 40, "converged_records": 40,
                       "exact_repetitions": 8, "planned_records": 40}:
            raise RuntimeError(f"rank {rank} summary mismatch: {summary}")
        for line in (rank_dir / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            raw = output / row["raw"]
            if not raw.is_file() or common.sha256(raw) != row["raw_sha256"]:
                raise RuntimeError(f"rank {rank} raw mismatch: {row}")
            records.append(row)
    if len(records) != 160:
        raise RuntimeError(f"checked {len(records)}/160 records")

    comparisons = {
        "amortized_over_grouped": {metric: [] for metric in METRICS},
        "grouped_over_old": {metric: [] for metric in METRICS},
        "amortized_over_old": {metric: [] for metric in METRICS},
        "grouped_right_over_left": {metric: [] for metric in METRICS},
        "old_right_over_left": {metric: [] for metric in METRICS},
    }
    per_repetition: list[dict[str, object]] = []
    for matrix, _ in MATRICES:
        for repetition in REPETITIONS:
            arms = {str(row["arm"]): row for row in records
                    if row["matrix"] == matrix and
                    row["repetition"] == repetition}
            bases = {
                "grouped": {
                    metric: math.sqrt(
                        float(arms["grouped-before"][metric]) *
                        float(arms["grouped-after"][metric]))
                    for metric in METRICS
                },
                "old": {
                    metric: math.sqrt(float(arms["old-before"][metric]) *
                                      float(arms["old-after"][metric]))
                    for metric in METRICS
                },
            }
            values: dict[str, dict[str, float]] = {}
            for comparison in comparisons:
                values[comparison] = {}
                for metric in METRICS:
                    if comparison == "amortized_over_grouped":
                        ratio = float(arms["amortized"][metric]) / \
                                bases["grouped"][metric]
                    elif comparison == "grouped_over_old":
                        ratio = bases["grouped"][metric] / bases["old"][metric]
                    elif comparison == "amortized_over_old":
                        ratio = float(arms["amortized"][metric]) / \
                                bases["old"][metric]
                    elif comparison == "grouped_right_over_left":
                        ratio = float(arms["grouped-after"][metric]) / \
                                float(arms["grouped-before"][metric])
                    else:
                        ratio = float(arms["old-after"][metric]) / \
                                float(arms["old-before"][metric])
                    comparisons[comparison][metric].append(ratio)
                    values[comparison][metric] = ratio
            per_repetition.append({"matrix": matrix, "repetition": repetition,
                                   "ratios": values})

    aggregates = {
        comparison: {metric: geomean(values)
                     for metric, values in metrics.items()}
        for comparison, metrics in comparisons.items()
    }
    arm_order = [name for name, _ in ARMS]
    with (output / "records.jsonl").open("w") as handle:
        for row in sorted(records, key=lambda item: (
                [name for name, _ in MATRICES].index(item["matrix"]),
                item["repetition"], arm_order.index(item["arm"]))):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "checked_records": 160, "planned_records": 160,
        "exact_repetitions": 32, "converged_records": 160,
        "geomean_ratios": aggregates, "per_repetition": per_repetition,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("checked 160/160; exact 32/32; converged 160/160")
    print(json.dumps(aggregates, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    for label in ("old", "grouped", "amortized"):
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

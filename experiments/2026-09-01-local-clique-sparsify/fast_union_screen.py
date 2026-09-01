#!/usr/bin/env python3
"""Bracket the output-identical local two-tree union optimization on Daint."""

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


MATRICES = common.MATRICES
SEEDS = (1, 17, 42, 73, 97)
ARMS = (
    ("production-before", "old", None, None),
    ("old-q-before", "old", 0.25, 1e-3),
    ("new-q", "new", 0.25, 1e-3),
    ("old-q-after", "old", 0.25, 1e-3),
    ("production-after", "old", None, None),
)
METRICS = (
    "setup_ms", "pcg_ms", "total_ms", "factor_nnz", "stored_nnz",
    "iterations", "raw_neighbors", "unique_neighbors", "emitted_edges",
)


def metadata_for(args: argparse.Namespace, rank: int,
                 matrices: list[tuple[str, Path]], runner) -> dict[str, object]:
    return {
        "hostname": socket.getfqdn(),
        "rank": rank,
        "tasks": 4,
        "matrices": [{"name": name, "path": str(path)}
                     for name, path in matrices],
        "threads": args.threads,
        "seeds": SEEDS,
        "arms": [
            {"name": name, "binary": binary, "keep": keep,
             "trigger_rel": trigger}
            for name, binary, keep, trigger in ARMS
        ],
        "new_source": str(args.new_source.resolve()),
        "new_head": common.command_text(
            ["git", "rev-parse", "HEAD"], args.new_source),
        "new_status": common.command_text(
            ["git", "status", "--short", "--untracked-files=no"],
            args.new_source),
        "new_binary": str(args.new_binary.resolve()),
        "new_binary_sha256": common.sha256(args.new_binary.resolve()),
        "old_source": str(args.old_source.resolve()),
        "old_head": common.command_text(
            ["git", "rev-parse", "HEAD"], args.old_source),
        "old_status": common.command_text(
            ["git", "status", "--short", "--untracked-files=no"],
            args.old_source),
        "old_binary": str(args.old_binary.resolve()),
        "old_binary_sha256": common.sha256(args.old_binary.resolve()),
        "clang": common.command_text(["clang++", "--version"]).splitlines()[0],
        "affinity": runner.benchmark_openmp_provenance(args.threads),
        "allocation_affinity": sorted(os.sched_getaffinity(0)),
    }


def signature(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["factor_nnz"], row["stored_nnz"], row["iterations"],
        row["residual_text"], row["pivots"], row["raw_neighbors"],
        row["unique_neighbors"], row["emitted_edges"],
    )


def run_rank(args: argparse.Namespace) -> int:
    rank = int(os.environ["SLURM_PROCID"])
    tasks = int(os.environ["SLURM_NTASKS"])
    if tasks != 4 or rank not in range(4):
        raise RuntimeError(f"expected four ranks, got rank={rank} tasks={tasks}")
    binaries = {
        "old": args.old_binary.resolve(),
        "new": args.new_binary.resolve(),
    }
    for label, binary in binaries.items():
        if not binary.is_file():
            raise RuntimeError(f"missing {label} binary: {binary}")
    matrices = [
        (name, args.data_root.resolve() / relative)
        for name, relative in MATRICES[rank::tasks]
    ]
    missing = [str(path) for _, path in matrices if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing matrices: {missing}")

    rank_dir = args.output.resolve() / f"rank-{rank}"
    if rank_dir.exists():
        raise RuntimeError(f"refusing to overwrite {rank_dir}")
    raw_dir = rank_dir / "raw"
    raw_dir.mkdir(parents=True)

    sys.path.insert(0, str(args.new_source.resolve() / "benchmarks"))
    import runner_common as runner

    (rank_dir / "metadata.json").write_text(
        json.dumps(metadata_for(args, rank, matrices, runner),
                   indent=2, sort_keys=True) + "\n")
    records: list[dict[str, object]] = []
    cpu_set = runner.affinity_spec(args.threads)
    for matrix_name, matrix in matrices:
        for seed in SEEDS:
            cell: list[dict[str, object]] = []
            for arm, binary_label, keep, trigger in ARMS:
                raw_path = raw_dir / f"{matrix_name}-s{seed}-{arm}.log"
                command = [
                    "taskset", "-c", cpu_set, str(binaries[binary_label]),
                    str(matrix), "--solve", "--graph-storage", "vec_pool_aos",
                    "--is", "block_greedy", "--seed", str(seed),
                ]
                started = time.time()
                completed = subprocess.run(
                    command,
                    env=common.clean_environment(
                        args.threads, keep, trigger, runner),
                    text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=args.timeout,
                )
                raw_path.write_text(completed.stdout)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{matrix_name}/seed{seed}/{arm} failed "
                        f"rc={completed.returncode}: {raw_path}")
                parsed = common.parse_output(completed.stdout)
                residual = float(parsed["residual"])
                if not math.isfinite(residual) or residual > 1e-8:
                    raise RuntimeError(
                        f"{matrix_name}/seed{seed}/{arm} did not converge: "
                        f"{parsed['residual_text']}")
                row = {
                    "rank": rank, "matrix": matrix_name,
                    "threads": args.threads, "seed": seed, "arm": arm,
                    "binary": binary_label, "keep": keep,
                    "trigger_rel": trigger,
                    "elapsed_wall_s": time.time() - started,
                    "raw": str(raw_path.relative_to(args.output.resolve())),
                    "raw_sha256": common.sha256(raw_path), **parsed,
                }
                records.append(row)
                cell.append(row)
                with (rank_dir / "records.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

            by_arm = {str(row["arm"]): row for row in cell}
            production = {
                signature(by_arm["production-before"]),
                signature(by_arm["production-after"]),
            }
            candidate = {
                signature(by_arm["old-q-before"]),
                signature(by_arm["new-q"]),
                signature(by_arm["old-q-after"]),
            }
            if len(production) != 1:
                raise RuntimeError(
                    f"{matrix_name}/seed{seed}: production mismatch")
            if len(candidate) != 1:
                raise RuntimeError(
                    f"{matrix_name}/seed{seed}: optimized output mismatch")

    expected = len(matrices) * len(SEEDS) * len(ARMS)
    cells = len(matrices) * len(SEEDS)
    summary = {
        "checked_records": expected, "planned_records": expected,
        "production_exact_cells": cells,
        "candidate_exact_cells": cells,
        "converged_records": expected,
    }
    (rank_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"rank {rank}: checked {expected}/{expected}; "
          f"production exact {cells}/{cells}; candidate exact {cells}/{cells}")
    return 0


def merge(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    records: list[dict[str, object]] = []
    expected_new_head = common.command_text(
        ["git", "rev-parse", "HEAD"], args.new_source)
    expected_old_head = common.command_text(
        ["git", "rev-parse", "HEAD"], args.old_source)
    expected_new_sha = common.sha256(args.new_binary.resolve())
    expected_old_sha = common.sha256(args.old_binary.resolve())
    for rank in range(4):
        rank_dir = output / f"rank-{rank}"
        metadata = json.loads((rank_dir / "metadata.json").read_text())
        if (metadata["new_head"] != expected_new_head or
                metadata["old_head"] != expected_old_head or
                metadata["new_status"] != "" or metadata["old_status"] != "" or
                metadata["new_binary_sha256"] != expected_new_sha or
                metadata["old_binary_sha256"] != expected_old_sha):
            raise RuntimeError(f"rank {rank} metadata mismatch")
        summary = json.loads((rank_dir / "summary.json").read_text())
        expected_rank = {
            "candidate_exact_cells": 5, "checked_records": 25,
            "converged_records": 25, "planned_records": 25,
            "production_exact_cells": 5,
        }
        if summary != expected_rank:
            raise RuntimeError(f"rank {rank} summary mismatch: {summary}")
        for line in (rank_dir / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            raw = output / row["raw"]
            if not raw.is_file() or common.sha256(raw) != row["raw_sha256"]:
                raise RuntimeError(f"rank {rank} raw mismatch: {row}")
            records.append(row)
    if len(records) != 100:
        raise RuntimeError(f"checked {len(records)}/100 records")

    ratios = {
        "new_over_old_q": {metric: [] for metric in METRICS},
        "new_q_over_production": {metric: [] for metric in METRICS},
    }
    per_cell: list[dict[str, object]] = []
    for matrix, _ in MATRICES:
        for seed in SEEDS:
            arms = {str(row["arm"]): row for row in records
                    if row["matrix"] == matrix and row["seed"] == seed}
            old_q = {
                metric: math.sqrt(float(arms["old-q-before"][metric]) *
                                  float(arms["old-q-after"][metric]))
                for metric in METRICS
            }
            production = {
                metric: math.sqrt(float(arms["production-before"][metric]) *
                                  float(arms["production-after"][metric]))
                for metric in METRICS
            }
            new = arms["new-q"]
            cell_ratios: dict[str, dict[str, float]] = {}
            for comparison, base in (
                    ("new_over_old_q", old_q),
                    ("new_q_over_production", production)):
                cell_ratios[comparison] = {}
                for metric in METRICS:
                    ratio = float(new[metric]) / base[metric]
                    ratios[comparison][metric].append(ratio)
                    cell_ratios[comparison][metric] = ratio
            per_cell.append({"matrix": matrix, "seed": seed,
                             "ratios": cell_ratios})

    arm_order = [name for name, *_ in ARMS]
    with (output / "records.jsonl").open("w") as handle:
        for row in sorted(records, key=lambda item: (
                [name for name, _ in MATRICES].index(item["matrix"]),
                SEEDS.index(item["seed"]), arm_order.index(item["arm"]))):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    aggregate = {
        comparison: {
            metric: common.geometric_mean(values)
            for metric, values in metrics.items()
        } for comparison, metrics in ratios.items()
    }
    summary = {
        "checked_records": 100, "planned_records": 100,
        "production_exact_cells": 20, "candidate_exact_cells": 20,
        "converged_records": 100,
        "geomean_ratios": aggregate, "per_cell": per_cell,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("checked 100/100; production exact 20/20; candidate exact 20/20")
    print(json.dumps(aggregate, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-source", required=True, type=Path)
    parser.add_argument("--old-source", required=True, type=Path)
    parser.add_argument("--new-binary", required=True, type=Path)
    parser.add_argument("--old-binary", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=72)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.threads < 2 or args.threads > 72:
        parser.error("threads must be in [2,72]")
    return merge(args) if args.merge else run_rank(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, subprocess.TimeoutExpired,
            ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

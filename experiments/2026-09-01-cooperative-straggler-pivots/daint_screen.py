#!/usr/bin/env python3
"""Run and merge the bounded cooperative-straggler-pivot Daint screen."""

from __future__ import annotations

import argparse
import csv
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
    ("com-Orkut", "matrices/com-Orkut.mtx"),
    ("com-LiveJournal", "matrices/com-LiveJournal.mtx"),
    ("grid_2000", "matrices/grid_2000.mtx"),
    ("iter0040", "ipm/iter0040/matrix.mtx"),
    ("com-Amazon", "matrices/com-Amazon.mtx"),
)
THREADS = (36, 72)
SEEDS = (1, 42, 97)
ARMS = (
    ("baseline-0", "baseline"),
    ("candidate", "candidate"),
    ("baseline-1", "baseline"),
)

SOLVE_RE = re.compile(
    r"solve_result setup_ms=(?P<setup>[0-9.]+) "
    r"pcg_ms=(?P<pcg>[0-9.]+) total_ms=(?P<total>[0-9.]+) "
    r"iterations=(?P<iterations>[0-9]+) residual=(?P<residual>[^ ]+)"
)
BREAKDOWN_RE = re.compile(
    r"setup_breakdown find_partition_ms=(?P<find_partition>[0-9.]+) "
    r"eliminate_ms=(?P<eliminate>[0-9.]+) "
    r"merge_is_ms=(?P<merge_is>[0-9.]+) "
    r"compute_ms=(?P<compute>[0-9.]+) apply_ms=(?P<apply>[0-9.]+) "
    r"compute_apply_fused_ms=(?P<compute_apply_fused>[0-9.]+) "
    r"elim_remaining_ms=(?P<elim_remaining>[0-9.]+) "
    r"assembly_ms=(?P<assembly>[0-9.]+) "
    r"sptrsv_setup_ms=(?P<sptrsv_setup>[0-9.]+) "
    r"spmv_setup_ms=(?P<spmv_setup>[0-9.]+)"
)
STORED_RE = re.compile(r"sptrsv storage[^\n]*stored_nnz=(?P<stored>[0-9]+)")
FACTOR_RE = re.compile(r"\[apxchol\] nnz\(L\) = (?P<factor>[0-9]+)")
COOPERATIVE_RE = re.compile(
    r"\[cooperative-pivot\] round=(?P<round>[0-9]+) "
    r"pivots=(?P<pivots>[0-9]+) team=(?P<team>[0-9]+) "
    r"assisted=(?P<assisted>[0-9]+) "
    r"sample_sources=(?P<sources>[0-9]+) tasks=(?P<tasks>[0-9]+) "
    r"max_helpers=(?P<helpers>[0-9]+)"
)

TIME_METRICS = (
    "setup_ms", "pcg_ms", "total_ms", "find_partition_ms",
    "eliminate_ms", "merge_is_ms", "compute_ms", "apply_ms",
    "compute_apply_fused_ms", "elim_remaining_ms", "assembly_ms",
    "sptrsv_setup_ms", "spmv_setup_ms",
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
    breakdown = BREAKDOWN_RE.search(text)
    stored = STORED_RE.search(text)
    factor = FACTOR_RE.search(text)
    if solve is None or breakdown is None or stored is None or factor is None:
        raise RuntimeError("missing solve, setup-breakdown, factor, or storage output")
    cooperative = [match.groupdict() for match in COOPERATIVE_RE.finditer(text)]
    parsed: dict[str, object] = {
        "setup_ms": float(solve.group("setup")),
        "pcg_ms": float(solve.group("pcg")),
        "total_ms": float(solve.group("total")),
        "iterations": int(solve.group("iterations")),
        "residual": float(solve.group("residual")),
        "residual_text": solve.group("residual"),
        "factor_nnz": int(factor.group("factor")),
        "stored_nnz": int(stored.group("stored")),
        "cooperative_rounds": len(cooperative),
        "assisted_pivots": sum(int(row["assisted"]) for row in cooperative),
        "sample_sources": sum(int(row["sources"]) for row in cooperative),
        "sample_tasks": sum(int(row["tasks"]) for row in cooperative),
        "max_helpers": max((int(row["helpers"]) for row in cooperative), default=0),
    }
    for name, value in breakdown.groupdict().items():
        parsed[f"{name}_ms"] = float(value)
    return parsed


def clean_environment(threads: int, candidate: bool, runner) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if (key.startswith("APXCHOL_") or key.startswith("OMP_") or
                key.startswith("KMP_") or key == "GOMP_CPU_AFFINITY"):
            env.pop(key)
    env.update({
        "APXCHOL_SPTRSV_FP16": "0",
        "APXCHOL_FACTOR_DROP": "1e-4",
        "APXCHOL_VERBOSE": "1",
        "APXCHOL_DUMP_NNZ": "1",
    })
    if candidate:
        env["APXCHOL_COOPERATIVE_PIVOT_TRACE"] = "1"
    return runner.benchmark_openmp_env(threads, base=env)


def source_metadata(source: Path, baseline: Path,
                    candidate: Path) -> dict[str, object]:
    return {
        "hostname": socket.getfqdn(),
        "head": command_text(["git", "rev-parse", "HEAD"], source),
        "tracked_status": command_text(
            ["git", "status", "--short", "--untracked-files=no"], source),
        "baseline_binary": str(baseline),
        "baseline_sha256": sha256(baseline),
        "candidate_binary": str(candidate),
        "candidate_sha256": sha256(candidate),
        "clang": command_text(["clang++", "--version"]).splitlines()[0],
    }


def all_cases() -> list[tuple[str, str, int, int]]:
    return [
        (name, relative, threads, seed)
        for name, relative in MATRICES
        for threads in THREADS
        for seed in SEEDS
    ]


def run_rank(args: argparse.Namespace) -> int:
    rank = int(os.environ["SLURM_PROCID"])
    tasks = int(os.environ["SLURM_NTASKS"])
    if tasks != 4 or rank not in range(4):
        raise RuntimeError(f"expected four ranks, got rank={rank} tasks={tasks}")

    source = args.source.resolve()
    baseline = args.baseline_binary.resolve()
    candidate = args.candidate_binary.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    cases = all_cases()[rank::tasks]
    if not baseline.is_file() or not candidate.is_file():
        raise RuntimeError(f"missing binaries: {baseline}, {candidate}")
    missing = [str(data_root / relative) for _, relative, _, _ in cases
               if not (data_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"missing matrices: {missing}")

    rank_dir = output / f"rank-{rank}"
    if rank_dir.exists():
        raise RuntimeError(f"refusing to overwrite {rank_dir}")
    raw_dir = rank_dir / "raw"
    raw_dir.mkdir(parents=True)

    sys.path.insert(0, str(source / "benchmarks"))
    import runner_common as runner

    metadata = {
        **source_metadata(source, baseline, candidate),
        "rank": rank,
        "tasks": tasks,
        "threads": THREADS,
        "seeds": SEEDS,
        "arms": [{"name": name, "binary_kind": kind} for name, kind in ARMS],
        "cases": [
            {"matrix": name, "path": str(data_root / relative),
             "threads": threads, "seed": seed}
            for name, relative, threads, seed in cases
        ],
        "allocation_affinity": sorted(os.sched_getaffinity(0)),
    }
    (rank_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    records: list[dict[str, object]] = []
    for matrix_name, relative, threads, seed in cases:
        matrix = data_root / relative
        cpu_set = runner.affinity_spec(threads)
        for arm, binary_kind in ARMS:
            executable = candidate if binary_kind == "candidate" else baseline
            raw_path = raw_dir / f"{matrix_name}-t{threads}-s{seed}-{arm}.log"
            command = [
                "taskset", "-c", cpu_set, str(executable), str(matrix),
                "--solve", "--graph-storage", "vec_pool_aos",
                "--is", "block_greedy", "--seed", str(seed),
            ]
            started = time.time()
            completed = subprocess.run(
                command,
                env=clean_environment(threads, binary_kind == "candidate", runner),
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            raw_path.write_text(completed.stdout)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{matrix_name}/t{threads}/s{seed}/{arm} failed "
                    f"rc={completed.returncode}: {raw_path}")
            parsed = parse_output(completed.stdout)
            if not math.isfinite(float(parsed["residual"])) or \
                    float(parsed["residual"]) > 1e-8:
                raise RuntimeError(
                    f"{matrix_name}/t{threads}/s{seed}/{arm} did not converge: "
                    f"{parsed['residual_text']}")
            record = {
                "rank": rank,
                "matrix": matrix_name,
                "threads": threads,
                "seed": seed,
                "arm": arm,
                "binary_kind": binary_kind,
                "elapsed_wall_s": time.time() - started,
                "raw": str(raw_path.relative_to(output)),
                "raw_sha256": sha256(raw_path),
                **parsed,
            }
            records.append(record)
            with (rank_dir / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        cell = [row for row in records
                if row["matrix"] == matrix_name and
                row["threads"] == threads and row["seed"] == seed]
        signatures = {
            (row["factor_nnz"], row["stored_nnz"], row["iterations"],
             row["residual_text"])
            for row in cell
        }
        if len(cell) != len(ARMS) or len(signatures) != 1:
            raise RuntimeError(
                f"{matrix_name}/t{threads}/s{seed}: output mismatch: {signatures}")

    expected = len(cases) * len(ARMS)
    exact_cells = len(cases)
    summary = {
        "checked_records": expected,
        "planned_records": expected,
        "output_exact_cells": exact_cells,
        "converged_records": expected,
    }
    (rank_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"rank {rank}: checked {expected}/{expected}; "
          f"outputs exact {exact_cells}/{exact_cells}; all converged")
    return 0


def bracket_ratio(cell: list[dict[str, object]], metric: str) -> float:
    by_arm = {str(row["arm"]): row for row in cell}
    values = [float(by_arm[arm][metric])
              for arm in ("baseline-0", "candidate", "baseline-1")]
    if values == [0.0, 0.0, 0.0]:
        return 1.0
    baseline = math.sqrt(
        values[0] * values[2])
    if baseline == 0.0:
        raise ValueError(f"zero baseline for nonzero {metric}: {values}")
    return values[1] / baseline


def merge(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    source = args.source.resolve()
    baseline = args.baseline_binary.resolve()
    candidate = args.candidate_binary.resolve()
    expected_head = command_text(["git", "rev-parse", "HEAD"], source)
    expected_baseline_sha = sha256(baseline)
    expected_candidate_sha = sha256(candidate)
    records: list[dict[str, object]] = []
    for rank in range(4):
        rank_dir = output / f"rank-{rank}"
        metadata = json.loads((rank_dir / "metadata.json").read_text())
        if metadata["head"] != expected_head or metadata["tracked_status"]:
            raise RuntimeError(f"rank {rank}: source metadata mismatch")
        if metadata["baseline_sha256"] != expected_baseline_sha or \
                metadata["candidate_sha256"] != expected_candidate_sha:
            raise RuntimeError(f"rank {rank}: binary metadata mismatch")
        for line in (rank_dir / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            raw_path = output / row["raw"]
            if sha256(raw_path) != row["raw_sha256"]:
                raise RuntimeError(f"raw hash mismatch: {raw_path}")
            records.append(row)

    expected_records = len(all_cases()) * len(ARMS)
    if len(records) != expected_records:
        raise RuntimeError(f"expected {expected_records} records, found {len(records)}")

    cells: list[list[dict[str, object]]] = []
    for matrix_name, _, threads, seed in all_cases():
        cell = [row for row in records
                if row["matrix"] == matrix_name and
                row["threads"] == threads and row["seed"] == seed]
        signatures = {
            (row["factor_nnz"], row["stored_nnz"], row["iterations"],
             row["residual_text"])
            for row in cell
        }
        if len(cell) != len(ARMS) or len(signatures) != 1:
            raise RuntimeError(
                f"{matrix_name}/t{threads}/s{seed}: output mismatch: {signatures}")
        cells.append(cell)

    summary_rows: list[dict[str, object]] = []
    for threads in THREADS:
        for matrix_name, _ in MATRICES:
            selected = [cell for cell in cells
                        if cell[0]["threads"] == threads and
                        cell[0]["matrix"] == matrix_name]
            candidates = [next(row for row in cell if row["arm"] == "candidate")
                          for cell in selected]
            row: dict[str, object] = {
                "scope": matrix_name,
                "threads": threads,
                "cells": len(selected),
                "activated_cells": sum(int(item["cooperative_rounds"]) > 0
                                       for item in candidates),
                "cooperative_rounds": sum(int(item["cooperative_rounds"])
                                          for item in candidates),
                "assisted_pivots": sum(int(item["assisted_pivots"])
                                       for item in candidates),
                "sample_sources": sum(int(item["sample_sources"])
                                      for item in candidates),
                "sample_tasks": sum(int(item["sample_tasks"])
                                    for item in candidates),
                "max_helpers": max(int(item["max_helpers"])
                                   for item in candidates),
            }
            for metric in TIME_METRICS:
                row[f"{metric}_ratio"] = geometric_mean(
                    [bracket_ratio(cell, metric) for cell in selected])
            null = []
            for cell in selected:
                by_arm = {str(item["arm"]): item for item in cell}
                null.append(float(by_arm["baseline-1"]["setup_ms"]) /
                            float(by_arm["baseline-0"]["setup_ms"]))
            row["setup_null_ratio"] = geometric_mean(null)
            summary_rows.append(row)

        selected = [cell for cell in cells if cell[0]["threads"] == threads]
        candidates = [next(row for row in cell if row["arm"] == "candidate")
                      for cell in selected]
        pooled: dict[str, object] = {
            "scope": "ALL",
            "threads": threads,
            "cells": len(selected),
            "activated_cells": sum(int(item["cooperative_rounds"]) > 0
                                   for item in candidates),
            "cooperative_rounds": sum(int(item["cooperative_rounds"])
                                      for item in candidates),
            "assisted_pivots": sum(int(item["assisted_pivots"])
                                   for item in candidates),
            "sample_sources": sum(int(item["sample_sources"])
                                  for item in candidates),
            "sample_tasks": sum(int(item["sample_tasks"])
                                for item in candidates),
            "max_helpers": max(int(item["max_helpers"])
                               for item in candidates),
        }
        for metric in TIME_METRICS:
            pooled[f"{metric}_ratio"] = geometric_mean(
                [bracket_ratio(cell, metric) for cell in selected])
        pooled["setup_null_ratio"] = geometric_mean([
            float(next(row for row in cell if row["arm"] == "baseline-1")["setup_ms"]) /
            float(next(row for row in cell if row["arm"] == "baseline-0")["setup_ms"])
            for cell in selected
        ])
        summary_rows.append(pooled)

    fieldnames = list(summary_rows[0])
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records))

    activated = sum(
        int(next(row for row in cell if row["arm"] == "candidate")
            ["cooperative_rounds"]) > 0
        for cell in cells
    )
    lines = [
        f"checked {len(records)}/{expected_records} records",
        f"output-exact cells {len(cells)}/{len(cells)}",
        f"converged {len(records)}/{len(records)}",
        f"candidate activated in {activated}/{len(cells)} cells",
        "",
        "scope threads cells active setup eliminate compute apply total null_setup "
        "coop_rounds sources tasks",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['scope']} {row['threads']} {row['cells']} "
            f"{row['activated_cells']} {row['setup_ms_ratio']:.6f} "
            f"{row['eliminate_ms_ratio']:.6f} {row['compute_ms_ratio']:.6f} "
            f"{row['apply_ms_ratio']:.6f} {row['total_ms_ratio']:.6f} "
            f"{row['setup_null_ratio']:.6f} {row['cooperative_rounds']} "
            f"{row['sample_sources']} {row['sample_tasks']}"
        )
    text = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(text)
    print(text, end="")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(merge(arguments) if arguments.merge else run_rank(arguments))

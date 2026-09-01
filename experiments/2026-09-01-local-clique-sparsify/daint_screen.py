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
    ("iter0030", "ipm/iter0030/matrix.mtx"),
    ("iter0040", "ipm/iter0040/matrix.mtx"),
    ("grid_500", "matrices/grid_500.mtx"),
    ("G3_circuit", "matrices/G3_circuit.mtx"),
    ("thermal2", "matrices/thermal2.mtx"),
    ("com-Amazon", "matrices/com-Amazon.mtx"),
)
ARMS = (
    ("baseline-0", None, None, False),
    ("approx-q0.20", 0.20, 1e-3, False),
    ("exact-q0.20", 0.20, 1e-3, True),
    ("baseline-1", None, None, False),
    ("approx-q0.25", 0.25, 1e-3, False),
    ("exact-q0.25", 0.25, 1e-3, True),
    ("baseline-2", None, None, False),
    ("approx-q0.30", 0.30, 1e-3, False),
    ("exact-q0.30", 0.30, 1e-3, True),
    ("baseline-3", None, None, False),
)
SEEDS = (42,)
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


def clean_environment(threads: int, keep: float | None,
                      trigger_rel: float | None,
                      exact_budget: bool,
                      runner) -> dict[str, str]:
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
    if trigger_rel is not None:
        env["APXCHOL_RESEARCH_LOCAL_TREE_TRIGGER_REL"] = str(trigger_rel)
    if exact_budget:
        env["APXCHOL_RESEARCH_LOCAL_TREE_EXACT_BUDGET"] = "1"
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
    assigned = MATRICES[rank::tasks]
    matrices = [(name, args.data_root.resolve() / relative)
                for name, relative in assigned]
    if not binary.is_file():
        raise RuntimeError(f"missing binary: {binary}")
    missing = [str(matrix) for _, matrix in matrices if not matrix.is_file()]
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
        **source_metadata(source, binary),
        "rank": rank,
        "tasks": tasks,
        "matrices": [
            {"name": name, "path": str(matrix)}
            for name, matrix in matrices
        ],
        "threads": args.threads,
        "seeds": SEEDS,
        "arms": [
            {"name": name, "keep": keep, "trigger_rel": trigger_rel,
             "exact_budget": exact_budget}
            for name, keep, trigger_rel, exact_budget in ARMS
        ],
        "affinity": runner.benchmark_openmp_provenance(args.threads),
        "allocation_affinity": sorted(os.sched_getaffinity(0)),
    }
    (rank_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    records: list[dict[str, object]] = []
    cpu_set = runner.affinity_spec(args.threads)
    for matrix_name, matrix in matrices:
        for seed in SEEDS:
            for arm, keep, trigger_rel, exact_budget in ARMS:
                raw_path = raw_dir / f"{matrix_name}-s{seed}-{arm}.log"
                command = [
                    "taskset", "-c", cpu_set, str(binary), str(matrix),
                    "--solve", "--graph-storage", "vec_pool_aos",
                    "--is", "block_greedy", "--seed", str(seed),
                ]
                started = time.time()
                completed = subprocess.run(
                    command, env=clean_environment(
                        args.threads, keep, trigger_rel, exact_budget, runner),
                    text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=args.timeout,
                )
                raw_path.write_text(completed.stdout)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{matrix_name}/seed{seed}/{arm} failed "
                        f"rc={completed.returncode}: {raw_path}")
                parsed = parse_output(completed.stdout)
                if not math.isfinite(float(parsed["residual"])) or \
                        float(parsed["residual"]) > 1e-8:
                    raise RuntimeError(
                        f"{matrix_name}/seed{seed}/{arm} did not converge: "
                        f"{parsed['residual_text']}")
                record = {
                    "rank": rank,
                    "matrix": matrix_name,
                    "threads": args.threads,
                    "seed": seed,
                    "arm": arm,
                    "keep": keep,
                    "trigger_rel": trigger_rel,
                    "exact_budget": exact_budget,
                    "elapsed_wall_s": time.time() - started,
                    "raw": str(raw_path.relative_to(output)),
                    "raw_sha256": sha256(raw_path),
                    **parsed,
                }
                records.append(record)
                with (rank_dir / "records.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

            baselines = [row for row in records
                         if row["matrix"] == matrix_name and
                         row["seed"] == seed and
                         str(row["arm"]).startswith("baseline-")]
            if len(baselines) != 4:
                raise RuntimeError(
                    f"{matrix_name}/seed{seed}: expected four baselines, "
                    f"found {len(baselines)}")
            signatures = {
                (row["factor_nnz"], row["stored_nnz"], row["iterations"],
                 row["residual_text"], row["raw_neighbors"],
                 row["unique_neighbors"], row["emitted_edges"])
                for row in baselines
            }
            if len(signatures) != 1:
                raise RuntimeError(
                    f"{matrix_name}/seed{seed}: baseline mismatch: "
                    f"{signatures}")

    expected = len(matrices) * len(SEEDS) * len(ARMS)
    exact_cells = len(matrices) * len(SEEDS)
    summary = {"checked_records": expected, "planned_records": expected,
               "baseline_exact_cells": exact_cells,
               "converged_records": expected}
    (rank_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"rank {rank}: checked {expected}/{expected}; "
          f"baseline exact {exact_cells}/{exact_cells}; all converged")
    return 0


def merge(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    source = args.source.resolve()
    binary = args.binary.resolve()
    expected_head = command_text(["git", "rev-parse", "HEAD"], source)
    expected_binary_sha = sha256(binary)
    expected_arms = [
        {"name": name, "keep": keep, "trigger_rel": trigger_rel,
         "exact_budget": exact_budget}
        for name, keep, trigger_rel, exact_budget in ARMS
    ]
    records: list[dict[str, object]] = []
    for rank in range(4):
        rank_dir = output / f"rank-{rank}"
        metadata = json.loads((rank_dir / "metadata.json").read_text())
        expected_matrices = [name for name, _ in MATRICES[rank::4]]
        observed_matrices = [item["name"] for item in metadata["matrices"]]
        if (metadata["head"] != expected_head or
                metadata["tracked_status"] != "" or
                metadata["binary_sha256"] != expected_binary_sha or
                metadata["threads"] != args.threads or
                metadata["seeds"] != list(SEEDS) or
                metadata["arms"] != expected_arms or
                observed_matrices != expected_matrices):
            raise RuntimeError(f"rank {rank} metadata mismatch: {metadata}")
        summary = json.loads((rank_dir / "summary.json").read_text())
        rank_records_expected = (
            len(expected_matrices) * len(SEEDS) * len(ARMS))
        rank_cells_expected = len(expected_matrices) * len(SEEDS)
        if summary != {
                "baseline_exact_cells": rank_cells_expected,
                "checked_records": rank_records_expected,
                "converged_records": rank_records_expected,
                "planned_records": rank_records_expected}:
            raise RuntimeError(f"rank {rank} summary mismatch: {summary}")
        rank_records = [json.loads(line) for line in
                        (rank_dir / "records.jsonl").read_text().splitlines()]
        for row in rank_records:
            raw = output / row["raw"]
            if (row["rank"] != rank or row["matrix"] not in expected_matrices or
                    row["seed"] not in SEEDS or
                    row["arm"] not in [name for name, *_ in ARMS] or
                    not raw.is_file() or sha256(raw) != row["raw_sha256"]):
                raise RuntimeError(f"rank {rank} record mismatch: {row}")
        records.extend(rank_records)
    expected_records = len(MATRICES) * len(SEEDS) * len(ARMS)
    expected_cells = len(MATRICES) * len(SEEDS)
    if len(records) != expected_records:
        raise RuntimeError(
            f"checked {len(records)}/{expected_records} records")

    metrics = ("setup_ms", "pcg_ms", "total_ms", "factor_nnz",
               "stored_nnz", "iterations", "raw_neighbors",
               "unique_neighbors", "emitted_edges")
    candidates = (
        "approx-q0.20", "exact-q0.20",
        "approx-q0.25", "exact-q0.25",
        "approx-q0.30", "exact-q0.30",
    )
    ratios = {candidate: {metric: [] for metric in metrics}
              for candidate in candidates}
    per_matrix: list[dict[str, object]] = []
    for matrix, _ in MATRICES:
        for seed in SEEDS:
            arms = {row["arm"]: row for row in records
                    if row["matrix"] == matrix and row["seed"] == seed}
            brackets = {
                "approx-q0.20": (arms["baseline-0"], arms["baseline-1"]),
                "exact-q0.20": (arms["baseline-0"], arms["baseline-1"]),
                "approx-q0.25": (arms["baseline-1"], arms["baseline-2"]),
                "exact-q0.25": (arms["baseline-1"], arms["baseline-2"]),
                "approx-q0.30": (arms["baseline-2"], arms["baseline-3"]),
                "exact-q0.30": (arms["baseline-2"], arms["baseline-3"]),
            }
            for candidate in candidates:
                before, after = brackets[candidate]
                row = arms[candidate]
                candidate_ratios = {}
                for metric in metrics:
                    base = math.sqrt(
                        float(before[metric]) * float(after[metric]))
                    value = float(row[metric]) / base
                    ratios[candidate][metric].append(value)
                    candidate_ratios[metric] = value
                per_matrix.append({
                    "matrix": matrix,
                    "seed": seed,
                    "candidate": candidate,
                    "ratios": candidate_ratios,
                    "iteration_delta": int(row["iterations"]) -
                                       int(before["iterations"]),
                })

    exact_over_approx = {}
    for keep in ("0.20", "0.25", "0.30"):
        approximate = [row for row in records
                       if row["arm"] == f"approx-q{keep}"]
        exact = [row for row in records if row["arm"] == f"exact-q{keep}"]
        approx_by_cell = {(row["matrix"], row["seed"]): row
                          for row in approximate}
        exact_by_cell = {(row["matrix"], row["seed"]): row for row in exact}
        if approx_by_cell.keys() != exact_by_cell.keys():
            raise RuntimeError(f"q={keep}: approximate/exact cell mismatch")
        exact_over_approx[keep] = {
            metric: geometric_mean([
                float(exact_by_cell[cell][metric]) /
                float(approx_by_cell[cell][metric])
                for cell in sorted(approx_by_cell)])
            for metric in metrics
        }

    arm_order = [name for name, *_ in ARMS]
    with (output / "records.jsonl").open("w") as handle:
        for row in sorted(records, key=lambda item: (
                [name for name, _ in MATRICES].index(item["matrix"]),
                SEEDS.index(item["seed"]),
                arm_order.index(item["arm"]))):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "checked_records": expected_records,
        "planned_records": expected_records,
        "baseline_exact_cells": expected_cells,
        "converged_records": expected_records,
        "per_matrix": per_matrix,
        "geomean_exact_over_approx": exact_over_approx,
        "geomean_candidate_over_bracket": {
            candidate: {metric: geometric_mean(values)
                        for metric, values in candidate_metrics.items()}
            for candidate, candidate_metrics in ratios.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"checked {expected_records}/{expected_records} records; "
          f"baseline exact {expected_cells}/{expected_cells}; "
          f"converged {expected_records}/{expected_records}")
    print(json.dumps(summary["geomean_candidate_over_bracket"], sort_keys=True))
    print(json.dumps(exact_over_approx, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
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
    except (KeyError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

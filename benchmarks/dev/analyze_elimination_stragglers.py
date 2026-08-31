#!/usr/bin/env python3
"""Analyze APXCHOL_ROUND_TRACE pivot concentration without timing claims.

The trace reports two structural work proxies per elimination round:

* ``neighbors``: post-dedup live neighbors processed by ``process_vertex``;
* ``sort_work``: sum of ``ceil(d * log2(d))`` over selected pivots.

Neither proxy is elapsed time.  In particular, the built-in GKS sampler uses
linear radix work on many large weighted pivots.  This analyzer therefore
reports both proxies and keeps two different bounds separate:

1. current-team divisibility: only remove LPT imbalance after the existing
   work-sized team has already been formed;
2. all-team divisibility: let every hardware thread help every round at zero
   fork/synchronization cost.  This is an intentionally unattainable upper
   bound, useful only for rejecting structurally tiny opportunities.

Input is a directory containing ``<matrix>/t<threads>/run.log`` files from an
``APXCHOL_ROUND_TRACE=1`` campaign.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


KV = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


@dataclass(frozen=True)
class Round:
    matrix: str
    threads: int
    index: int
    pivots: int
    team: int
    linear: int
    max_linear: int
    linear_lpt: float
    sorting: int
    max_sorting: int
    sort_lpt: float

    @staticmethod
    def _span(work: int, workers: int, efficiency: float) -> float:
        if work == 0:
            return 0.0
        if workers <= 0 or not (0.0 < efficiency <= 1.000001):
            raise ValueError(
                f"invalid LPT metadata: work={work}, workers={workers}, "
                f"efficiency={efficiency}"
            )
        return work / (workers * efficiency)

    @property
    def linear_span(self) -> float:
        return self._span(self.linear, self.team, self.linear_lpt)

    @property
    def sort_span(self) -> float:
        return self._span(self.sorting, self.team, self.sort_lpt)

    @property
    def ideal_current_linear_span(self) -> float:
        return self.linear / self.team if self.team else 0.0

    @property
    def ideal_current_sort_span(self) -> float:
        return self.sorting / self.team if self.team else 0.0

    @property
    def ideal_all_linear_span(self) -> float:
        return self.linear / self.threads if self.threads else 0.0

    @property
    def ideal_all_sort_span(self) -> float:
        return self.sorting / self.threads if self.threads else 0.0


def parse_trace(path: Path) -> list[Round]:
    matrix = path.parents[1].name
    thread_dir = path.parent.name
    if not thread_dir.startswith("t"):
        raise ValueError(f"unexpected thread directory: {path}")
    threads = int(thread_dir[1:])
    rounds: list[Round] = []
    for line in path.read_text().splitlines():
        if not line.startswith("[pivot-probe]"):
            continue
        values = dict(KV.findall(line))
        rounds.append(
            Round(
                matrix=matrix,
                threads=threads,
                index=int(values["round"]),
                pivots=int(values["pivots"]),
                team=int(values["team"]),
                linear=int(values["neighbors"]),
                max_linear=int(values["max_neighbors"]),
                linear_lpt=float(values["lpt_linear"]),
                sorting=int(values["sort_work"]),
                max_sorting=int(values["max_sort_work"]),
                sort_lpt=float(values["lpt_sort"]),
            )
        )
    if not rounds:
        raise ValueError(f"no [pivot-probe] records in {path}")
    return rounds


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_record(rounds: list[Round]) -> dict[str, object]:
    first = rounds[0]
    if any((r.matrix, r.threads) != (first.matrix, first.threads)
           for r in rounds):
        raise ValueError("record mixes matrices or thread counts")

    total_linear = sum(r.linear for r in rounds)
    total_sort = sum(r.sorting for r in rounds)
    current_linear_span = sum(r.linear_span for r in rounds)
    current_sort_span = sum(r.sort_span for r in rounds)
    ideal_current_linear = sum(r.ideal_current_linear_span for r in rounds)
    ideal_current_sort = sum(r.ideal_current_sort_span for r in rounds)
    ideal_all_linear = sum(r.ideal_all_linear_span for r in rounds)
    ideal_all_sort = sum(r.ideal_all_sort_span for r in rounds)

    return {
        "matrix": first.matrix,
        "threads": first.threads,
        "rounds": len(rounds),
        "pivots": sum(r.pivots for r in rounds),
        "linear_work": total_linear,
        "sort_proxy": total_sort,
        "team1_linear_fraction": ratio(
            sum(r.linear for r in rounds if r.team == 1), total_linear),
        "team1_sort_fraction": ratio(
            sum(r.sorting for r in rounds if r.team == 1), total_sort),
        "imbalanced_parallel_linear_fraction": ratio(
            sum(r.linear for r in rounds
                if r.team > 1 and r.linear_lpt < 0.8), total_linear),
        "imbalanced_parallel_sort_fraction": ratio(
            sum(r.sorting for r in rounds
                if r.team > 1 and r.sort_lpt < 0.8), total_sort),
        # Fractions of the current structural span removable if work inside an
        # already-created team were perfectly divisible.  No new team is funded.
        "current_team_linear_span_headroom": ratio(
            current_linear_span - ideal_current_linear, current_linear_span),
        "current_team_sort_span_headroom": ratio(
            current_sort_span - ideal_current_sort, current_sort_span),
        # Unattainable zero-overhead bounds which also parallelize team==1 rounds.
        "all_team_linear_upper_speedup": ratio(
            current_linear_span, ideal_all_linear),
        "all_team_sort_upper_speedup": ratio(
            current_sort_span, ideal_all_sort),
        "max_pivot_degree": max(r.max_linear for r in rounds),
        "max_pivot_sort_proxy": max(r.max_sorting for r in rounds),
        "worst_parallel_linear_lpt": min(
            (r.linear_lpt for r in rounds if r.team > 1), default=1.0),
        "worst_parallel_sort_lpt": min(
            (r.sort_lpt for r in rounds if r.team > 1), default=1.0),
    }


def top_rounds(rounds: list[Round], count: int) -> list[Round]:
    def removable_sort_span(row: Round) -> float:
        return row.sort_span - row.ideal_current_sort_span

    return sorted(
        (row for row in rounds if row.team > 1),
        key=removable_sort_span,
        reverse=True,
    )[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expect-records", type=int)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--top-rounds", type=int, default=12)
    args = parser.parse_args()

    paths = sorted(args.root.glob("*/t*/run.log"))
    if args.expect_records is not None and len(paths) != args.expect_records:
        raise RuntimeError(
            f"expected {args.expect_records} trace records, found {len(paths)}"
        )
    if not paths:
        raise RuntimeError(f"no <matrix>/t*/run.log records under {args.root}")

    records: list[tuple[Path, list[Round]]] = [
        (path, parse_trace(path)) for path in paths
    ]
    all_rounds = [row for _, rows in records for row in rows]
    summaries = [summarize_record(rows) for _, rows in records]

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

    matrices = sorted({row.matrix for row in all_rounds})
    thread_counts = sorted({row.threads for row in all_rounds})
    print(
        f"checked {len(paths)}/{args.expect_records or len(paths)} records; "
        f"matrices={len(matrices)} ({', '.join(matrices)}); "
        f"threads={thread_counts}; rounds={len(all_rounds)}; "
        f"selected_pivots={sum(row.pivots for row in all_rounds)}"
    )
    print()
    print(
        "matrix,T,rounds,team1_linear%,team1_sort%,"
        "imbalanced_parallel_linear%,imbalanced_parallel_sort%,"
        "current_team_linear_headroom%,current_team_sort_headroom%,"
        "all_team_linear_upper_speedup,all_team_sort_upper_speedup,"
        "max_degree,worst_parallel_lpt"
    )
    for row in summaries:
        print(
            f"{row['matrix']},{row['threads']},{row['rounds']},"
            f"{100 * float(row['team1_linear_fraction']):.3f},"
            f"{100 * float(row['team1_sort_fraction']):.3f},"
            f"{100 * float(row['imbalanced_parallel_linear_fraction']):.3f},"
            f"{100 * float(row['imbalanced_parallel_sort_fraction']):.3f},"
            f"{100 * float(row['current_team_linear_span_headroom']):.3f},"
            f"{100 * float(row['current_team_sort_span_headroom']):.3f},"
            f"{float(row['all_team_linear_upper_speedup']):.3f},"
            f"{float(row['all_team_sort_upper_speedup']):.3f},"
            f"{row['max_pivot_degree']},"
            f"{float(row['worst_parallel_sort_lpt']):.6f}"
        )

    print()
    print("top already-parallel rounds by removable dlogd span proxy:")
    print(
        "matrix,T,round,pivots,team,sort_proxy,max_sort,max_fraction,lpt,"
        "current_span,ideal_divisible_span"
    )
    for row in top_rounds(all_rounds, args.top_rounds):
        print(
            f"{row.matrix},{row.threads},{row.index},{row.pivots},{row.team},"
            f"{row.sorting},{row.max_sorting},"
            f"{ratio(row.max_sorting, row.sorting):.6f},{row.sort_lpt:.6f},"
            f"{row.sort_span:.1f},{row.ideal_current_sort_span:.1f}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

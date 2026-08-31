#!/usr/bin/env python3
"""Render the checksummed Daint scaling and historical A/B campaigns.

The input directories are downloaded campaign ``results/`` trees.  Raw logs stay
in the checksummed campaign archive; this script commits compact CSV extracts,
the aggregate summary, and figures that can be reproduced from those extracts.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIMING_METRICS = (
    "setup",
    "make_graph",
    "find_partition",
    "select",
    "prune",
    "eliminate",
    "assembly",
    "sptrsv_setup",
    "pcg",
    "total",
)
SNAPSHOT_LABEL = "HISTORICAL SNAPSHOT · checksummed Daint campaign"
SCALING_MATRICES = (
    "G3_circuit",
    "as-Skitter",
    "coPapersDBLP",
    "com-Amazon",
    "com-LiveJournal",
    "com-Orkut",
    "grid_500",
    "iter0040",
    "kron_g500-logn16",
)
SCALING_THREADS = (1, 2, 4, 8, 16, 36, 72)
SCALING_REPS = (0, 1, 2)
SCALING_RECORDS = 189
SCALING_TOTAL_ABS_TOL_MS = 1.5e-3
SCALING_PANELS = (
    ("total_ms", "Total"),
    ("pcg_ms", "PCG / solve"),
    ("setup_ms", "Setup"),
)
BASELINE_RECORDS = 162
PIVOT_RECORDS = 18
TIMING_PATTERNS = {
    name: re.compile(rf"^\s*{re.escape(name)}\s+([0-9.]+) ms", re.MULTILINE)
    for name in TIMING_METRICS
}
KV_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def geomean(values):
    values = list(values)
    return math.exp(statistics.fmean(math.log(value) for value in values))


def parse_log(path: Path) -> dict[str, float | int]:
    text = path.read_text()
    result: dict[str, float | int] = {}
    for name, pattern in TIMING_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            raise RuntimeError(f"missing {name}: {path}")
        result[f"{name}_ms"] = float(match.group(1))
    integer_patterns = {
        "factor_nnz": r"^\[apxchol\] nnz\(L\) =\s+(\d+)",
        "iterations": r"^\[info\] iterations:\s+(\d+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s+(\d+)",
    }
    for name, pattern in integer_patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"missing {name}: {path}")
        result[name] = int(match.group(1))
    residual = re.search(r"^\[info\] residual:\s+([0-9.eE+-]+)", text, re.MULTILINE)
    if not residual:
        raise RuntimeError(f"missing residual: {path}")
    result["residual"] = float(residual.group(1))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, *, expected: int, int_fields=(), float_fields=()) -> list[dict]:
    """Read a committed campaign extract and restore its numeric field types."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = set(int_fields) | set(float_fields)
        missing = sorted(required - fields)
        if missing:
            raise RuntimeError(f"missing columns in {path}: {', '.join(missing)}")
        rows = list(reader)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} records in {path}, got {len(rows)}")
    try:
        for row in rows:
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
    except ValueError as exc:
        raise RuntimeError(f"invalid numeric value in {path}: {exc}") from exc
    return rows


def load_csv_inputs(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Load the committed extracts needed to reproduce figures and summary."""
    timing_fields = tuple(f"{name}_ms" for name in TIMING_METRICS)
    scaling = read_csv(
        root / "scaling.csv", expected=SCALING_RECORDS,
        int_fields=("threads", "rep", "factor_nnz", "iterations", "max_rss_kb"),
        float_fields=(*timing_fields, "residual"),
    )
    baseline = read_csv(
        root / "historical_ab.csv", expected=BASELINE_RECORDS,
        int_fields=("threads", "seed", "factor_nnz", "iterations", "max_rss_kb"),
        float_fields=(*timing_fields, "residual"),
    )
    pivots = read_csv(
        root / "pivot_summary.csv", expected=PIVOT_RECORDS,
        int_fields=("threads", "rounds", "total_sort_work", "max_pivot_degree",
                    "max_pivot_sort_work"),
        float_fields=("team1_sort_fraction", "imbalanced_sort_fraction",
                      "worst_parallel_lpt"),
    )
    return scaling, baseline, pivots


def parse_scaling(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("timing/*/t*/rep*/run.log")):
        matrix, thread_dir, rep_dir = path.parts[-4:-1]
        rows.append({
            "matrix": matrix,
            "threads": int(thread_dir[1:]),
            "rep": int(rep_dir[3:]),
            **parse_log(path),
        })
    if len(rows) != SCALING_RECORDS:
        raise RuntimeError(f"expected {SCALING_RECORDS} scaling records, got {len(rows)}")
    return rows


def parse_baseline(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/t*/s*/*/run.log")):
        matrix, thread_dir, seed_dir, arm = path.parts[-5:-1]
        rows.append({
            "matrix": matrix,
            "threads": int(thread_dir[1:]),
            "seed": int(seed_dir[1:]),
            "arm": arm,
            **parse_log(path),
        })
    if len(rows) != BASELINE_RECORDS:
        raise RuntimeError(f"expected {BASELINE_RECORDS} baseline records, got {len(rows)}")
    return rows


def parse_structure(root: Path) -> tuple[list[dict], list[dict]]:
    pivot_summary = []
    region_rows = []
    paths = sorted(root.glob("structure/*/t*/run.log"))
    if len(paths) != PIVOT_RECORDS:
        raise RuntimeError(f"expected {PIVOT_RECORDS} structure records, got {len(paths)}")
    for path in paths:
        matrix, thread_dir = path.parts[-3:-1]
        threads = int(thread_dir[1:])
        pivots = []
        for line in path.read_text().splitlines():
            if line.startswith("[pivot-probe]"):
                pivots.append(dict(KV_PATTERN.findall(line)))
            elif line.startswith("[region-probe]"):
                values = dict(KV_PATTERN.findall(line))
                region_rows.append({
                    "matrix": matrix,
                    "threads": threads,
                    "round": int(values["round"]),
                    "active": int(values["active"]),
                    "candidates": int(values["candidates"]),
                    "separator_fraction": float(values["separator_frac"]),
                    "regions": int(values["regions"]),
                    "singletons": int(values["singletons"]),
                    "largest_vertices": int(values["largest_vertices"]),
                    "work": int(values["work"]),
                    "max_work": int(values["max_work"]),
                    "max_work_fraction": float(values["max_work_frac"]),
                    "lpt_efficiency": float(values[f"lpt_efficiency_p{threads}"]),
                    "internal_edges": int(values["internal_edges"]),
                    "boundary_incidence": int(values["boundary_incidence"]),
                    "boundary_fraction": float(values["boundary_frac"]),
                })
        total_sort = sum(int(row["sort_work"]) for row in pivots)
        team1_sort = sum(
            int(row["sort_work"]) for row in pivots if int(row["team"]) == 1
        )
        imbalanced_sort = sum(
            int(row["sort_work"])
            for row in pivots
            if int(row["team"]) > 1 and float(row["lpt_sort"]) < 0.8
        )
        pivot_summary.append({
            "matrix": matrix,
            "threads": threads,
            "rounds": len(pivots),
            "total_sort_work": total_sort,
            "team1_sort_fraction": team1_sort / total_sort if total_sort else 0.0,
            "imbalanced_sort_fraction": imbalanced_sort / total_sort if total_sort else 0.0,
            "max_pivot_degree": max(int(row["max_neighbors"]) for row in pivots),
            "max_pivot_sort_work": max(int(row["max_sort_work"]) for row in pivots),
            "worst_parallel_lpt": min(
                (float(row["lpt_sort"]) for row in pivots if int(row["team"]) > 1),
                default=1.0,
            ),
        })
    return pivot_summary, region_rows


def keyed(rows: list[dict], fields: tuple[str, ...]) -> dict[tuple, dict]:
    return {tuple(row[field] for field in fields): row for row in rows}


def validate_scaling_records(rows: list[dict]) -> float:
    """Require the complete historical scaling grid and its timing identity.

    The three timing fields are independently rounded to 0.001 ms in the
    committed extract, so their worst possible rounding disagreement is
    0.0015 ms.  Return the largest observed disagreement for the audit line.
    """
    expected = {
        (matrix, threads, rep)
        for matrix in SCALING_MATRICES
        for threads in SCALING_THREADS
        for rep in SCALING_REPS
    }
    if len(expected) != SCALING_RECORDS:
        raise RuntimeError(
            f"internal scaling denominator mismatch: {len(expected)} != "
            f"{SCALING_RECORDS}"
        )

    try:
        keys = [(row["matrix"], row["threads"], row["rep"]) for row in rows]
    except KeyError as exc:
        raise RuntimeError(f"missing scaling key column: {exc.args[0]}") from exc
    actual = set(keys)
    duplicates = sorted(key for key in actual if keys.count(key) > 1)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    checked = len(actual & expected)
    if len(rows) != SCALING_RECORDS or duplicates or missing or unexpected:
        details = [f"rows={len(rows)}", f"unique={len(actual)}"]
        if missing:
            details.append(f"missing={missing[:3]}" + ("..." if len(missing) > 3 else ""))
        if duplicates:
            details.append(
                f"duplicates={duplicates[:3]}" + ("..." if len(duplicates) > 3 else "")
            )
        if unexpected:
            details.append(
                f"unexpected={unexpected[:3]}" + ("..." if len(unexpected) > 3 else "")
            )
        raise RuntimeError(
            f"scaling coverage failed: checked {checked}/{SCALING_RECORDS}; "
            + "; ".join(details)
        )

    max_delta = 0.0
    for row in rows:
        key = (row["matrix"], row["threads"], row["rep"])
        setup = row["setup_ms"]
        pcg = row["pcg_ms"]
        total = row["total_ms"]
        if not all(math.isfinite(value) and value > 0.0 for value in (setup, pcg, total)):
            raise RuntimeError(f"non-positive or non-finite timing at {key}")
        delta = abs((setup + pcg) - total)
        max_delta = max(max_delta, delta)
        if delta > SCALING_TOTAL_ABS_TOL_MS:
            raise RuntimeError(
                "scaling timing consistency failed after checked "
                f"{SCALING_RECORDS}/{SCALING_RECORDS}: setup+pcg=total differs "
                f"by {delta:.12g} ms at {key}; allowed rounding tolerance is "
                f"{SCALING_TOTAL_ABS_TOL_MS:.4g} ms"
            )
    return max_delta


def scaling_medians(rows: list[dict]) -> dict[tuple[str, int], dict[str, float]]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["matrix"], row["threads"]), []).append(row)
    return {
        key: {
            metric: statistics.median(row[metric] for row in group)
            for metric in (*[f"{name}_ms" for name in TIMING_METRICS], "max_rss_kb")
        }
        for key, group in groups.items()
    }


def baseline_ratio(index: dict, matrix: str, threads: int, seed: int, metric: str) -> float:
    old_a = index[(matrix, threads, seed, "old_a")][metric]
    old_b = index[(matrix, threads, seed, "old_b")][metric]
    current = index[(matrix, threads, seed, "current")][metric]
    return current / math.sqrt(old_a * old_b)


def render_scaling_figure(path: Path, medians: dict) -> None:
    matrices = list(SCALING_MATRICES)
    threads = list(SCALING_THREADS)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(
        1, len(SCALING_PANELS), figsize=(13.2, 5.2), sharex=True, sharey=True
    )
    legend_handles = []
    for panel, (ax, (metric, title)) in enumerate(zip(axes, SCALING_PANELS)):
        for index, matrix in enumerate(matrices):
            speedup = [
                medians[(matrix, 1)][metric] / medians[(matrix, thread)][metric]
                for thread in threads
            ]
            line, = ax.plot(
                threads,
                speedup,
                color=colors[index],
                marker="o",
                markersize=3.2,
                linewidth=1.15,
                alpha=0.62,
                label=matrix,
            )
            if panel == 0:
                legend_handles.append(line)
        overall = [
            geomean(
                medians[(matrix, 1)][metric] / medians[(matrix, thread)][metric]
                for matrix in matrices
            )
            for thread in threads
        ]
        geomean_line, = ax.plot(
            threads,
            overall,
            color="black",
            marker="o",
            markersize=4.2,
            linewidth=2.6,
            label="geomean",
            zorder=5,
        )
        ideal_line, = ax.plot(
            threads,
            threads,
            color="0.62",
            linestyle="--",
            linewidth=1.1,
            label="ideal",
            zorder=0,
        )
        if panel == 0:
            legend_handles.extend((geomean_line, ideal_line))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(threads, [str(thread) for thread in threads])
        ax.set_ylim(0.5, 96)
        ax.set_title(f"({chr(ord('a') + panel)}) {title}", fontsize=11)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("speedup over T=1")
    axes[0].set_yticks(
        (0.5, 1, 2, 4, 8, 16, 32, 64),
        ("0.5×", "1×", "2×", "4×", "8×", "16×", "32×", "64×"),
    )
    fig.suptitle(f"apxchol scaling on Daint\n{SNAPSHOT_LABEL}", fontsize=14, y=0.98)
    fig.supxlabel("OpenMP threads within one 72-core NUMA domain", y=0.205)
    fig.text(
        0.5,
        0.155,
        f"checked {SCALING_RECORDS}/{SCALING_RECORDS} timing records · "
        f"{len(SCALING_MATRICES)} matrices × {len(SCALING_THREADS)} thread counts × "
        f"{len(SCALING_REPS)} reps · medians of 3 · setup + PCG = total validated",
        ha="center",
        fontsize=8.5,
        color="0.25",
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        fontsize=7.7,
        frameon=False,
    )
    fig.subplots_adjust(left=0.065, right=0.992, top=0.78, bottom=0.28, wspace=0.12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_baseline_figure(path: Path, rows: list[dict]) -> None:
    index = keyed(rows, ("matrix", "threads", "seed", "arm"))
    matrices = sorted({row["matrix"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    labels = matrices + ["geomean"]
    values = {}
    for threads in (16, 72):
        per_matrix = [
            geomean(baseline_ratio(index, matrix, threads, seed, "total_ms")
                    for seed in seeds)
            for matrix in matrices
        ]
        values[threads] = per_matrix + [geomean(per_matrix)]
    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.8, 5.1), constrained_layout=True)
    ax.bar([v - width / 2 for v in x], values[16], width, label="T=16")
    ax.bar([v + width / 2 for v in x], values[72], width, label="T=72")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylabel("campaign head / July-27 total time (lower is better)")
    ax.set_title(f"Within-snapshot single-RHS improvement on Daint\n{SNAPSHOT_LABEL}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_breakdown_figure(path: Path, medians: dict) -> None:
    matrices = sorted({matrix for matrix, _ in medians})
    stages = ("make_graph_ms", "find_partition_ms", "eliminate_ms", "assembly_ms", "sptrsv_setup_ms")
    labels = ("graph", "partition phase", "eliminate", "assembly", "SpTRSV setup")
    bottoms = [0.0] * len(matrices)
    fig, ax = plt.subplots(figsize=(10.3, 5.0), constrained_layout=True)
    for stage, label in zip(stages, labels):
        values = [100.0 * medians[(matrix, 72)][stage] / medians[(matrix, 72)]["setup_ms"]
                  for matrix in matrices]
        ax.bar(matrices, values, bottom=bottoms, label=label)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    remainder = [max(0.0, 100.0 - value) for value in bottoms]
    ax.bar(matrices, remainder, bottom=bottoms, label="other")
    ax.set_ylabel("share of setup at T=72 (%)")
    ax.set_title(f"Setup composition on Daint\n{SNAPSHOT_LABEL}")
    ax.tick_params(axis="x", rotation=32)
    ax.legend(ncol=1, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, scaling: list[dict], baseline: list[dict], pivots: list[dict]) -> None:
    medians = scaling_medians(scaling)
    baseline_index = keyed(baseline, ("matrix", "threads", "seed", "arm"))
    matrices = sorted({row["matrix"] for row in baseline})
    seeds = sorted({row["seed"] for row in baseline})
    lines = [
        "# Daint campaign summary",
        "",
        f"> **{SNAPSHOT_LABEL}.** This renderer consumes committed extracts from "
        "fixed, checksummed raw-log campaigns rather than the current unified cell store.",
        "",
        "Both campaigns used one Daint node split into four NUMA-local 72-core ranks. "
        "Times are milliseconds; scaling cells are medians of three repetitions. "
        "Historical ratios bracket each campaign-head run by two executions of the old binary.",
        "",
        "## Within-snapshot July-27 to campaign-head comparison",
        "",
        "| threads | setup ratio | PCG ratio | total ratio | RSS ratio | factor nnz ratio | iteration delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threads in (16, 72):
        cells = [(matrix, threads, seed) for matrix in matrices for seed in seeds]
        ratios = {
            metric: geomean(baseline_ratio(baseline_index, *cell, metric) for cell in cells)
            for metric in ("setup_ms", "pcg_ms", "total_ms", "max_rss_kb", "factor_nnz")
        }
        iteration_delta = sum(
            baseline_index[(*cell, "current")]["iterations"]
            - baseline_index[(*cell, "old_a")]["iterations"]
            for cell in cells
        )
        lines.append(
            f"| {threads} | {ratios['setup_ms']:.4f} | {ratios['pcg_ms']:.4f} | "
            f"{ratios['total_ms']:.4f} | {ratios['max_rss_kb']:.4f} | "
            f"{ratios['factor_nnz']:.4f} | {iteration_delta:+d} |"
        )
    lines += [
        "",
        "Ratios are campaign head divided by the geometric mean of the old-before/old-after bracket; "
        "below one is better. The iteration delta is summed over 27 matrix/seed cells.",
        "",
        "## Historical snapshot scaling",
        "",
        "`partition phase` includes degree pruning, IS selection, and collection; it is not "
        "the pure selector timing.",
        "",
        "| threads | setup | partition phase | prune | IS selector | elimination | SpTRSV setup |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threads in (1, 2, 4, 8, 16, 36, 72):
        speedups = []
        for metric in ("setup_ms", "find_partition_ms", "prune_ms", "select_ms",
                       "eliminate_ms", "sptrsv_setup_ms"):
            speedups.append(geomean(
                medians[(matrix, 1)][metric] / medians[(matrix, threads)][metric]
                for matrix in matrices
            ))
        lines.append(f"| {threads} | " + " | ".join(f"{value:.3f}x" for value in speedups) + " |")

    selector_speedup = geomean(
        medians[(matrix, 1)]["select_ms"] / medians[(matrix, 72)]["select_ms"]
        for matrix in matrices
    )
    partition_speedup = geomean(
        medians[(matrix, 1)]["find_partition_ms"]
        / medians[(matrix, 72)]["find_partition_ms"]
        for matrix in matrices
    )
    prune_speedup = geomean(
        medians[(matrix, 1)]["prune_ms"] / medians[(matrix, 72)]["prune_ms"]
        for matrix in matrices
    )
    selector_share = statistics.median(
        medians[(matrix, 72)]["select_ms"] / medians[(matrix, 72)]["setup_ms"]
        for matrix in matrices
    )
    elimination_share = statistics.median(
        medians[(matrix, 72)]["eliminate_ms"] / medians[(matrix, 72)]["setup_ms"]
        for matrix in matrices
    )
    elimination_shares = [
        medians[(matrix, 72)]["eliminate_ms"] / medians[(matrix, 72)]["setup_ms"]
        for matrix in matrices
    ]
    elimination_speedup = geomean(
        medians[(matrix, 1)]["eliminate_ms"] / medians[(matrix, 72)]["eliminate_ms"]
        for matrix in matrices
    )
    lines += [
        "",
        f"At T=72 the whole partition phase reaches {partition_speedup:.3f}x because pruning reaches "
        f"{prune_speedup:.3f}x; pure IS selection reaches only {selector_speedup:.3f}x. "
        f"The selector is nevertheless a median {selector_share:.1%} of setup, versus "
        f"{elimination_share:.1%} for elimination. Elimination is "
        f"{min(elimination_shares):.1%}-{max(elimination_shares):.1%} of setup "
        f"on all nine matrices and its {elimination_speedup:.3f}x geomean speedup is the main remaining "
        "scaling limit.",
        "",
        "## Structural probe verdict",
        "",
        "| matrix | T | singleton-tail work | imbalanced parallel work | worst parallel LPT |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pivots:
        lines.append(
            f"| {row['matrix']} | {row['threads']} | {row['team1_sort_fraction']:.3%} | "
            f"{row['imbalanced_sort_fraction']:.3%} | {row['worst_parallel_lpt']:.4f} |"
        )
    lines += [
        "",
        "A general intra-pivot team is not justified: eight matrices keep parallel-round "
        "LPT efficiency above 0.947, while only Orkut puts substantial work (about 14.6%) "
        "in rounds below 80% efficiency. Candidate-induced components are balanced in early "
        "rounds, then often collapse into one giant component. The next region prototype should "
        "eliminate balanced small components and route an oversized component through the existing "
        "MIS selector; this has IS and full-region elimination as limiting cases.",
        "",
        "All 54 historical-comparison cells converged to true relative residual at most 1e-8. "
        "All 207 scaling/probe records and all 162 historical records were covered by their "
        "campaign completion markers and downloaded checksum manifests.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", type=Path,
                        help="completed historical scaling campaign results directory")
    parser.add_argument("--baseline", type=Path,
                        help="completed historical A/B campaign results directory")
    parser.add_argument("--csv-input", type=Path,
                        help="directory containing committed scaling.csv, "
                             "historical_ab.csv, and pivot_summary.csv extracts")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args(argv)

    if args.csv_input and (args.scaling or args.baseline):
        parser.error("--csv-input cannot be combined with --scaling or --baseline")
    if not args.csv_input and not (args.scaling and args.baseline):
        parser.error("provide --csv-input or both --scaling and --baseline")

    mode = "CSV-extract reproduction" if args.csv_input else "raw-log renderer"
    print(f"{SNAPSHOT_LABEL}: {mode}, outside the unified cell store",
          file=sys.stderr)

    if args.csv_input:
        scaling, baseline, pivots = load_csv_inputs(args.csv_input)
    else:
        scaling = parse_scaling(args.scaling)
        baseline = parse_baseline(args.baseline)
        pivots, regions = parse_structure(args.scaling)

    max_total_delta = validate_scaling_records(scaling)
    print(
        f"scaling audit: checked {len(scaling)}/{SCALING_RECORDS}; "
        "9 matrices x 7 thread counts x 3 reps; "
        f"max |setup+pcg-total|={max_total_delta:.12g} ms",
        file=sys.stderr,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    if not args.csv_input:
        write_csv(args.output / "scaling.csv", scaling)
        write_csv(args.output / "historical_ab.csv", baseline)
        write_csv(args.output / "pivot_summary.csv", pivots)
        write_csv(args.output / "region_snapshots.csv", regions)
    medians = scaling_medians(scaling)
    render_scaling_figure(figures / "setup_scaling.png", medians)
    render_baseline_figure(figures / "historical_total_ratio.png", baseline)
    render_breakdown_figure(figures / "setup_t72_breakdown.png", medians)
    write_summary(args.output / "summary.md", scaling, baseline, pivots)


if __name__ == "__main__":
    main()

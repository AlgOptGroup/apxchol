#!/usr/bin/env python3
"""benchmarks/update_readme.py – Auto-update benchmarks/README.md with latest results.

Reads a CSV and generates a summary table between markers in README.md.
"""

import sys
import os
import pandas as pd
from datetime import datetime


def format_time(s):
    if s < 1e-3:
        return f"{s*1e6:.0f}µs"
    if s < 1.0:
        return f"{s*1e3:.1f}ms"
    return f"{s:.2f}s"


def format_residual(r):
    if r < 1e-2:
        return f"{r:.1e}"
    return f"{r:.2e}"


EXCLUDE_SOLVERS = {"AMG+CG [AMGCL]", "CG+ICC [Eigen]", "pRCHOL+PCG [Chen20;par]", "AC [Kyng16;Jl]"}


def generate_table(df):
    """Generate a markdown summary table from benchmark CSV."""
    df = df[~df["solver"].isin(EXCLUDE_SOLVERS)]
    lines = []

    # Summary: best result per solver on largest checkerboard κ=1
    mask = df["graph"].str.contains("checker", na=False) & df["graph"].str.contains("k1_t4", na=False)
    sub = df[mask].copy()
    if sub.empty:
        return "No checkerboard (κ=1) results found.\n"

    largest_n = sub["n"].max()
    sub = sub[sub["n"] == largest_n].sort_values("total_s")

    lines.append(f"### Checkerboard Grid (n={largest_n:,}, κ=1, tile=4)\n")
    lines.append("| Solver | Setup | Solve | Total | Iters | Rel. Residual | µs/nnz |")
    lines.append("|--------|-------|-------|-------|------:|---------------|-------:|")

    for _, row in sub.iterrows():
        lines.append(
            f"| {row['solver']} "
            f"| {format_time(row['setup_s'])} "
            f"| {format_time(row['solve_s'])} "
            f"| {format_time(row['total_s'])} "
            f"| {int(row['iters'])} "
            f"| {format_residual(row['rel_res'])} "
            f"| {row['us_per_nnz']:.2f} |"
        )

    return "\n".join(lines) + "\n"


def update_readme(csv_path):
    readme_path = os.path.join(os.path.dirname(csv_path), "..", "README.md")
    if not os.path.isfile(readme_path):
        # Try from benchmarks dir
        readme_path = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "README.md")
    if not os.path.isfile(readme_path):
        readme_path = os.path.join(os.path.dirname(csv_path), "..", "README.md")

    # Ensure benchmarks/README.md exists
    bench_readme = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "README.md")
    if os.path.basename(os.path.dirname(os.path.dirname(csv_path))) != "benchmarks":
        bench_readme = os.path.join(os.path.dirname(csv_path), "..", "README.md")

    df = pd.read_csv(csv_path)
    table = generate_table(df)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    START_MARKER = "<!-- BENCHMARK_RESULTS_START -->"
    END_MARKER = "<!-- BENCHMARK_RESULTS_END -->"

    if os.path.isfile(bench_readme):
        with open(bench_readme, "r") as f:
            content = f.read()

        if START_MARKER in content and END_MARKER in content:
            before = content[:content.index(START_MARKER) + len(START_MARKER)]
            after = content[content.index(END_MARKER):]
            content = before + f"\n*Last updated: {timestamp}*\n\n" + table + "\n" + after
        else:
            content += f"\n{START_MARKER}\n*Last updated: {timestamp}*\n\n{table}\n{END_MARKER}\n"

        with open(bench_readme, "w") as f:
            f.write(content)
        print(f"Updated {bench_readme}")
    else:
        print(f"README not found at {bench_readme}, skipping update", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 update_readme.py <csv_file>", file=sys.stderr)
        sys.exit(1)
    update_readme(sys.argv[1])

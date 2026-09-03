#!/usr/bin/env python3
"""Plot the recovered, quality-only iter0010 exponent sweep.

The input deliberately has no timing columns.  The historical workstation was
not isolated, so this artifact records only PCG iterations and raw factor size.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: pathlib.Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    with path.open(newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    if len(raw) != 12:
        raise RuntimeError(f"expected 12 recovered rows, found {len(raw)}")

    baseline: dict[str, object] | None = None
    points: list[dict[str, object]] = []
    for row in raw:
        seeds = tuple(int(value) for value in row["seeds"].split(","))
        iterations = tuple(int(value) for value in row["iterations"].split(","))
        if seeds != (1, 17, 42, 73, 97) or len(iterations) != len(seeds):
            raise RuntimeError(f"bad seed/iteration payload: {row}")
        parsed: dict[str, object] = {
            "sampler": row["sampler"],
            "alpha": float(row["alpha"]) if row["alpha"] else None,
            "iterations": iterations,
            "mean_iterations": float(row["mean_iterations"]),
            "mean_raw_nnz": float(row["mean_raw_nnz"]),
        }
        if not np.isclose(np.mean(iterations), parsed["mean_iterations"]):
            raise RuntimeError(f"iteration mean does not match samples: {row}")
        if row["sampler"] == "gks":
            baseline = parsed
        else:
            points.append(parsed)
    if baseline is None:
        raise RuntimeError("missing GKS baseline")
    points.sort(key=lambda row: float(row["alpha"]))
    return baseline, points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    baseline, rows = read_rows(args.input)
    alpha = np.asarray([row["alpha"] for row in rows], dtype=float)
    means = np.asarray([row["mean_iterations"] for row in rows], dtype=float)
    nnz = np.asarray([row["mean_raw_nnz"] for row in rows], dtype=float) / 1e6

    plt.rcParams.update(
        {
            "font.size": 9,
            "svg.fonttype": "none",
            "svg.hashsalt": "apxchol-bkz26-alpha-sweep",
        }
    )
    figure, (iterations_axis, nnz_axis) = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.1),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )

    seed_offsets = np.linspace(-0.018, 0.018, 5)
    for index, row in enumerate(rows):
        iterations_axis.scatter(
            float(row["alpha"]) + seed_offsets,
            row["iterations"],
            color="#5d82b3",
            alpha=0.50,
            s=16,
            label="individual seeds" if index == 0 else None,
        )
    iterations_axis.plot(
        alpha,
        means,
        color="#1f4e79",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="p-tree mean (5 seeds)",
    )
    iterations_axis.axhline(
        float(baseline["mean_iterations"]),
        color="#2d7f5e",
        linestyle="--",
        linewidth=1.4,
        label="GKS mean",
    )
    best_index = int(np.argmin(means))
    iterations_axis.scatter(
        alpha[best_index],
        means[best_index],
        marker="D",
        s=45,
        color="#b84a3a",
        zorder=5,
        label=r"best tested: $\alpha=1.75$",
    )
    iterations_axis.annotate(
        "58.4 vs 90.4 at BKZ α=1\nGKS remains 42.0",
        xy=(alpha[best_index], means[best_index]),
        xytext=(2.05, 105),
        arrowprops={"arrowstyle": "->", "color": "#555"},
        fontsize=8,
        ha="left",
    )
    iterations_axis.set_ylabel("PCG iterations")
    iterations_axis.set_title("iter0010 exponent sweep — solver quality only")
    iterations_axis.grid(alpha=0.20)
    iterations_axis.legend(frameon=False, ncol=2, loc="upper left")

    nnz_axis.plot(
        alpha,
        nnz,
        color="#8b5a2b",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="p-tree raw factor",
    )
    nnz_axis.axhline(
        float(baseline["mean_raw_nnz"]) / 1e6,
        color="#2d7f5e",
        linestyle="--",
        linewidth=1.4,
        label="GKS raw factor",
    )
    nnz_axis.scatter(
        alpha[best_index], nnz[best_index], marker="D", s=45, color="#b84a3a", zorder=5
    )
    nnz_axis.set_xlabel(r"Prüfer-symbol exponent $\alpha$ in $q_i \propto a_i^\alpha$")
    nnz_axis.set_ylabel("mean raw nnz(L), millions")
    nnz_axis.grid(alpha=0.20)
    nnz_axis.legend(frameon=False, ncol=2, loc="best")

    figure.text(
        0.5,
        0.012,
        "Seeds 1, 17, 42, 73, 97. No setup or solve-time claim is made from this sweep.",
        ha="center",
        fontsize=8,
        color="#444",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output, metadata={"Date": None, "Creator": "plot_alpha_sweep.py"}
    )
    # Matplotlib 3.11 leaves spaces at the ends of SVG path-data lines.
    # Canonicalize them so the committed text asset passes git whitespace checks.
    svg = args.output.read_text()
    args.output.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


if __name__ == "__main__":
    main()

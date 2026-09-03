#!/usr/bin/env python3
"""Optimize the exact p-tree local-degree variance over q for sample stars."""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
from scipy.optimize import minimize, minimize_scalar

CASES = {
    "uniform": (1, 1, 1, 1, 1, 1),
    "mild": (1, 1.5, 2, 3, 5, 8),
    "geometric": (1, 2, 4, 8, 16, 32),
    "two-scale": (1, 1, 1, 1, 100, 100),
    "six-decades": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1),
}


def exact_degrees(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    return weights * (total - weights) / total


def gks_rms_relative_l2(weights: np.ndarray) -> float:
    weights = np.sort(weights)
    total = float(np.sum(weights))
    target = exact_degrees(weights)
    suffix = np.cumsum(weights[::-1])[::-1]
    suffix_square = np.cumsum((weights * weights)[::-1])[::-1]
    variance = 0.0
    for source in range(len(weights) - 1):
        remaining = suffix[source + 1]
        remaining_square = suffix_square[source + 1]
        value = weights[source] * remaining / total
        variance += value * value * (1.0 - remaining_square / (remaining * remaining))
    return math.sqrt(max(0.0, variance) / float(np.dot(target, target)))


def ptree_relative_l2_squared(weights: np.ndarray, q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
        return float("inf")
    q = q / np.sum(q)
    total = float(np.sum(weights))
    target = exact_degrees(weights)
    variance_sum = 0.0
    for center in range(len(weights)):
        mask = np.arange(len(weights)) != center
        q_other = q[mask]
        w_other = weights[mask]
        probability = q[center] + q_other
        value = weights[center] * w_other / (total * probability)
        # Var(sum_j h_j I_ij), using Cov(I_ij,I_ik)=-q_j q_k.
        diagonal = float(np.sum(value * value * probability * (1.0 - probability)))
        weighted = value * q_other
        covariance = float(np.sum(weighted) ** 2 - np.dot(weighted, weighted))
        variance_sum += diagonal - covariance
    return max(0.0, variance_sum) / float(np.dot(target, target))


def power_q(weights: np.ndarray, alpha: float) -> np.ndarray:
    scaled = np.power(weights / float(np.max(weights)), alpha)
    return scaled / float(np.sum(scaled))


def best_power(weights: np.ndarray) -> tuple[float, float]:
    result = minimize_scalar(
        lambda alpha: ptree_relative_l2_squared(weights, power_q(weights, alpha)),
        bounds=(-1.0, 8.0), method="bounded", options={"xatol": 1e-11},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x), math.sqrt(float(result.fun))


def optimize_q(weights: np.ndarray, leverage_floor: float = 0.0) -> tuple[np.ndarray, float]:
    n = len(weights)
    leverage_q = weights / float(np.sum(weights))
    lower = np.maximum(1e-10, leverage_floor * leverage_q)
    starts = [
        (1.0 - leverage_floor) * power_q(weights, alpha) + leverage_floor * leverage_q
        for alpha in (0.0, 0.5, 1.0, 1.5, 1.75, 2.0, 3.0, 5.0)
    ]
    rng = np.random.default_rng(20260903)
    starts.extend(
        (1.0 - leverage_floor) * rng.dirichlet(np.ones(n)) + leverage_floor * leverage_q
        for _ in range(8)
    )
    best = None
    for start in starts:
        result = minimize(
            lambda q: ptree_relative_l2_squared(weights, q),
            start,
            method="SLSQP",
            bounds=[(float(value), 1.0) for value in lower],
            constraints={"type": "eq", "fun": lambda q: float(np.sum(q) - 1.0)},
            options={"ftol": 1e-14, "maxiter": 3000},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("all q optimizations failed")
    q = np.maximum(np.asarray(best.x, dtype=float), lower)
    q /= np.sum(q)
    return q, math.sqrt(ptree_relative_l2_squared(weights, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=pathlib.Path)
    parser.add_argument("--plot", type=pathlib.Path)
    args = parser.parse_args()

    rows: list[tuple[object, ...]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for name, raw in CASES.items():
        weights = np.asarray(raw, dtype=float)
        gks = gks_rms_relative_l2(weights)
        bkz = math.sqrt(ptree_relative_l2_squared(weights, power_q(weights, 1.0)))
        alpha, power = best_power(weights)
        q, optimum = optimize_q(weights)
        max_ht = 1.0 / min(
            q[left] + q[right]
            for left in range(len(q)) for right in range(left + 1, len(q))
        )
        protected = []
        for floor in (0.01, 0.05, 0.25):
            floor_q, floor_error = optimize_q(weights, floor)
            floor_max_ht = 1.0 / min(
                floor_q[left] + floor_q[right]
                for left in range(len(floor_q)) for right in range(left + 1, len(floor_q))
            )
            protected.append((floor, floor_error, floor_max_ht))
        rows.append((name, gks, bkz, alpha, power, optimum, max_ht,
                     ",".join(f"{value:.7g}" for value in q), protected))
        alphas = np.linspace(-0.5, 4.0, 181)
        curve = np.asarray([
            math.sqrt(ptree_relative_l2_squared(weights, power_q(weights, value)))
            for value in alphas
        ])
        curves[name] = (alphas, curve, gks, optimum)

    text = "case\tgks_l2\tbkz_alpha1_l2\tbest_alpha\tbest_power_l2\tbest_q_l2\tbest_q_max_ht\tbest_q\tfloor1_l2/maxht\tfloor5_l2/maxht\tfloor25_l2/maxht\n"
    for name, gks, bkz, alpha, power, optimum, max_ht, q, protected in rows:
        floor_text = "\t".join(f"{error:.8f}/{ht:.3g}" for _, error, ht in protected)
        text += f"{name}\t{gks:.8f}\t{bkz:.8f}\t{alpha:.8f}\t{power:.8f}\t{optimum:.8f}\t{max_ht:.3g}\t{q}\t{floor_text}\n"
    print(text, end="")
    if args.tsv:
        args.tsv.write_text(text)

    if args.plot:
        import matplotlib.pyplot as plt

        plt.rcParams["svg.hashsalt"] = "apxchol-bkz26-local-degree"
        figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
        for axis, name in zip(axes.flat, ("mild", "geometric", "two-scale", "six-decades")):
            alphas, curve, gks, optimum = curves[name]
            axis.plot(alphas, curve, color="#3465a4", label="p-tree power law")
            axis.axhline(gks, color="#2e8b57", linestyle="--", label="GKS")
            axis.axhline(optimum, color="#a33", linestyle=":", label="best unrestricted q")
            axis.axvline(1.0, color="#777", linewidth=0.8)
            axis.set_yscale("log")
            axis.set_title(name)
            axis.set_xlabel(r"power $\alpha$ in $q_i\propto w_i^\alpha$")
            axis.set_ylabel("exact RMS relative degree error")
            axis.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3,
                      bbox_to_anchor=(0.5, 0.995))
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=180, metadata={"Date": None})
        # Matplotlib 3.11 leaves spaces at the ends of SVG path-data lines.
        # Canonicalize them so regenerating the committed asset is byte-stable.
        svg = args.plot.read_text()
        args.plot.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
        )


if __name__ == "__main__":
    main()

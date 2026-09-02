#!/usr/bin/env python3
"""Exact local-degree error comparison for small product-clique stars.

This is deliberately a local estimator diagnostic, not a solver benchmark.
It enumerates every GKS outcome and every Pruefer code, so the reported L1
expectations have no Monte Carlo error.  Closed-form variances independently
check the enumerated L2 values.
"""

from __future__ import annotations

import itertools
import math
import random


def targets(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(weights)
    return tuple(w * (total - w) / total for w in weights)


def errors(sampled: list[float], exact: tuple[float, ...]) -> tuple[float, float]:
    scale1 = sum(exact)
    scale2 = sum(value * value for value in exact)
    return (
        sum(abs(value - target) for value, target in zip(sampled, exact)) / scale1,
        sum((value - target) ** 2 for value, target in zip(sampled, exact)) / scale2,
    )


def enumerate_gks(weights: tuple[float, ...]) -> tuple[float, float]:
    weights = tuple(sorted(weights))
    n = len(weights)
    exact = targets(weights)
    choices = [tuple(range(source + 1, n)) for source in range(n - 1)]
    expected_l1 = expected_l2_squared = probability_sum = 0.0
    for partners in itertools.product(*choices):
        sampled = [0.0] * n
        probability = 1.0
        for source, partner in enumerate(partners):
            suffix = sum(weights[source + 1 :])
            probability *= weights[partner] / suffix
            value = weights[source] * suffix / sum(weights)
            sampled[source] += value
            sampled[partner] += value
        l1, l2_squared = errors(sampled, exact)
        expected_l1 += probability * l1
        expected_l2_squared += probability * l2_squared
        probability_sum += probability
    assert math.isclose(probability_sum, 1.0, rel_tol=2e-12, abs_tol=2e-12)
    return expected_l1, math.sqrt(expected_l2_squared)


def decode_pruefer(code: tuple[int, ...], n: int) -> list[tuple[int, int]]:
    degree = [1] * n
    for symbol in code:
        degree[symbol] += 1
    edges: list[tuple[int, int]] = []
    for symbol in code:
        leaf = next(vertex for vertex, value in enumerate(degree) if value == 1)
        edges.append((leaf, symbol))
        degree[leaf] -= 1
        degree[symbol] -= 1
    leaves = [vertex for vertex, value in enumerate(degree) if value == 1]
    assert len(leaves) == 2
    edges.append((leaves[0], leaves[1]))
    return edges


def enumerate_ptree(weights: tuple[float, ...], alpha: float) -> tuple[float, float]:
    n = len(weights)
    exact = targets(weights)
    raw_q = [weight**alpha for weight in weights]
    q_sum = sum(raw_q)
    q = [value / q_sum for value in raw_q]
    expected_l1 = expected_l2_squared = probability_sum = 0.0
    for code in itertools.product(range(n), repeat=n - 2):
        probability = math.prod(q[symbol] for symbol in code)
        sampled = [0.0] * n
        for left, right in decode_pruefer(code, n):
            marginal = q[left] + q[right]
            value = weights[left] * weights[right] / (sum(weights) * marginal)
            sampled[left] += value
            sampled[right] += value
        l1, l2_squared = errors(sampled, exact)
        expected_l1 += probability * l1
        expected_l2_squared += probability * l2_squared
        probability_sum += probability
    assert math.isclose(probability_sum, 1.0, rel_tol=2e-12, abs_tol=2e-12)
    return expected_l1, math.sqrt(expected_l2_squared)


def gks_l2_formula(weights: tuple[float, ...]) -> float:
    weights = tuple(sorted(weights))
    total = sum(weights)
    exact = targets(weights)
    variance_sum = 0.0
    for target_index in range(len(weights)):
        for source in range(target_index):
            suffix = sum(weights[source + 1 :])
            probability = weights[target_index] / suffix
            value = weights[source] * suffix / total
            variance_sum += value * value * probability * (1.0 - probability)
    return math.sqrt(variance_sum / sum(value * value for value in exact))


def ptree_l2_formula(weights: tuple[float, ...], alpha: float) -> float:
    total = sum(weights)
    exact = targets(weights)
    raw_q = [weight**alpha for weight in weights]
    q_sum = sum(raw_q)
    q = [value / q_sum for value in raw_q]
    variance_sum = 0.0
    for center in range(len(weights)):
        incident: list[tuple[int, float, float]] = []
        for other in range(len(weights)):
            if other == center:
                continue
            probability = q[center] + q[other]
            value = weights[center] * weights[other] / (total * probability)
            variance_sum += value * value * probability * (1.0 - probability)
            incident.append((other, value, probability))
        for first in range(len(incident)):
            other_first, value_first, _ = incident[first]
            for second in range(first + 1, len(incident)):
                other_second, value_second, _ = incident[second]
                # Product-clique spanning-tree edge indicators are determinantal.
                # For two edges sharing `center`, Cov(I_ci,I_cj) = -q_i q_j.
                variance_sum -= 2.0 * value_first * value_second * q[other_first] * q[other_second]
    return math.sqrt(max(0.0, variance_sum) / sum(value * value for value in exact))


def main() -> None:
    cases = {
        "uniform": (1, 1, 1, 1, 1, 1),
        "mild": (1, 1.5, 2, 3, 5, 8),
        "geometric": (1, 2, 4, 8, 16, 32),
        "two-scale": (1, 1, 1, 1, 100, 100),
        "six-decades": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1),
    }
    print("case\tsampl\tE_rel_L1\trms_rel_L2")
    for name, raw_weights in cases.items():
        weights = tuple(float(value) for value in raw_weights)
        results = {
            "gks": enumerate_gks(weights),
            "bkz-a1": enumerate_ptree(weights, 1.0),
            "tilt-a1.75": enumerate_ptree(weights, 1.75),
        }
        assert math.isclose(results["gks"][1], gks_l2_formula(weights), rel_tol=2e-11, abs_tol=2e-11)
        for alpha, label in ((1.0, "bkz-a1"), (1.75, "tilt-a1.75")):
            assert math.isclose(results[label][1], ptree_l2_formula(weights, alpha), rel_tol=2e-11, abs_tol=2e-11)
        for sampler, (l1, l2) in results.items():
            print(f"{name}\t{sampler}\t{l1:.8f}\t{l2:.8f}")

    rng = random.Random(20260903)
    print("\nspread\tstars\tGKS_lower_L2\tBKZ_lower_L2\tties")
    for decades in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        counts = [0, 0, 0]
        for _ in range(2000):
            weights = tuple(10.0 ** rng.uniform(-decades, 0.0) for _ in range(8))
            gks = gks_l2_formula(weights)
            bkz = ptree_l2_formula(weights, 1.0)
            if gks + 1e-12 < bkz:
                counts[0] += 1
            elif bkz + 1e-12 < gks:
                counts[1] += 1
            else:
                counts[2] += 1
        print(f"{decades:.1f}\t2000\t{counts[0]}\t{counts[1]}\t{counts[2]}")


if __name__ == "__main__":
    main()

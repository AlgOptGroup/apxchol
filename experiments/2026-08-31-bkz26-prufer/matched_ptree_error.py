#!/usr/bin/env python3
"""Exact small-star probe for a matching-conditioned product-clique tree.

The sampler makes a deterministic matching exact, contracts every matched
pair, samples a weighted product-clique spanning tree on the components, and
expands every component edge to endpoints proportionally to their weights.
Every outcome is still a tree with d-1 edges.  Exact enumeration below checks
unbiasedness and compares local degree error; it is a research probe, not a
production sampler.
"""

from __future__ import annotations

import itertools
import math

from small_star_error import decode_pruefer, enumerate_gks, enumerate_ptree


CASES = {
    "uniform": (1, 1, 1, 1, 1, 1),
    "mild": (1, 1.5, 2, 3, 5, 8),
    "geometric": (1, 2, 4, 8, 16, 32),
    "two-scale": (1, 1, 1, 1, 100, 100),
    "six-decades": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1),
}


def partial_matchings(vertices: tuple[int, ...]):
    if not vertices:
        yield ()
        return
    first = vertices[0]
    for suffix in partial_matchings(vertices[1:]):
        yield suffix
    for offset in range(1, len(vertices)):
        second = vertices[offset]
        rest = vertices[1:offset] + vertices[offset + 1 :]
        for suffix in partial_matchings(rest):
            yield ((first, second),) + suffix


def matching_key(matching: tuple[tuple[int, int], ...], weights: tuple[float, ...]):
    return tuple(sorted((weights[left], weights[right]) for left, right in matching))


def enumerate_matching_ptree(
    weights: tuple[float, ...], matching: tuple[tuple[int, int], ...]
) -> tuple[float, float]:
    n = len(weights)
    total = sum(weights)
    exact = tuple(weight * (total - weight) / total for weight in weights)
    matched_vertices = {vertex for pair in matching for vertex in pair}
    components = tuple(tuple(pair) for pair in matching) + tuple(
        (vertex,) for vertex in range(n) if vertex not in matched_vertices
    )
    assert sorted(vertex for component in components for vertex in component) == list(range(n))
    component_weights = tuple(sum(weights[vertex] for vertex in component) for component in components)
    q = tuple(value / total for value in component_weights)

    deterministic = [0.0] * n
    deterministic_edges = []
    for left, right in matching:
        value = weights[left] * weights[right] / total
        deterministic[left] += value
        deterministic[right] += value
        deterministic_edges.append((left, right, value))

    mean = [0.0] * n
    mean_edges = [[0.0] * n for _ in range(n)]
    expected_l1 = expected_l2_squared = probability_sum = 0.0
    component_codes = itertools.product(range(len(components)), repeat=max(0, len(components) - 2))
    for code in component_codes:
        code_probability = math.prod(q[symbol] for symbol in code)
        component_edges = decode_pruefer(code, len(components)) if len(components) > 1 else []
        endpoint_choices = []
        for left_component, right_component in component_edges:
            choices = []
            left_weight = component_weights[left_component]
            right_weight = component_weights[right_component]
            edge_value = total * left_weight * right_weight / (
                total * (left_weight + right_weight)
            )
            for left in components[left_component]:
                for right in components[right_component]:
                    probability = (
                        weights[left] / left_weight * weights[right] / right_weight
                    )
                    choices.append((left, right, edge_value, probability))
            endpoint_choices.append(tuple(choices))

        for expanded_edges in itertools.product(*endpoint_choices):
            sampled = deterministic.copy()
            sampled_edges = deterministic_edges.copy()
            probability = code_probability
            for left, right, value, endpoint_probability in expanded_edges:
                sampled[left] += value
                sampled[right] += value
                sampled_edges.append((left, right, value))
                probability *= endpoint_probability
            probability_sum += probability
            for vertex, value in enumerate(sampled):
                mean[vertex] += probability * value
            for left, right, value in sampled_edges:
                left, right = sorted((left, right))
                mean_edges[left][right] += probability * value
            scale1 = sum(exact)
            scale2 = sum(value * value for value in exact)
            expected_l1 += probability * sum(
                abs(value - target) for value, target in zip(sampled, exact)
            ) / scale1
            expected_l2_squared += probability * sum(
                (value - target) ** 2 for value, target in zip(sampled, exact)
            ) / scale2

    assert math.isclose(probability_sum, 1.0, rel_tol=2e-12, abs_tol=2e-12)
    for observed, target in zip(mean, exact):
        assert math.isclose(observed, target, rel_tol=2e-11, abs_tol=2e-11)
    for left in range(n):
        for right in range(left + 1, n):
            target = weights[left] * weights[right] / total
            assert math.isclose(
                mean_edges[left][right], target,
                rel_tol=2e-11, abs_tol=2e-11,
            )
    return expected_l1, math.sqrt(expected_l2_squared)


def main() -> None:
    print("case\tgks_l2\tbkz_l2\ta1.75_l2\ttop_pair_l2\tfull_assortative_l2\tbest_matching_l2\tbest_matching")
    for name, raw_weights in CASES.items():
        weights = tuple(float(value) for value in raw_weights)
        order = tuple(sorted(range(len(weights)), key=lambda vertex: weights[vertex], reverse=True))
        top_pair = ((order[0], order[1]),)
        assortative = tuple((order[offset], order[offset + 1]) for offset in range(0, len(order), 2))
        top_pair_error = enumerate_matching_ptree(weights, top_pair)[1]
        assortative_error = enumerate_matching_ptree(weights, assortative)[1]
        evaluated = [
            (enumerate_matching_ptree(weights, matching)[1], matching)
            for matching in partial_matchings(tuple(range(len(weights))))
        ]
        best_error, best_matching = min(evaluated)
        print(
            f"{name}\t{enumerate_gks(weights)[1]:.8f}"
            f"\t{enumerate_ptree(weights, 1.0)[1]:.8f}"
            f"\t{enumerate_ptree(weights, 1.75)[1]:.8f}"
            f"\t{top_pair_error:.8f}\t{assortative_error:.8f}\t{best_error:.8f}"
            f"\t{matching_key(best_matching, weights)}"
        )


if __name__ == "__main__":
    main()

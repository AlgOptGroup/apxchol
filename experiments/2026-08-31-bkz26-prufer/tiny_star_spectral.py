#!/usr/bin/env python3
"""Exact tiny-star comparison plus a numerical all-tree comparator.

Named samplers and every alpha curve are evaluated by enumerating all 16
labeled trees of K4.  The "best-found all-tree" rows are deterministic
multi-start numerical searches over all tree probabilities.  They are useful
lower comparators, not certified global optima.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

CASES = {
    "uniform": (1.0, 1.0, 1.0, 1.0),
    "graded": (1.0, 2.0, 4.0, 8.0),
    "two-scale": (1.0, 1.0, 8.0, 8.0),
}
METRICS = ("degree_rms", "expected_normalized_spectral")


def pruefer_decode(code: tuple[int, ...], size: int) -> tuple[tuple[int, int], ...]:
    degree = [1] * size
    for symbol in code:
        degree[symbol] += 1
    edges: list[tuple[int, int]] = []
    for symbol in code:
        leaf = next(vertex for vertex, value in enumerate(degree) if value == 1)
        edges.append((min(leaf, symbol), max(leaf, symbol)))
        degree[leaf] -= 1
        degree[symbol] -= 1
    leaves = [vertex for vertex, value in enumerate(degree) if value == 1]
    edges.append((min(leaves), max(leaves)))
    return tuple(sorted(edges))


class Star:
    def __init__(self, raw_weights: tuple[float, ...]):
        self.weights = np.asarray(raw_weights, dtype=float)
        self.size = len(self.weights)
        self.total = float(np.sum(self.weights))
        self.edges = tuple(itertools.combinations(range(self.size), 2))
        self.edge_index = {edge: index for index, edge in enumerate(self.edges)}
        self.trees = tuple(
            sorted(
                {
                    pruefer_decode(code, self.size)
                    for code in itertools.product(
                        range(self.size), repeat=self.size - 2
                    )
                }
            )
        )
        if len(self.trees) != self.size ** (self.size - 2):
            raise RuntimeError("Prüfer enumeration did not produce every labeled tree")

        self.membership = np.zeros((len(self.trees), len(self.edges)))
        for tree_index, tree in enumerate(self.trees):
            for edge in tree:
                self.membership[tree_index, self.edge_index[edge]] = 1.0

        self.vertex_edge = np.zeros((self.size, len(self.edges)))
        for edge_index, (left, right) in enumerate(self.edges):
            self.vertex_edge[left, edge_index] = 1.0
            self.vertex_edge[right, edge_index] = 1.0
        self.tree_degree = self.membership @ self.vertex_edge.T
        self.clique_weight = np.asarray(
            [
                self.weights[left] * self.weights[right] / self.total
                for left, right in self.edges
            ]
        )
        self.exact_degree = self.weights * (self.total - self.weights) / self.total

        clique = np.zeros((self.size, self.size))
        edge_laplacians = []
        for weight, (left, right) in zip(self.clique_weight, self.edges):
            laplacian = np.zeros_like(clique)
            laplacian[left, left] = laplacian[right, right] = 1.0
            laplacian[left, right] = laplacian[right, left] = -1.0
            clique += weight * laplacian
            edge_laplacians.append(laplacian)
        eigenvalues, eigenvectors = np.linalg.eigh(clique)
        keep = eigenvalues > 1e-12 * float(np.max(eigenvalues))
        inverse_sqrt = (
            eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
        ) @ eigenvectors[:, keep].T
        self.normalized_exact = inverse_sqrt @ clique @ inverse_sqrt
        self.normalized_edges = np.asarray(
            [inverse_sqrt @ laplacian @ inverse_sqrt for laplacian in edge_laplacians]
        )

    def gks_distribution(self) -> np.ndarray:
        order = list(np.argsort(self.weights, kind="stable"))
        choices: list[list[tuple[int, float]]] = []
        for position in range(self.size - 1):
            later = order[position + 1 :]
            suffix = float(np.sum(self.weights[later]))
            choices.append(
                [(partner, self.weights[partner] / suffix) for partner in later]
            )
        probabilities = np.zeros(len(self.trees))
        tree_index = {tree: index for index, tree in enumerate(self.trees)}
        for outcome in itertools.product(*choices):
            edges = []
            probability = 1.0
            for position, (partner, factor) in enumerate(outcome):
                source = order[position]
                edges.append((min(source, partner), max(source, partner)))
                probability *= float(factor)
            probabilities[tree_index[tuple(sorted(edges))]] += probability
        rank = {vertex: position for position, vertex in enumerate(order)}
        expected_marginals = []
        for left, right in self.edges:
            source, parent = (
                (left, right) if rank[left] < rank[right] else (right, left)
            )
            later = order[rank[source] + 1 :]
            expected_marginals.append(
                self.weights[parent] / np.sum(self.weights[later])
            )
        if not np.allclose(
            probabilities @ self.membership, expected_marginals, rtol=1e-13, atol=1e-13
        ):
            raise RuntimeError("GKS enumeration disagrees with its edge marginals")
        return probabilities

    def ptree_distribution(self, alpha: float) -> np.ndarray:
        log_q = alpha * np.log(self.weights / float(np.max(self.weights)))
        q = np.exp(log_q - float(np.max(log_q)))
        q /= float(np.sum(q))
        log_probability = (self.tree_degree - 1.0) @ np.log(q)
        probabilities = np.exp(log_probability - float(np.max(log_probability)))
        probabilities /= float(np.sum(probabilities))
        expected_marginals = np.asarray(
            [q[left] + q[right] for left, right in self.edges]
        )
        if not np.allclose(
            probabilities @ self.membership, expected_marginals, rtol=1e-13, atol=1e-13
        ):
            raise RuntimeError("Prüfer enumeration disagrees with q_i + q_j")
        return probabilities

    def statistics(self, probabilities: np.ndarray) -> dict[str, float]:
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities = probabilities / float(np.sum(probabilities))
        marginals = probabilities @ self.membership
        if float(np.min(marginals)) <= 0.0:
            return {metric: float("inf") for metric in METRICS} | {
                "edge_marginal_min": float(np.min(marginals))
            }
        sampled_edges = self.membership * (self.clique_weight / marginals)
        if not np.allclose(
            probabilities @ sampled_edges, self.clique_weight, rtol=1e-13, atol=1e-13
        ):
            raise RuntimeError("tree reweighting is not edgewise unbiased")
        sampled_degree = sampled_edges @ self.vertex_edge.T
        error = sampled_degree - self.exact_degree
        degree_rms = math.sqrt(
            float(probabilities @ np.sum(error * error, axis=1))
            / float(np.dot(self.exact_degree, self.exact_degree))
        )
        normalized = (
            np.einsum("te,eij->tij", sampled_edges, self.normalized_edges)
            - self.normalized_exact[None, :, :]
        )
        spectral_per_tree = np.max(np.abs(np.linalg.eigvalsh(normalized)), axis=1)
        return {
            "degree_rms": degree_rms,
            "expected_normalized_spectral": float(probabilities @ spectral_per_tree),
            "edge_marginal_min": float(np.min(marginals)),
        }

    def best_found(self, metric: str) -> tuple[np.ndarray, bool]:
        rng = np.random.default_rng(20260903)
        starts = [
            np.full(len(self.trees), 1.0 / len(self.trees)),
            self.gks_distribution(),
            *(self.ptree_distribution(alpha) for alpha in (0.0, 1.0, 1.75, 2.0, 3.0)),
            *(rng.dirichlet(np.ones(len(self.trees))) for _ in range(8)),
        ]
        best_probability = starts[0]
        best_value = self.statistics(best_probability)[metric]
        converged = False

        def objective(theta: np.ndarray) -> float:
            centered = theta - float(np.max(theta))
            probability = np.exp(centered)
            probability /= float(np.sum(probability))
            stats = self.statistics(probability)
            shortfall = max(0.0, 1e-6 - stats["edge_marginal_min"])
            return stats[metric] + 1e6 * shortfall * shortfall

        for start in starts:
            direct_value = self.statistics(start)[metric]
            if direct_value < best_value:
                best_probability, best_value = start, direct_value
            regularized = np.maximum(start, 1e-12)
            regularized /= float(np.sum(regularized))
            result = minimize(
                objective,
                np.log(regularized),
                method="L-BFGS-B",
                bounds=[(-30.0, 30.0)] * len(self.trees),
                options={"maxiter": 1200, "ftol": 1e-14, "gtol": 1e-10},
            )
            candidate = np.exp(result.x - float(np.max(result.x)))
            candidate /= float(np.sum(candidate))
            value = self.statistics(candidate)[metric]
            if value < best_value:
                best_probability, best_value = candidate, value
                converged = bool(result.success)
        return best_probability, converged


def write_outputs(tsv_path: pathlib.Path | None, plot_path: pathlib.Path | None) -> str:
    alpha_grid = np.unique(np.append(np.linspace(0.0, 3.0, 61), 1.75))
    rows: list[dict[str, object]] = []
    plot_data: dict[tuple[str, str], dict[str, object]] = {}

    for case, weights in CASES.items():
        star = Star(weights)
        gks = star.gks_distribution()
        best_by_metric = {metric: star.best_found(metric) for metric in METRICS}
        curves = [
            (float(alpha), star.statistics(star.ptree_distribution(float(alpha))))
            for alpha in alpha_grid
        ]
        for metric in METRICS:
            gks_stats = star.statistics(gks)
            best_probability, converged = best_by_metric[metric]
            best_stats = star.statistics(best_probability)
            rows.append(
                {
                    "case": case,
                    "weights": ",".join(f"{value:g}" for value in weights),
                    "metric": metric,
                    "sampler": "gks",
                    "alpha": "",
                    "value": gks_stats[metric],
                    "edge_marginal_min": gks_stats["edge_marginal_min"],
                    "optimizer_converged": "not-applicable",
                }
            )
            rows.append(
                {
                    "case": case,
                    "weights": ",".join(f"{value:g}" for value in weights),
                    "metric": metric,
                    "sampler": "best-found-all-tree",
                    "alpha": "",
                    "value": best_stats[metric],
                    "edge_marginal_min": best_stats["edge_marginal_min"],
                    "optimizer_converged": str(converged).lower(),
                }
            )
            for alpha, stats in curves:
                rows.append(
                    {
                        "case": case,
                        "weights": ",".join(f"{value:g}" for value in weights),
                        "metric": metric,
                        "sampler": "ptree",
                        "alpha": alpha,
                        "value": stats[metric],
                        "edge_marginal_min": stats["edge_marginal_min"],
                        "optimizer_converged": "not-applicable",
                    }
                )
            plot_data[(case, metric)] = {
                "alpha": np.asarray([alpha for alpha, _ in curves]),
                "curve": np.asarray([stats[metric] for _, stats in curves]),
                "gks": gks_stats[metric],
                "best": best_stats[metric],
            }

    fieldnames = tuple(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        printable = dict(row)
        printable["value"] = f"{float(row['value']):.10g}"
        printable["edge_marginal_min"] = f"{float(row['edge_marginal_min']):.10g}"
        if row["alpha"] != "":
            printable["alpha"] = f"{float(row['alpha']):.2f}"
        writer.writerow(printable)
    text = buffer.getvalue()
    if tsv_path:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        tsv_path.write_text(text)

    if plot_path:
        plt.rcParams.update(
            {
                "font.size": 8.5,
                "svg.fonttype": "none",
                "svg.hashsalt": "apxchol-bkz26-tiny-star",
            }
        )
        figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.2), sharex=True)
        titles = {
            "uniform": "uniform: 1, 1, 1, 1",
            "graded": "graded: 1, 2, 4, 8",
            "two-scale": "two-scale: 1, 1, 8, 8",
        }
        labels = {
            "degree_rms": "RMS relative degree error",
            "expected_normalized_spectral": "expected normalized spectral error",
        }
        for column, case in enumerate(CASES):
            for row_index, metric in enumerate(METRICS):
                axis = axes[row_index, column]
                data = plot_data[(case, metric)]
                axis.plot(
                    data["alpha"],
                    data["curve"],
                    color="#315f9b",
                    linewidth=1.6,
                    label="p-tree across α",
                )
                axis.axhline(
                    data["gks"],
                    color="#2d7f5e",
                    linestyle="--",
                    linewidth=1.4,
                    label="GKS",
                )
                axis.axhline(
                    data["best"],
                    color="#222",
                    linestyle=":",
                    linewidth=1.5,
                    label="best-found all-tree",
                )
                for alpha, marker, color in (
                    (1.0, "o", "#d07a21"),
                    (1.75, "D", "#b43c52"),
                ):
                    index = int(np.argmin(np.abs(data["alpha"] - alpha)))
                    axis.scatter(
                        alpha,
                        data["curve"][index],
                        marker=marker,
                        color=color,
                        s=30,
                        zorder=4,
                        label=("BKZ α=1" if alpha == 1.0 else "α=1.75"),
                    )
                axis.set_title(titles[case])
                axis.set_ylabel(labels[metric])
                axis.grid(alpha=0.20)
                if row_index == 1:
                    axis.set_xlabel(r"symbol exponent $\alpha$")
        handles, labels_ = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels_,
            loc="upper center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
        figure.suptitle(
            "Tiny-star model: exact named samplers and a numerical all-tree comparator",
            y=0.94,
            fontsize=11,
        )
        figure.text(
            0.5,
            0.012,
            "All 16 labeled K4 trees are enumerated. The all-tree search is deterministic but not a certified global optimum.",
            ha="center",
            fontsize=8,
            color="#444",
        )
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.89))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            plot_path, metadata={"Date": None, "Creator": "tiny_star_spectral.py"}
        )
        # Matplotlib 3.11 leaves spaces at the ends of SVG path-data lines.
        # Canonicalize them so the committed text asset passes git whitespace checks.
        svg = plot_path.read_text()
        plot_path.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
        )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=pathlib.Path)
    parser.add_argument("--plot", type=pathlib.Path)
    args = parser.parse_args()
    text = write_outputs(args.tsv, args.plot)
    if not args.tsv:
        print(text, end="")


if __name__ == "__main__":
    main()

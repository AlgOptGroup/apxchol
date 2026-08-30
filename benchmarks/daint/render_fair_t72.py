#!/usr/bin/env python3
"""Import and render the repaired T=72 fair-solver campaign from CSCS Daint.

With ``--cells``, validate a downloaded unified cell store and refresh the
committed compact CSV.  Without it, reproduce the summary and figures from that
CSV alone.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
sys.path.insert(0, str(BENCHMARKS))

import combined_charts as combined  # noqa: E402
import fair_charts as cpu  # noqa: E402
import gpu_charts as gpu  # noqa: E402
from runner_common import mat_labels  # noqa: E402


CORE_CELLS = 588
EXPECTED_CELLS = 750
EXPECTED_MATRICES = 27
CORE_STATUS = {"complete": 578, "timeout": 8, "failed": 2}
SOURCE_SHA = "2b75599748604439c2b51027ab2c6a428fed9c66"
EXTENSION_SOURCE_ID = "portable-competitors-20260828-r1"
EXTENSION_SERIES = {
    ("cpu", "rchol", ""),
    ("cpu", "rchol_par", ""),
    ("cpu", "ac", ""),
    ("cpu", "ac2", ""),
    ("gpu", "parac_graph", ""),
    ("gpu", "parac_physics", ""),
}
DEFAULT_CONFIG = "bg+tree[vec_pool_aos]"
FAMILIES = ("grids", "ipm", "suitesparse")

CSV_FIELDS = (
    "family", "matrix_id", "device", "solver", "config", "threads", "status",
    "n", "nnz", "setup_s", "solve_s", "total_s", "iters", "rel_res", "fillin",
    "us_per_nnz", "solve_rss_mb", "max_rss_mb", "max_vram_mb", "cuda_init_s",
    "timeout_cap_s",
    "returncode", "diagnostic", "git_sha", "source_id", "repeat", "compiler",
    "compiler_version", "openmp_runtime", "node_index_bits", "edge_index_bits",
    "hypre_cuda", "rchol_pcg", "cuda_host_compiler", "arch_flags",
)
INT_FIELDS = {"threads", "n", "nnz", "iters", "returncode", "repeat",
              "node_index_bits", "edge_index_bits"}
FLOAT_FIELDS = {"setup_s", "solve_s", "total_s", "rel_res", "fillin", "us_per_nnz",
                "solve_rss_mb", "max_rss_mb", "max_vram_mb", "cuda_init_s",
                "timeout_cap_s"}


def geomean(values) -> float:
    values = list(values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean needs positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _diagnostic(record: dict) -> str:
    status = record["status"]
    metrics = record.get("metrics", {})
    if record["status"] == "timeout":
        cap = record.get("timeout_cap_s")
        return f"wall-clock cap {cap:g}s" if cap else "wall-clock timeout"
    if status == "not_converged":
        residual = metrics.get("rel_res")
        return (f"true residual {residual:.3g} > 1e-8" if residual is not None
                else "independent true-residual check exceeded 1e-8")
    blob = "\n".join(str(metrics.get(key, ""))
                     for key in ("stdout_tail", "stderr_tail")).lower()
    if "insufficient resources" in blob and "cusparse" in blob:
        return "cuSPARSE insufficient resources during SpGEMM"
    if "invalid factor diagonal" in blob:
        return "portable PCG rejected an invalid factor diagonal"
    returncode = metrics.get("returncode")
    if returncode in (139, -11):
        return "process terminated with SIGSEGV"
    reason = str(record.get("matrix_meta", {}).get("na_reason", ""))
    if status == "failed" and reason:
        return "Julia driver failed" if "bench_laplacians.jl" in reason else reason[:160]
    if status == "failed" and returncode is not None:
        return f"process exit status {returncode}"
    return ""


def flatten(record: dict) -> dict[str, object]:
    cell = record["cell"]
    metrics = record.get("metrics", {})
    matrix = record.get("matrix_meta", {})
    provenance = record.get("provenance", {})
    row = {field: "" for field in CSV_FIELDS}
    row.update({field: cell.get(field, "")
                for field in ("family", "matrix_id", "device", "solver", "config", "threads")})
    row["status"] = record["status"]
    for field in ("n", "nnz"):
        row[field] = metrics.get(field, matrix.get(field, ""))
    for field in ("setup_s", "solve_s", "total_s", "iters", "rel_res", "fillin",
                  "us_per_nnz", "solve_rss_mb", "max_rss_mb", "max_vram_mb",
                  "cuda_init_s",
                  "returncode"):
        row[field] = metrics.get(field, "")
    row["timeout_cap_s"] = record.get("timeout_cap_s", "")
    row["diagnostic"] = _diagnostic(record)
    for field in ("git_sha", "source_id", "repeat", "compiler", "compiler_version", "openmp_runtime",
                  "node_index_bits", "edge_index_bits", "hypre_cuda",
                  "rchol_pcg", "cuda_host_compiler", "arch_flags"):
        row[field] = provenance.get(field, "")
    return row


def load_cells(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.glob("**/*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid cell {path}: {error}") from error
        if isinstance(record, dict) and "cell" in record and "status" in record:
            records.append(record)
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flatten(record) for record in records)


def read_csv(path: Path) -> list[dict]:
    records = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            typed: dict[str, object] = dict(raw)
            for field in INT_FIELDS:
                typed[field] = int(raw[field]) if raw.get(field) else None
            for field in FLOAT_FIELDS:
                typed[field] = float(raw[field]) if raw.get(field) else None
            metrics = {
                field: typed[field]
                for field in ("n", "nnz", "setup_s", "solve_s", "total_s", "iters",
                              "rel_res", "fillin", "us_per_nnz", "solve_rss_mb",
                              "max_rss_mb", "max_vram_mb", "cuda_init_s", "returncode")
                if typed.get(field) is not None
            }
            provenance = {
                field: typed[field]
                for field in ("git_sha", "source_id", "repeat", "compiler", "compiler_version",
                              "openmp_runtime", "node_index_bits", "edge_index_bits",
                              "hypre_cuda", "rchol_pcg", "cuda_host_compiler", "arch_flags")
                if typed.get(field) not in (None, "")
            }
            record = {
                "cell": {field: typed[field]
                         for field in ("family", "matrix_id", "device", "solver",
                                       "config", "threads")},
                "matrix_meta": {field: typed[field] for field in ("n", "nnz")},
                "metrics": metrics,
                "provenance": provenance,
                "status": typed["status"],
                "diagnostic": typed.get("diagnostic", ""),
            }
            if typed.get("timeout_cap_s") is not None:
                record["timeout_cap_s"] = typed["timeout_cap_s"]
            records.append(record)
    return records


def validate(records: list[dict]) -> None:
    if len(records) != EXPECTED_CELLS:
        raise RuntimeError(f"expected {EXPECTED_CELLS} cells, got {len(records)}")
    keys = set()
    matrices = set()
    core_status = Counter()
    extension_counts = Counter()
    for record in records:
        cell = record["cell"]
        key = tuple(cell[field] for field in
                    ("family", "matrix_id", "device", "solver", "config", "threads"))
        if key in keys:
            raise RuntimeError(f"duplicate cell {key}")
        keys.add(key)
        matrices.add((cell["family"], cell["matrix_id"]))
        series = (cell["device"], cell["solver"], cell.get("config", ""))
        if series in EXTENSION_SERIES:
            extension_counts[series] += 1
            if record["provenance"].get("source_id") != EXTENSION_SOURCE_ID:
                raise RuntimeError(f"extension cell lacks source id for {key}")
            if record["provenance"].get("repeat") != 3:
                raise RuntimeError(f"extension cell lacks repeat=3 provenance: {key}")
            if cell["solver"] in ("rchol", "rchol_par") \
                    and record["provenance"].get("rchol_pcg") != "portable-eigen":
                raise RuntimeError(f"RCHOL cell lacks portable-PCG provenance: {key}")
        else:
            core_status[record["status"]] += 1
        if cell["threads"] != 72:
            raise RuntimeError(f"non-T72 cell {key}")
        if record["provenance"].get("git_sha") != SOURCE_SHA:
            raise RuntimeError(f"wrong source commit for {key}")
        if cell["solver"] == "hypre_boomeramg_gpu" \
                and record["provenance"].get("hypre_cuda") != "on":
            raise RuntimeError(f"Hypre GPU cell lacks CUDA provenance: {key}")
        if record["status"] == "complete":
            metrics = record["metrics"]
            required = ("setup_s", "solve_s", "total_s", "iters", "rel_res")
            if any(metrics.get(field) is None for field in required):
                raise RuntimeError(f"incomplete metrics for {key}")
            if abs(metrics["setup_s"] + metrics["solve_s"] - metrics["total_s"]) \
                    > max(1e-6, 2e-5 * metrics["total_s"]):
                raise RuntimeError(f"timing arithmetic mismatch for {key}")
            if metrics["rel_res"] > 1e-8:
                raise RuntimeError(f"true residual exceeds tolerance for {key}")
        elif record["status"] == "timeout" and record.get("timeout_cap_s") is None:
            raise RuntimeError(f"timeout without exact cap for {key}")
    if len(matrices) != EXPECTED_MATRICES:
        raise RuntimeError(f"expected {EXPECTED_MATRICES} matrices, got {len(matrices)}")
    if sum(core_status.values()) != CORE_CELLS or dict(core_status) != CORE_STATUS:
        raise RuntimeError(f"core denominator changed: {dict(core_status)}")
    expected_extension = Counter({series: EXPECTED_MATRICES for series in EXTENSION_SERIES})
    if extension_counts != expected_extension:
        raise RuntimeError(f"extension denominator changed: {dict(extension_counts)}")


def gpu_views(records: list[dict]):
    rows = defaultdict(dict)
    outcomes = {}
    for record in records:
        cell = record["cell"]
        if cell["device"] != "gpu":
            continue
        label = gpu.LABELS.get((cell["solver"], cell.get("config", "")))
        if not label:
            continue
        outcome_key = (cell["family"], cell["matrix_id"], label)
        outcomes[outcome_key] = {
            "status": record["status"],
            "timeout_cap_s": record.get("timeout_cap_s"),
        }
        if record["status"] != "complete":
            continue
        metrics = record["metrics"]
        rows[(cell["family"], cell["matrix_id"])][label] = {
            "setup": metrics["setup_s"], "solve": metrics["solve_s"],
            "total": metrics["total_s"], "iters": metrics["iters"],
            "rel_res": metrics["rel_res"], "nnz": metrics.get("nnz", 0),
            "rss_peak": metrics.get("max_rss_mb"),
            "rss_solve": metrics.get("solve_rss_mb"),
            "vram_peak": metrics.get("max_vram_mb"),
        }
    return rows, outcomes


def render_headline(records: list[dict], out: Path) -> None:
    cpu_records = [record for record in records if record["cell"]["device"] == "cpu"]
    gpu_rows, outcomes = gpu_views(records)
    for family in FAMILIES:
        matrices = cpu.matrix_order(cpu_records, family)
        combined.overview_heatmap(
            cpu_records, gpu_rows, family,
            out / f"fair_t72_total_{family}.png",
            mode="combined", metric="total", mats=matrices,
            goutcomes=outcomes, thread_label="T=72",
        )
        combined.overview_heatmap(
            cpu_records, gpu_rows, family,
            out / f"fair_t72_solve_{family}.png",
            mode="combined", metric="solve", mats=matrices,
            goutcomes=outcomes, thread_label="T=72",
        )

        # Linear setup+solve bars remain useful alongside the ratio heatmaps:
        # they show whether a total-time loss comes from setup or the iterative
        # solve.  Reuse the same family splits as the laptop renderer so large
        # social graphs cannot flatten the smaller SuiteSparse matrices.
        headline = cpu.headline_mats(cpu_records, family)
        for suffix, group in cpu.family_groups(cpu_records, family, headline, split=True):
            selected = set(group)
            subset = [record for record in cpu_records
                      if record["cell"]["family"] != family
                      or record["cell"]["matrix_id"] in selected]
            combined._bars(
                subset, gpu_rows, family,
                out / f"fair_t72_breakdown_{family}{suffix}.png",
                stacked=True,
                val=lambda values: (values["setup"], values["solve"]),
                goutcomes=outcomes,
                device="both",
                ycompress=suffix.startswith("_giants"),
                ylabel="time (s) — T=72  [solid = setup, /// = solve]",
                title=f"{family}{suffix}: CPU+GPU setup and solve",
            )


def _index(records: list[dict]) -> dict[tuple, dict]:
    return {
        (r["cell"]["matrix_id"], r["cell"]["device"], r["cell"]["solver"],
         r["cell"].get("config", "")): r
        for r in records
    }


def render_apx_device_crossover(records: list[dict], out: Path) -> None:
    index = _index(records)
    cpu_records = [record for record in records if record["cell"]["device"] == "cpu"]
    ordered = []
    spans = []
    for family in FAMILIES:
        family_matrices = cpu.matrix_order(cpu_records, family)
        begin = len(ordered)
        ordered.extend(family_matrices)
        spans.append((family, begin, len(ordered) - 1))
    phase = (("setup", "setup_s", "#4c78a8"),
             ("solve", "solve_s", "#f58518"),
             ("total", "total_s", "#222222"))
    fig, ax = plt.subplots(figsize=(16, 6.2), constrained_layout=True)
    x = np.arange(len(ordered))
    all_values = []
    for label, metric, color in phase:
        values = []
        for matrix in ordered:
            c = index[(matrix, "cpu", "apxchol_v1", DEFAULT_CONFIG)]
            g = index[(matrix, "gpu", "apxchol_v1", DEFAULT_CONFIG)]
            values.append(c["metrics"][metric] / g["metrics"][metric])
        all_values.extend(values)
        ax.plot(x, values, marker="o", markersize=4.2, linewidth=1.5,
                color=color, label=label)
    ax.axhline(1.0, color="0.25", linewidth=1.1, linestyle="--")
    ax.set_yscale("log", base=2)
    lo, hi = min(all_values), max(all_values)
    ax.set_ylim(2 ** math.floor(math.log2(lo) - 0.25),
                2 ** math.ceil(math.log2(hi) + 0.25))
    ax.set_xticks(x, mat_labels(ordered), rotation=42, ha="right", fontsize=8)
    ax.set_ylabel("CPU time / GPU time  (above 1 = GPU faster)")
    ax.set_title("apxchol default CPU/GPU crossover on Daint, T=72")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    for family, begin, end in spans:
        if begin:
            ax.axvline(begin - 0.5, color="0.75", linewidth=1)
        ax.text((begin + end) / 2, 1.015, family, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9)
    ax.legend(ncol=3, loc="lower right")
    fig.savefig(out, dpi=180)
    plt.close(fig)


CONFIG_ORDER = (
    DEFAULT_CONFIG,
    "bg+tree[vec_pool]", "greedy+tree[vec_pool]", "bk+tree[vec_pool]",
    "bg+tree[fwd_star]", "greedy+tree[fwd_star]", "bk+tree[fwd_star]",
    "bg+tree[vec]", "greedy+tree[vec]", "bk+tree[vec]",
    "bg+tree[bstr]", "greedy+tree[bstr]", "bk+tree[bstr]",
)
CONFIG_LABEL = {
    DEFAULT_CONFIG: "bg / vec_pool_aos (default)",
    **{config: config.replace("+tree[", " / ").rstrip("]") for config in CONFIG_ORDER[1:]},
}


def ablation_ratios(records: list[dict], metric: str):
    index = _index(records)
    cpu_records = [record for record in records if record["cell"]["device"] == "cpu"]
    matrices_by_group = {
        family: cpu.matrix_order(cpu_records, family) for family in FAMILIES
    }
    matrices_by_group["all"] = [matrix for family in FAMILIES
                                 for matrix in matrices_by_group[family]]
    values = np.full((len(CONFIG_ORDER), len(matrices_by_group)), np.nan)
    counts = np.zeros_like(values, dtype=int)
    groups = tuple(matrices_by_group)
    for i, config in enumerate(CONFIG_ORDER):
        for j, group in enumerate(groups):
            ratios = []
            for matrix in matrices_by_group[group]:
                record = index.get((matrix, "cpu", "apxchol_v1", config))
                default = index.get((matrix, "cpu", "apxchol_v1", DEFAULT_CONFIG))
                if (record and default and record["status"] == default["status"] == "complete"
                        and record["metrics"].get(metric) and default["metrics"].get(metric)):
                    ratios.append(record["metrics"][metric] / default["metrics"][metric])
            if ratios:
                values[i, j] = geomean(ratios)
                counts[i, j] = len(ratios)
    return groups, values, counts


def render_ablation(records: list[dict], out: Path) -> None:
    panels = (("setup", "setup_s"), ("solve", "solve_s"),
              ("total", "total_s"), ("iterations", "iters"))
    prepared = [(name, *ablation_ratios(records, metric)) for name, metric in panels]
    finite = np.concatenate([values[np.isfinite(values)] for _, _, values, _ in prepared])
    norm = mcolors.TwoSlopeNorm(vmin=min(0.75, float(finite.min())), vcenter=1.0,
                                vmax=max(1.25, float(finite.max())))
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad("0.86")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5), constrained_layout=True)
    image = None
    for ax, (name, groups, values, counts) in zip(axes.flat, prepared):
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(groups)), ["SuiteSparse" if g == "suitesparse" else g.upper()
                                              if g == "ipm" else g.title() for g in groups])
        ax.set_yticks(range(len(CONFIG_ORDER)), [CONFIG_LABEL[c] for c in CONFIG_ORDER], fontsize=8)
        ax.set_title(f"{name}: geometric-mean ratio to default", fontsize=10)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.2f}×\nn={counts[i, j]}",
                            ha="center", va="center", fontsize=7)
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.78)
    colorbar.set_label("ratio to bg / vec_pool_aos on the same matrices (lower is better)")
    fig.suptitle("apxchol CPU configuration ablation on Daint, T=72\n"
                 "paired by matrix; Orkut omits fwd_star/bstr by the declared sweep gate",
                 fontsize=12)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def paired_device_rows(records: list[dict]):
    index = _index(records)
    matrices = sorted({r["cell"]["matrix_id"] for r in records})
    pairs = (
        ("apxchol/bg", "apxchol_v1", DEFAULT_CONFIG, "apxchol_v1"),
        ("AMGCL", "amgcl", "", "amgcl_cuda"),
        ("BoomerAMG", "hypre_boomeramg", "", "hypre_boomeramg_gpu"),
        ("BoomerAMG/cut", "hypre_boomeramg", "cut", "hypre_boomeramg_gpu"),
    )
    rows = []
    for label, cpu_solver, config, gpu_solver in pairs:
        paired = []
        for matrix in matrices:
            c = index.get((matrix, "cpu", cpu_solver, config))
            g = index.get((matrix, "gpu", gpu_solver, config))
            if c and g and c["status"] == g["status"] == "complete":
                paired.append((c, g))
        rows.append((label, len(paired), *(
            geomean(c["metrics"][metric] / g["metrics"][metric] for c, g in paired)
            for metric in ("setup_s", "solve_s", "total_s")
        )))
    return rows


def competitor_rows(records: list[dict], device: str):
    index = _index(records)
    matrices = sorted({r["cell"]["matrix_id"] for r in records})
    comparisons = ((
        ("AMGCL", "amgcl", ""),
        ("BoomerAMG", "hypre_boomeramg", ""),
        ("BoomerAMG/cut", "hypre_boomeramg", "cut"),
        ("RCHOL (portable PCG)", "rchol", ""),
        ("pRCHOL (portable PCG)", "rchol_par", ""),
        ("AC", "ac", ""),
        ("AC2", "ac2", ""),
    ) if device == "cpu" else (
        ("AMGCL", "amgcl_cuda", ""),
        ("BoomerAMG", "hypre_boomeramg_gpu", ""),
        ("BoomerAMG/cut", "hypre_boomeramg_gpu", "cut"),
        ("ParAC Graph", "parac_graph", ""),
        ("ParAC Physics", "parac_physics", ""),
    ))
    rows = []
    for label, solver, config in comparisons:
        ratios = []
        for matrix in matrices:
            apx = index.get((matrix, device, "apxchol_v1", DEFAULT_CONFIG))
            competitor = index.get((matrix, device, solver, config))
            if apx and competitor and apx["status"] == competitor["status"] == "complete":
                ratios.append(competitor["metrics"]["total_s"] / apx["metrics"]["total_s"])
        rows.append((label, len(ratios), geomean(ratios), sum(ratio > 1 for ratio in ratios),
                     sum(ratio < 1 for ratio in ratios)))
    return rows


def write_summary(path: Path, records: list[dict]) -> None:
    status = Counter(record["status"] for record in records)
    status_order = ("complete", "not_converged", "timeout", "failed", "oom", "n/a")
    status_text = ", ".join(f"**{status[name]} {name.replace('_', ' ')}**"
                            for name in status_order if status[name])
    max_residual = max(record["metrics"]["rel_res"] for record in records
                       if record["status"] == "complete")
    lines = [
        "# Daint T=72 fair-solver campaign",
        "",
        f"The combined campaign contains **{len(records)}/{EXPECTED_CELLS} planned cells** over "
        f"{EXPECTED_MATRICES} matrices: {status_text}. "
        f"Every completed cell has true relative residual at most `{max_residual:.7g}`.",
        "",
        "Times are medians of three full setup+solve repetitions at T=72. The CPU/GPU "
        "ratio below is CPU time divided by GPU time, so values above one favor the GPU.",
        "",
        "## CPU/GPU crossover on paired completed cells",
        "",
        "| solver | pairs | setup CPU/GPU | solve CPU/GPU | total CPU/GPU |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, count, setup, solve, total in paired_device_rows(records):
        lines.append(f"| {label} | {count} | {setup:.3f}× | {solve:.3f}× | {total:.3f}× |")
    lines += [
        "",
        "The GPU frequently accelerates the iterative solve but pays a larger setup "
        "interval (device preparation, factor upload, and GPU triangular-solve setup). "
        "For the apxchol default it wins total time on the largest 2D grid and the two "
        "largest social graphs, while the 72-core CPU path remains faster on most smaller inputs.",
        "",
        "## Headline total-time comparison",
        "",
        "Ratios are competitor/apxchol-default on paired completed cells; above one "
        "favors apxchol. Timeouts and failures are excluded from the geometric mean, "
        "but remain visible in the heatmaps and outcome table.",
        "",
        "| device | competitor | pairs | competitor/apxchol | apxchol wins | competitor wins |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for device in ("cpu", "gpu"):
        for label, count, ratio, apx_wins, competitor_wins in competitor_rows(records, device):
            lines.append(f"| {device.upper()} | {label} | {count} | {ratio:.3f}× | "
                         f"{apx_wins} | {competitor_wins} |")
    lines += [
        "",
        "## Non-complete outcomes",
        "",
        "| matrix | device | solver/config | status | evidence |",
        "|---|---|---|---|---|",
    ]
    for record in sorted((r for r in records if r["status"] not in ("complete", "n/a")),
                         key=lambda r: (r["cell"]["matrix_id"], r["cell"]["device"],
                                        r["cell"]["solver"], r["cell"].get("config", ""))):
        cell = record["cell"]
        label = cell["solver"] + (f"/{cell['config']}" if cell.get("config") else "")
        evidence = record.get("diagnostic") or "recorded process failure"
        lines.append(f"| {cell['matrix_id']} | {cell['device'].upper()} | `{label}` | "
                     f"{record['status']} | {evidence} |")
    lines += [
        "",
        "## Coverage boundary",
        "",
        "Daint's ARM64 environment supports the 13 CPU apxchol configurations, AMGCL, "
        "BoomerAMG default/cut, portable-PCG RCHOL/pRCHOL, AC/AC2 under native ARM64 "
        "Julia, three GPU apxchol configurations, AMGCL-CUDA, genuine Hypre-CUDA "
        "default/cut, and ParAC's CUDA graph/physics drivers. The existing main sweep "
        "intentionally omits six fwd_star/bstr Orkut ablations. ParAC-CPU remains "
        "excluded because its upstream driver requires oneMKL; CMG remains excluded "
        "because native MATLAB for Linux is x86-64 and Octave timing is not the same series.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path,
                        help="downloaded campaign results tree; also refreshes --csv")
    parser.add_argument("--csv", type=Path, default=HERE / "fair_t72.csv")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    parser.add_argument("--summary", type=Path, default=HERE / "fair_t72_summary.md")
    args = parser.parse_args()

    if args.cells:
        records = load_cells(args.cells)
        validate(records)
        write_csv(args.csv, records)
        records = read_csv(args.csv)  # prove the committed extract is sufficient
    else:
        records = read_csv(args.csv)
    validate(records)
    args.out.mkdir(parents=True, exist_ok=True)
    render_headline(records, args.out)
    render_apx_device_crossover(records, args.out / "fair_t72_apxchol_cpu_gpu.png")
    render_ablation(records, args.out / "fair_t72_apxchol_ablation.png")
    write_summary(args.summary, records)
    print(f"fair_t72: {len(records)} cells -> {args.csv}, {args.summary}, {args.out}")


if __name__ == "__main__":
    main()

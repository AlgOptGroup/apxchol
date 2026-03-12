#!/usr/bin/env python3
"""benchmarks/plot.py – Generate benchmark charts from CSV results.

Usage:
    python3 benchmarks/plot.py benchmarks/results/latest.csv
    python3 benchmarks/plot.py benchmarks/results/bench_*.csv  # merge multiple

Outputs PDF+PNG charts into results/plots/.
Following SDDM2023 methodology: separate scaling charts per graph family.
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

SOLVER_COLORS = {
    "ApxChol+PCG [Kyng16]":     "#1f77b4",
    "CG [Eigen]":               "#aec7e8",
    "LDLT [Eigen]":             "#2ca02c",
    "RCHOL+PCG [Chen20]":       "#d62728",
    "RCHOL+MKL [Chen20]":       "#e41a1c",
    "pRCHOL+PCG [Chen20;par]":  "#ff6961",
    "CHOLMOD [SuiteSparse]":    "#9467bd",
    "GPU-RCHOL+PCG [Liang25]":  "#ff69b4",
    "AC [Kyng16;Jl]":           "#e377c2",
    "AC2 [Kyng16;Jl]":          "#bcbd22",
    "CG [Julia]":               "#17becf",
    "Chol [Julia]":             "#7f7f7f",
}

# Solvers excluded from ALL charts
EXCLUDE_SOLVERS = {"AMG+CG [AMGCL]", "CG+ICC [Eigen]", "pRCHOL+PCG [Chen20;par]"}

# Solvers excluded from bar comparison charts (extreme outliers)
BAR_EXCLUDE_SOLVERS = {"GPU-RCHOL+PCG [Liang25]"}

# Solver-specific maxiter thresholds — hitting maxiter = not converged
# (CG maxiter=500, GPU maxiter=300; others always converge before limit)
SOLVER_MAXITER = {
    "CG [Eigen]": 500,
    "GPU-RCHOL+PCG [Liang25]": 300,
}

# Families where CG scaling is misleading — exclude from scaling charts
# (maxiter hits, or erratic per-nnz cost from cache effects + growing iter count)
CG_BAD_FAMILIES = {"chimera", "wchimera", "aniso", "wgrid", "uniform_grid"}

# Solvers excluded from SDDM parameter sweep charts (too many maxiter hits)
SDDM_PARAM_EXCLUDE = {"CG [Eigen]", "GPU-RCHOL+PCG [Liang25]"}

def _hit_maxiter(row):
    """Return True if this row hit its solver's iteration limit."""
    limit = SOLVER_MAXITER.get(row["solver"])
    return limit is not None and row["iters"] >= limit


def solver_color(name):
    return SOLVER_COLORS.get(name, "#333333")


def save_fig(fig, outdir, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.{{pdf,png}}")


def load_csv(paths):
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"Warning: skipping {p}: {e}", file=sys.stderr)
    if not frames:
        print("No valid CSV files.", file=sys.stderr)
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    # Normalize graph names: strip .mtx suffix for consistent matching
    df["graph"] = df["graph"].str.replace(r"\.mtx$", "", regex=True)
    df = df[~df["solver"].isin(EXCLUDE_SOLVERS)]
    df = df[~df["solver"].str.contains("FAIL|warning|file:|stopped for rho", na=False, regex=True)]
    return df


def _grid(df):
    """Filter to uniform grid graphs."""
    return df[df["graph"].str.match(r"grid_\d+$", na=False)].copy()


def _checker_k100000_t4(df):
    """Filter to checkerboard κ=100000, tile=4."""
    return df[df["graph"].str.match(r"checker_\d+_k100000_t4$", na=False)].copy()


def _dedup_by_graph(df):
    """Keep only one row per (solver, graph) — take the median total_s run."""
    if df.empty:
        return df
    return df.loc[df.groupby(["solver", "graph"])["total_s"].idxmin()]





# ══════════════════════════════════════════════════════
# Iterations vs kappa — preconditioned iterative only
# ══════════════════════════════════════════════════════

def plot_iterations(df, outdir):
    # Only tile=4 checkerboard to avoid duplicate kappa points from varying-tile runs
    checker = df[df["graph"].str.match(r"checker_\d+_k\d+_t4$", na=False)].copy()
    if checker.empty:
        return

    largest_n = checker["n"].max()
    sub = checker[checker["n"] == largest_n].copy()
    sub["kappa"] = sub["graph"].str.extract(r"k(\d+)").astype(float)
    sub = sub.dropna(subset=["kappa"])

    # Remove direct solvers and unpreconditioned CG
    drop = {"LDLT [Eigen]", "CHOLMOD [SuiteSparse]",
            "CG [Eigen]", "Chol [Julia]", "CG [Julia]"}
    sub = sub[(sub["iters"] > 1) & ~sub["solver"].isin(drop)]
    # Deduplicate: one point per (solver, kappa)
    sub = sub.loc[sub.groupby(["solver", "kappa"])["total_s"].idxmin()]
    if sub.empty:
        return

    fig, ax = plt.subplots()
    for solver, grp in sub.groupby("solver"):
        grp = grp.sort_values("kappa")
        ax.plot(grp["kappa"], grp["iters"], "o-", label=solver,
                color=solver_color(solver), markersize=5)
    ax.set_xlabel("Condition parameter \u03ba")
    ax.set_ylabel("PCG iterations")
    ax.set_title(f"Preconditioned Iterations vs \u03ba (n={int(largest_n)})")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    save_fig(fig, outdir, "iterations_vs_kappa")


# ══════════════════════════════════════════════════════
# Bar comparison — setup + solve at largest n
# ══════════════════════════════════════════════════════

def _plot_bar_ax(ax, sub, title):
    """Draw a horizontal setup+solve bar chart on the given axes."""
    sub = sub.sort_values("total_s")
    if sub.empty:
        ax.set_visible(False)
        return
    y = range(len(sub))
    setup_vals = sub["setup_s"].values
    solve_vals = sub["solve_s"].values
    total_vals = sub["total_s"].values
    colors = [solver_color(s) for s in sub["solver"]]

    ax.barh(list(y), setup_vals, color="#999999", alpha=0.7)
    ax.barh(list(y), solve_vals, left=setup_vals, color=colors, alpha=1.0)
    for i, (s, sv, t) in enumerate(zip(setup_vals, solve_vals, total_vals)):
        pct = s / t * 100 if t > 0 else 0
        if pct > 2:
            ax.text(s / 2, i, f"{pct:.0f}%", va="center", ha="center",
                    fontsize=7, color="white", fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(sub["solver"], fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title(title, fontsize=10)


def plot_bar_comparison(df, outdir):
    sub = _grid(df)
    if sub.empty:
        return
    sub = sub[~sub["solver"].isin(BAR_EXCLUDE_SOLVERS)]
    largest_n = sub["n"].max()
    sub = sub[sub["n"] == largest_n]
    sub = sub.loc[sub.groupby("solver")["total_s"].idxmin()]

    fig, ax = plt.subplots(figsize=(11, max(4, len(sub) * 0.45)))
    _plot_bar_ax(ax, sub,
                 f"Setup + Solve Breakdown \u2014 grid n={int(largest_n)}")
    ax.legend(["Setup", "Solve"], fontsize=9)
    save_fig(fig, outdir, "bar_comparison")


# ══════════════════════════════════════════════════════
# Residual accuracy
# ══════════════════════════════════════════════════════

def plot_residual(df, outdir):
    sub = _grid(df)
    if sub.empty:
        return
    largest_n = sub["n"].max()
    sub = sub[sub["n"] == largest_n]
    sub = sub.loc[sub.groupby("solver")["total_s"].idxmin()]
    sub = sub.sort_values("rel_res")

    fig, ax = plt.subplots(figsize=(10, max(4, len(sub) * 0.4)))
    colors = [solver_color(s) for s in sub["solver"]]
    ax.barh(sub["solver"], sub["rel_res"], color=colors)
    ax.set_xlabel("Relative residual ||Ax\u2212b||/||b||")
    ax.set_title(f"Solution Accuracy \u2014 grid n={int(largest_n)}")
    ax.set_xscale("log")
    ax.axvline(1e-8, color="red", ls="--", alpha=0.5, label="tol=1e\u207b\u2078")
    ax.legend(fontsize=9)
    save_fig(fig, outdir, "residual")


# ══════════════════════════════════════════════════════
# Efficiency — µs/nnz on checkerboard κ=1000 with IQR
# ══════════════════════════════════════════════════════

def plot_efficiency(df, outdir):
    sub = _grid(df)
    # Exclude solvers that hit maxiter (non-convergence)
    sub = sub[~sub.apply(_hit_maxiter, axis=1)].copy()
    if sub.empty:
        return

    # Aggregate: median and IQR per (solver, nnz)
    agg = sub.groupby(["solver", "nnz"]).agg(
        med=("us_per_nnz", "median"),
        q25=("us_per_nnz", lambda x: x.quantile(0.25)),
        q75=("us_per_nnz", lambda x: x.quantile(0.75)),
    ).reset_index()

    fig, ax = plt.subplots()
    for solver, grp in agg.groupby("solver"):
        grp = grp.sort_values("nnz")
        ax.plot(grp["nnz"], grp["med"], "o-", label=solver,
                color=solver_color(solver), markersize=4, alpha=0.9)
        # IQR shaded band
        ax.fill_between(grp["nnz"], grp["q25"], grp["q75"],
                         color=solver_color(solver), alpha=0.15)
    ax.set_xlabel("nnz")
    ax.set_ylabel("\u00b5s / nnz (median)")
    ax.set_title("Efficiency \u2014 grid (median \u00b1 IQR)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, outdir, "efficiency")





# ══════════════════════════════════════════════════════
# GPU RCHOL Paper — SuiteSparse matrix comparison
# ══════════════════════════════════════════════════════

GPU_PAPER_MATRICES = {"parabolic_fem", "ecology1", "ecology2", "apache2", "G3_circuit"}

# Known GPU failures — exclude from charts (crashes or degenerate results)
GPU_FAILURES = {
    ("GPU-RCHOL+PCG [Liang25]", "apache2"),    # CUDA illegal memory access (linear elimination tree)
    ("GPU-RCHOL+PCG [Liang25]", "G3_circuit"),  # tree depth ~1.46M → extremely slow / garbage factor
}


def plot_gpu_paper(df, outdir):
    """Per-matrix horizontal bar charts for GPU RCHOL paper matrices."""
    gpu_df = df[df["graph"].isin(GPU_PAPER_MATRICES)].copy()
    if gpu_df.empty:
        return
    # Filter known GPU failures
    gpu_df = gpu_df[~gpu_df.apply(
        lambda r: (r["solver"], r["graph"]) in GPU_FAILURES, axis=1)]
    gpu_df = _dedup_by_graph(gpu_df)

    matrices = sorted(gpu_df["graph"].unique())
    for mat in matrices:
        sub = gpu_df[gpu_df["graph"] == mat]
        sub = sub.loc[sub.groupby("solver")["total_s"].idxmin()]
        fig, ax = plt.subplots(figsize=(11, max(4, len(sub) * 0.45)))
        _plot_bar_ax(ax, sub, f"Setup + Solve \u2014 {mat}")
        ax.legend(["Setup", "Solve"], fontsize=9)
        save_fig(fig, outdir, f"gpu_bar_{mat}")

    # Combined: one panel per matrix
    ncols = min(len(matrices), 3)
    nrows = (len(matrices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, mat in enumerate(matrices):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sub = gpu_df[gpu_df["graph"] == mat]
        sub = sub.loc[sub.groupby("solver")["total_s"].idxmin()]
        _plot_bar_ax(ax, sub, mat)

    for idx in range(len(matrices), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#999999", alpha=0.7, label="Setup"),
                       Patch(facecolor="#4488cc", alpha=1.0, label="Solve")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("GPU RCHOL Paper Matrices \u2014 Setup + Solve Comparison",
                 fontsize=14, y=1.01)
    save_fig(fig, outdir, "combined_gpu_paper")


# ══════════════════════════════════════════════════════
# SDDM2023 — plots for generated benchmark instances
# ══════════════════════════════════════════════════════

def _is_sddm(graph):
    """Return True if graph name is from SDDM2023 generator."""
    prefixes = ("uniform_grid_", "checkered_", "aniso_", "wgrid_",
                "chimera_", "wchimera_", "star_")
    return any(graph.startswith(p) for p in prefixes)


def _sddm_family(graph):
    """Classify SDDM2023 graph name into family."""
    for prefix in ("uniform_grid", "checkered", "aniso", "wgrid",
                   "wchimera", "chimera", "star"):
        if graph.startswith(prefix + "_"):
            return prefix
    return "other"


def _prepare_sddm(df):
    """Common SDDM2023 data preparation."""
    sddm = df[df["graph"].apply(_is_sddm)].copy()
    if sddm.empty:
        return sddm
    sddm["graph_base"] = sddm["graph"]
    sddm["family"] = sddm["graph_base"].apply(_sddm_family)
    sddm = _dedup_by_graph(sddm)
    sddm["tve"] = sddm["total_s"] / (sddm["nnz"] / 2).replace(0, np.nan)
    return sddm


def _plot_sddm_scaling_ax(ax, fam_df, xlabel="n"):
    """Plot time-per-edge scaling on a given axes.
    For chimera/wchimera uses nnz as x-axis (instances differ wildly in density).
    Excludes CG from families where it mostly doesn't converge.
    Marks maxiter hits with × markers."""
    if fam_df.empty:
        return
    family = fam_df["family"].iloc[0] if "family" in fam_df.columns else ""
    use_nnz = family in ("chimera", "wchimera")
    xcol = "nnz" if use_nnz else "n"
    real_xlabel = "nnz" if use_nnz else xlabel

    # Exclude CG from families where it mostly hits maxiter
    plot_df = fam_df
    if family in CG_BAD_FAMILIES:
        plot_df = fam_df[fam_df["solver"] != "CG [Eigen]"]

    has_nonconv = False
    for solver, grp in plot_df.groupby("solver"):
        grp = grp.sort_values(xcol)
        color = solver_color(solver)
        ax.plot(grp[xcol], grp["tve"], "o-", label=solver,
                color=color, markersize=4, linewidth=1.5)
        nconv = grp[grp.apply(_hit_maxiter, axis=1)]
        if not nconv.empty:
            has_nonconv = True
            ax.scatter(nconv[xcol], nconv["tve"], marker="x", c=color,
                       s=60, zorder=5, linewidths=2)
    if has_nonconv:
        ax.scatter([], [], marker="x", c="black", s=60, linewidths=2,
                   label="\u00d7 = hit maxiter")
    ax.set_xlabel(real_xlabel)
    ax.set_ylabel("Time per edge (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")


def _plot_sddm_params_ax(ax, fam_df, param_col, xlabel="param"):
    """Plot time-per-edge vs parameter on a given axes.
    Uses only the largest size (nnz2M) for cleaner charts.
    Excludes CG and GPU (too many maxiter hits on param sweeps)."""
    if fam_df.empty:
        return
    # Filter to largest size only and exclude noisy solvers
    fam_df = fam_df.copy()
    fam_df["size"] = fam_df["graph_base"].str.extract(r"(nnz\w+?)_")[0]
    sizes = sorted(fam_df["size"].dropna().unique())
    if sizes:
        fam_df = fam_df[fam_df["size"] == sizes[-1]]  # largest size
    fam_df = fam_df[~fam_df["solver"].isin(SDDM_PARAM_EXCLUDE)]
    if fam_df.empty:
        return

    for solver, grp in fam_df.groupby("solver"):
        grp = grp.sort_values(param_col)
        color = solver_color(solver)
        ax.plot(grp[param_col], grp["tve"], "o-", label=solver,
                color=color, markersize=4, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time per edge (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_sddm_combined(df, outdir):
    """Combined SDDM2023 dashboard: 2×4 subplots — all families + profile."""
    sddm = _prepare_sddm(df)
    if sddm.empty:
        return

    fig, axes = plt.subplots(2, 4, figsize=(26, 12))

    # ── Row 1: Scaling families (x=n, y=time-per-edge) ──
    scaling = [
        ("uniform_grid", "Uniform Grid (Poisson)"),
        ("chimera", "Chimera (unweighted)"),
        ("wchimera", "Chimera (weighted)"),
        ("star", "Sachdeva Star"),
    ]
    for col, (family, title) in enumerate(scaling):
        ax = axes[0, col]
        fam_df = sddm[sddm["family"] == family]
        _plot_sddm_scaling_ax(ax, fam_df, xlabel="n")
        ax.set_title(title, fontsize=11)

    # ── Row 2: Parameter sweeps + performance profile ──
    param_configs = [
        ("checkered", r"_b(\d+)_", "block count", "Checkered Grid (blocks, nnz\u22482M)"),
        ("aniso", r"_s([\d.eE+-]+)$", "stretch", "Anisotropic Grid (stretch, nnz\u22482M)"),
        ("wgrid", r"_w([\d.eE+-]+)$", "weight", "Weighted Grid (weight, nnz\u22482M)"),
    ]
    for col, (family, extract_re, xlabel, title) in enumerate(param_configs):
        ax = axes[1, col]
        fam_df = sddm[sddm["family"] == family].copy()
        if fam_df.empty:
            ax.set_visible(False)
            continue
        fam_df["param"] = fam_df["graph_base"].str.extract(extract_re).astype(float)
        fam_df = fam_df.dropna(subset=["param"])
        _plot_sddm_params_ax(ax, fam_df, "param", xlabel=xlabel)
        ax.set_title(title, fontsize=11)

    # ── Bottom-right: Performance profile ──
    ax = axes[1, 3]
    best_per_graph = sddm.groupby("graph")["tve"].min()
    merged = sddm.merge(best_per_graph.rename("best_tve"), on="graph")
    merged["ratio"] = merged["tve"] / merged["best_tve"]
    alphas = np.logspace(0, 2, 300)
    for solver, grp in merged.groupby("solver"):
        profile = [(grp["ratio"] <= a).mean() for a in alphas]
        ax.plot(alphas, profile, label=solver,
                color=solver_color(solver), linewidth=1.5)
    ax.set_xlabel("Performance ratio \u03b1 (\u00d7 best)")
    ax.set_ylabel("Fraction of instances \u2264 \u03b1\u00d7best")
    ax.set_title("Performance Profile", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.axvline(1, color="gray", ls=":", alpha=0.5)

    # ── Shared legend at bottom ──
    handles, labels = [], []
    for a in axes.flat:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("SDDM2023 Benchmark Suite \u2014 Time per Edge", fontsize=15, y=1.01)
    save_fig(fig, outdir, "sddm_combined")


def plot_sddm_scaling(df, outdir):
    """Individual scaling plots for SDDM2023 families with varying n."""
    sddm = _prepare_sddm(df)
    if sddm.empty:
        return

    scaling_families = {
        "uniform_grid": "Uniform Grid (Poisson)",
        "chimera":      "Chimera (unweighted)",
        "wchimera":     "Chimera (weighted)",
        "star":         "Sachdeva Star",
    }
    for family, title in scaling_families.items():
        fam_df = sddm[sddm["family"] == family]
        if fam_df["graph"].nunique() < 2:
            continue
        fig, ax = plt.subplots()
        _plot_sddm_scaling_ax(ax, fam_df, xlabel="n")
        ax.set_title(f"SDDM2023 \u2014 {title}")
        ax.legend(fontsize=8, ncol=2)
        save_fig(fig, outdir, f"sddm_{family}")


def plot_sddm_params(df, outdir):
    """Parameter sweep plots for SDDM2023 families (fixed size, varying params)."""
    sddm = _prepare_sddm(df)
    if sddm.empty:
        return

    param_configs = [
        ("aniso", "stretch", r"_s([\d.eE+-]+)$", "Anisotropic Grid \u2014 stretch (nnz\u22482M)"),
        ("wgrid", "weight", r"_w([\d.eE+-]+)$", "Weighted Grid \u2014 weight (nnz\u22482M)"),
        ("checkered", "blocks", r"_b(\d+)_", "Checkered Grid \u2014 block count (nnz\u22482M)"),
    ]
    for family, param_name, extract_re, title in param_configs:
        fam_df = sddm[sddm["family"] == family].copy()
        if fam_df.empty:
            continue
        fam_df["param"] = fam_df["graph_base"].str.extract(extract_re).astype(float)
        fam_df = fam_df.dropna(subset=["param"])
        if fam_df["param"].nunique() < 2:
            continue

        fig, ax = plt.subplots()
        _plot_sddm_params_ax(ax, fam_df, "param", xlabel=param_name)
        ax.set_title(f"SDDM2023 \u2014 {title}")
        ax.legend(fontsize=8, ncol=2)
        save_fig(fig, outdir, f"sddm_{family}_params")


def plot_sddm_overview(df, outdir):
    """Performance profile across all SDDM2023 instances."""
    sddm = _prepare_sddm(df)
    if sddm.empty:
        return

    best_per_graph = sddm.groupby("graph")["tve"].min()
    merged = sddm.merge(best_per_graph.rename("best_tve"), on="graph")
    merged["ratio"] = merged["tve"] / merged["best_tve"]

    alphas = np.logspace(0, 2, 300)
    fig, ax = plt.subplots(figsize=(10, 6))
    for solver, grp in merged.groupby("solver"):
        profile = [(grp["ratio"] <= a).mean() for a in alphas]
        ax.plot(alphas, profile, label=solver,
                color=solver_color(solver), linewidth=2)
    ax.set_xlabel("Performance ratio \u03b1 (\u00d7 best)")
    ax.set_ylabel("Fraction of instances \u2264 \u03b1\u00d7best")
    ax.set_title("SDDM2023 \u2014 Performance Profile (all instances)")
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.axvline(1, color="gray", ls=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, outdir, "sddm_overview")


# ══════════════════════════════════════════════════════
# SDDM2023 — bar comparison per family (largest instance)
# ══════════════════════════════════════════════════════

def plot_sddm_bar_comparison(df, outdir):
    """Individual bar comparison chart for each SDDM2023 family."""
    sddm = _prepare_sddm(df)
    if sddm.empty:
        return
    sddm = sddm[~sddm["solver"].isin(BAR_EXCLUDE_SOLVERS)]

    for family in sorted(sddm["family"].unique()):
        fam_df = sddm[sddm["family"] == family]
        if fam_df.empty:
            continue
        largest_graph = fam_df.loc[fam_df["nnz"].idxmax(), "graph"]
        sub = fam_df[fam_df["graph"] == largest_graph]
        sub = sub.loc[sub.groupby("solver")["total_s"].idxmin()]

        fig, ax = plt.subplots(figsize=(11, max(4, len(sub) * 0.45)))
        nice = family.replace("_", " ").title()
        _plot_bar_ax(ax, sub, f"Setup + Solve \u2014 {nice} ({largest_graph})")
        ax.legend(["Setup", "Solve"], fontsize=9)
        save_fig(fig, outdir, f"sddm_bar_{family}")


# ══════════════════════════════════════════════════════
# Combined multi-panel charts
# ══════════════════════════════════════════════════════

def plot_combined_scaling(df, outdir):
    """Combined 1\u00d73 panel: checker + grid + erdos scaling with shared legend."""
    # CG excluded from checker/grid (75% maxiter); kept in erdos (0% maxiter)
    no_cg = df[df["solver"] != "CG [Eigen]"]
    configs = [
        (_checker_k100000_t4(no_cg), "Checkerboard (\u03ba=10\u2075)"),
        (no_cg[no_cg["graph"].str.match(r"grid_\d+$", na=False)], "Uniform Grid"),
        (df[df["graph"].str.contains("erdos", na=False)], "Erd\u0151s\u2013R\u00e9nyi"),
    ]
    # Filter out empty panels
    configs = [(sub, title) for sub, title in configs if not _dedup_by_graph(sub).empty]
    if not configs:
        return

    ncols = len(configs)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols + 1, 6), sharey=True)
    if ncols == 1:
        axes = [axes]
    seen = {}
    has_nonconv = False
    for ax, (sub, title) in zip(axes, configs):
        sub = _dedup_by_graph(sub)
        for solver, grp in sub.groupby("solver"):
            grp = grp.sort_values("nnz")
            color = solver_color(solver)
            line, = ax.plot(grp["nnz"], grp["total_s"], "o-",
                            color=color, markersize=5)
            if solver not in seen:
                seen[solver] = line
            nconv = grp[grp.apply(_hit_maxiter, axis=1)]
            if not nconv.empty:
                has_nonconv = True
                ax.scatter(nconv["nnz"], nconv["total_s"], marker="x", c=color,
                           s=60, zorder=5, linewidths=2)
        ax.set_xlabel("nnz")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].set_ylabel("Total time (s)")

    if has_nonconv:
        seen["\u00d7 = hit maxiter"] = ax.scatter([], [], marker="x",
            c="black", s=60, linewidths=2)
    fig.legend(seen.values(), seen.keys(),
               loc="lower center", ncol=min(len(seen), 6), fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Scaling Comparison", fontsize=14, y=1.01)
    save_fig(fig, outdir, "combined_scaling")


def plot_combined_sddm_bar(df, outdir):
    """Combined SDDM2023 bar chart: one panel per family, largest instance."""
    sddm = _prepare_sddm(df)
    if sddm.empty:
        return
    sddm = sddm[~sddm["solver"].isin(BAR_EXCLUDE_SOLVERS)]

    families = sorted(sddm["family"].unique())
    families = [f for f in families if not sddm[sddm["family"] == f].empty]
    if not families:
        return

    nf = len(families)
    ncols = min(nf, 4)
    nrows = (nf + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, family in enumerate(families):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        fam_df = sddm[sddm["family"] == family]
        largest_graph = fam_df.loc[fam_df["nnz"].idxmax(), "graph"]
        sub = fam_df[fam_df["graph"] == largest_graph]
        sub = sub.loc[sub.groupby("solver")["total_s"].idxmin()]
        nice = family.replace("_", " ").title()
        _plot_bar_ax(ax, sub, f"{nice}\n({largest_graph})")

    for idx in range(nf, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#999999", alpha=0.7, label="Setup"),
                       Patch(facecolor="#4488cc", alpha=1.0, label="Solve")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("SDDM2023 \u2014 Setup + Solve Breakdown (largest instance per family)",
                 fontsize=14, y=1.01)
    save_fig(fig, outdir, "combined_sddm_bar")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot.py <csv_file> [csv_file ...]",
              file=sys.stderr)
        sys.exit(1)

    csv_paths = sys.argv[1:]
    df = load_csv(csv_paths)

    base_outdir = os.path.join(os.path.dirname(csv_paths[0]), "plots")
    dirs = {}
    for sub in ("comparison", "gpu", "sddm", "combined"):
        d = os.path.join(base_outdir, sub)
        os.makedirs(d, exist_ok=True)
        dirs[sub] = d

    print(f"Loaded {len(df)} rows, {df['solver'].nunique()} solvers")
    print(f"Solvers: {sorted(df['solver'].unique())}")
    print(f"Generating plots in {base_outdir}/")

    # Comparison / analysis (checker-based)
    plot_bar_comparison(df, dirs["comparison"])
    plot_residual(df, dirs["comparison"])
    plot_efficiency(df, dirs["comparison"])

    # GPU paper matrices
    plot_gpu_paper(df, dirs["gpu"])

    # SDDM2023
    plot_sddm_scaling(df, dirs["sddm"])
    plot_sddm_params(df, dirs["sddm"])
    plot_sddm_overview(df, dirs["sddm"])
    plot_sddm_bar_comparison(df, dirs["sddm"])

    # Combined multi-panel
    plot_sddm_combined(df, dirs["combined"])
    plot_combined_scaling(df, dirs["combined"])
    plot_combined_sddm_bar(df, dirs["combined"])

    print("Done.")


if __name__ == "__main__":
    main()

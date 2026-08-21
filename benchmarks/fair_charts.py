#!/usr/bin/env python3
"""Generate the illustrative benchmark artifacts for benchmarks/latest/ from the
per-cell results (results/cells). Produces, per family (grids / ipm / suitesparse):
  - a grouped bar chart of total solve time per matrix, one bar per solver (t16)
  - a log-log scaling curve (total_s vs nnz), one line per solver
plus a flat results.csv export and a markdown summary table.

Non-converged / failed cells are drawn as hatched 'x' bars so the reader sees
coverage honestly. Run AFTER the fair sweep + ParAC merge:
  PYTHONPATH=benchmarks python3 benchmarks/fair_charts.py --out benchmarks/latest
"""
import argparse, csv, json, glob, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import numpy as np
import gpu_charts as gpu   # reuse the GPU CSV loader + colours/order for overlays
# Matrix axis labels come from the registry, so a matrix we ASSEMBLED an operator
# for (a graph file -> L = D - A) never appears under its bare file name.
from runner_common import (APXCHOL_DEFAULT_CONFIG, mat_labels,
                           require_injective_labels, timeout_cap)

TOL = 1e-8

# canonical solver label + display order + colour
#
# ── THE SERIES RULE (uniform across every solver, CPU and GPU) ───────────────────
#   Every charted series is EXACTLY ONE (solver, configuration). No series is a
#   per-cell minimum over several configurations. Where a figure structurally needs
#   one row per method (the cross-method poster heatmap), it names a configuration
#   DECLARED a priori (APX_DEFAULT below), never the per-cell fastest.
#
# This replaces the old rule, which mapped all four apxchol IS-selectors
# (bg/luby/root/bk) onto one "apxchol" label and let _pick() take the FASTEST of the
# four per matrix -- while BoomerAMG's two configurations (default / cut) stayed two
# separate series that each got no such minimum. That was best-of-4 for us against
# best-of-1 for them: measured over the 27 CPU matrices in the store it handed
# apxchol a geomean 5.6% (max 23.4%) discount versus its own declared default, and
# 9.7% (max 2.33x) on the GPU axis -- and, because the timed-out-competitor cap is
# 10x apxchol's total, it also shrank the bound every capped competitor is drawn at.
# The four selectors keep their own dedicated per-matrix figures (ablation_*,
# selector_*, poster_selectors_*), so nothing is hidden by declaring a default.
LABELS = {
    ("apxchol_v1", "bg+tree[vec_pool]"): "apxchol/bg",
    ("apxchol_v1", "luby+tree[vec_pool]"): "apxchol/luby",
    ("apxchol_v1", "root+tree[vec_pool]"): "apxchol/root",
    ("apxchol_v1", "bk+tree[vec_pool]"): "apxchol/bk",
    ("rchol", ""): "RCHOL",
    ("rchol_par", ""): "pRCHOL",
    ("hypre_boomeramg", ""): "BoomerAMG",
    ("hypre_boomeramg", "cut"): "BoomerAMG/cut",
    ("amgcl", ""): "AMGCL",
    ("cmg", ""): "CMG (MATLAB)†",
    ("parac", ""): "ParAC Graph",
    ("parac_physics", ""): "ParAC Physics",
    ("ac", ""): "AC (Jl ref)†",
    ("ac2", ""): "AC2 (Jl ref)†",
}
# This is a real runtime check, not an `assert` that disappears under `python -O`.
require_injective_labels(LABELS, "CPU chart")

# apxchol's DECLARED default configuration (sweep_fair.py: "The single headline
# apxchol config is bg+tree"). Used wherever a figure needs exactly one apxchol row.
# It is fixed a priori; it is never re-chosen from the timings.
APX_DEFAULT = LABELS[("apxchol_v1", APXCHOL_DEFAULT_CONFIG)]
APX_SERIES = ["apxchol/bg", "apxchol/luby", "apxchol/root", "apxchol/bk"]

# Geometric MG excluded (structured-grid only, not a general-matrix solver).
# AMGCL re-added: a distinct algebraic-MG competitor
# (smoothed-aggregation + spai0), not redundant with BoomerAMG's classical AMG.
# Every apxchol IS-selector is its own series, exactly as BoomerAMG's two
# configurations are -- the reader sees the spread instead of only its minimum.
ORDER = APX_SERIES + [
         "RCHOL", "pRCHOL",
         "BoomerAMG", "BoomerAMG/cut", "AMGCL", "CMG (MATLAB)†", "ParAC Graph", "ParAC Physics",
         "AC (Jl ref)†", "AC2 (Jl ref)†"]
COLORS = {"apxchol/bg": "#0b5394",     # declared default = the darkest blue
          "apxchol/luby": "#3d7ebf",   # the other three selectors: lighter blues,
          "apxchol/root": "#6fa8dc",   # same family, same treatment as
          "apxchol/bk": "#a4c2f4",     # BoomerAMG / BoomerAMG-cut's two greens
          "RCHOL": "#d62728", "pRCHOL": "#ff9896",
          "BoomerAMG": "#2ca02c", "BoomerAMG/cut": "#74c476",  # cut = lighter green
          "AMGCL": "#8c564b",   # brown (matches thread_scaling)
          "CMG (MATLAB)†": "#e377c2", "ParAC Graph": "#ff8c00", "ParAC Physics": "#e6550d",
          "AC (Jl ref)†": "#7f7f7f", "AC2 (Jl ref)†": "#bcbd22"}

# CMG runs as the canonical MATLAB CMG (Koutis' cmg-solver, MEX recompiled, in the
# matlab-deps container). Its
# wall-clock isn't cross-language-comparable to the C++ solvers (MATLAB-pcg vector ops),
# so it stays out of the speed bar/scaling charts; its ITERATION COUNT is the comparable
# signal and IS shown (iters chart + summary).
# AC/AC2 (the Julia reference of OUR method) ARE shown -- they're only ~2-3x slower
# than apxchol, so the comparison "our C++ vs the Julia reference" is meaningful.
CHART_EXCLUDE = {"CMG (MATLAB)†"}

def label(cell):
    return LABELS.get((cell["solver"], cell.get("config", "")))

def load(root):
    # CPU charts read ONLY device=cpu cells. The store now also holds device=gpu
    # cells (same solver/config labels), so without this filter _pick() could
    # return a GPU cell for a CPU series — which made apxchol's CPU bar equal its
    # GPU bar and dropped it from the scaling chart (GPU cells carry no nnz).
    recs = []
    for p in glob.glob(f"{root}/**/*.json", recursive=True):
        try:
            c = json.load(open(p))
            if c.get("cell", {}).get("device", "cpu") != "gpu":
                recs.append(c)
        except: pass
    return recs

def load_gpu_cfg(root):
    """device=gpu apxchol_v1 cells grouped by (family, config) -> {matrix: metrics}.
    Unlike gpu_charts.load this does NOT collapse the four IS-selectors onto one
    label, so the ablation can show the GPU number per individual config. Keeping it
    per-matrix lets the ablation median GPU over the SAME matrix set as CPU. (Only the
    [vec_pool] selectors are swept on the GPU axis, so [vec] configs get no GPU bar.)"""
    rows = defaultdict(dict)
    if not os.path.isdir(root):
        return rows
    for p in glob.glob(f"{root}/**/*__gpu.json", recursive=True):
        try:
            c = json.load(open(p))
        except Exception:
            continue
        cell = c["cell"]
        if (cell.get("device") != "gpu" or c.get("status") != "complete"
                or cell.get("solver") != "apxchol_v1"):
            continue
        rows[(cell["family"], cell.get("config", ""))][cell["matrix_id"]] = c.get("metrics", {})
    return rows

def matrix_order(recs, fam):
    mats = {r["cell"]["matrix_id"] for r in recs if r["cell"]["family"] == fam}
    def key(m):
        for r in recs:
            if r["cell"]["matrix_id"] == m and r.get("metrics", {}).get("nnz"):
                return r["metrics"]["nnz"]
        return 0
    # Tie-break on the id. Sorting a SET by nnz alone left tied matrices in set-iteration
    # order, which varies with PYTHONHASHSEED -- the four IPM iters all have nnz=7867398,
    # so every ipm chart's column order (and summary.md's row order) was reshuffled on
    # each run. Deterministic now.
    return sorted(mats, key=lambda m: (key(m), m))

def headline_mats(recs, fam):
    """Per-matrix column charts: ordered by nnz ascending (the SAME order GPU +
    combined use, so they line up matrix-for-matrix). For grids show ONLY the
    largest 2D and largest 3D grid tested (the smaller ones are redundant here;
    the full size ladder lives in the scaling chart)."""
    mats = matrix_order(recs, fam)  # ascending by nnz
    if fam != "grids":
        return mats
    twod = [m for m in mats if not m.startswith("grid3d")]
    threed = [m for m in mats if m.startswith("grid3d")]
    # Largest THREE 2D + THREE 3D: the very largest (grid_5000 / grid3d_250) OOM
    # Hypre-GPU, and ParAC only ran up to grid_3000 / grid3d_150 -- so showing the top
    # three of each keeps a Hypre-GPU bar AND a ParAC bar in view.
    return twod[-3:] + threed[-3:]

# SuiteSparse mixes small FEM/planar matrices (a few M nnz, sub-second solves)
# with social/scale-free GIANTS (com-LiveJournal 73M, com-Orkut 237M, ... ; many
# seconds). On a shared linear axis the giants flatten everything else to
# invisible nubs. Split the per-matrix charts into a "small" and a "giants" group
# (by nnz) so each is legible. Tunable; ~20M cleanly separates FEM (<10M) from the
# social graphs (>=24M). Only suitesparse splits; grids/ipm stay single-group.
GIANT_NNZ = 20_000_000
# Explicit social-giant set: the large scale-free graphs grouped as "_giants" regardless
# of nnz. com-Youtube (~6M nnz) is below GIANT_NNZ but is a large social graph (and is
# in the deferred-refresh giant set), so name it here so it joins the giants rather than
# sitting stale among the freshly-refreshed small/FEM matrices.
GIANT_MATS = {"as-Skitter", "coPapersDBLP", "com-LiveJournal", "com-Orkut", "com-Youtube"}
# The two heaviest giants (com-LiveJournal 73M, com-Orkut 237M nnz) dwarf the mid
# giants (com-Youtube/as-Skitter/coPapersDBLP) on a shared bar axis, so chart them in
# their own "_giants_xl" group; the mid giants stay in "_giants".
GIANT_XL_MATS = {"com-LiveJournal", "com-Orkut"}

def _nnz_of(recs, fam, mat):
    for r in recs:
        if r["cell"]["family"] == fam and r["cell"]["matrix_id"] == mat:
            m = r.get("metrics", {})
            if m.get("nnz"): return m["nnz"]
            mm = r.get("matrix_meta", {})
            if mm.get("nnz_estimate"): return mm["nnz_estimate"]
    return 0

def family_groups(recs, fam, mats, split=False):
    """[(suffix, submats)]. split=True is for BAR charts only -- grouped bars get cramped
    and different scales flatten each other, so grids split into _2d / _3d and suitesparse
    into _small / _giants / _giants_xl. split=False (heatmaps: ratio-coloured per column,
    legible across the full ladder) keeps every family a single unsuffixed group.
    Falls back to one group if a split would leave a side empty."""
    if not split:
        return [("", mats)]
    if fam == "grids":
        twod   = [m for m in mats if not m.startswith("grid3d")]
        threed = [m for m in mats if m.startswith("grid3d")]
        if twod and threed:
            return [("_2d", twod), ("_3d", threed)]
        return [("", mats)]
    if fam != "suitesparse":
        return [("", mats)]
    def is_giant(m): return _nnz_of(recs, fam, m) >= GIANT_NNZ or m in GIANT_MATS
    giants = [m for m in mats if is_giant(m)]
    small  = [m for m in mats if not is_giant(m)]
    if not giants or not small:
        return [("", mats)]
    # Split the giants: the two heaviest (com-LiveJournal/com-Orkut) into _giants_xl,
    # the mid giants (com-Youtube/as-Skitter/coPapersDBLP) stay in _giants.
    xl  = [m for m in giants if m in GIANT_XL_MATS]
    mid = [m for m in giants if m not in GIANT_XL_MATS]
    groups = [("_small", small)]
    if mid: groups.append(("_giants", mid))
    if xl:  groups.append(("_giants_xl", xl))
    return groups

def scaling_chart(recs, fam, out, gpu_rows=None, device="cpu"):
    # total-time-vs-nnz LINE only makes sense for a genuine size ladder (grids).
    # Split by device (like the bar charts): device="cpu" plots the CPU solvers,
    # device="gpu" plots the GPU solvers — separate figures, not one overlay.
    if fam != "grids":
        return
    nnz_by_mat = {}
    for r in recs:
        if r["cell"]["family"] == fam and r.get("metrics", {}).get("nnz"):
            nnz_by_mat[r["cell"]["matrix_id"]] = r["metrics"]["nnz"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    drawn = False
    capped_any = False
    if device == "cpu":
        for lab in ORDER:
            if lab in CHART_EXCLUDE: continue
            pts, capped = [], []
            for mat in matrix_order(recs, fam):
                best = _pick(recs, fam, mat, lab)
                if best is None: continue
                nz = nnz_by_mat.get(mat)
                if best["status"] != "complete":
                    # A dropped point used to end the line with no marker, so a
                    # solver that HIT THE WALL-CLOCK CAP looked simply "not
                    # measured". Draw it at its known lower bound instead.
                    lo = timeout_cap(best)
                    if best["status"] == "timeout" and nz and lo:
                        capped.append((nz, lo))
                    continue
                m = best["metrics"]
                if m.get("nnz") and m.get("total_s"):
                    pts.append((m["nnz"], m["total_s"]))
            if pts:
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                        label=lab, color=COLORS.get(lab, "#888")); drawn = True
            if capped:
                capped.sort(); capped_any = True
                ax.scatter([p[0] for p in capped], [p[1] for p in capped], marker="x",
                           s=55, linewidths=1.6, color=COLORS.get(lab, "#888"),
                           zorder=5, label=None if pts else lab)
                drawn = True
    else:  # gpu
        gpu_series = defaultdict(list)
        for (gf, mat), d in (gpu_rows or {}).items():
            if gf != fam: continue
            nz = nnz_by_mat.get(mat)
            if not nz: continue
            for glab, gv in d.items():
                # THE GRADING RULE (benchmarks/README.md): exactly TOL, no grace factor.
                if gv.get("total") and gv.get("rel_res", 1.0) <= TOL:
                    gpu_series[glab].append((nz, gv["total"]))
        for glab in gpu.ORDER:
            if glab not in gpu_series: continue
            pts = sorted(gpu_series[glab])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s",
                    label=glab, color=gpu.COLORS.get(glab, "#888")); drawn = True
    if not drawn:
        plt.close(fig); return
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nnz"); ax.set_ylabel(f"total solve time (s) — t16 ({device.upper()})")
    cap_note = ("\n× = timed out; plotted at the exact persisted wall-clock cap "
                "true time unknown" if capped_any else "")
    ax.set_title(f"{fam} ({device.upper()}): scaling vs nnz (per-solver grounding, tol 1e-8)"
                 + cap_note, fontsize=10.5)
    if capped_any:   # one neutral legend key for the marker, not one per solver
        ax.scatter([], [], marker="x", s=55, linewidths=1.6, color="0.25",
                   label="× = timed out (exact persisted lower bound)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

def convergence_chart(recs, fam, out, mats=None):
    """Final relative residual reached per solver per matrix (log y, 1e-8 line).
    This is the one chart that DOES show non-converged runs -- the whole point is
    to expose who reaches the tolerance and who floors above it."""
    if mats is None: mats = headline_mats(recs, fam)
    data = defaultdict(dict)
    for lab in ORDER:                      # include CMG/AC here: precision IS meaningful
        for mat in mats:
            best = _pick(recs, fam, mat, lab)
            if best is None: continue
            rr = best.get("metrics", {}).get("rel_res")
            if rr and rr > 0: data[lab][mat] = rr
    labs = [l for l in ORDER if l in data]
    if not labs or not mats: return
    maxb = max((sum(1 for l in labs if m in data[l]) for m in mats), default=1)
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(mats)), 5))
    w = 0.8 / max(1, maxb); x = np.arange(len(mats)); legended = set()
    for i, mat in enumerate(mats):
        present = sorted((l for l in labs if mat in data[l]), key=lambda l: data[l][mat])
        for j, lab in enumerate(present):
            ax.bar(x[i] + j * w, data[lab][mat], w, color=COLORS.get(lab, "#888"),
                   label=(lab if lab not in legended else None)); legended.add(lab)
    ax.axhline(1e-8, color="k", ls="--", alpha=0.6, label="tol 1e-8")
    ax.set_yscale("log"); ax.set_ylim(1e-10, 1)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(mat_labels(mats), rotation=30, ha="right")
    ax.set_ylabel("final ‖b−Ax‖/‖b‖ (lower = better)")
    ax.set_title(f"{fam}: solution accuracy (bars above the 1e-8 line did not converge)")
    ax.legend(ncol=3, fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

import re as _re
_ABL_RX = _re.compile(r"^(bg|luby|root|bk)\+tree\[(fwd_star|vec|bstr|vec_pool)\]$")
_ABL_SELS = ["bg", "luby", "root", "bk"]               # heatmap rows (IS selector)
_ABL_STOS = ["fwd_star", "vec", "bstr", "vec_pool"]  # cols (storage progression)

def ablation_heatmap(recs, gpu_cfg, fam, out):
    """apxchol selector x storage ablation as small-multiple heatmaps (total / setup /
    solve / iters), t16. Rows = IS selector (bg/luby/root/bk); cols = storage backend
    in the progression fwd_star -> vec -> bstr -> vec_pool, plus a vec_pool
    GPU column (the GPU axis swept vec_pool only) so the default backend's CPU->GPU shift is
    visible. Each metric is medianed over the matrix set COMMON to all CPU configs
    (fair); colour = value / best-in-grid (green = best), cells annotated with the
    absolute value. Blank (grey) = config not swept / did not converge."""
    from statistics import median
    cpu = defaultdict(dict)   # cpu[(sel,sto)][matrix] = metrics dict
    for r in recs:
        c = r["cell"]
        if (c["family"] != fam or c["solver"] != "apxchol_v1"
                or c.get("threads") != 16 or r["status"] != "complete"):
            continue
        m = _ABL_RX.match(c.get("config", ""))
        if m: cpu[(m.group(1), m.group(2))][c["matrix_id"]] = r["metrics"]
    if not cpu:
        return
    gpu = defaultdict(dict)   # gpu[sel][matrix] = metrics (vec_pool only)
    for (gf, cfg), per_mat in gpu_cfg.items():
        if gf != fam: continue
        m = _ABL_RX.match(cfg)
        if m and m.group(2) == "vec_pool":
            for mat, met in per_mat.items():
                gpu[m.group(1)][mat] = met
    # Median over the matrix set common to all CPU configs (apples-to-apples).
    common = set.intersection(*[set(d) for d in cpu.values()])
    if not common:
        common = set.union(*[set(d) for d in cpu.values()])
    cols = _ABL_STOS + ["vec_pool\n(GPU)"]
    cmap = plt.cm.RdYlGn_r.copy(); cmap.set_bad("0.85")

    def med_of(d, key):
        vals = [d[mt][key] for mt in common if mt in d and d[mt].get(key) is not None]
        return median(vals) if vals else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, key, unit) in zip(
            axes.flat, [("total", "total_s", "s"), ("setup", "setup_s", "s"),
                        ("solve", "solve_s", "s"), ("iters", "iters", "")]):
        M = np.full((len(_ABL_SELS), len(cols)), np.nan)
        for i, sel in enumerate(_ABL_SELS):
            for j, sto in enumerate(_ABL_STOS):
                if (sel, sto) in cpu:
                    M[i, j] = med_of(cpu[(sel, sto)], key)
            if sel in gpu:
                M[i, len(_ABL_STOS)] = med_of(gpu[sel], key)
        # Colour-normalise by the best CPU cell (the 5 CPU storage cols), NOT the
        # global min -- otherwise the much-faster GPU column flattens the whole CPU
        # grid to one colour and the selector x storage signal (the point of the
        # chart) is lost. A GPU cell that beats every CPU config just clamps to the
        # green end; its real value is always printed in the annotation.
        cpu_cells = M[:, :len(_ABL_STOS)]
        cpu_finite = cpu_cells[np.isfinite(cpu_cells)]
        best = cpu_finite.min() if cpu_finite.size else 1.0
        vmax = min(float(cpu_finite.max() / best) if cpu_finite.size else 2.0, 3.0)
        norm = np.ma.masked_invalid(M / best)
        ax.imshow(norm, cmap=cmap, aspect="auto", vmin=1.0, vmax=max(vmax, 1.01))
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7.5)
        ax.set_yticks(range(len(_ABL_SELS)))
        ax.set_yticklabels([f"{s}+tree" for s in _ABL_SELS], fontsize=8.5)
        ax.set_title(f"{name}" + (f" ({unit})" if unit else " (count)"), fontsize=10)
        for i in range(len(_ABL_SELS)):
            for j in range(len(cols)):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}" if unit else f"{int(round(v))}",
                            ha="center", va="center", fontsize=7.5)
    fig.suptitle(f"apxchol  IS-selector x storage  ablation ({fam}, t16, tol 1e-8) — "
                 f"median over {len(common)} matrices · colour = ×best (green=fastest)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

# Cross-family selector x graph heatmap (for the poster): which IS selector wins on
# which GRAPH TYPE, at the default vec_pool storage. Columns span structured grids ->
# FEM/planar -> social/scale-free so the selector spread by graph type is the message
# (bg on structured, luby/root on scale-free). Curated representative set (a poster
# can't show all 26); ordered along the structured->irregular axis.
_SELMAT_COLS = [
    ("grid_3000",        "grid 2D"),
    ("grid3d_150",       "grid 3D"),
    ("parabolic_fem",    "parabolic_fem"),
    ("ecology1",         "ecology1"),
    ("G3_circuit",       "G3_circuit"),
    ("thermal2",         "thermal2"),
    ("iter0040",         "iter0040 (IPM)"),
    ("com-Amazon",       "com-Amazon"),
    ("coAuthorsDBLP",    "coAuthorsDBLP"),
    ("kron_g500-logn16", "kron_g500"),
    ("com-Youtube",      "com-Youtube"),
    ("coPapersDBLP",     "coPapersDBLP"),
    ("as-Skitter",       "as-Skitter"),
    ("com-LiveJournal",  "com-LiveJournal"),
]

def selector_matrix_heatmap(recs, out, metric="total_s", mats=None):
    """apxchol IS-selector x GRAPH heatmap at vec_pool storage, t16. Rows = the four
    IS selectors (bg/luby/root/bk); columns = matrices spanning grid -> FEM -> social.
    Colour normalises PER COLUMN (per graph) so green = the best selector for that
    graph regardless of the graph's absolute scale -- the point is which selector wins
    where, not absolute time. Cells annotated with the absolute value. metric =
    total_s | setup_s | solve_s | iters."""
    cols = mats or _SELMAT_COLS
    sels = _ABL_SELS  # bg luby root bk
    by = defaultdict(dict)   # by[(sel,matrix)] = value
    for r in recs:
        c = r["cell"]
        if (c["solver"] != "apxchol_v1" or c.get("threads") != 16
                or r["status"] != "complete"):
            continue
        m = _ABL_RX.match(c.get("config", ""))
        if m and m.group(2) == "vec_pool":
            v = r["metrics"].get(metric)
            if v is not None:
                by[(m.group(1), c["matrix_id"])] = v
    M = np.full((len(sels), len(cols)), np.nan)
    for j, (mid, _lab) in enumerate(cols):
        for i, sel in enumerate(sels):
            if (sel, mid) in by:
                M[i, j] = by[(sel, mid)]
    # per-column normalise by the best (min) selector in that column.
    norm = np.full_like(M, np.nan)
    for j in range(M.shape[1]):
        col = M[:, j]; fin = col[np.isfinite(col)]
        if fin.size:
            norm[:, j] = col / fin.min()
    is_iters = (metric == "iters")
    unit = "" if is_iters else "s"
    cmap = plt.cm.RdYlGn_r.copy(); cmap.set_bad("0.85")
    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(cols) + 2), 3.4))
    vmax = np.nanmax(norm) if np.isfinite(norm).any() else 1.5
    ax.imshow(np.ma.masked_invalid(norm), cmap=cmap, aspect="auto",
              vmin=1.0, vmax=max(min(float(vmax), 2.0), 1.05))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab for _mid, lab in cols], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(sels)))
    ax.set_yticklabels([f"{s}+tree" for s in sels], fontsize=10)
    for i in range(len(sels)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isfinite(v):
                txt = f"{int(round(v))}" if is_iters else (f"{v:.2f}" if v < 100 else f"{v:.0f}")
                # bold the best (min) selector in each column
                best = np.nanmin(M[:, j])
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        fontweight="bold" if v == best else "normal")
    metric_name = {"total_s": "total time", "setup_s": "setup time",
                   "solve_s": "solve time", "iters": "PCG iterations"}[metric]
    ax.set_title(f"apxchol  IS-selector × graph  —  {metric_name} (vec_pool, t16, tol 1e-8)\n"
                 f"colour = × best selector in each column (green = best);  bold = best per graph",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def selector_family_panel(recs, gpu_cfg, fam, out, device="cpu"):
    """Per-family IS-selector × matrix small-multiple heatmaps (total/setup/solve/iters)
    at vec_pool storage, t16, for ONE family and ONE device. Rows = bg/luby/root/bk;
    columns = this family's matrices (by nnz). Each metric normalises PER COLUMN so
    green = the best selector for that matrix and bold = the per-matrix winner."""
    sels = _ABL_SELS
    by = {}   # by[(sel, matrix)] = metrics dict
    if device == "cpu":
        for r in recs:
            c = r["cell"]
            if (c["family"] != fam or c["solver"] != "apxchol_v1"
                    or c.get("threads") != 16 or r["status"] != "complete"):
                continue
            m = _ABL_RX.match(c.get("config", ""))
            if m and m.group(2) == "vec_pool":
                by[(m.group(1), c["matrix_id"])] = r["metrics"]
    else:
        for (gf, cfg), per_mat in gpu_cfg.items():
            if gf != fam:
                continue
            m = _ABL_RX.match(cfg)
            if m and m.group(2) == "vec_pool":
                for mt, met in per_mat.items():
                    by[(m.group(1), mt)] = met
    mats = [mt for mt in matrix_order(recs, fam) if any((s, mt) in by for s in sels)]
    if not mats:
        return
    metrics = [("total", "total_s", "s"), ("setup", "setup_s", "s"),
               ("solve", "solve_s", "s"), ("iters", "iters", "")]
    cmap = plt.cm.RdYlGn_r.copy(); cmap.set_bad("0.85")
    fig, axes = plt.subplots(2, 2, figsize=(max(11, 0.7 * len(mats) + 3), 7))
    for ax, (name, key, unit) in zip(axes.flat, metrics):
        M = np.full((len(sels), len(mats)), np.nan)
        for i, sel in enumerate(sels):
            for j, mt in enumerate(mats):
                v = by.get((sel, mt), {}).get(key)
                if v is not None:
                    M[i, j] = v
        norm = np.full_like(M, np.nan)
        for j in range(M.shape[1]):
            col = M[:, j]; fin = col[np.isfinite(col)]
            if fin.size:
                norm[:, j] = col / fin.min()
        is_it = (name == "iters")
        vmax = np.nanmax(norm) if np.isfinite(norm).any() else 1.5
        ax.imshow(np.ma.masked_invalid(norm), cmap=cmap, aspect="auto",
                  vmin=1.0, vmax=max(min(float(vmax), 2.0), 1.05))
        ax.set_xticks(range(len(mats))); ax.set_xticklabels(mat_labels(mats), rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(sels))); ax.set_yticklabels([f"{s}+tree" for s in sels], fontsize=8.5)
        ax.set_title(f"{name}" + (f" ({unit})" if unit else " (count)"), fontsize=10)
        for i in range(len(sels)):
            for j in range(len(mats)):
                v = M[i, j]
                if np.isfinite(v):
                    txt = f"{int(round(v))}" if is_it else (f"{v:.2f}" if v < 100 else f"{v:.0f}")
                    best = np.nanmin(M[:, j])
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            fontweight="bold" if v == best else "normal")
    fig.suptitle(f"apxchol  IS-selector × matrix  ablation — {fam} ({device.upper()}, vec_pool, t16, tol 1e-8)\n"
                 f"colour = × best selector per column (green = best);  bold = best per matrix", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ── Poster headline pair ─────────────────────────────────────────────────────────
# TWO CPU-only heatmaps that share the SAME columns so they read side-by-side on the
# poster: (1) the IS-selector question (which eliminator wins per graph) and (2) the
# cross-method question (apxchol at its DECLARED default selector vs the multigrid/
# Cholesky competitors -- figure (1) is where the selector spread is read). One
# curated, representative column set spanning the structured→irregular axis with BOTH
# matrices apxchol loses (structured grids / FEM / IPM) and wins (the social giants).
# IPM is one aggregated column (geomean over the iter ladder). Total time, t16, CPU.
IPM_ITERS = ["iter0010", "iter0020", "iter0030", "iter0040"]
POSTER_COLS = [                       # (col_id, family, display);  "__ipm__" = geomean
    # structured -> irregular. A balanced, popular spread: apxchol LOSES on the
    # structured/FEM/circuit/scale-free left (multigrid territory) and WINS on the
    # social/citation giants on the right where classical-AMG coarsening is expensive.
    ("grid_5000",        "grids",       "grid 2D (25M)"),
    ("grid3d_250",       "grids",       "grid 3D (15.6M)"),
    ("__ipm__",          "ipm",         "IPM (geomean)"),
    ("G3_circuit",       "suitesparse", "G3_circuit"),
    ("thermal2",         "suitesparse", "thermal2"),
    ("kron_g500-logn16", "suitesparse", "kron_g500"),
    ("coPapersDBLP",     "suitesparse", "coPapersDBLP"),
    ("coAuthorsDBLP",    "suitesparse", "coAuthorsDBLP"),
    ("com-Amazon",       "suitesparse", "com-Amazon"),
    ("com-Youtube",      "suitesparse", "com-Youtube"),
    ("as-Skitter",       "suitesparse", "as-Skitter"),
    ("com-LiveJournal",  "suitesparse", "com-LiveJournal"),
]
# The headline carries exactly one declared apxchol configuration; competitor
# configurations remain separate series rather than being collapsed into minima.
POSTER_SOLVERS = [APX_DEFAULT, "BoomerAMG", "BoomerAMG/cut", "AMGCL", "CMG (MATLAB)†",
                  "ParAC Physics", "RCHOL"]
def _poster_cell(recs, row, fam, mid):
    """(total_s | None, status) for one method on one real matrix, CPU t16. row is
    ('sel', selector) for an apxchol IS-selector at vec_pool, or ('lab', label) for a
    competitor. Both resolve through the same declared-thread `_pick` rule."""
    kind, key = row
    if kind == "sel":
        key = f"apxchol/{key}"
    b = _pick(recs, fam, mid, key)
    if b is None:
        return None, "missing", None
    if b["status"] != "complete":
        return None, b["status"], timeout_cap(b)
    return b.get("metrics", {}).get("total_s"), "complete", None

def _poster_value(recs, row, fam, col_id):
    """As _poster_cell, but col_id '__ipm__' returns the geomean of total_s over the IPM
    iter ladder (a single representative IPM column). Partial coverage still aggregates
    what converged (flagged), so a method that times out on one iter isn't dropped."""
    if col_id != "__ipm__":
        return _poster_cell(recs, row, fam, col_id)
    vals, caps = [], []
    for it in IPM_ITERS:
        v, st, cap = _poster_cell(recs, row, "ipm", it)
        if st == "complete" and v is not None:
            vals.append(v)
        if st == "timeout" and cap is not None:
            caps.append(cap)
    if not vals:
        if len(caps) == len(IPM_ITERS):
            return None, "timeout", float(np.exp(np.mean(np.log(caps))))
        return None, "missing", None
    g = float(np.exp(np.mean(np.log(vals))))
    return g, ("complete" if len(vals) == len(IPM_ITERS) else "partial"), None

def _poster_heatmap(recs, rows, out, title):
    """Shared renderer for the poster pair. rows = [(display_label, (kind, key))].
    Per-column normalised by the fastest method in that column (green = best); cells
    annotated with absolute seconds; bold = per-column winner; T = ran-but-timed-out."""
    n, ncol = len(rows), len(POSTER_COLS)
    M = np.full((n, ncol), np.nan)
    CAPS = np.full((n, ncol), np.nan)
    ST = [["missing"] * ncol for _ in range(n)]
    for i, (_lab, spec) in enumerate(rows):
        for j, (cid, fam, _disp) in enumerate(POSTER_COLS):
            v, st, cap = _poster_value(recs, spec, fam, cid)
            ST[i][j] = st
            if v is not None and st in ("complete", "partial"):
                M[i, j] = v
            if cap is not None:
                CAPS[i, j] = cap
    # A timeout is numerical only when its exact cap was persisted in that cell.
    capped = np.zeros((n, ncol), dtype=bool)
    known = np.isfinite(CAPS)
    M[known] = CAPS[known]
    capped[known] = True
    ratio = np.full_like(M, np.nan)
    for j in range(ncol):
        col = M[:, j]
        completed = col[np.isfinite(col) & ~capped[:, j]]
        if completed.size:
            ratio[:, j] = col / completed.min()
    cmap = plt.cm.RdYlGn_r.copy(); cmap.set_bad("0.9")
    # LOG colour scale on the ratio (same as combined_charts.overview_heatmap), NOT a
    # capped linear one: on the social-giant columns the competitors are 100x+ best, so
    # a linear scale washes the whole right half to one indistinguishable red. Log
    # spreads 1x..16x across the ramp so near-ties (e.g. coPapersDBLP) stay readable.
    finite = ratio[np.isfinite(ratio)]
    vmax = min(float(finite.max()), 16.0) if finite.size else 4.0
    norm = mcolors.LogNorm(vmin=1.0, vmax=max(vmax, 1.6))
    fig, ax = plt.subplots(figsize=(max(10.5, 1.25 * ncol + 3.0), 0.66 * n + 2.3))
    im = ax.imshow(np.ma.masked_invalid(ratio), cmap=cmap, aspect="auto", norm=norm)
    ax.set_xticks(range(ncol)); ax.set_xticklabels([d for *_r, d in POSTER_COLS],
                                                   rotation=30, ha="right", fontsize=9.5)
    ax.set_yticks(range(n)); ax.set_yticklabels([l for l, _ in rows], fontsize=10.5)
    for i in range(n):
        for j in range(ncol):
            v = M[i, j]
            if not np.isfinite(v):
                if ST[i][j] == "timeout":
                    ax.text(j, i, "Timeout\ncap unknown", ha="center", va="center",
                            fontsize=7.0, color="#8b0000", fontweight="bold")
                continue
            pre = "≥" if capped[i, j] else ""
            is_best = (not capped[i, j]) and ratio[i, j] == 1.0
            ax.text(j, i, f"{pre}{v:.2f}s\n{pre}{ratio[i, j]:.1f}×", ha="center", va="center",
                    fontsize=7.5, fontweight="bold" if is_best else "normal")
            if capped[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=1.3))
    cb = fig.colorbar(im, ax=ax, ticks=[1, 1.5, 2, 3, 4, 6, 8, 12, 16], fraction=0.022, pad=0.012)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
    cb.set_label("× best solver in column (log scale)")
    ax.set_title(title, fontsize=11.5)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def poster_charts(recs, outdir):
    """The two side-by-side poster heatmaps (CPU, total time, shared columns)."""
    sel_rows = [(f"{s}+tree", ("sel", s)) for s in _ABL_SELS]
    _poster_heatmap(recs, sel_rows, f"{outdir}/poster_selectors_cpu.png",
        "apxchol  IS-selector × graph  —  CPU total time (vec_pool, t16, tol 1e-8)\n"
        "colour = × fastest selector per column (green = best);  bold = best per graph")
    cmp_rows = [(s, ("lab", s)) for s in POSTER_SOLVERS]
    _poster_heatmap(recs, cmp_rows, f"{outdir}/poster_comparison_cpu.png",
        f"apxchol ({APX_DEFAULT.split('/')[1]}+tree, the declared default — NOT the per-column best selector)"
        "  vs multigrid / Cholesky\nCPU total time (t16, tol 1e-8);  one configuration per row for every solver.  "
        "colour = × fastest per column;  bold = best;  ≥ = timed out at the cap")


def export_csv(recs, path):
    rows = []
    for r in recs:
        c = r["cell"]; m = r.get("metrics", {})
        rows.append({"family": c["family"], "matrix": c["matrix_id"],
                     "solver": c["solver"], "config": c.get("config", ""),
                     "threads": c.get("threads"), "status": r["status"],
                     "timeout_cap_s": timeout_cap(r),
                     "setup_s": m.get("setup_s"), "solve_s": m.get("solve_s"),
                     "total_s": m.get("total_s"), "iters": m.get("iters"),
                     "rel_res": m.get("rel_res"), "nnz": m.get("nnz")})
    rows.sort(key=lambda x: (x["family"], x["matrix"], x["solver"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def _pick(recs, fam, mat, lab):
    """The one cell that represents `lab` on this matrix.

    Thread count is selected a priori: t16, with t1 as the historical fallback.
    There is then exactly one cell. Duplicate cells are an error; no timing or
    status is ever used to choose among them."""
    cands = [r for r in recs if r["cell"]["family"] == fam
             and r["cell"]["matrix_id"] == mat and label(r["cell"]) == lab]
    if not cands:
        return None
    t16 = [r for r in cands if r["cell"].get("threads") == 16]
    t1 = [r for r in cands if r["cell"].get("threads") == 1]
    chosen = t16 or t1
    if not chosen and len(cands) == 1:
        chosen = cands
    if len(chosen) != 1:
        details = [(r["cell"].get("threads"), r.get("status")) for r in cands]
        raise RuntimeError(f"ambiguous chart series {fam}/{mat} [{lab}]: {details}")
    return chosen[0]

def summary_md(recs, path):
    fams = sorted({r["cell"]["family"] for r in recs})
    lines = ["# Latest benchmark summary — t16, tol 1e-8, original singular L (per-solver grounding; ParAC per-component-consistent RHS scored vs original L, CMG reg-rel)\n",
             "`† CMG (MATLAB)` = canonical Koutis CMG (MEX, matlab-deps container). MATLAB-pcg wall-time isn't cross-language-comparable, so its **iteration count** is the comparable signal — see below.",
             "Blank = not run; `X` = ran but did not reach 1e-8; `T` = timed out "
             "without a recoverable cap; `T≥seconds` = timed out at the exact cap "
             "persisted by the runner; `—` = solver doesn't support "
             "that de-singularization cell.\n",
             "**Series rule (uniform).** Every column is exactly ONE (solver, configuration); "
             "no column is a per-cell minimum over configurations. apxchol's four IS-selectors "
             f"are four columns (`{'`, `'.join(APX_SERIES)}`), exactly as BoomerAMG's two "
             "configurations are two columns; its declared default is "
             f"`{APX_DEFAULT}`. The chart thread count is selected a priori (t16, with "
             "a t1 fallback); duplicate cells are rejected, so neither status nor time "
             "can select the representative.\n",
             "## Total solve time (s)\n"]
    for fam in fams:
        mats = matrix_order(recs, fam)
        lines.append(f"\n### {fam}\n")
        lines.append("| matrix | " + " | ".join(ORDER) + " |")
        lines.append("|" + "---|" * (len(ORDER) + 1))
        for mat in mats:
            cells = []
            for lab in ORDER:
                best = _pick(recs, fam, mat, lab)
                if best is None: cells.append("")
                elif best["status"] == "complete": cells.append(f"{best['metrics']['total_s']:.2f}")
                elif best["status"] == "timeout":
                    cap = timeout_cap(best)
                    cells.append(f"T≥{cap:g}" if cap is not None else "T")
                elif best["status"] == "n/a": cells.append("—")       # solver doesn't support this de-sing cell
                else: cells.append("X")
            lines.append(f"| {mat} | " + " | ".join(cells) + " |")
    # iteration-count table — the metric on which CMG is fairly comparable
    lines.append("\n## PCG iterations (preconditioner quality, threads-independent)\n")
    for fam in fams:
        mats = matrix_order(recs, fam)
        lines.append(f"\n### {fam}\n")
        lines.append("| matrix | " + " | ".join(ORDER) + " |")
        lines.append("|" + "---|" * (len(ORDER) + 1))
        for mat in mats:
            cells = []
            for lab in ORDER:
                best = _pick(recs, fam, mat, lab)
                if best is None or not best.get("metrics", {}).get("iters"): cells.append("")
                else: cells.append(str(best["metrics"]["iters"]))
            lines.append(f"| {mat} | " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/cells")
    ap.add_argument("--out", default="benchmarks/latest")
    a = ap.parse_args()
    recs = load(a.root)
    # GPU overlay for the grid scaling chart. Default: the SAME per-cell store the
    # gpu_charts / combined_charts read (gpu.load filters device=gpu), so ParAC and
    # every other GPU solver appear. The old default pointed at a stale flat CSV,
    # which silently froze this chart (missing ParAC, pre-patch numbers).
    gpu_rows = gpu.load(a.root)
    gpu_cfg = load_gpu_cfg(a.root)   # per-config GPU cells for the ablation CPU/GPU split
    os.makedirs(f"{a.out}/figures", exist_ok=True)
    for fam in sorted({r["cell"]["family"] for r in recs}):
        # Column charts are the LINEAR setup+solve breakdown (every family, incl IPM);
        # the log-scale view is the scaling line chart. Plus a PCG-iteration chart.
        # Per-matrix charts (breakdown / iters / accuracy) split suitesparse into
        # _small + _giants groups so the social giants don't flatten the FEM
        # matrices; grids/ipm render as one unsuffixed chart (suffix == "").
        # breakdown + accuracy stay GROUPED BARS on the headline matrices (grids reduced
        # to largest 2D+3D; bars get cramped past a handful). The iters chart is now a
        # HEATMAP, so it takes the FULL matrix_order -- every measured grid shows.
        # breakdown + iters now come from combined_charts (device-filtered CPU-only /
        # GPU-only views, one consistent style); fair_charts keeps accuracy/scaling/ablation.
        for suffix, gmats in family_groups(recs, fam, headline_mats(recs, fam), split=True):
            convergence_chart(recs, fam, f"{a.out}/figures/accuracy_{fam}{suffix}.png", gmats)
        scaling_chart(recs, fam, f"{a.out}/figures/scaling_{fam}.png", gpu_rows, "cpu")
        scaling_chart(recs, fam, f"{a.out}/figures/scaling_gpu_{fam}.png", gpu_rows, "gpu")
        # ablation: IS-selector x storage as small-multiple heatmaps (total/setup/
        # solve/iters), with a vec_pool GPU column.
        ablation_heatmap(recs, gpu_cfg, fam, f"{a.out}/figures/ablation_{fam}.png")
        # Per-family IS-selector x matrix panels (total/setup/solve/iters), CPU + GPU.
        for dev in ("cpu", "gpu"):
            selector_family_panel(recs, gpu_cfg, fam,
                                  f"{a.out}/figures/selector_{fam}_{dev}.png", device=dev)
    # Cross-family selector x graph heatmaps (poster): which IS selector wins per
    # GRAPH TYPE at vec_pool. total-time + iters variants.
    selector_matrix_heatmap(recs, f"{a.out}/figures/selector_graph_total.png", metric="total_s")
    selector_matrix_heatmap(recs, f"{a.out}/figures/selector_graph_iters.png", metric="iters")
    # Poster headline pair: IS-selector heatmap + cross-method comparison, shared
    # columns, CPU total time (the two poster figures).
    poster_charts(recs, f"{a.out}/figures")
    export_csv(recs, f"{a.out}/results.csv")
    summary_md(recs, f"{a.out}/summary.md")
    print(f"charts: {len(recs)} cells -> {a.out}")

if __name__ == "__main__":
    main()

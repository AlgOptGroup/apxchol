#!/usr/bin/env python3
"""GPU comparison figures from the per-cell store (results/cells, device=gpu).

Separate from the CPU fair charts (fair_charts.py) on purpose: CPU and GPU times
are apples-to-oranges, so GPU gets its own figure set (breakdown / iters /
accuracy) per family. Every solver here attacks the same matrix to tol 1e-8 (see
the de-singularization protocol in benchmarks/README.md). GPU cells are produced by
`python3 benchmarks/sweep_fair.py --device gpu` (ParAC in-process via
parac_runner.py), the same runner/store as the CPU axis.

  python3 benchmarks/gpu_charts.py --root results/cells --out benchmarks/latest/figures
"""
import argparse, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import numpy as np
import chart_cells
# Matrix axis labels come from the registry, so a matrix we ASSEMBLED an operator
# for (a graph file -> L = D - A) never appears under its bare file name.
from runner_common import (APXCHOL_DEFAULT_CONFIG, mat_labels,
                           require_injective_labels, timeout_cap)

# apxchol (GPU) = the GPU-RESIDENT PCG (cuda_pcg: all PCG vectors stay on device,
# cuBLAS axpy/dot/nrm2 + cuSPARSE SpMV, precond via cuda_sptrsv::solve_LLt_dev). It
# is 2-4.5x faster per iter than the host-PCG hybrid (measured RTX 4090, locked
# clocks), so the hybrid is dropped. apxchol stays blue; competitors match the CPU
# chart colours (BoomerAMG green, AMGCL brown, ParAC orange). Geometric MG is excluded
# (geometric MG is structured-grid-only, not a general-matrix solver).

# ── THE SERIES RULE (same as fair_charts) ────────────────────────────────────────
# Every series is EXACTLY ONE (solver, configuration); no series is a per-cell
# minimum over configurations. Previously all apxchol IS-selectors mapped to a
# single "apxchol (GPU)" label and load() kept the fastest selector per matrix,
# while BoomerAMG's two configurations stayed two separate series -- best-of-4 for
# us, best-of-1 for them. On the 27 GPU matrices in the store that minimum was worth
# a geomean 9.7% (max 2.33x on parabolic_fem) against apxchol's own declared
# default, and it also shrank the 10x-apxchol cap that every timed-out competitor
# bar is clamped to. Headline charts therefore show only the declared bg default;
# selector spread lives in dedicated selector/ablation figures.
APX_SERIES  = ["apxchol/bg (GPU)"]
ORDER  = APX_SERIES + ["ParAC Graph (GPU)", "ParAC Physics (GPU)",
          "BoomerAMG (GPU)", "BoomerAMG/cut (GPU)", "AMGCL (GPU)"]
COLORS = {"apxchol/bg (GPU)": "#0b5394",     # declared default = darkest blue
          "ParAC Graph (GPU)": "#ff8c00", "ParAC Physics (GPU)": "#e6550d",
          "BoomerAMG (GPU)": "#2ca02c", "BoomerAMG/cut (GPU)": "#74c476",
          "AMGCL (GPU)": "#8c564b"}
# (solver, config) -> chart label for GPU cells. MUST stay injective (asserted below).
LABELS = {
    ("apxchol_v1", "bg+tree[vec_pool_aos]"): "apxchol/bg (GPU)",
    ("hypre_boomeramg_gpu", ""): "BoomerAMG (GPU)",
    ("hypre_boomeramg_gpu", "cut"): "BoomerAMG/cut (GPU)",
    ("amgcl_cuda", ""): "AMGCL (GPU)",
    ("parac_graph", ""): "ParAC Graph (GPU)",
    ("parac_physics", ""): "ParAC Physics (GPU)",
}
require_injective_labels(LABELS, "GPU chart")
APX_DEFAULT = LABELS[("apxchol_v1", APXCHOL_DEFAULT_CONFIG)]
FAMS = ["grids", "ipm", "suitesparse"]
# Social-giant set (match fair_charts.GIANT_MATS): grouped as _giants regardless of nnz.
GIANT_MATS = {"as-Skitter", "coPapersDBLP", "com-LiveJournal", "com-Orkut", "com-Youtube"}
TOL = 1e-8

def load(root):
    """Load GPU cells (device=gpu) from the per-cell JSON store into
    rows[(family,matrix)][label] = dict(setup,solve,total,iters,rel_res).

    Legacy flat CSVs are intentionally rejected: they omit solver configuration,
    status, and timeout-cap provenance, so they cannot satisfy the series rule."""
    rows = defaultdict(dict)
    if os.path.isfile(root):
        raise ValueError("legacy GPU CSVs lack configuration/status provenance; "
                         "pass the unified per-cell store instead")
    seen = set()
    records, _ = chart_cells.load_current_records(
        root,
        pattern="**/*__gpu.json",
        include=lambda c: (c.get("cell", {}).get("device") == "gpu"
                           and (c.get("cell", {}).get("solver"),
                                c.get("cell", {}).get("config", "")) in LABELS),
        stale_policy="filter",
        source="gpu_charts measurements",
    )
    for c in records:
        cell = c["cell"]
        if cell.get("device") != "gpu":
            continue
        lab = LABELS.get((cell["solver"], cell.get("config", "")))
        if not lab:
            continue
        key = (cell["family"], cell["matrix_id"], lab, cell.get("threads"))
        if key in seen:
            raise RuntimeError(f"duplicate GPU chart series {key}")
        seen.add(key)
        if c.get("status") != "complete":
            continue
        m = c.get("metrics", {})
        if m.get("total_s") is None:
            continue
        prev = rows[(cell["family"], cell["matrix_id"])].get(lab)
        if prev is not None:
            # LABELS is injective, so two cells can only collide here if the store
            # holds a duplicate for one (solver, config). Taking the faster one
            # would be a min-over-draws; refuse instead of silently picking a winner.
            raise RuntimeError(
                f"duplicate GPU cell for {cell['family']}/{cell['matrix_id']} "
                f"[{lab}] (config {cell.get('config','')!r}) -- refusing to choose "
                f"between {prev['total']}s and {m['total_s']}s")
        rows[(cell["family"], cell["matrix_id"])][lab] = dict(
            setup=m.get("setup_s", 0.0), solve=m.get("solve_s", 0.0),
            total=m["total_s"], iters=m.get("iters", 0),
            rel_res=m.get("rel_res", 1.0), nnz=m.get("nnz", 0),
            rss_peak=m.get("max_rss_mb"), rss_solve=m.get("solve_rss_mb"),
            vram_peak=m.get("max_vram_mb"))
    return rows

def load_outcomes(root):
    """Outcome metadata for every charted GPU cell, including exact timeout caps."""
    out = {}
    if os.path.isfile(root):
        raise ValueError("legacy GPU CSVs lack status and timeout-cap provenance")
    records, _ = chart_cells.load_current_records(
        root,
        pattern="**/*__gpu.json",
        include=lambda c: (c.get("cell", {}).get("device") == "gpu"
                           and (c.get("cell", {}).get("solver"),
                                c.get("cell", {}).get("config", "")) in LABELS),
        stale_policy="filter",
        source="gpu_charts outcomes",
    )
    for c in records:
        cell = c["cell"]
        if cell.get("device") != "gpu":
            continue
        lab = LABELS.get((cell["solver"], cell.get("config", "")))
        if lab:
            key = (cell["family"], cell["matrix_id"], lab)
            if key in out:
                raise RuntimeError(f"duplicate GPU outcome for {key}")
            out[key] = {"status": c.get("status"),
                        "timeout_cap_s": timeout_cap(c)}
    return out

def load_status(root):
    """Compatibility view used by GPU-only marker charts."""
    return {key: value["status"] for key, value in load_outcomes(root).items()}


def mats_of(rows, fam, reduce_grids=True):
    # Order by nnz ascending — the SAME order/selection as the CPU fair charts
    # (fair_charts.headline_mats), so CPU / GPU / combined figures line up. For the
    # grouped-bar charts, grids reduce to ONLY the largest 2D and largest 3D grid
    # (smaller ones redundant); the iters HEATMAP passes reduce_grids=False to show
    # every measured grid.
    def key(m):
        d = rows[(fam, m)]
        return (max((v.get("nnz", 0) for v in d.values()), default=0), m)
    ms = sorted({m for (f, m) in rows if f == fam}, key=key)
    if fam != "grids" or not reduce_grids:
        return ms
    twod = [m for m in ms if not m.startswith("grid3d")]
    threed = [m for m in ms if m.startswith("grid3d")]
    return ([twod[-1]] if twod else []) + ([threed[-1]] if threed else [])

def present_labels(rows, fam, mats):
    have = {l for m in mats for l in rows[(fam, m)]}
    return [l for l in ORDER if l in have]

def value_heatmap(mats, row_labels, M, out, *, title, is_iters=False, cell_fmt=None,
                  cbar="× best in column (log scale)", status=None, device="cpu"):
    """Generic matrix × method heatmap (the house style, shared so the iteration and
    fill charts are heatmaps, not cramped grouped bars): rows = method, cols = matrix;
    colour = value / best (lowest) in that column (green = best, log scale); each cell
    annotated with the absolute value over the ratio. M is a rows×cols array with NaN
    for missing cells (rendered grey). cell_fmt(v) formats the absolute annotation
    (default: integer for is_iters, else "{v:.2f}s")."""
    M = np.asarray(M, dtype=float)
    if M.size == 0 or not np.isfinite(M).any():
        return
    if cell_fmt is None:
        cell_fmt = (lambda v: f"{int(round(v))}") if is_iters else (lambda v: f"{v:.2f}s")
    ratio = np.full_like(M, np.nan)
    for j in range(M.shape[1]):
        col = M[:, j]; pos = col[np.isfinite(col) & (col > 0)]   # >0 only: no /0 -> inf/nan
        if pos.size:
            ratio[:, j] = col / pos.min()
    cmap = plt.cm.RdYlGn_r.copy(); cmap.set_bad("white")   # never-run -> white
    finite = ratio[np.isfinite(ratio)]
    vmax = min(float(finite.max()), 16.0) if finite.size else 4.0
    norm = mcolors.LogNorm(vmin=1.0, vmax=max(vmax, 1.6))
    fig, ax = plt.subplots(figsize=(max(8, 1.25*len(mats)+3.5), 0.55*len(row_labels)+2))
    im = ax.imshow(np.ma.masked_invalid(ratio), cmap=cmap, aspect="auto", norm=norm)
    ax.set_xticks(range(len(mats))); ax.set_xticklabels(mat_labels(mats), rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{cell_fmt(M[i, j])}\n{ratio[i, j]:.1f}×",
                        ha="center", va="center", fontsize=6.3)
            elif status is not None and status[i][j] in ("timeout", "oom", "failed"):
                st = status[i][j]                       # ran-but-no-result -> red flag
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2a0a0",
                                           edgecolor="#a00000", lw=0.9))
                txt = ("Timeout" if st == "timeout"
                       else f"OOM\n{'GPU' if device == 'gpu' else 'RAM'}" if st == "oom"
                       else "FAIL")
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.8,
                        color="black", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, ticks=[1, 1.5, 2, 3, 4, 6, 8, 12, 16])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
    cb.set_label(cbar)
    ax.set_title(title, fontsize=9.5)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

def breakdown(rows, fam, out):
    mats = mats_of(rows, fam); labs = present_labels(rows, fam, mats)
    if not mats or not labs: return
    fig, ax = plt.subplots(figsize=(max(8, 1.7*len(mats)), 5.5))
    w = 0.8/max(1, len(labs)); x = np.arange(len(mats)); legended = set()
    for i, m in enumerate(mats):
        d = rows[(fam, m)]
        # bars within a matrix sorted fastest->slowest (converged only get a bar)
        present = sorted((l for l in labs if l in d), key=lambda l: d[l]["total"])
        for j, lab in enumerate(present):
            # THE GRADING RULE (benchmarks/README.md): exactly TOL, no grace factor.
            conv = d[lab]["rel_res"] <= TOL
            col = COLORS[lab]; xx = x[i] + j*w
            ax.bar(xx, d[lab]["setup"], w, color=col,
                   label=(lab if lab not in legended else None)); legended.add(lab)
            b = ax.bar(xx, d[lab]["solve"], w, bottom=d[lab]["setup"], color=col,
                       alpha=0.45, hatch="///")
            if not conv:
                for bb in b: bb.set_hatch("xxx")
    ax.set_xticks(x + 0.4 - w/2); ax.set_xticklabels(mat_labels(mats), rotation=30, ha="right")
    ax.set_ylabel("time (s) — t16 GPU   [solid = setup, /// = solve]")
    ax.set_title(f"{fam} (GPU): setup + solve breakdown (per-solver grounding, tol 1e-8)")
    ax.legend(ncol=3, fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

def iters(rows, fam, out, gstatus=None):
    """PCG iterations as a matrix × solver heatmap (was grouped bars)."""
    gstatus = gstatus or {}
    mats = mats_of(rows, fam, reduce_grids=False); labs = present_labels(rows, fam, mats)
    if not mats or not labs: return
    M = np.full((len(labs), len(mats)), np.nan)
    stats = [[None] * len(mats) for _ in labs]
    for i, lab in enumerate(labs):
        for j, m in enumerate(mats):
            d = rows[(fam, m)].get(lab)
            if d and d.get("iters"):
                M[i, j] = d["iters"]
            else:                                   # OOM / timeout / failed -> red flag
                stats[i][j] = gstatus.get((fam, m, lab))
    value_heatmap(mats, labs, M, out, is_iters=True, status=stats, device="gpu",
                  title=f"{fam} (GPU): PCG iterations — matrix × solver, t16 "
                        f"(green = fewest per matrix; cell = iters / ×best; "
                        f"Timeout / OOM / FAIL = ran but no result; blank = not run)")

def accuracy(rows, fam, out):
    mats = mats_of(rows, fam); labs = present_labels(rows, fam, mats)
    if not mats or not labs: return
    fig, ax = plt.subplots(figsize=(max(8, 1.7*len(mats)), 5))
    w = 0.8/max(1, len(labs)); x = np.arange(len(mats)); legended = set()
    for i, m in enumerate(mats):
        d = rows[(fam, m)]
        present = sorted((l for l in labs if l in d), key=lambda l: d[l]["rel_res"])
        for j, lab in enumerate(present):
            rr = max(d[lab]["rel_res"], 1e-12)
            ax.bar(x[i] + j*w, rr, w, color=COLORS[lab],
                   label=(lab if lab not in legended else None)); legended.add(lab)
    ax.axhline(TOL, color="k", ls="--", alpha=0.6, label="tol 1e-8")
    ax.set_yscale("log"); ax.set_ylim(1e-10, 1)
    ax.set_xticks(x + 0.4 - w/2); ax.set_xticklabels(mat_labels(mats), rotation=30, ha="right")
    ax.set_ylabel("final ‖b−Ax‖/‖b‖ (lower = better)")
    ax.set_title(f"{fam} (GPU): solution accuracy (all should sit on/under the 1e-8 line)")
    ax.legend(ncol=3, fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/cells",
                    help="unified per-cell JSON store containing device=gpu cells")
    ap.add_argument("--out", default="benchmarks/latest/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.root)
    gstatus = load_status(a.root)   # non-complete GPU cells (oom/timeout/failed) for marks
    GIANT_NNZ = 20_000_000   # match fair_charts: split SuiteSparse small vs social giants
    def _nnz(fam, m):
        return max((v.get("nnz", 0) for v in rows[(fam, m)].values()), default=0)
    n = 0
    for fam in FAMS:
        if not any(f == fam for (f, _) in rows): continue
        # SuiteSparse splits into _small/_giants (the social giants flatten the FEM
        # matrices); grids/ipm stay unsuffixed. Matches the CPU + combined charts.
        if fam == "suitesparse":
            mats = [m for (f, m) in rows if f == fam]
            is_giant = lambda m: _nnz(fam, m) >= GIANT_NNZ or m in GIANT_MATS
            giants = [m for m in mats if is_giant(m)]
            small  = [m for m in mats if not is_giant(m)]
            groups = [("_small", set(small)), ("_giants", set(giants))] if (giants and small) else [("", None)]
        else:
            groups = [("", None)]
        # breakdown + iters now come from combined_charts (device-filtered GPU-only view,
        # one consistent style); gpu_charts keeps accuracy/scaling.
        for suffix, sub in groups:   # accuracy bars split by scale (small / giants)
            rsub = rows if sub is None else defaultdict(dict,
                   {k: v for k, v in rows.items() if k[0] != fam or k[1] in sub})
            accuracy(rsub, fam, f"{a.out}/accuracy_gpu_{fam}{suffix}.png")
            n += 1
    print(f"gpu_charts: {len(rows)} (family,matrix) cells -> {n} figures in {a.out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cholesky-family fill-ratio chart from results/fill_cells (see fill_pass.py).

fill = 2*offdiag(L)/offdiag(A) on ONE consistent definition across the Cholesky-type
solvers: apxchol (per IS selector), the AC/AC2 reference, RCHOL/pRCHOL, and ParAC
(graph/physics × CPU/GPU). It is the fill-vs-iterations tradeoff axis (more fill -> a
stronger preconditioner -> fewer PCG iterations). AMG solvers (BoomerAMG/AMGCL) have no
triangular factor and so no comparable number -- they are intentionally absent.

One figure per family: a matrix × method heatmap (fill_heatmap_{fam}.png, every
measured grid).

  python3 benchmarks/fill_chart.py --out benchmarks/latest/figures
"""
import argparse, glob, json, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gpu_charts as gpu          # shared matrix × method heatmap (value_heatmap)

FAMS = ["grids", "ipm", "suitesparse"]
# solver key -> (label, colour), drawn/ordered in this list order:
#   - apxchol per IS selector (bg/greedy/bk): the order changes the factor density
#     (blues, darkest = bg);
#   - AC/AC2 = Kyng16 approximate-Cholesky reference, Laplacian-only (so absent on the
#     SDDM IPM matrices);
#   - RCHOL/pRCHOL;
#   - ParAC graph/physics on CPU vs GPU -- the CPU (AMD-reordered) and GPU (ParAC's own
#     random-nnz-sort) implementations factor differently, so their fill genuinely
#     differs and each is its own series (oranges).
SOLVERS = [
    ("apxchol_bg",   "apxchol bg",   "#0b5394"),
    ("apxchol_greedy", "apxchol priority-greedy", "#3d85c6"),
    ("apxchol_bk",   "apxchol bk",   "#9fc5e8"),
    ("ac",  "AC [Kyng16]",  "#2ca02c"),
    ("ac2", "AC2 [Kyng16]", "#9467bd"),
    ("rchol",     "RCHOL",  "#d62728"),
    ("rchol_par", "pRCHOL", "#e377c2"),
    ("parac_graph_cpu",   "ParAC graph (CPU)",   "#ff8c00"),
    ("parac_physics_cpu", "ParAC physics (CPU)", "#e6550d"),
    ("parac_graph_gpu",   "ParAC graph (GPU)",   "#fdae6b"),
    ("parac_physics_gpu", "ParAC physics (GPU)", "#8c2d04"),
]


def load(root):
    rows = defaultdict(dict)  # rows[(family,matrix)][solver] = fill
    for f in glob.glob(f"{root}/*.json"):
        d = json.load(open(f))
        rows[(d["family"], d["matrix_id"])][d["solver"]] = d["fill"]
    return rows


def _grid_key(m):
    return int(m.split("_")[1]) if "_" in m else 0

def _ordered(rows, fam):
    """matrix columns for fam: 2D grids by n, then 3D grids by n, then the rest."""
    mats = sorted({m for (f, m) in rows if f == fam and rows.get((fam, m))})
    g2 = sorted((m for m in mats if m.startswith("grid_")), key=_grid_key)
    g3 = sorted((m for m in mats if m.startswith("grid3d")), key=_grid_key)
    rest = [m for m in mats if not m.startswith("grid")]
    return g2 + g3 + rest


def heatmap(rows, fam, out):
    """Fill as a matrix × method heatmap (every measured grid): colour = fill relative
    to the SPARSEST factor in that matrix (per column, green = sparsest, log scale),
    each cell annotated with the absolute fill over the ×sparsest ratio."""
    mats = _ordered(rows, fam)
    if not mats:
        return False
    rowlabs, M = [], []
    for key, lab, _ in SOLVERS:
        vals = [rows.get((fam, m), {}).get(key, np.nan) for m in mats]
        arr = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        if np.isfinite(arr).any():
            rowlabs.append(lab); M.append(arr)
    if not rowlabs:
        return False
    gpu.value_heatmap(mats, rowlabs, np.array(M, dtype=float), out,
                      cell_fmt=lambda v: f"{v:.2f}",
                      cbar="× sparsest in column (log scale)",
                      title=f"{fam}: Cholesky-family fill — matrix × method "
                            f"(green = sparsest factor per matrix; cell = fill / ×sparsest)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="results/fill_cells")
    ap.add_argument("--out", default="benchmarks/latest/figures")
    a = ap.parse_args()
    rows = load(a.cells)
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for fam in FAMS:
        if heatmap(rows, fam, f"{a.out}/fill_heatmap_{fam}.png"):
            n += 1
    print(f"fill_chart: {len(rows)} fill cells -> {n} figures in {a.out}")


if __name__ == "__main__":
    main()

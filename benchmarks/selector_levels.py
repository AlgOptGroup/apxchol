#!/usr/bin/env python3
"""Collect SpTRSV level counts per (IS selector, matrix) at vec_pool, then plot a
selector x matrix heatmap of the back-solve level count. The level count is the
direct explainer for the selector x graph timing: a deep elimination tree (many
SpTRSV levels) makes the triangular solve slow, so a selector's depth blow-up on
dense social graphs shows up here as a high level count. Runs the bench with
APXCHOL_LEVEL_DUMP=1
maxiter=1 (factorize + dump only, no PCG) for each selector. CPU-heavy on the
giants; run from the repository top-level directory. Writes
results/selector_levels.csv + the heatmap.
"""
import csv, os, re, subprocess, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import chart_cells
import runner_common as rc
from runner_common import (benchmark_openmp_env, git_sha, margs_for, sh,
                           taskset_prefix)

ROOT = str(Path(__file__).resolve().parents[1])   # repo root
BIN = f"{ROOT}/benchmarks/build/benchmark"
OUT_CSV = f"{ROOT}/results/selector_levels.csv"
SELS = ["bg", "greedy", "bk"]
PAT = re.compile(r"fwd_lvls=(\d+) \(max=(\d+)\) bck_lvls=(\d+)")
LEVEL_SCHEMA = 2

# (matrix_id, family, bench-args, needs_reg) — the cross-family poster set + the
# per-family ladders, minus com-Orkut to keep this already CPU-heavy sweep bounded.
MATS = [
    ("grid_2000", "g", "--graph grid --n 2000", False),
    ("grid_3000", "g", "--graph grid --n 3000", False),
    ("grid3d_100", "g", "--graph grid3d --n 100", False),
    ("grid3d_150", "g", "--graph grid3d --n 150", False),
    ("parabolic_fem", "ss", margs_for("parabolic_fem"), True),
    ("ecology1", "ss", margs_for("ecology1"), True),
    ("G3_circuit", "ss", margs_for("G3_circuit"), True),
    ("thermal2", "ss", margs_for("thermal2"), True),
    ("iter0010", "ipm", margs_for("iter0010"), False),
    ("iter0040", "ipm", margs_for("iter0040"), False),
    ("com-Amazon", "ss", margs_for("com-Amazon"), True),
    ("coAuthorsDBLP", "ss", margs_for("coAuthorsDBLP"), True),
    ("kron_g500-logn16", "ss", margs_for("kron_g500-logn16"), True),
    ("com-Youtube", "ss", margs_for("com-Youtube"), True),
    ("coPapersDBLP", "ss", margs_for("coPapersDBLP"), True),
    ("as-Skitter", "ss", margs_for("as-Skitter"), True),
    ("com-LiveJournal", "ss", margs_for("com-LiveJournal"), True),
]

def collect():
    rows = []
    source_sha = git_sha()
    for mid, fam, args, reg in MATS:
        regf = "--reg-rel 1e-6" if reg else ""
        for sel in SELS:
            env = benchmark_openmp_env(
                16, dict(os.environ, APXCHOL_LEVEL_DUMP="1"))
            cmd = (f"{taskset_prefix(16)} {BIN} {args} {regf} --solver apxchol_v1 "
                   f"--v1-configs '{sel}+tree[vec_pool]' --threads 16 --tol 1e-8 "
                   f"--maxiter 1 --repeat 1 --csv")
            try:
                o = sh(cmd, env=env, timeout=900).stderr
            except subprocess.TimeoutExpired:
                print(f"  {mid:16} {sel:5} TIMEOUT", flush=True); continue
            m = PAT.search(o)
            if not m:
                print(f"  {mid:16} {sel:5} no level line", flush=True); continue
            fwd, bck = int(m.group(1)), int(m.group(3))
            rows.append(dict(schema=LEVEL_SCHEMA, matrix_id=mid, family=fam,
                             selector=sel, config=f"{sel}+tree[vec_pool]",
                             device="cpu", git_sha=source_sha,
                             fwd_lvls=fwd, bck_lvls=bck))
            print(f"  {mid:16} {sel:5} fwd={fwd:5} bck={bck:5}", flush=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["schema", "matrix_id", "family",
                                          "selector", "config", "device",
                                          "git_sha", "fwd_lvls", "bck_lvls"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT_CSV}")
    return rows


def validate_rows(rows):
    """Reject old plot-only CSV rows that current stale rules invalidate."""
    stale = []
    for row in rows:
        if int(row.get("schema", 0)) != LEVEL_SCHEMA:
            raise chart_cells.CellStoreError(
                "selector-level CSV lacks current schema/provenance; rerun "
                "selector_levels.py before plotting")
        matrix_id = row["matrix_id"]
        record = {
            "cell": {"family": row["family"], "matrix_id": matrix_id,
                     "solver": "apxchol_v1", "config": row["config"],
                     "device": row["device"], "threads": 16},
            "matrix_meta": {"kind": rc.kind_of(matrix_id)},
            "metrics": {"fwd_lvls": int(row["fwd_lvls"]),
                        "bck_lvls": int(row["bck_lvls"])},
            "status": "complete",
            "provenance": {"git_sha": row["git_sha"]},
        }
        reasons = chart_cells.stale_reasons(record)
        if reasons:
            stale.append((matrix_id, row["selector"], reasons[0]))
    print(f"selector_levels plot input: checked {len(rows)}/{len(rows)} rows; "
          f"current={len(rows) - len(stale)}, stale={len(stale)}", file=sys.stderr)
    if stale:
        examples = "; ".join(f"{m}/{s}: {reason}" for m, s, reason in stale[:3])
        raise chart_cells.StaleCellError(
            f"selector_levels plot input contains {len(stale)} stale rows: {examples}")

def plot(rows, out=f"{ROOT}/benchmarks/latest/figures/selector_levels.png"):
    validate_rows(rows)
    mats = [m for m, *_ in MATS if any(r["matrix_id"] == m for r in rows)]
    by = {(r["selector"], r["matrix_id"]): r["bck_lvls"] for r in rows}
    M = np.full((len(SELS), len(mats)), np.nan)
    for i, s in enumerate(SELS):
        for j, mt in enumerate(mats):
            if (s, mt) in by: M[i, j] = by[(s, mt)]
    norm = np.full_like(M, np.nan)
    for j in range(M.shape[1]):
        col = M[:, j]; fin = col[np.isfinite(col)]
        if fin.size: norm[:, j] = col / fin.min()
    cmap = plt.cm.RdYlGn_r.copy(); cmap.set_bad("0.85")
    fig, ax = plt.subplots(figsize=(max(10, 0.8*len(mats)+2), 3.4))
    vmax = np.nanmax(norm) if np.isfinite(norm).any() else 2.0
    ax.imshow(np.ma.masked_invalid(norm), cmap=cmap, aspect="auto", vmin=1.0,
              vmax=max(min(float(vmax), 4.0), 1.05))
    ax.set_xticks(range(len(mats))); ax.set_xticklabels(mats, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(SELS))); ax.set_yticklabels([f"{s}+tree" for s in SELS], fontsize=10)
    for i in range(len(SELS)):
        for j in range(len(mats)):
            v = M[i, j]
            if np.isfinite(v):
                best = np.nanmin(M[:, j])
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7.5,
                        fontweight="bold" if v == best else "normal")
    ax.set_title("apxchol  IS-selector × graph  —  SpTRSV back-solve LEVEL COUNT (vec_pool, t16)\n"
                 "fewer levels = shallower factor = faster triangular solve;  colour = ×fewest per column",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"plot -> {out}")

if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        with open(OUT_CSV, newline="") as handle:
            rows = [dict(r, fwd_lvls=int(r["fwd_lvls"]),
                         bck_lvls=int(r["bck_lvls"]))
                    for r in csv.DictReader(handle)]
    else:
        rows = collect()
    plot(rows)

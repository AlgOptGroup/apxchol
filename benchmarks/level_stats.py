#!/usr/bin/env python3
"""Collect SpTRSV level statistics per matrix (level counts + work concentration).

Runs the benchmark with APXCHOL_LEVEL_DUMP=1 (apxchol bg+tree factor) on each
matrix and parses the "[trsv-levels]" stderr line, which reports the back-solve
level count, max level size, and the WORK-concentration signals:
  bck_work_top1_frac    share of off-diagonal SpTRSV work in the single fattest level
  bck_work_in_tiny_frac share of work in levels below the legacy size threshold

Historical purpose: screening for the sync-free back solve (since REMOVED from
omp_sptrsv -- level-set won everywhere; power-law graphs have a tiny average
level size but a fat work head). The "syncfree_candidate" column keeps that
screen's verdict for continuity: work genuinely SPREAD across many tiny levels
with NO dominant head, i.e. bck_work_in_tiny_frac HIGH AND bck_work_top1_frac
LOW. The dump also prints per-direction thin-level counts, thin-run length
histograms and barriers/solve (one per level) on extra "[trsv-levels]
fwd:/bck:" lines. Writes results/level_stats.csv. Run from repo root:
  python3 benchmarks/level_stats.py
"""
import csv, os, re

from runner_common import ROOT, sh

BIN = f"{ROOT}/benchmarks/build/benchmark"
OUT = f"{ROOT}/results/level_stats.csv"

# (matrix_id, family, apxchol-args, needs_reg)
MATS = [
    ("grid_500", "grids", "--graph grid --n 500", False),
    ("grid_2000", "grids", "--graph grid --n 2000", False),
    ("grid3d_100", "grids", "--graph grid3d --n 100", False),
    ("grid3d_150", "grids", "--graph grid3d --n 150", False),
    ("parabolic_fem", "suitesparse", f"--mtx {ROOT}/data/matrices/parabolic_fem.mtx", True),
    ("apache2", "suitesparse", f"--mtx {ROOT}/data/matrices/apache2.mtx", True),
    ("ecology1", "suitesparse", f"--mtx {ROOT}/data/matrices/ecology1.mtx", True),
    ("G3_circuit", "suitesparse", f"--mtx {ROOT}/data/matrices/G3_circuit.mtx", True),
    ("thermal2", "suitesparse", f"--mtx {ROOT}/data/matrices/thermal2.mtx", True),
    ("com-Amazon", "suitesparse", f"--mtx {ROOT}/data/matrices/com-Amazon.mtx", True),
    ("coAuthorsDBLP", "suitesparse", f"--mtx {ROOT}/data/matrices/coAuthorsDBLP.mtx", True),
    ("kron_g500-logn16", "suitesparse", f"--mtx {ROOT}/data/matrices/kron_g500-logn16.mtx", True),
    ("iter0010", "ipm", f"--mtx {ROOT}/data/ipm/iter0010/matrix.mtx", False),
    ("iter0040", "ipm", f"--mtx {ROOT}/data/ipm/iter0040/matrix.mtx", False),
]

PAT = re.compile(
    r"m=(\d+) fwd_lvls=(\d+) \(max=(\d+)\) bck_lvls=(\d+) \(max=(\d+)\) "
    r"bck_work_top1_frac=([0-9.]+) bck_work_in_tiny_frac=([0-9.]+)")


def candidate(top1, tiny):
    # The (removed) sync-free schedule only had a chance when work is spread
    # across tiny levels with no fat head. Kept as a descriptive screen; these
    # cutoffs are deliberately permissive.
    return "MAYBE" if (tiny > 0.5 and top1 < 0.1) else "no"


def main():
    rows = []
    for mid, family, args, reg in MATS:
        regflag = "--reg-rel 1e-6" if reg else ""
        env = dict(os.environ, APXCHOL_LEVEL_DUMP="1")
        cmd = (f"{BIN} {args} {regflag} --solver apxchol_v1 --v1-configs 'bg+tree[vec_pool_aos]' "
               f"--threads 16 --tol 1e-8 --maxiter 1 --repeat 1 --csv")
        # runner_common.sh = hardened (process-group kill on timeout)
        o = sh(cmd, timeout=900, env=env)
        m = PAT.search(o.stderr)
        if not m:
            print(f"  {mid:16} FAILED (no [trsv-levels] line)"); continue
        mm, fl, fmax, bl, bmax, top1, tiny = m.groups()
        mm, bl = int(mm), int(bl)
        top1, tiny = float(top1), float(tiny)
        avg = mm / bl if bl else 0
        verdict = candidate(top1, tiny)
        rows.append(dict(matrix_id=mid, family=family, m=mm, fwd_lvls=int(fl),
                         bck_lvls=bl, bck_max=int(bmax), avg_bck_lvl=round(avg, 1),
                         bck_work_top1_frac=top1, bck_work_in_tiny_frac=tiny,
                         syncfree_candidate=verdict))
        print(f"  {mid:16} bck_lvls={bl:5} avg={avg:8.1f} top1={top1:.3f} "
              f"tiny={tiny:.3f}  syncfree_candidate={verdict}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"level stats -> {OUT}")


if __name__ == "__main__":
    main()

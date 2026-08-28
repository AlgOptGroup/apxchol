#!/usr/bin/env python3
import argparse
import math
import os
import re
import statistics
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("matrix")
parser.add_argument("--threads", type=int, default=16)
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
parser.add_argument("--maxiter", type=int, default=1)
parser.add_argument("--candidate-binary")
parser.add_argument("--control-binary")
args = parser.parse_args()

root = Path(__file__).resolve().parents[2]
binary = root / "build-clang" / "apxchol"
candidate_binary = (Path(args.candidate_binary).resolve()
                    if args.candidate_binary else binary)
control_binary = (Path(args.control_binary).resolve()
                  if args.control_binary else binary)
matrix = Path(args.matrix).resolve()
timing_names = (
    "setup", "find_partition", "eliminate", "sptrsv_setup", "pcg", "total",
)

def metric(text, name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s+([0-9.]+) ms\b", text)
    if not match:
        raise RuntimeError(f"missing timing {name}\n{text[-4000:]}")
    return float(match.group(1))

def run(seed, candidate):
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(args.threads), "APXCHOL_VERBOSE": "1"})
    command = [
        str(candidate_binary if candidate else control_binary), str(matrix),
        "--input-kind", "adjacency", "--random-rhs",
        "--is", "block_greedy", "--graph-storage", "vec_pool_aos",
        "--maxiter", str(args.maxiter), "--seed", str(seed), "--verbose",
    ]
    completed = subprocess.run(command, env=env, text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"command failed rc={completed.returncode}\n{completed.stdout[-4000:]}")
    text = completed.stdout
    row = {name: metric(text, name) for name in timing_names}
    stored = re.search(r"stored_nnz=([0-9]+)", text)
    residual = re.search(r"\[info\] residual:\s+([0-9.eE+-]+)", text)
    iterations = re.search(r"\[info\] iterations:\s+([0-9]+)", text)
    row.update({
        "stored_nnz": int(stored.group(1)) if stored else -1,
        "residual": float(residual.group(1)) if residual else math.nan,
        "iterations": int(iterations.group(1)) if iterations else -1,
    })
    return row

ratios = []
for seed in args.seeds:
    before = run(seed, False)
    candidate = run(seed, True)
    after = run(seed, False)
    ratio = {}
    for key in timing_names:
        control = math.sqrt(before[key] * after[key])
        ratio[key] = candidate[key] / control if control else math.nan
    ratios.append(ratio)
    print(
        f"seed={seed} T={args.threads} "
        f"setup={ratio['setup']:.4f} eliminate={ratio['eliminate']:.4f} "
        f"total={ratio['total']:.4f} "
        f"nnz={before['stored_nnz']}/{candidate['stored_nnz']}/{after['stored_nnz']} "
        f"iters={before['iterations']}/{candidate['iterations']}/{after['iterations']} "
        f"residual={before['residual']:.6g}/{candidate['residual']:.6g}/"
        f"{after['residual']:.6g}"
    )

print("median " + " ".join(
    f"{key}={statistics.median(row[key] for row in ratios):.4f}"
    for key in timing_names
))

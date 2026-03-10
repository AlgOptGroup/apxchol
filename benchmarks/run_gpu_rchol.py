#!/usr/bin/env python3
"""Run Liang et al. GPU RCHOL benchmarks and output CSV rows.

This script:
  1. Generates a Laplacian matrix in Matrix Market format using our benchmark tool.
  2. Runs the GPU driver (gpu_rchol_gpu_driver) which includes PCG solve.
  3. Parses stdout for timing and outputs CSV rows compatible with our benchmark format.

Usage:
    python3 run_gpu_rchol.py --build-dir benchmarks/build --graph checkerboard --n 300 --kappa 1000 --tile 4
    python3 run_gpu_rchol.py --build-dir benchmarks/build --graph grid --n 300

Requires: scipy (for MMX generation), numpy
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from scipy import sparse
from scipy.io import mmwrite


def make_grid_laplacian(n_side: int, kappa: float = 1.0, tile: int = 4,
                        checkerboard: bool = False) -> tuple:
    """Build a grid Laplacian, optionally with checkerboard weights."""
    N = n_side * n_side
    rows, cols, data = [], [], []
    for i in range(n_side):
        for j in range(n_side):
            v = i * n_side + j
            neighbors = []
            if i > 0: neighbors.append((i - 1) * n_side + j)
            if j > 0: neighbors.append(i * n_side + j - 1)
            if i < n_side - 1: neighbors.append((i + 1) * n_side + j)
            if j < n_side - 1: neighbors.append(i * n_side + j + 1)
            for u in neighbors:
                w = 1.0
                if checkerboard and kappa > 1.0:
                    ti, tj = i // tile, j // tile
                    ui, uj = u // n_side, u % n_side
                    uti, utj = ui // tile, uj // tile
                    if (ti + tj) % 2 != (uti + utj) % 2:
                        w = kappa
                rows.append(v)
                cols.append(u)
                data.append(-w)
    L = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    diag = np.array(-L.sum(axis=1)).flatten()
    L = L + sparse.diags(diag) + 1e-6 * sparse.eye(N)
    return L, N


def make_erdos_renyi_laplacian(n: int, p: float, seed: int = 42) -> tuple:
    """Build an Erdős-Rényi Laplacian."""
    rng = np.random.default_rng(seed)
    rows, cols, data = [], [], []
    # For moderate n, direct sampling is fine
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([-1.0, -1.0])

    if not rows:
        for i in range(n - 1):
            rows.extend([i, i + 1])
            cols.extend([i + 1, i])
            data.extend([-1.0, -1.0])

    L = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    diag = np.array(-L.sum(axis=1)).flatten()
    L = L + sparse.diags(diag) + 1e-6 * sparse.eye(n)
    return L, n


def run_gpu_driver(driver_path: str, mtx_path: str, n_blocks: int = 512,
                   tol: float = 1e-8) -> dict:
    """Run gpu_rchol_gpu_driver and parse output."""
    cmd = [driver_path, mtx_path, str(n_blocks), "1", f"{tol:.0e}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    info = {}
    m = re.search(r"number of nodes:\s*(\d+)", output)
    if m: info["n"] = int(m.group(1))
    m = re.search(r"number of nonzeros:\s*(\d+)", output)
    if m: info["nnz"] = int(m.group(1))
    m = re.search(r"Kernel execution time:\s*([\d.]+)\s*ms", output)
    if m: info["setup_ms"] = float(m.group(1))
    m = re.search(r"pcg time:\s*([\d.]+)\s*ms", output)
    if m: info["solve_ms"] = float(m.group(1))
    m = re.search(r"final iteration:\s*(\d+)", output)
    if m: info["iters"] = int(m.group(1))
    m = re.search(r"normalized diff norm:\s*([\d.eE+-]+)", output)
    if m: info["rel_res"] = float(m.group(1))
    m = re.search(r"nnz ratio:\s*([\d.]+)", output)
    if m: info["fillin"] = float(m.group(1))
    m = re.search(r"solve preprocess time:\s*([\d.]+)\s*ms", output)
    if m: info["preprocess_ms"] = float(m.group(1))

    return info


def main():
    parser = argparse.ArgumentParser(description="GPU RCHOL benchmark")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--graph", default="checkerboard",
                        choices=["grid", "checkerboard", "erdos"])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--kappa", type=float, default=1000)
    parser.add_argument("--tile", type=int, default=4)
    parser.add_argument("--er-p", type=float, default=0.02)
    parser.add_argument("--mtx", type=str, default=None,
                        help="Path to pre-existing MTX file (skips generation)")
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--csv-header", action="store_true")
    args = parser.parse_args()

    gpu_driver = os.path.join(args.build_dir, "gpu_rchol_gpu_driver")
    if not os.path.isfile(gpu_driver):
        print(f"GPU driver not found: {gpu_driver}", file=sys.stderr)
        sys.exit(1)

    if args.csv_header:
        print("solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz")

    if args.mtx:
        # Use pre-existing MTX file
        mtx_path = args.mtx
        graph_name = os.path.splitext(os.path.basename(mtx_path))[0]
        # Read to get N for fallback
        from scipy.io import mmread
        M = mmread(mtx_path)
        N = M.shape[0]
        nnz_fallback = M.nnz if hasattr(M, 'nnz') else 0
        info = run_gpu_driver(gpu_driver, mtx_path, args.blocks, args.tol)
    else:
        is_checker = args.graph == "checkerboard"
        if args.graph == "erdos":
            L, N = make_erdos_renyi_laplacian(args.n, getattr(args, 'er_p', 0.02))
            graph_name = f"erdos_{args.n}_p{args.er_p:.4g}"
        else:
            L, N = make_grid_laplacian(args.n, args.kappa, args.tile, is_checker)
            if is_checker:
                graph_name = f"checker_{args.n}_k{int(args.kappa)}_t{args.tile}"
            else:
                graph_name = f"grid_{args.n}"

        nnz_fallback = L.nnz
        with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as f:
            mtx_path = f.name
        try:
            mmwrite(mtx_path, L.tocoo(), symmetry="general")
            info = run_gpu_driver(gpu_driver, mtx_path, args.blocks, args.tol)
        finally:
            os.unlink(mtx_path)

    if "setup_ms" not in info or "solve_ms" not in info:
        print(f"Failed to parse GPU driver output for {graph_name}",
              file=sys.stderr)
        sys.exit(1)

    setup_s = info["setup_ms"] / 1000.0
    # Include preprocess in solve time if available
    solve_s = info["solve_ms"] / 1000.0
    if "preprocess_ms" in info:
        solve_s += info["preprocess_ms"] / 1000.0
    total_s = setup_s + solve_s
    n = info.get("n", N)
    nnz = info.get("nnz", nnz_fallback)
    iters = info.get("iters", 0)
    rel_res = info.get("rel_res", 0.0)
    fillin = info.get("fillin", 0.0)
    us_per_nnz = total_s / nnz * 1e6 if nnz > 0 else 0.0

    solver_name = "GPU-RCHOL+PCG [Liang25]"
    print(f"{solver_name},{graph_name},{n},{nnz},"
          f"{setup_s:.6e},{solve_s:.6e},{total_s:.6e},"
          f"{iters},{rel_res:.6e},{fillin:.4f},{us_per_nnz:.4f}")


if __name__ == "__main__":
    main()

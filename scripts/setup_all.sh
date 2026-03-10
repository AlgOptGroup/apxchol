#!/usr/bin/env bash
# setup_all.sh — One-shot setup: build core, benchmarks, fetch matrices
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Scalable Approximate Cholesky — Full Setup"
echo "============================================"
echo ""

# ── Core library + tests ──────────────────────────
echo "[1/4] Building core library + tests..."
cd "$PROJECT_DIR"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j"$(nproc)" 2>&1 | tail -5
echo "  Running tests..."
ctest --output-on-failure 2>&1 | tail -5

# ── Julia packages ─────────────────────────────────
echo ""
echo "[2/4] Setting up Julia environment..."
if command -v julia &>/dev/null; then
    julia --project="$PROJECT_DIR/benchmarks/julia" -e '
        using Pkg
        Pkg.add(["Laplacians", "SparseArrays", "LinearAlgebra", "Statistics",
                  "MatrixMarket", "Printf", "Random", "DelimitedFiles"])
        Pkg.precompile()
        println("[ok] Julia packages installed and precompiled")
    '
else
    echo "[skip] Julia not found. Install Julia for Laplacians.jl benchmarks."
fi

# ── Benchmarks ─────────────────────────────────────
echo ""
echo "[3/4] Building benchmark suite (FetchContent will download deps)..."
cd "$PROJECT_DIR"
mkdir -p benchmarks/build && cd benchmarks/build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -10
make -j"$(nproc)" benchmark 2>&1 | tail -5

# ── Test matrices ──────────────────────────────────
echo ""
echo "[4/4] Downloading test matrices..."
cd "$PROJECT_DIR"
bash scripts/download_graphs.sh

echo ""
echo "============================================"
echo " Setup complete."
echo ""
echo " Run benchmarks:  bash scripts/run_benchmarks.sh"
echo " Run tests:       cd build && ctest --output-on-failure"
echo "============================================"

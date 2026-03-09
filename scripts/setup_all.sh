#!/usr/bin/env bash
# setup_all.sh — One-shot setup: install Julia deps, build rchol, build our code
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Scalable Approximate Cholesky — Full Setup"
echo "============================================"
echo ""

# ── Submodules ─────────────────────────────────────
echo "[1/5] Initializing git submodules..."
cd "$PROJECT_DIR"
git submodule update --init --recursive

# ── Julia packages ─────────────────────────────────
echo ""
echo "[2/5] Setting up Julia environment..."
if command -v julia &>/dev/null; then
    julia --project="$PROJECT_DIR/bench/julia" -e '
        using Pkg
        Pkg.develop(path=joinpath(@__DIR__, "../../extern/Laplacians.jl"))
        Pkg.add(["SparseArrays", "LinearAlgebra", "Statistics",
                  "MatrixMarket", "Printf", "Random", "DelimitedFiles"])
        Pkg.precompile()
        println("[ok] Julia packages installed and precompiled")
    '
else
    echo "[skip] Julia not found. Install Julia for Laplacians.jl benchmarks."
fi

# ── RCHOL ──────────────────────────────────────────
echo ""
echo "[3/5] Building RCHOL (extern/rchol)..."
RCHOL_DIR="$PROJECT_DIR/extern/rchol/c++"
if [[ -d "$RCHOL_DIR" ]]; then
    # Check for METIS
    METIS_OK=0
    if pkg-config --exists metis 2>/dev/null; then
        METIS_OK=1
    elif [[ -f /usr/include/metis.h ]] || [[ -f /usr/local/include/metis.h ]]; then
        METIS_OK=1
    fi

    if (( METIS_OK )); then
        echo "  METIS found."
    else
        echo "  [warn] METIS not found. Parallel rchol will not build."
        echo "  Install: pacman -S metis / apt install libmetis-dev"
    fi

    # Try building with g++ (the original Makefile uses icc)
    echo "  Attempting build with g++..."
    cd "$RCHOL_DIR"
    if [[ -f Makefile.gcc ]]; then
        make -f Makefile.gcc -j"$(nproc)" 2>&1 | tail -5
    else
        echo "  [info] No Makefile.gcc found. See bench/rchol/README.md for manual build."
    fi
else
    echo "  [skip] extern/rchol not found."
fi

# ── Our code ───────────────────────────────────────
echo ""
echo "[4/5] Building our solver + benchmarks..."
cd "$PROJECT_DIR"
mkdir -p build && cd build
# Try to find system Eigen, fall back to FetchContent
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j"$(nproc)" 2>&1 | tail -5

# ── Test matrices ──────────────────────────────────
echo ""
echo "[5/5] Downloading test matrices..."
cd "$PROJECT_DIR"
bash scripts/download_graphs.sh

echo ""
echo "============================================"
echo " Setup complete."
echo ""
echo " Run benchmarks:  bash scripts/run_benchmarks.sh"
echo " Run tests:       cd build && ctest --output-on-failure"
echo "============================================"

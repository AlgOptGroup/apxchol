#!/usr/bin/env bash
# setup_all.sh — One-shot setup: build core, benchmarks, fetch matrices
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Scalable Approximate Cholesky — Full Setup"
echo "============================================"
echo ""

# ── Core library + tests ──────────────────────────
echo "[1/4] Building core library + tests..."
cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$PROJECT_DIR/build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-6}"
echo "  Running tests..."
ctest --test-dir "$PROJECT_DIR/build" --output-on-failure

# ── Julia packages ─────────────────────────────────
echo ""
echo "[2/4] Setting up Julia environment..."
if command -v julia &>/dev/null; then
    julia --project="$PROJECT_DIR/benchmarks/julia" -e '
        using Pkg
        Pkg.instantiate()
        Pkg.precompile()
        println("[ok] Julia packages installed and precompiled")
    '
else
    echo "[skip] Julia not found. Install Julia for Laplacians.jl benchmarks."
fi

# ── Benchmarks ─────────────────────────────────────
echo ""
echo "[3/4] Building benchmark suite (FetchContent will download deps)..."
cmake -S "$PROJECT_DIR/benchmarks" -B "$PROJECT_DIR/benchmarks/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$PROJECT_DIR/benchmarks/build" --target benchmark --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-6}"

# ── Test matrices ──────────────────────────────────
echo ""
echo "[4/4] Downloading test matrices..."
cd "$PROJECT_DIR"
bash scripts/download_graphs.sh

echo ""
echo "============================================"
echo " Setup complete."
echo ""
echo " Run benchmarks:  python3 benchmarks/sweep_fair.py --device cpu"
echo " Run tests:       ctest --test-dir build --output-on-failure"
echo "============================================"

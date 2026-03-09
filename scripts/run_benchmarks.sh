#!/usr/bin/env bash
# run_benchmarks.sh – run all benchmark configurations and collect results
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
DATA_DIR="$PROJECT_DIR/data/matrices"
RESULTS_DIR="$PROJECT_DIR/results"

mkdir -p "$RESULTS_DIR"

BENCH="$BUILD_DIR/benchmark"
if [[ ! -x "$BENCH" ]]; then
    echo "Benchmark binary not found. Building..."
    cd "$BUILD_DIR"
    cmake .. -DEigen3_DIR=/usr/share/eigen3/cmake 2>/dev/null
    make -j"$(nproc)" benchmark
    cd "$PROJECT_DIR"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$RESULTS_DIR/bench_${TIMESTAMP}.csv"

echo "Writing results to: $OUTFILE"
echo ""

# Header
"$BENCH" --graph grid --n 10 --solver cg --csv --maxiter 1 2>/dev/null | head -1 > "$OUTFILE"

run() {
    echo "[run] $*" >&2
    "$BENCH" "$@" --csv 2>/dev/null | tail -n +2 >> "$OUTFILE" || echo "[warn] failed: $*" >&2
}

# ── Synthetic benchmarks ──────────────────────────────

echo "=== Grid graphs ==="
for N in 100 200 500; do
    run --graph grid --n "$N"
done

echo "=== Checkerboard graphs (varying condition) ==="
for N in 100 200 500; do
    for KAPPA in 10 100 1000 10000; do
        run --graph checkerboard --n "$N" --kappa "$KAPPA" --tile 4
    done
done

echo "=== Checkerboard (varying tile) ==="
for TILE in 1 2 4 8 16; do
    run --graph checkerboard --n 300 --kappa 1000 --tile "$TILE"
done

echo "=== Erdős-Rényi ==="
for N in 500 1000 2000; do
    run --graph erdos --n "$N" --er-p 0.02
done

# ── Matrix Market benchmarks ─────────────────────────
if [[ -d "$DATA_DIR" ]]; then
    echo "=== SuiteSparse matrices ==="
    for mtx in "$DATA_DIR"/*.mtx; do
        [[ -f "$mtx" ]] || continue
        # Skip very large matrices (> 200k vertices)
        n=$(grep -v '^%' "$mtx" | head -1 | awk '{print $1}')
        if (( n > 200000 )); then
            echo "[skip] $mtx (n=$n too large)"
            continue
        fi
        run --mtx "$mtx"
    done
else
    echo "[info] No data/matrices/ directory. Run scripts/download_graphs.sh first."
fi

echo ""
echo "Results saved to: $OUTFILE"
echo ""

# ── Julia / Laplacians.jl benchmarks ─────────────────
JULIA_BENCH="$PROJECT_DIR/bench/julia/bench_laplacians.jl"
JULIA_OUT="$RESULTS_DIR/bench_julia_${TIMESTAMP}.csv"

if command -v julia &>/dev/null && [[ -f "$JULIA_BENCH" ]]; then
    echo "=== Laplacians.jl benchmarks ==="

    julia_run() {
        echo "[julia] $*" >&2
        julia --project="$PROJECT_DIR/bench/julia" "$JULIA_BENCH" "$@" --csv 2>/dev/null | tail -n +2 >> "$JULIA_OUT" || echo "[warn] julia failed: $*" >&2
    }

    # Header
    julia --project="$PROJECT_DIR/bench/julia" "$JULIA_BENCH" --graph grid --n 10 --csv --maxiter 1 2>/dev/null | head -1 > "$JULIA_OUT"

    for N in 100 200 500; do
        julia_run --graph grid --n "$N"
    done
    for N in 100 200 500; do
        for KAPPA in 10 100 1000 10000; do
            julia_run --graph checkerboard --n "$N" --kappa "$KAPPA" --tile 4
        done
    done
    for N in 500 1000 2000; do
        julia_run --graph erdos --n "$N" --er-p 0.02
    done

    echo "Julia results saved to: $JULIA_OUT"
else
    echo "[skip] Julia or bench_laplacians.jl not available."
fi

# ── RCHOL benchmarks ─────────────────────────────────
RCHOL_BENCH="$BUILD_DIR/bench_rchol"
RCHOL_OUT="$RESULTS_DIR/bench_rchol_${TIMESTAMP}.csv"

if [[ -x "$RCHOL_BENCH" ]]; then
    echo "=== RCHOL benchmarks ==="
    echo "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz" > "$RCHOL_OUT"

    for N in 50 100 200; do
        for KAPPA in 10 100 1000 10000; do
            echo "[rchol] --n $N --kappa $KAPPA --tile 4" >&2
            "$RCHOL_BENCH" --n "$N" --kappa "$KAPPA" --tile 4 --csv 2>/dev/null | tail -n +2 >> "$RCHOL_OUT" || echo "[warn] rchol failed" >&2
        done
    done

    echo "RCHOL results saved to: $RCHOL_OUT"
else
    echo "[skip] RCHOL benchmark binary not found. Build it first (see bench/rchol/README.md)."
fi

echo ""
echo "============================================"
echo " All results in: $RESULTS_DIR/"
echo "============================================"
echo ""

echo "C++ summary:"
column -t -s',' "$OUTFILE" | head -30 || true

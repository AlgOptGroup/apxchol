#!/usr/bin/env bash
# benchmarks/run.sh – Run benchmark configurations and collect CSV results
#
# Usage:
#   bash benchmarks/run.sh                              # full suite (all bench sets, all solvers)
#   bash benchmarks/run.sh --quick                      # quick smoke test (smaller sizes)
#   bash benchmarks/run.sh --bench grid,checker         # only grid + checkerboard
#   bash benchmarks/run.sh --solver apxchol,rchol_mkl   # only these C++ solvers
#   bash benchmarks/run.sh --solver ac,ac2              # only Julia solvers
#   bash benchmarks/run.sh --solver gpu_rchol           # only GPU RCHOL
#   bash benchmarks/run.sh --bench checker --solver cg  # specific combo
#   bash benchmarks/run.sh --append results/prev.csv    # append to existing CSV
#
# Bench sets: grid, checker, erdos, tile, mtx, gpu_paper, sddm2023
# Solvers (C++):  apxchol, cg, ldlt, rchol, rchol_mkl, rchol_mkl1, cholmod, amgcl
# Solvers (Julia): ac, ac2
# Solvers (GPU):   gpu_rchol
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/build"
DATA_DIR="$PROJECT_DIR/data/matrices"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

BENCH="$BUILD_DIR/benchmark"
if [[ ! -x "$BENCH" ]]; then
    echo "Benchmark binary not found. Building from benchmarks/..."
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
    make -j"$(nproc)" benchmark
    cd "$PROJECT_DIR"
fi

# ── Parse arguments ──────────────────────────────────
QUICK=false
BENCH_SETS=""
SOLVER_ARG=""
APPEND_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)   QUICK=true; shift ;;
        --bench)   BENCH_SETS="$2"; shift 2 ;;
        --solver)  SOLVER_ARG="$2"; shift 2 ;;
        --append)  APPEND_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Default: all bench sets
ALL_BENCH_SETS="grid,checker,erdos,tile,mtx,gpu_paper,sddm2023"
if [[ -z "$BENCH_SETS" || "$BENCH_SETS" == "all" ]]; then
    BENCH_SETS="$ALL_BENCH_SETS"
fi

# ── Classify solvers into C++, Julia, GPU ────────────
ALL_CPP_SOLVERS="apxchol apxchol_v1 cg ldlt rchol rchol_mkl rchol_mkl1 cholmod amgcl"
ALL_JULIA_SOLVERS="ac ac2"
ALL_GPU_SOLVERS="gpu_rchol"

# Parse SOLVER_ARG into separate lists
CPP_SOLVERS=""
JULIA_SOLVERS=""
RUN_GPU_RCHOL=false

if [[ -z "$SOLVER_ARG" ]]; then
    # Default: run everything
    CPP_SOLVERS="$ALL_CPP_SOLVERS"
    JULIA_SOLVERS="$ALL_JULIA_SOLVERS"
    RUN_GPU_RCHOL=true
else
    IFS=',' read -ra REQUESTED <<< "$SOLVER_ARG"
    for S in "${REQUESTED[@]}"; do
        case "$S" in
            ac|ac2)       JULIA_SOLVERS="$JULIA_SOLVERS${JULIA_SOLVERS:+,}$S" ;;
            gpu_rchol)    RUN_GPU_RCHOL=true ;;
            *)            CPP_SOLVERS="$CPP_SOLVERS${CPP_SOLVERS:+ }$S" ;;
        esac
    done
fi

# Build solver flag for the C++ benchmark binary
SOLVER_FLAG=""
if [[ -n "$CPP_SOLVERS" ]]; then
    SOLVER_FLAG="--solver $(echo "$CPP_SOLVERS" | tr ' ' ',')"
fi

# ── Output file ──────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$APPEND_FILE" ]]; then
    OUTFILE="$APPEND_FILE"
    echo "Appending to: $OUTFILE"
else
    OUTFILE="$RESULTS_DIR/bench_${TIMESTAMP}.csv"
    echo "Writing results to: $OUTFILE"
    # CSV header
    "$BENCH" --graph grid --n 10 --solver cg --csv --maxiter 1 2>/dev/null | head -1 > "$OUTFILE"
fi
LATEST="$RESULTS_DIR/latest.csv"

# Solvers to skip for the current batch (set before calling run(), reset after)
SKIP_SOLVERS=""

run() {
    # Run each solver in its own process to isolate crashes (e.g., METIS segfault)
    # --repeat 3 takes the median of 3 runs to reduce noise (cold-cache, freq. scaling)
    [[ -z "$CPP_SOLVERS" ]] && return 0
    if [[ -n "$SOLVER_FLAG" ]]; then
        # Filter out skipped solvers from the explicit list
        local filtered="$CPP_SOLVERS"
        if [[ -n "$SKIP_SOLVERS" ]]; then
            for S in $SKIP_SOLVERS; do
                filtered=$(echo "$filtered" | sed "s/\b$S\b//g" | xargs)
            done
        fi
        [[ -z "$filtered" ]] && return 0
        echo "[run] $* --solver $(echo "$filtered" | tr ' ' ',')" >&2
        for S in $filtered; do
            if [[ "$S" == "apxchol_v1" ]]; then
                # Run v1 combos at 1 thread and 16 threads
                for T in 1 16; do
                    "$BENCH" "$@" --csv --repeat 3 --solver "$S" --threads "$T" 2>/dev/null | tail -n +2 >> "$OUTFILE" || true
                done
            else
                "$BENCH" "$@" --csv --repeat 3 --solver "$S" 2>/dev/null | tail -n +2 >> "$OUTFILE" || true
            fi
        done
    else
        for S in $ALL_CPP_SOLVERS; do
            if [[ -n "$SKIP_SOLVERS" && " $SKIP_SOLVERS " == *" $S "* ]]; then
                continue
            fi
            if [[ "$S" == "apxchol_v1" ]]; then
                for T in 1 16; do
                    "$BENCH" "$@" --csv --repeat 3 --solver "$S" --threads "$T" 2>/dev/null | tail -n +2 >> "$OUTFILE" || true
                done
            else
                "$BENCH" "$@" --csv --repeat 3 --solver "$S" 2>/dev/null | tail -n +2 >> "$OUTFILE" || true
            fi
        done
        echo "[run] $*" >&2
    fi
}

JULIA_BENCH="$SCRIPT_DIR/julia/bench_laplacians.jl"
HAS_JULIA=false
command -v julia &>/dev/null && [[ -f "$JULIA_BENCH" ]] && HAS_JULIA=true

run_julia() {
    # Run Julia solvers (ac, ac2) with the given graph arguments
    [[ -z "$JULIA_SOLVERS" ]] && return 0
    $HAS_JULIA || return 0
    echo "[run] julia $*" >&2
    julia --project="$SCRIPT_DIR/julia" "$JULIA_BENCH" \
        "$@" --solver "$JULIA_SOLVERS" --csv 2>/dev/null \
        | tail -n +2 >> "$OUTFILE" || echo "[warn] Julia failed: $*" >&2
}

GPU_RCHOL_SCRIPT="$SCRIPT_DIR/run_gpu_rchol.py"
GPU_DRIVER="$BUILD_DIR/gpu_rchol_gpu_driver"
HAS_GPU=false
[[ -x "$GPU_DRIVER" ]] && command -v python3 &>/dev/null && HAS_GPU=true

run_gpu() {
    # Run GPU RCHOL with the given graph arguments
    $RUN_GPU_RCHOL || return 0
    $HAS_GPU || return 0
    echo "[run] gpu_rchol $*" >&2
    python3 "$GPU_RCHOL_SCRIPT" --build-dir "$BUILD_DIR" \
        "$@" 2>/dev/null \
        | tail -n +1 >> "$OUTFILE" || echo "[warn] GPU RCHOL failed: $*" >&2
}

# ── Size configurations ──────────────────────────────
# Unified ranges: all families use comparable n² sizes.
# quick: n up to 300 (n²=90K)   full: n up to 1000 (n²=1M)
if $QUICK; then
    GRID_SIZES=(50 100 150 200 300)
    CHECKER_SIZES=(50 100 150 200 300)
    KAPPAS=(100 1000 10000)
    # (vertex_count, edge_probability) pairs — target ~100K edges each
    ER_CONFIGS=("500 0.8" "1000 0.2" "2000 0.05" "3000 0.022" "5000 0.008")
else
    # Full mode: grid n up to 3000 → n²=9M vertices, ~36M nnz
    GRID_SIZES=(100 200 500 1000 1500 2000 3000)
    CHECKER_SIZES=(100 200 500 1000 1500 2000 3000)
    KAPPAS=(100000)
    # (vertex_count, edge_probability) — scaling from ~200 edges to ~10K×0.02=1M edges
    ER_CONFIGS=("500 0.8" "1000 0.5" "3000 0.1" "5000 0.04" "10000 0.02")
fi

# ── Bench sets ───────────────────────────────────────
IFS=',' read -ra SETS <<< "$BENCH_SETS"

for SET in "${SETS[@]}"; do
    case "$SET" in
    grid)
        echo "=== Grid graphs ==="
        for N in "${GRID_SIZES[@]}"; do
            # Skip slow solvers on large grids: LDLT is O(n^3/2), CG hits maxiter
            if (( N >= 2000 )); then
                SKIP_SOLVERS="ldlt cg"
            elif (( N >= 1500 )); then
                SKIP_SOLVERS="cg"
            else
                SKIP_SOLVERS=""
            fi
            run --graph grid --n "$N"
            run_julia --graph grid --n "$N"
            run_gpu --graph grid --n "$N"
        done
        SKIP_SOLVERS=""
        ;;
    checker)
        echo "=== Checkerboard graphs (varying κ) ==="
        for N in "${CHECKER_SIZES[@]}"; do
            if (( N >= 2000 )); then
                SKIP_SOLVERS="ldlt cg"
            elif (( N >= 1500 )); then
                SKIP_SOLVERS="cg"
            else
                SKIP_SOLVERS="cg"
            fi
            # Non-CG solvers: run at kappa=100000 (kappa=1 ≡ grid, already benchmarked)
            run --graph checkerboard --n "$N" --kappa 100000 --tile 4
            run_julia --graph checkerboard --n "$N" --kappa 100000 --tile 4
            run_gpu --graph checkerboard --n "$N" --kappa 100000 --tile 4
            # CG: sensitive to kappa (~4x variation), run at multiple kappas
            if (( N < 1500 )); then
                for KAPPA in "${KAPPAS[@]}"; do
                    "$BENCH" --graph checkerboard --n "$N" --kappa "$KAPPA" --tile 4 \
                        --csv --repeat 3 --solver cg 2>/dev/null | tail -n +2 >> "$OUTFILE" || true
                done
            fi
        done
        SKIP_SOLVERS=""
        ;;
    erdos)
        echo "=== Erdős-Rényi ==="
        for CFG in "${ER_CONFIGS[@]}"; do
            read -r N P <<< "$CFG"
            # Dense ER graphs cause fill-in explosion in direct solvers
            if (( N >= 10000 )); then
                SKIP_SOLVERS="ldlt cholmod"
            elif (( N >= 5000 )); then
                SKIP_SOLVERS="ldlt"
            else
                SKIP_SOLVERS=""
            fi
            run --graph erdos --n "$N" --er-p "$P"
            run_julia --graph erdos --n "$N" --er-p "$P"
            run_gpu --graph erdos --n "$N" --er-p "$P"
        done
        SKIP_SOLVERS=""
        ;;
    tile)
        echo "=== Checkerboard (varying tile) ==="
        TILES=(1 2 4 8 16)
        $QUICK && TILES=(2 4 8)
        for TILE in "${TILES[@]}"; do
            run --graph checkerboard --n 300 --kappa 1000 --tile "$TILE"
        done
        ;;
    mtx)
        if [[ -d "$DATA_DIR" ]]; then
            echo "=== SuiteSparse matrices ==="
            for mtx in "$DATA_DIR"/*.mtx; do
                [[ -f "$mtx" ]] || continue
                n=$(grep -v '^%' "$mtx" | head -1 | awk '{print $1}')
                if (( n > 2000000 )); then
                    echo "[skip] $mtx (n=$n too large)" >&2
                    continue
                fi
                # Skip LDLT on large matrices (O(n^3/2) too slow)
                if (( n >= 30000 )); then
                    SKIP_SOLVERS="ldlt"
                else
                    SKIP_SOLVERS=""
                fi
                run --mtx "$mtx"
            done
        else
            echo "[skip] No data/matrices directory" >&2
        fi
        ;;
    gpu_paper)
        # GPU RCHOL paper matrices — large SuiteSparse/FEM (ecology, apache2, G3_circuit, etc.)
        GPU_PAPER_MATRICES="parabolic_fem ecology1 ecology2 apache2 G3_circuit"
        if [[ -d "$DATA_DIR" ]]; then
            echo "=== GPU RCHOL Paper Matrices ==="
            for NAME in $GPU_PAPER_MATRICES; do
                mtx="$DATA_DIR/${NAME}.mtx"
                [[ -f "$mtx" ]] || { echo "[skip] $NAME.mtx not found (run scripts/download_graphs.sh)" >&2; continue; }
                # All GPU paper matrices are large (n>=500K), skip LDLT
                SKIP_SOLVERS="ldlt"
                run --mtx "$mtx"
                run_julia --mtx "$mtx"
                run_gpu --mtx "$mtx"
            done
            SKIP_SOLVERS=""
        else
            echo "[skip] No data/matrices directory (run scripts/download_graphs.sh)" >&2
        fi
        ;;
    sddm2023)
        SDDM_DIR="$PROJECT_DIR/data/sddm2023"
        if [[ -d "$SDDM_DIR" ]] && ls "$SDDM_DIR"/*.mtx &>/dev/null; then
            echo "=== SDDM2023 instances ==="
            for mtx in "$SDDM_DIR"/*.mtx; do
                [[ -f "$mtx" ]] || continue
                read -r n _ nnz <<< "$(grep -v '^%' "$mtx" | head -1)"
                if (( n > 2000000 )); then
                    echo "[skip] $(basename "$mtx") (n=$n too large)" >&2
                    continue
                fi
                if (( nnz > 10000000 )); then
                    echo "[skip] $(basename "$mtx") (nnz=$nnz too large)" >&2
                    continue
                fi
                # Skip LDLT on large SDDM instances (O(n^3/2) too slow)
                if (( n >= 30000 )); then
                    SKIP_SOLVERS="ldlt"
                else
                    SKIP_SOLVERS=""
                fi
                run --mtx "$mtx"
                run_julia --mtx "$mtx"
                run_gpu --mtx "$mtx"
            done
            SKIP_SOLVERS=""
        else
            echo "[skip] No SDDM2023 instances. Run: julia --project=benchmarks/julia scripts/generate_sddm_instances.jl" >&2
        fi
        ;;
    *)
        echo "[warn] Unknown bench set: $SET (valid: grid,checker,erdos,tile,mtx,gpu_paper,sddm2023)" >&2
        ;;
    esac
done

# Symlink to latest (avoid self-referencing symlink when appending to latest.csv)
REAL_OUTFILE="$(readlink -f "$OUTFILE")"
if [[ "$(readlink -f "$LATEST")" != "$REAL_OUTFILE" ]]; then
    ln -sf "$(basename "$REAL_OUTFILE")" "$LATEST"
fi

echo ""
echo "Results: $OUTFILE"
echo "Symlink: $LATEST"

# Auto-generate plots if Python available
if command -v python3 &>/dev/null && [[ -f "$SCRIPT_DIR/plot.py" ]]; then
    echo ""
    echo "Generating plots..."
    python3 "$SCRIPT_DIR/plot.py" "$OUTFILE"

    # Copy latest results and plots for embedding in docs
    EMBED_DIR="$SCRIPT_DIR/latest"
    mkdir -p "$EMBED_DIR"
    cp "$OUTFILE" "$EMBED_DIR/results.csv"
    if [[ -d "$RESULTS_DIR/plots" ]]; then
        for sub in comparison gpu sddm combined; do
            if [[ -d "$RESULTS_DIR/plots/$sub" ]]; then
                mkdir -p "$EMBED_DIR/$sub"
                cp "$RESULTS_DIR/plots/$sub/"*.png "$EMBED_DIR/$sub/" 2>/dev/null
            fi
        done
    fi
    echo "Embedded results: $EMBED_DIR/"
fi

# Auto-update README with summary table
if command -v python3 &>/dev/null && [[ -f "$SCRIPT_DIR/update_readme.py" ]]; then
    python3 "$SCRIPT_DIR/update_readme.py" "$OUTFILE"
fi

echo "Done."

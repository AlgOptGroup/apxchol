#!/usr/bin/env bash
# benchmarks/run.sh – Run benchmark configurations and collect CSV results
#
# Usage:
#   bash benchmarks/run.sh                              # full suite (all bench sets, all solvers)
#   bash benchmarks/run.sh --quick                      # quick smoke test (smaller sizes)
#   bash benchmarks/run.sh --bench grid,checker         # only grid + checkerboard
#   bash benchmarks/run.sh --solver apxchol,rchol       # only these solvers
#   bash benchmarks/run.sh --bench checker --solver cg  # specific combo
#   bash benchmarks/run.sh --append results/prev.csv    # append to existing CSV
#
# Bench sets: grid, checker, erdos, tile, julia, gpu, mtx, gpu_paper, sddm2023
# Solvers: apxchol, cg, ldlt, rchol, rchol_mkl, rchol_par, cholmod, amgcl, icc
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
if [[ -z "$BENCH_SETS" ]]; then
    BENCH_SETS="grid,checker,erdos,julia,gpu,mtx,gpu_paper,sddm2023"
fi

# Build solver flag for the C++ benchmark binary
SOLVER_FLAG=""
if [[ -n "$SOLVER_ARG" ]]; then
    SOLVER_FLAG="--solver $SOLVER_ARG"
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

run() {
    # Run each solver in its own process to isolate crashes (e.g., METIS segfault)
    # --repeat 3 takes the median of 3 runs to reduce noise (cold-cache, freq. scaling)
    if [[ -n "$SOLVER_FLAG" ]]; then
        echo "[run] $* $SOLVER_FLAG" >&2
        "$BENCH" "$@" --csv --repeat 3 $SOLVER_FLAG 2>/dev/null | tail -n +2 >> "$OUTFILE" || true
    else
        local ALL_SOLVERS="apxchol cg ldlt rchol rchol_mkl rchol_par cholmod amgcl icc"
        for S in $ALL_SOLVERS; do
            "$BENCH" "$@" --csv --repeat 3 --solver "$S" 2>/dev/null | tail -n +2 >> "$OUTFILE" || true
        done
        echo "[run] $*" >&2
    fi
}

# ── Size configurations ──────────────────────────────
# Unified ranges: all families use comparable n² sizes.
# quick: n up to 300 (n²=90K)   full: n up to 1000 (n²=1M)
if $QUICK; then
    GRID_SIZES=(50 100 150 200 300)
    CHECKER_SIZES=(50 100 150 200 300)
    KAPPAS=(1 10 100 1000 10000)
    # (vertex_count, edge_probability) pairs — target ~100K edges each
    ER_CONFIGS=("500 0.8" "1000 0.2" "2000 0.05" "3000 0.022" "5000 0.008")
else
    GRID_SIZES=(50 100 200 300 500 700 1000)
    CHECKER_SIZES=(50 100 200 300 500 700 1000)
    KAPPAS=(1 10 100 1000 10000 100000)
    # (vertex_count, edge_probability) pairs — target ~1M edges each
    ER_CONFIGS=("500 0.8" "1000 0.5" "2000 0.2" "3000 0.1" "5000 0.04" "10000 0.02")
fi

# ── Bench sets ───────────────────────────────────────
IFS=',' read -ra SETS <<< "$BENCH_SETS"

for SET in "${SETS[@]}"; do
    case "$SET" in
    grid)
        echo "=== Grid graphs ==="
        for N in "${GRID_SIZES[@]}"; do
            run --graph grid --n "$N"
        done
        ;;
    checker)
        echo "=== Checkerboard graphs (varying κ) ==="
        for N in "${CHECKER_SIZES[@]}"; do
            for KAPPA in "${KAPPAS[@]}"; do
                run --graph checkerboard --n "$N" --kappa "$KAPPA" --tile 4
            done
        done
        ;;
    erdos)
        echo "=== Erdős-Rényi ==="
        for CFG in "${ER_CONFIGS[@]}"; do
            read -r N P <<< "$CFG"
            run --graph erdos --n "$N" --er-p "$P"
        done
        ;;
    tile)
        echo "=== Checkerboard (varying tile) ==="
        TILES=(1 2 4 8 16)
        $QUICK && TILES=(2 4 8)
        for TILE in "${TILES[@]}"; do
            run --graph checkerboard --n 300 --kappa 1000 --tile "$TILE"
        done
        ;;
    julia)
        JULIA_BENCH="$SCRIPT_DIR/julia/bench_laplacians.jl"
        if command -v julia &>/dev/null && [[ -f "$JULIA_BENCH" ]]; then
            echo "=== Julia (Laplacians.jl) ==="
            # Checkerboard with varying kappa
            for N in "${CHECKER_SIZES[@]}"; do
                for KAPPA in "${KAPPAS[@]}"; do
                    echo "[run] julia checker n=$N kappa=$KAPPA" >&2
                    julia --project="$SCRIPT_DIR/julia" "$JULIA_BENCH" \
                        --graph checkerboard --n "$N" --kappa "$KAPPA" --tile 4 --csv 2>/dev/null \
                        | tail -n +2 >> "$OUTFILE" || echo "[warn] Julia failed checker n=$N k=$KAPPA" >&2
                done
            done
            # Grid
            for N in "${GRID_SIZES[@]}"; do
                echo "[run] julia grid n=$N" >&2
                julia --project="$SCRIPT_DIR/julia" "$JULIA_BENCH" \
                    --graph grid --n "$N" --csv 2>/dev/null \
                    | tail -n +2 >> "$OUTFILE" || echo "[warn] Julia failed grid n=$N" >&2
            done
            # Erdős-Rényi
            for CFG in "${ER_CONFIGS[@]}"; do
                read -r N P <<< "$CFG"
                echo "[run] julia erdos n=$N p=$P" >&2
                julia --project="$SCRIPT_DIR/julia" "$JULIA_BENCH" \
                    --graph erdos --n "$N" --er-p "$P" --csv 2>/dev/null \
                    | tail -n +2 >> "$OUTFILE" || echo "[warn] Julia failed erdos n=$N p=$P" >&2
            done
        else
            echo "[skip] Julia not available or bench script missing" >&2
        fi
        ;;
    gpu)
        GPU_RCHOL_SCRIPT="$SCRIPT_DIR/run_gpu_rchol.py"
        GPU_DRIVER="$BUILD_DIR/gpu_rchol_gpu_driver"
        if [[ -x "$GPU_DRIVER" ]] && command -v python3 &>/dev/null; then
            echo "=== GPU RCHOL (Liang et al.) ==="
            for N in "${CHECKER_SIZES[@]}"; do
                for KAPPA in "${KAPPAS[@]}"; do
                    echo "[run] gpu_rchol checker n=$N kappa=$KAPPA" >&2
                    python3 "$GPU_RCHOL_SCRIPT" --build-dir "$BUILD_DIR" \
                        --graph checkerboard --n "$N" --kappa "$KAPPA" --tile 4 2>/dev/null \
                        | tail -n +1 >> "$OUTFILE" || echo "[warn] GPU RCHOL failed n=$N k=$KAPPA" >&2
                done
            done
            for N in "${GRID_SIZES[@]}"; do
                echo "[run] gpu_rchol grid n=$N" >&2
                python3 "$GPU_RCHOL_SCRIPT" --build-dir "$BUILD_DIR" \
                    --graph grid --n "$N" 2>/dev/null \
                    | tail -n +1 >> "$OUTFILE" || echo "[warn] GPU RCHOL failed grid n=$N" >&2
            done
            for CFG in "${ER_CONFIGS[@]}"; do
                read -r N P <<< "$CFG"
                echo "[run] gpu_rchol erdos n=$N p=$P" >&2
                python3 "$GPU_RCHOL_SCRIPT" --build-dir "$BUILD_DIR" \
                    --graph erdos --n "$N" --er-p "$P" 2>/dev/null \
                    | tail -n +1 >> "$OUTFILE" || echo "[warn] GPU RCHOL failed erdos n=$N p=$P" >&2
            done
        else
            echo "[skip] GPU RCHOL driver not built or python3 missing" >&2
        fi
        ;;
    mtx)
        if [[ -d "$DATA_DIR" ]]; then
            echo "=== SuiteSparse matrices ==="
            for mtx in "$DATA_DIR"/*.mtx; do
                [[ -f "$mtx" ]] || continue
                n=$(grep -v '^%' "$mtx" | head -1 | awk '{print $1}')
                if (( n > 200000 )); then
                    echo "[skip] $mtx (n=$n too large)" >&2
                    continue
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
            echo "=== GPU RCHOL Paper Matrices (C++) ==="
            for NAME in $GPU_PAPER_MATRICES; do
                mtx="$DATA_DIR/${NAME}.mtx"
                [[ -f "$mtx" ]] || { echo "[skip] $NAME.mtx not found (run scripts/download_graphs.sh)" >&2; continue; }
                run --mtx "$mtx"
            done
            # Julia solvers on GPU paper matrices
            JULIA_BENCH="$SCRIPT_DIR/julia/bench_laplacians.jl"
            if command -v julia &>/dev/null && [[ -f "$JULIA_BENCH" ]]; then
                echo "=== GPU RCHOL Paper Matrices (Julia) ==="
                for NAME in $GPU_PAPER_MATRICES; do
                    mtx="$DATA_DIR/${NAME}.mtx"
                    [[ -f "$mtx" ]] || continue
                    echo "[run] julia mtx $NAME" >&2
                    julia --project="$SCRIPT_DIR/julia" "$JULIA_BENCH" \
                        --mtx "$mtx" --csv 2>/dev/null \
                        | tail -n +2 >> "$OUTFILE" || true
                done
            fi
            # GPU RCHOL on its own paper matrices
            GPU_RCHOL_SCRIPT="$SCRIPT_DIR/run_gpu_rchol.py"
            GPU_DRIVER="$BUILD_DIR/gpu_rchol_gpu_driver"
            if [[ -x "$GPU_DRIVER" ]] && command -v python3 &>/dev/null; then
                echo "=== GPU RCHOL Paper Matrices (GPU RCHOL) ==="
                for NAME in $GPU_PAPER_MATRICES; do
                    mtx="$DATA_DIR/${NAME}.mtx"
                    [[ -f "$mtx" ]] || continue
                    echo "[run] gpu_rchol mtx $NAME" >&2
                    python3 "$GPU_RCHOL_SCRIPT" --build-dir "$BUILD_DIR" \
                        --mtx "$mtx" 2>/dev/null \
                        | tail -n +1 >> "$OUTFILE" || true
                done
            fi
        else
            echo "[skip] No data/matrices directory (run scripts/download_graphs.sh)" >&2
        fi
        ;;
    sddm2023)
        SDDM_DIR="$PROJECT_DIR/data/sddm2023"
        if [[ -d "$SDDM_DIR" ]] && ls "$SDDM_DIR"/*.mtx &>/dev/null; then
            echo "=== SDDM2023 instances (C++) ==="
            for mtx in "$SDDM_DIR"/*.mtx; do
                [[ -f "$mtx" ]] || continue
                n=$(grep -v '^%' "$mtx" | head -1 | awk '{print $1}')
                if (( n > 500000 )); then
                    echo "[skip] $(basename "$mtx") (n=$n too large for default run)" >&2
                    continue
                fi
                run --mtx "$mtx"
            done
            # Julia solvers on SDDM2023 instances
            JULIA_BENCH="$SCRIPT_DIR/julia/bench_laplacians.jl"
            if command -v julia &>/dev/null && [[ -f "$JULIA_BENCH" ]]; then
                echo "=== SDDM2023 instances (Julia) ==="
                for mtx in "$SDDM_DIR"/*.mtx; do
                    [[ -f "$mtx" ]] || continue
                    n=$(grep -v '^%' "$mtx" | head -1 | awk '{print $1}')
                    if (( n > 500000 )); then continue; fi
                    echo "[run] julia mtx $(basename "$mtx")" >&2
                    julia --project="$SCRIPT_DIR/julia" "$JULIA_BENCH" \
                        --mtx "$mtx" --csv 2>/dev/null \
                        | tail -n +2 >> "$OUTFILE" || true
                done
            fi
            # GPU RCHOL on SDDM2023 instances
            GPU_RCHOL_SCRIPT="$SCRIPT_DIR/run_gpu_rchol.py"
            GPU_DRIVER="$BUILD_DIR/gpu_rchol_gpu_driver"
            if [[ -x "$GPU_DRIVER" ]] && command -v python3 &>/dev/null; then
                echo "=== SDDM2023 instances (GPU RCHOL) ==="
                for mtx in "$SDDM_DIR"/*.mtx; do
                    [[ -f "$mtx" ]] || continue
                    n=$(grep -v '^%' "$mtx" | head -1 | awk '{print $1}')
                    if (( n > 500000 )); then continue; fi
                    echo "[run] gpu_rchol mtx $(basename "$mtx")" >&2
                    python3 "$GPU_RCHOL_SCRIPT" --build-dir "$BUILD_DIR" \
                        --mtx "$mtx" 2>/dev/null \
                        | tail -n +1 >> "$OUTFILE" || true
                done
            fi
        else
            echo "[skip] No SDDM2023 instances. Run: julia --project=benchmarks/julia scripts/generate_sddm_instances.jl" >&2
        fi
        ;;
    *)
        echo "[warn] Unknown bench set: $SET (valid: grid,checker,erdos,tile,julia,gpu,mtx,gpu_paper,sddm2023)" >&2
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
        cp "$RESULTS_DIR/plots/"*.png "$EMBED_DIR/" 2>/dev/null
    fi
    echo "Embedded results: $EMBED_DIR/"
fi

# Auto-update README with summary table
if command -v python3 &>/dev/null && [[ -f "$SCRIPT_DIR/update_readme.py" ]]; then
    python3 "$SCRIPT_DIR/update_readme.py" "$OUTFILE"
fi

echo "Done."

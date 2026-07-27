#!/usr/bin/env bash
# run_cmg.sh — invoke bench_cmg.m via Octave (preferred) or MATLAB
# Usage: bash run_cmg.sh --mtx <path> [--tol <T>] [--max_iter <N>] [--seed <S>] [--csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_M_DIR="$SCRIPT_DIR/cmg"

MTX=""
TOL="1e-8"
MAXITER="500"
SEED="42"
REGREL="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mtx)             MTX="$2";     shift 2 ;;
        --tol)             TOL="$2";     shift 2 ;;
        --max_iter|--maxiter) MAXITER="$2"; shift 2 ;;
        --seed)            SEED="$2";    shift 2 ;;
        --reg-rel)         REGREL="$2";  shift 2 ;;
        --csv)             shift ;;
        --solver)          shift 2 ;;
        *)                 shift ;;
    esac
done

if [[ -z "$MTX" ]]; then
    echo "Error: --mtx required" >&2
    exit 1
fi

# Resolve to absolute path.
MTX="$(realpath "$MTX" 2>/dev/null || echo "$MTX")"

# Try Octave first (via conda env if available), then system Octave, then MATLAB.
OCT_CMD=""
if conda env list 2>/dev/null | grep -q "^octave_env "; then
    OCT_CMD="conda run -n octave_env octave --no-gui --quiet"
elif command -v octave &>/dev/null; then
    OCT_CMD="octave --no-gui --quiet"
fi

if [[ -n "$OCT_CMD" ]]; then
    cd "$BENCH_M_DIR"
    $OCT_CMD --eval "bench_cmg('$MTX', $TOL, $MAXITER, $SEED, $REGREL)"
elif command -v matlab &>/dev/null; then
    cd "$BENCH_M_DIR"
    matlab -batch "bench_cmg('$MTX', $TOL, $MAXITER, $SEED, $REGREL)"
else
    echo "Error: neither Octave nor MATLAB available" >&2
    exit 1
fi

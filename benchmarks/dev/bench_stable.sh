#!/usr/bin/env bash
# Thermal-stable benchmark harness for apxchol.
#
# Why this exists: on a thermally constrained machine (a laptop-class CPU, say)
# session-level variance can reach 25-30%, so any perf change <20% gets lost in
# the noise. Naive 3-rep or 5-rep medians can't distinguish a 5% win from random
# jitter, and same-session A/B pairings only catch local differences (not
# cumulative session-vs-baseline regressions).
#
# Methodology:
#   * Wait for thermal cooldown to a defined ceiling before each measurement
#   * Pin to specific cores via taskset (default: all 16 cores both CCDs)
#   * N warmup runs (results discarded — primes caches, page tables, OMP teams)
#   * M measured runs collected sorted; median, P25, P75 (IQR) reported
#   * IQR/median ratio flags whether the signal is reliable (<5% = good)
#
# Usage:
#   bash benchmarks/dev/bench_stable.sh <workload> [binary] [threads]
#
# Example:
#   bash benchmarks/dev/bench_stable.sh iter0040
#   bash benchmarks/dev/bench_stable.sh grid_2000 ./benchmarks/build-cuda/benchmark
#
# Workloads:
#   iter0001..iter0045 — the LP-IPM ladder (data/ipm/...)
#   grid_500, grid_1000, grid_2000 — synthetic uniform grids
#
# Env knobs:
#   WARMUP      number of discarded warmup runs (default 3)
#   REPS        number of measured runs (default 15)
#   COOLDOWN_C  Tctl ceiling in °C before each rep (default 70)
#   POLL_SECS   sleep between cooldown checks (default 5)
#   TASKSET     core list for taskset -c (default "0-15", which is all 16 P-cores)
#   SOLVER      bench --solver name (default "apxchol_v1" with --v1-configs).
#               For competitors: "apxchol", "amgcl", "hypre_boomeramg",
#               "hypre_boomeramg_gpu",
#               "cg" (no precond Eigen CG), "ldlt", "cholmod", "rchol*".
#   CONFIG      v1-config name (default "bg+tree[vec_pool]", only used
#               when SOLVER=apxchol_v1)
#   QUIET       if set, suppress per-rep progress lines
#
# Output columns (per stage):  median  P25  P75  IQR%
#   IQR% = (P75-P25)/median × 100. Below 5% means the measurement is stable
#   and any A/B delta larger than IQR% is likely real.

set -euo pipefail

WORKLOAD=${1:-iter0040}
BINARY=${2:-./benchmarks/build/benchmark}
THREADS=${3:-16}
WARMUP=${WARMUP:-3}
REPS=${REPS:-15}
COOLDOWN_C=${COOLDOWN_C:-70}
POLL_SECS=${POLL_SECS:-5}
TASKSET=${TASKSET:-0-15}
SOLVER=${SOLVER:-apxchol_v1}
CONFIG=${CONFIG:-bg+tree[vec_pool]}
QUIET=${QUIET:-}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=openmp_affinity.sh
source "$SCRIPT_DIR/openmp_affinity.sh"
apxchol_benchmark_openmp_env "$THREADS" "$TASKSET"

# Resolve workload to bench args.
case "$WORKLOAD" in
    iter*)
        MTX_DIR="data/ipm/${WORKLOAD}"
        if [[ ! -f "$MTX_DIR/matrix.mtx" ]]; then
            echo "ERROR: workload $WORKLOAD not found at $MTX_DIR" >&2
            exit 1
        fi
        BENCH_ARGS="--mtx $MTX_DIR/matrix.mtx --kind operator --class sddm"
        ;;
    grid_*)
        N=${WORKLOAD#grid_}
        BENCH_ARGS="--graph grid --n $N"
        ;;
    *)
        echo "ERROR: unknown workload '$WORKLOAD'" >&2
        exit 1
        ;;
esac

# Read current Tctl in degrees C (integer). Returns 0 if sensors unavailable.
read_tctl() {
    local raw
    raw=$(sensors 2>/dev/null | awk '/Tctl/ {gsub(/[+°C]/,"",$2); print $2; exit}')
    if [[ -z "$raw" ]]; then
        echo 0
        return
    fi
    printf '%.0f' "$raw"
}

# Wait until Tctl drops below threshold (or hits a hard deadline).
# Returns silently if cooldown succeeds; emits one final status line if the
# deadline fires (i.e. we're running hot because of ambient/sustained load).
wait_cooldown() {
    local target=$1
    local t deadline first=1
    deadline=$((SECONDS + 60))
    while true; do
        t=$(read_tctl)
        if (( t <= target )); then
            return 0
        fi
        if (( SECONDS >= deadline )); then
            [[ -n "$QUIET" ]] || echo "  [cooldown deadline; running at Tctl=${t}°C > ${target}°C target]" >&2
            return 0
        fi
        if [[ -z "$QUIET" && $first -eq 1 ]]; then
            printf "  [cooldown waiting Tctl=%d°C → %d°C]\r" "$t" "$target" >&2
            first=0
        fi
        sleep "$POLL_SECS"
    done
}

# Run one benchmark; emit "setup solve total iters" on stdout.
# Output parser: skip header lines, take the FIRST data line whose last 8
# fields are numeric (setup, solve, total, iters, relres, fillin, us_per_nnz).
# Works for apxchol_v1, apxchol, amgcl, hypre_*, cg, ldlt, cholmod, rchol*.
run_once() {
    local solver_args
    if [[ "$SOLVER" == "apxchol_v1" ]]; then
        solver_args="--solver apxchol_v1 --v1-configs $CONFIG"
    else
        solver_args="--solver $SOLVER"
    fi
    taskset -c "$TASKSET" "$BINARY" \
        $BENCH_ARGS $solver_args --repeat 1 2>/dev/null \
        | awk 'NF >= 10 && $(NF-6) ~ /^[0-9.]+$/ && $(NF-5) ~ /^[0-9.]+$/ && $(NF-4) ~ /^[0-9.]+$/ && $(NF-3) ~ /^[0-9]+$/ {
            print $(NF-6), $(NF-5), $(NF-4), $(NF-3); exit
        }'
}

# Compute median + P25 + P75 from sorted values on stdin.
stats() {
    awk '{
        a[NR]=$1
    } END {
        n=NR
        # sort
        for (i=1; i<=n; ++i) for (j=i+1; j<=n; ++j) if (a[i] > a[j]) { t=a[i]; a[i]=a[j]; a[j]=t }
        med = (n % 2) ? a[(n+1)/2] : (a[n/2] + a[n/2+1]) / 2
        p25 = a[int(0.25 * n) + 1]
        p75 = a[int(0.75 * n) + 1]
        iqr_pct = (med > 0) ? (p75 - p25) / med * 100 : 0
        printf "%.4f %.4f %.4f %.1f", med, p25, p75, iqr_pct
    }'
}

# ── Main ───────────────────────────────────────────────────

if [[ -z "$QUIET" ]]; then
    echo "================================================================"
    echo " bench_stable.sh"
    echo "   workload : $WORKLOAD"
    echo "   binary   : $BINARY"
    echo "   threads  : $THREADS  (taskset -c $TASKSET)"
    echo "   config   : $CONFIG"
    echo "   warmup   : $WARMUP runs"
    echo "   measured : $REPS runs"
    echo "   cooldown : Tctl ≤ ${COOLDOWN_C}°C between reps"
    echo "================================================================"

    # Stability preconditions.
    boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a)
    gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)
    if [[ "$boost" == "1" ]]; then
        echo ""
        echo "WARNING: CPU boost is ENABLED. Core frequencies will swing between"
        echo "         base and turbo depending on thermal + power + sibling-core"
        echo "         state. Expect 20-30% variance even with cooldown gating."
        echo "         To stabilize:"
        echo "             sudo bash benchmarks/dev/bench_stable_setup.sh"
        echo ""
    fi
    if [[ "$gov" != "performance" ]]; then
        echo "WARNING: governor is '$gov' (expected 'performance')."
    fi
fi

# Verify binary exists.
[[ -x "$BINARY" ]] || { echo "ERROR: $BINARY not executable" >&2; exit 1; }

# Warmup phase (discarded).
for ((w=1; w<=WARMUP; ++w)); do
    [[ -n "$QUIET" ]] || printf "  warmup %d/%d\r" "$w" "$WARMUP" >&2
    run_once >/dev/null
done

# Measured phase.
declare -a setups solves totals iters
for ((r=1; r<=REPS; ++r)); do
    wait_cooldown "$COOLDOWN_C"
    row=$(run_once)
    if [[ -z "$row" ]]; then
        echo "ERROR: bench produced no output on rep $r" >&2
        exit 1
    fi
    read -r s o t i <<<"$row"
    setups+=("$s")
    solves+=("$o")
    totals+=("$t")
    iters+=("$i")
    [[ -n "$QUIET" ]] || printf "  rep %2d/%-2d : setup=%-7s solve=%-7s total=%-7s iters=%-3s\n" \
        "$r" "$REPS" "$s" "$o" "$t" "$i" >&2
done

# Aggregate.
setup_line=$(printf '%s\n' "${setups[@]}" | stats)
solve_line=$(printf '%s\n' "${solves[@]}" | stats)
total_line=$(printf '%s\n' "${totals[@]}" | stats)
iter_line=$(printf '%s\n' "${iters[@]}" | stats)

if [[ -z "$QUIET" ]]; then
    printf "\n  %-7s %8s %8s %8s %6s\n" "stage" "median" "P25" "P75" "IQR%"
    printf "  %-7s %s\n" "setup" "$setup_line"
    printf "  %-7s %s\n" "solve" "$solve_line"
    printf "  %-7s %s\n" "total" "$total_line"
    printf "  %-7s %s\n" "iters" "$iter_line"
else
    # Machine-readable single line: workload,setup_med,setup_p25,setup_p75,setup_iqr,solve_*,total_*,iters_*
    printf '%s,%s,%s,%s\n' "$WORKLOAD" "$setup_line" "$solve_line" "$total_line"
fi

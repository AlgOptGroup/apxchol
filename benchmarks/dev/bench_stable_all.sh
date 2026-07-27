#!/usr/bin/env bash
# Run bench_stable.sh across the standard workload set (the LP-IPM ladder +
# grid_2000) on both CPU and CUDA builds. Emits a compact comparison table.
#
# Assumes bench_stable_setup.sh has been run (boost off, perf governor).
#
# Usage:
#   bash benchmarks/dev/bench_stable_all.sh                   # default WARMUP=2 REPS=8
#   WARMUP=3 REPS=15 bash benchmarks/dev/bench_stable_all.sh  # tighter IQR
#
# Each (workload, build) measurement takes WARMUP+REPS bench invocations
# plus cooldown gating between reps. With defaults: ~10 reps × ~5 s × 10
# cells ≈ 10 min wall time, plus cooldowns.

set -euo pipefail

WARMUP=${WARMUP:-2}
REPS=${REPS:-8}
COOLDOWN_C=${COOLDOWN_C:-75}
WORKLOADS=(${WORKLOADS:-iter0010 iter0020 iter0030 iter0040 grid_2000})

BENCH_CPU=./benchmarks/build/benchmark
BENCH_GPU=./benchmarks/build-cuda/benchmark

# Use bench_stable.sh in QUIET mode to get a single CSV-ish line per call.
# Schema:  workload,setup_med setup_p25 setup_p75 setup_iqr,solve_*,total_*
collect() {
    local wl=$1 binary=$2
    WARMUP=$WARMUP REPS=$REPS COOLDOWN_C=$COOLDOWN_C QUIET=1 \
        bash "$(dirname "$0")/bench_stable.sh" "$wl" "$binary" 16
}

fmt_cell() {
    # Parse "setup_med setup_p25 setup_p75 setup_iqr" 4-tuple and emit
    # "median±IQR% (P25..P75)". Expects 4 space-separated floats.
    local m p25 p75 iqr
    read -r m p25 p75 iqr <<<"$1"
    printf "%.2fs ±%.0f%%" "$m" "$iqr"
}

echo "================================================================"
echo "bench_stable_all: ${#WORKLOADS[@]} workloads, CPU + CUDA"
echo "  warmup=$WARMUP reps=$REPS Tctl≤${COOLDOWN_C}°C"
boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a)
echo "  boost=$boost   governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "================================================================"
printf "\n%-10s | %-22s | %-22s | %-22s | %-22s\n" \
    "workload" "CPU setup" "CPU solve" "CPU total" "GPU total"
printf "%-10s-+-%-22s-+-%-22s-+-%-22s-+-%-22s\n" \
    "----------" "----------------------" "----------------------" "----------------------" "----------------------"

for wl in "${WORKLOADS[@]}"; do
    cpu_line=$(collect "$wl" "$BENCH_CPU")
    gpu_line=$(collect "$wl" "$BENCH_GPU")
    # Each line is: "workload,s_med s_p25 s_p75 s_iqr,o_med o_p25 o_p75 o_iqr,t_med t_p25 t_p75 t_iqr"
    IFS=',' read -r _ cpu_s cpu_o cpu_t <<<"$cpu_line"
    IFS=',' read -r _ gpu_s gpu_o gpu_t <<<"$gpu_line"
    printf "%-10s | %-22s | %-22s | %-22s | %-22s\n" \
        "$wl" "$(fmt_cell "$cpu_s")" "$(fmt_cell "$cpu_o")" "$(fmt_cell "$cpu_t")" "$(fmt_cell "$gpu_t")"
done

echo ""
echo "Notes:"
echo "  - All measurements at the fixed base clock (boost disabled by setup)."
echo "  - Cell format: median ±IQR%. IQR% < 10% means trustworthy."
echo "  - Compare future runs against THIS table; do not compare to boost-on numbers."

#!/usr/bin/env bash
# Reproduce the reference comparison table (apxchol + AMGCL + BoomerAMG
# at 16T, 3-rep median, on iter0010..iter0040 + grid_2000) with cooldown gating
# between reps to mitigate thermal-throttling variance at boost-on. Works for
# either boost-on (turbo) or locked (benchmarks/dev/bench_stable_setup.sh) state
# — the cooldown gate makes early-rep boost behavior approximately consistent
# across reps.
#
# Usage:
#   bash benchmarks/dev/bench_reference_table.sh [output_file]
#
# Output: markdown-formatted tables + a delta-vs-baseline computation if the
# user has the original numbers handy.

set -euo pipefail

OUTPUT=${1:-/tmp/bench_reference_table.txt}
COOLDOWN_C=${COOLDOWN_C:-75}
THREADS=16
TASKSET=0-15

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=openmp_affinity.sh
source "$SCRIPT_DIR/openmp_affinity.sh"
apxchol_benchmark_openmp_env "$THREADS" "$TASKSET"

# nnz for each workload (for µs/nnz calc).
declare -A NNZ=([iter0010]=7867398 [iter0020]=7867398 [iter0030]=7867398
                [iter0040]=7867398 [grid_2000]=19956001)

# Cooldown to target Tctl, deadline 90s so we don't stall forever.
wait_cool() {
    local target=$1
    local t deadline=$((SECONDS+90))
    while true; do
        t=$(sensors 2>/dev/null | awk '/Tctl/ {gsub(/[+°C]/,"",$2); print $2; exit}')
        [[ -z "$t" ]] && return 0
        local ti=${t%.*}
        (( ti <= target )) && return 0
        (( SECONDS >= deadline )) && return 0
        sleep 3
    done
}

# Run N reps + median for one (workload, binary, solver) cell.
median_run() {
    local wl=$1 bin=$2 solver=$3 envk=$4
    local args
    if [[ "$wl" == "grid_2000" ]]; then args="--graph grid --n 2000"
    else args="--mtx data/ipm/${wl}/matrix.mtx --kind operator --class sddm"
    fi
    local v1cfg=""
    [[ "$solver" == "apxchol_v1" ]] && v1cfg="--v1-configs bg+tree[vec_pool]"
    # Warmup.
    wait_cool "$COOLDOWN_C"
    env $envk taskset -c "$TASKSET" $bin $args --solver $solver $v1cfg --repeat 1 >/dev/null 2>&1
    local -a sa oa ta ia
    sa=(); oa=(); ta=(); ia=()
    for r in 1 2 3; do
        wait_cool "$COOLDOWN_C"
        local OUT
        OUT=$(env $envk taskset -c "$TASKSET" $bin $args --solver $solver $v1cfg --repeat 1 2>/dev/null \
              | awk 'NF >= 10 && $(NF-6) ~ /^[0-9.]+$/ {print $(NF-6), $(NF-5), $(NF-4), $(NF-3); exit}')
        [[ -z "$OUT" ]] && { echo "FAIL"; return 1; }
        sa+=($(echo $OUT|awk '{print $1}'))
        oa+=($(echo $OUT|awk '{print $2}'))
        ta+=($(echo $OUT|awk '{print $3}'))
        ia+=($(echo $OUT|awk '{print $4}'))
    done
    local s=$(printf '%s\n' "${sa[@]}" | sort -n | sed -n '2p')
    local o=$(printf '%s\n' "${oa[@]}" | sort -n | sed -n '2p')
    local t=$(printf '%s\n' "${ta[@]}" | sort -n | sed -n '2p')
    local i=$(printf '%s\n' "${ia[@]}" | sort -n | sed -n '2p')
    echo "$s $o $t $i"
}

fmt_cell() {
    local s=$1 o=$2 t=$3 i=$4 nnz=$5
    # ns/nnz instead of µs/nnz: keeps everything as small integers, no
    # precision loss for fast GPU solves (BoomerAMG GPU ~5 ns/nnz would be
    # 0.005 µs/nnz which rounds to "0.00" at 2 decimals). Uses awk for
    # rounding (vs bc scale=N which truncates).
    local sn=$(awk -v s=$s -v n=$nnz 'BEGIN { printf "%.0f", s * 1e9 / n }')
    local on=$(awk -v s=$o -v n=$nnz 'BEGIN { printf "%.0f", s * 1e9 / n }')
    local tn=$(awk -v s=$t -v n=$nnz 'BEGIN { printf "%.0f", s * 1e9 / n }')
    printf "%.2f/%.2f/%.2f s (%s it; %s/%s/%s ns/nnz)" $s $o $t $i $sn $on $tn
}

WORKLOADS=(iter0010 iter0020 iter0030 iter0040 grid_2000)

: > "$OUTPUT"
{
    echo "# Reference comparison table"
    echo ""
    echo "- boost=$(cat /sys/devices/system/cpu/cpufreq/boost) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "- commit: $(git log --oneline -1)"
    echo "- cooldown gate: Tctl ≤ ${COOLDOWN_C}°C between reps"
    echo "- 3-rep median, 16 threads (taskset -c 0-15), OMP_NUM_THREADS=16"
    echo ""
} | tee -a "$OUTPUT"

# CPU table.
echo "## CPU" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"
echo "| Workload  | apxchol bg+tree[vec_pool] | AMGCL | BoomerAMG [Hypre] |" | tee -a "$OUTPUT"
echo "|-----------|---------------------------|-------|-------------------|" | tee -a "$OUTPUT"
for wl in "${WORKLOADS[@]}"; do
    nnz=${NNZ[$wl]}
    row="| $wl  |"
    for sv in apxchol_v1 amgcl hypre_boomeramg; do
        R=$(median_run "$wl" "./benchmarks/build/benchmark" "$sv" "")
        if [[ "$R" == "FAIL" ]]; then
            row="$row n/a |"
            continue
        fi
        read -r s o t i <<<"$R"
        row="$row $(fmt_cell "$s" "$o" "$t" "$i" "$nnz") |"
    done
    echo "$row" | tee -a "$OUTPUT"
done
echo "" | tee -a "$OUTPUT"

# GPU table.
echo "## GPU" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"
echo "| Workload  | apxchol + GPU-PCG | AMGCL CUDA | BoomerAMG CUDA |" | tee -a "$OUTPUT"
echo "|-----------|-------------------|------------|----------------|" | tee -a "$OUTPUT"
for wl in "${WORKLOADS[@]}"; do
    nnz=${NNZ[$wl]}
    row="| $wl  |"
    for cfg in "apxchol_v1:" "amgcl_cuda:" "hypre_boomeramg_gpu:"; do
        IFS=':' read -r sv envk <<<"$cfg"
        R=$(median_run "$wl" "./benchmarks/build-cuda/benchmark" "$sv" "$envk")
        if [[ "$R" == "FAIL" ]]; then
            row="$row n/a |"
            continue
        fi
        read -r s o t i <<<"$R"
        row="$row $(fmt_cell "$s" "$o" "$t" "$i" "$nnz") |"
    done
    echo "$row" | tee -a "$OUTPUT"
done
echo "" | tee -a "$OUTPUT"
echo "DONE. Saved to $OUTPUT" >&2

#!/usr/bin/env bash
# Shell counterpart of runner_common.benchmark_openmp_env(). Source this file,
# then call apxchol_benchmark_openmp_env THREADS CPUSET before launching the
# monolithic C++ benchmark. CPUSET uses taskset syntax (for example 0-15,32-47).

apxchol_benchmark_openmp_env() {
    local threads=$1 cpuset=$2 item lo hi cpu
    local -a items cpus selected places
    local -A seen=()

    if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: benchmark thread count must be positive, got '$threads'" >&2
        return 2
    fi

    IFS=',' read -r -a items <<< "$cpuset"
    for item in "${items[@]}"; do
        if [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            lo=${BASH_REMATCH[1]}
            hi=${BASH_REMATCH[2]}
            if (( hi < lo )); then
                echo "ERROR: descending CPU range '$item'" >&2
                return 2
            fi
            for ((cpu=lo; cpu<=hi; ++cpu)); do
                if [[ -n "${seen[$cpu]:-}" ]]; then
                    echo "ERROR: duplicate CPU $cpu in '$cpuset'" >&2
                    return 2
                fi
                seen[$cpu]=1
                cpus+=("$cpu")
            done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            if [[ -n "${seen[$item]:-}" ]]; then
                echo "ERROR: duplicate CPU $item in '$cpuset'" >&2
                return 2
            fi
            seen[$item]=1
            cpus+=("$item")
        else
            echo "ERROR: invalid taskset CPU item '$item'" >&2
            return 2
        fi
    done

    if (( ${#cpus[@]} < threads )); then
        echo "ERROR: $threads threads requested from only ${#cpus[@]} CPUs: $cpuset" >&2
        return 2
    fi
    selected=("${cpus[@]:0:threads}")
    for cpu in "${selected[@]}"; do places+=("{$cpu}"); done

    export OMP_NUM_THREADS=$threads
    export OMP_DYNAMIC=FALSE
    export OMP_PROC_BIND=close
    export OMP_PLACES
    OMP_PLACES=$(IFS=,; echo "${places[*]}")
    export KMP_AFFINITY=norespect
    unset GOMP_CPU_AFFINITY
}

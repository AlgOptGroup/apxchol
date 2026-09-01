#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

: "${APXCHOL_CAMPAIGN:?}"
readonly rank=${SLURM_PROCID:?}
readonly binary="${APXCHOL_CAMPAIGN}/source/build/tests/bkz26_quality_probe"
source_commit=$(<"${APXCHOL_CAMPAIGN}/SOURCE_COMMIT")
readonly source_commit
read -r -a seeds < "${APXCHOL_CAMPAIGN}/SEEDS"
readonly seeds
readonly seed_count=${#seeds[@]}
[[ ${seed_count} -eq 5 ]]
seed_list=$(IFS=,; printf '%s' "${seeds[*]}")
readonly seed_list
readonly records_per_matrix=$((3 * seed_count))

labels=(iter0010 iter0020 iter0030 iter0040 grid_500 G3_circuit thermal2 com-Amazon)
paths=(
    /capstor/scratch/cscs/okulkov/apxchol/data/ipm/iter0010/matrix.mtx
    /capstor/scratch/cscs/okulkov/apxchol/data/ipm/iter0020/matrix.mtx
    /capstor/scratch/cscs/okulkov/apxchol/data/ipm/iter0030/matrix.mtx
    /capstor/scratch/cscs/okulkov/apxchol/data/ipm/iter0040/matrix.mtx
    /capstor/scratch/cscs/okulkov/apxchol/data/matrices/grid_500.mtx
    /capstor/scratch/cscs/okulkov/apxchol/data/matrices/G3_circuit.mtx
    /capstor/scratch/cscs/okulkov/apxchol/data/matrices/thermal2.mtx
    /capstor/scratch/cscs/okulkov/apxchol/data/matrices/com-Amazon.mtx
)

[[ ${#labels[@]} -eq 8 ]]
[[ ${#paths[@]} -eq 8 ]]
test -x "${binary}"
(cd "${APXCHOL_CAMPAIGN}" && sha256sum -c BINARY.sha256)

export OMP_NUM_THREADS=72
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export APXCHOL_FACTOR_DROP=1e-4

for ((index=rank; index<${#labels[@]}; index+=4)); do
    label=${labels[index]}
    matrix=${paths[index]}
    out="${APXCHOL_CAMPAIGN}/results/${label}"
    mkdir -p "${out}"
    [[ ! -e ${out}/DONE ]]
    test -s "${matrix}"
    {
        printf 'source_commit=%s\n' "${source_commit}"
        printf 'node=%s rank=%s cpus=%s start=%s\n' \
            "$(hostname)" "${rank}" "${SLURM_CPUS_PER_TASK}" "$(date -Is)"
        printf 'matrix=%s\nsize_bytes=%s\nseeds=%s\n' \
            "${matrix}" "$(stat -c %s "${matrix}")" "${seed_list}"
        printf 'omp_num_threads=%s\nfactor_drop=%s\n' \
            "${OMP_NUM_THREADS}" "${APXCHOL_FACTOR_DROP}"
    } > "${out}/meta.txt"

    set +e
    /usr/bin/time -v -o "${out}/time.txt" \
        timeout 600 "${binary}" "${label}" "${matrix}" "${seeds[@]}" \
        > "${out}/run.tsv" 2> "${out}/run.err"
    rc=$?
    set -e
    printf 'end=%s\nexit_code=%s\n' "$(date -Is)" "${rc}" >> "${out}/meta.txt"

    if [[ ${rc} -ne 0 ]] || \
       ! awk -F '\t' -v expected="${records_per_matrix}" -v seeds="${seed_count}" '
           NR == 1 { if (NF != 9 || $1 != "matrix" || $9 != "solve_s") exit 1 }
           NR > 1 { if (NF != 9) exit 1; rows++; arms[$3]++ }
           END {
               if (NR != expected + 1 || rows != expected ||
                   arms["gks_before"] != seeds || arms["bkz26"] != seeds ||
                   arms["gks_after"] != seeds) exit 1
           }
       ' "${out}/run.tsv"; then
        printf 'BKZ26_BROAD_RECORD_FAILED label=%s rc=%s\n' \
            "${label}" "${rc}" > "${out}/FAILED"
        exit 1
    fi
    (cd "${out}" && sha256sum meta.txt time.txt run.tsv run.err > SHA256SUMS)
    printf 'BKZ26_BROAD_RECORD_OK label=%s rows=%s/%s\n' \
        "${label}" "${records_per_matrix}" "${records_per_matrix}" \
        > "${out}/DONE"
done

printf 'BKZ26_BROAD_WORKER_OK rank=%s matrices=2/2 rows=%s/%s\n' \
    "${rank}" "$((2 * records_per_matrix))" "$((2 * records_per_matrix))" \
    > "${APXCHOL_CAMPAIGN}/WORKER_${rank}_OK"

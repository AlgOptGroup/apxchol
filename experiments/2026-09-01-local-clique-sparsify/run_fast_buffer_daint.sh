#!/bin/bash -l
# One-node repeated old/grouped/amortized exact bracket.
#SBATCH --job-name=apx-local-buffer
#SBATCH --account=prep34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --output=apx-local-buffer-%j.out
#SBATCH --error=apx-local-buffer-%j.err

set -euo pipefail

for name in OLD_SOURCE OLD_BINARY GROUPED_SOURCE GROUPED_BINARY \
            AMORTIZED_SOURCE DATA OUTPUT EXPECTED_OLD_HEAD \
            EXPECTED_GROUPED_HEAD EXPECTED_AMORTIZED_HEAD; do
    variable=APXCHOL_DAINT_${name}
    if test -z "${!variable:-}"; then
        echo "missing ${variable}" >&2
        exit 2
    fi
done

old_source=$(realpath "${APXCHOL_DAINT_OLD_SOURCE}")
old_binary=$(realpath "${APXCHOL_DAINT_OLD_BINARY}")
grouped_source=$(realpath "${APXCHOL_DAINT_GROUPED_SOURCE}")
grouped_binary=$(realpath "${APXCHOL_DAINT_GROUPED_BINARY}")
amortized_source=$(realpath "${APXCHOL_DAINT_AMORTIZED_SOURCE}")
amortized_build=${APXCHOL_DAINT_AMORTIZED_BUILD:-"${amortized_source}/build-daint"}
uenv_spec=prgenv-gnu/26.3:v1
llvm_root=/capstor/scratch/cscs/okulkov/llvm/LLVM-22.1.8-Linux-ARM64
run_path=${llvm_root}/bin:/user-environment/env/default/bin:/usr/bin:/bin

hostname -f
test "$(git -C "${old_source}" rev-parse HEAD)" = \
    "${APXCHOL_DAINT_EXPECTED_OLD_HEAD}"
test "$(git -C "${grouped_source}" rev-parse HEAD)" = \
    "${APXCHOL_DAINT_EXPECTED_GROUPED_HEAD}"
test "$(git -C "${amortized_source}" rev-parse HEAD)" = \
    "${APXCHOL_DAINT_EXPECTED_AMORTIZED_HEAD}"
for source in "${old_source}" "${grouped_source}" "${amortized_source}"; do
    git -C "${source}" diff --quiet HEAD --
    git -C "${source}" diff --cached --quiet
done
test -x "${old_binary}"
test -x "${grouped_binary}"

srun --uenv="${uenv_spec}" --view=default -N1 -n1 -c72 \
    --exclusive --exact --hint=nomultithread --cpu-bind=cores \
    "${amortized_source}/experiments/2026-09-01-local-clique-sparsify/daint_build.sh" \
    "${amortized_source}" "${amortized_build}"

if test -e "${APXCHOL_DAINT_OUTPUT}"; then
    echo "refusing to overwrite ${APXCHOL_DAINT_OUTPUT}" >&2
    exit 2
fi
mkdir -p "${APXCHOL_DAINT_OUTPUT}"

runner=${amortized_source}/experiments/2026-09-01-local-clique-sparsify/fast_buffer_screen.py
amortized_binary=${amortized_build}/tests/analyze_factor
srun --uenv="${uenv_spec}" --view=default -n4 -c72 \
    --distribution=block:block --cpu-bind=cores --mem-bind=local \
    --kill-on-bad-exit=1 \
    env PATH="${run_path}" python3 "${runner}" \
    --old-source "${old_source}" --old-binary "${old_binary}" \
    --grouped-source "${grouped_source}" --grouped-binary "${grouped_binary}" \
    --amortized-source "${amortized_source}" \
    --amortized-binary "${amortized_binary}" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}" --threads 72 --timeout 600

srun --uenv="${uenv_spec}" --view=default -N1 -n1 -c1 \
    --exclusive --exact --hint=nomultithread --cpu-bind=cores \
    env PATH="${run_path}" python3 "${runner}" --merge \
    --old-source "${old_source}" --old-binary "${old_binary}" \
    --grouped-source "${grouped_source}" --grouped-binary "${grouped_binary}" \
    --amortized-source "${amortized_source}" \
    --amortized-binary "${amortized_binary}" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}" --threads 72

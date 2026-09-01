#!/bin/bash -l
# One-node output-identical old/new local-union bracket.
#SBATCH --job-name=apx-local-union
#SBATCH --account=prep34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --output=apx-local-union-%j.out
#SBATCH --error=apx-local-union-%j.err

set -euo pipefail

: "${APXCHOL_DAINT_NEW_SOURCE:?set to the optimized fixed-source checkout}"
: "${APXCHOL_DAINT_OLD_SOURCE:?set to the verified q=.25 checkout}"
: "${APXCHOL_DAINT_OLD_BINARY:?set to its verified analyze_factor binary}"
: "${APXCHOL_DAINT_DATA:?set to the matrix root}"
: "${APXCHOL_DAINT_OUTPUT:?set to a new result directory}"
: "${APXCHOL_DAINT_EXPECTED_NEW_HEAD:?pin the optimized commit}"
: "${APXCHOL_DAINT_EXPECTED_OLD_HEAD:?pin the old q=.25 commit}"

new_source=$(realpath "${APXCHOL_DAINT_NEW_SOURCE}")
old_source=$(realpath "${APXCHOL_DAINT_OLD_SOURCE}")
old_binary=$(realpath "${APXCHOL_DAINT_OLD_BINARY}")
new_build=${APXCHOL_DAINT_NEW_BUILD:-"${new_source}/build-daint"}
uenv_spec=prgenv-gnu/26.3:v1
llvm_root=/capstor/scratch/cscs/okulkov/llvm/LLVM-22.1.8-Linux-ARM64
run_path=${llvm_root}/bin:/user-environment/env/default/bin:/usr/bin:/bin

hostname -f
test "$(git -C "${new_source}" rev-parse HEAD)" = \
    "${APXCHOL_DAINT_EXPECTED_NEW_HEAD}"
test "$(git -C "${old_source}" rev-parse HEAD)" = \
    "${APXCHOL_DAINT_EXPECTED_OLD_HEAD}"
git -C "${new_source}" diff --quiet HEAD --
git -C "${new_source}" diff --cached --quiet
git -C "${old_source}" diff --quiet HEAD --
git -C "${old_source}" diff --cached --quiet
test -x "${old_binary}"

srun --uenv="${uenv_spec}" --view=default -N1 -n1 -c72 \
    --exclusive --exact --hint=nomultithread --cpu-bind=cores \
    "${new_source}/experiments/2026-09-01-local-clique-sparsify/daint_build.sh" \
    "${new_source}" "${new_build}"

if test -e "${APXCHOL_DAINT_OUTPUT}"; then
    echo "refusing to overwrite ${APXCHOL_DAINT_OUTPUT}" >&2
    exit 2
fi
mkdir -p "${APXCHOL_DAINT_OUTPUT}"

runner=${new_source}/experiments/2026-09-01-local-clique-sparsify/fast_union_screen.py
new_binary=${new_build}/tests/analyze_factor
srun --uenv="${uenv_spec}" --view=default -n4 -c72 \
    --distribution=block:block --cpu-bind=cores --mem-bind=local \
    --kill-on-bad-exit=1 \
    env PATH="${run_path}" python3 "${runner}" \
    --new-source "${new_source}" --old-source "${old_source}" \
    --new-binary "${new_binary}" --old-binary "${old_binary}" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}" --threads 72 --timeout 600

srun --uenv="${uenv_spec}" --view=default -N1 -n1 -c1 \
    --exclusive --exact --hint=nomultithread --cpu-bind=cores \
    env PATH="${run_path}" python3 "${runner}" --merge \
    --new-source "${new_source}" --old-source "${old_source}" \
    --new-binary "${new_binary}" --old-binary "${old_binary}" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}" --threads 72

#!/bin/bash -l
# One-node, four-module Daint screen. Submission is external to this script.
#SBATCH --job-name=apx-coop-pivot
#SBATCH --account=prep34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --output=apx-coop-pivot-%j.out
#SBATCH --error=apx-coop-pivot-%j.err

set -euo pipefail

: "${APXCHOL_DAINT_SOURCE:?set to the fixed-source clone}"
: "${APXCHOL_DAINT_DATA:?set to the matrix-data root}"
: "${APXCHOL_DAINT_OUTPUT:?set to a new result directory}"
: "${APXCHOL_DAINT_EXPECTED_HEAD:?set to the source commit}"

source_dir=$(realpath "${APXCHOL_DAINT_SOURCE}")
build_root=${APXCHOL_DAINT_BUILD_ROOT:-"${source_dir}/build-daint-cooperative-pivot"}
baseline_build=${build_root}-baseline
candidate_build=${build_root}-candidate
uenv_spec=prgenv-gnu/26.3:v1
llvm_root=/capstor/scratch/cscs/okulkov/llvm/LLVM-22.1.8-Linux-ARM64
run_path=${llvm_root}/bin:/user-environment/env/default/bin:/usr/bin:/bin

hostname -f
test "$(git -C "${source_dir}" rev-parse HEAD)" = \
    "${APXCHOL_DAINT_EXPECTED_HEAD}"
git -C "${source_dir}" diff --quiet HEAD --
git -C "${source_dir}" diff --cached --quiet

srun --uenv="${uenv_spec}" --view=default -N1 -n1 -c72 \
    --exclusive --exact --hint=nomultithread --cpu-bind=cores \
    "${source_dir}/experiments/2026-09-01-cooperative-straggler-pivots/daint_build.sh" \
    "${source_dir}" "${baseline_build}" "${candidate_build}"

if test -e "${APXCHOL_DAINT_OUTPUT}"; then
    echo "refusing to overwrite ${APXCHOL_DAINT_OUTPUT}" >&2
    exit 2
fi
mkdir -p "${APXCHOL_DAINT_OUTPUT}"

runner=${source_dir}/experiments/2026-09-01-cooperative-straggler-pivots/daint_screen.py
srun --uenv="${uenv_spec}" --view=default -n4 -c72 \
    --distribution=block:block --cpu-bind=cores --mem-bind=local \
    --kill-on-bad-exit=1 \
    env PATH="${run_path}" python3 "${runner}" \
    --source "${source_dir}" \
    --baseline-binary "${baseline_build}/tests/analyze_factor" \
    --candidate-binary "${candidate_build}/tests/analyze_factor" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}" --timeout 900

srun --uenv="${uenv_spec}" --view=default -N1 -n1 -c1 \
    --exclusive --exact --hint=nomultithread --cpu-bind=cores \
    env PATH="${run_path}" python3 "${runner}" --merge \
    --source "${source_dir}" \
    --baseline-binary "${baseline_build}/tests/analyze_factor" \
    --candidate-binary "${candidate_build}/tests/analyze_factor" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}"


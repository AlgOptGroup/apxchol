#!/bin/bash -l
# Structural-only one-node Daint trace; no timing claims are made from it.
#SBATCH --job-name=apx-pivot-trace
#SBATCH --account=prep34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=00:10:00
#SBATCH --output=apx-pivot-trace-%j.out
#SBATCH --error=apx-pivot-trace-%j.err

set -euo pipefail

: "${APXCHOL_DAINT_SOURCE:?set to the fixed-source clone}"
: "${APXCHOL_DAINT_EXPECTED_HEAD:?set to the trace-script source commit}"
: "${APXCHOL_DAINT_DATA:?set to the matrix-data root}"
: "${APXCHOL_DAINT_OUTPUT:?set to a new result directory}"
: "${APXCHOL_DAINT_BINARY:?set to the tested baseline analyze_factor}"
: "${APXCHOL_DAINT_BINARY_SHA256:?set to its expected SHA-256}"

source_dir=$(realpath "${APXCHOL_DAINT_SOURCE}")
binary=$(realpath "${APXCHOL_DAINT_BINARY}")
uenv_spec=prgenv-gnu/26.3:v1
llvm_root=/capstor/scratch/cscs/okulkov/llvm/LLVM-22.1.8-Linux-ARM64
run_path=${llvm_root}/bin:/user-environment/env/default/bin:/usr/bin:/bin
runner=${source_dir}/experiments/2026-09-01-cooperative-straggler-pivots/daint_trace.py

test "$(git -C "${source_dir}" rev-parse HEAD)" = \
    "${APXCHOL_DAINT_EXPECTED_HEAD}"
git -C "${source_dir}" diff --quiet HEAD --
git -C "${source_dir}" diff --cached --quiet
test "$(sha256sum "${binary}" | cut -d' ' -f1)" = \
    "${APXCHOL_DAINT_BINARY_SHA256}"
if test -e "${APXCHOL_DAINT_OUTPUT}"; then
    echo "refusing to overwrite ${APXCHOL_DAINT_OUTPUT}" >&2
    exit 2
fi
mkdir -p "${APXCHOL_DAINT_OUTPUT}"

srun --uenv="${uenv_spec}" --view=default -n4 -c72 \
    --distribution=block:block --cpu-bind=cores --mem-bind=local \
    --kill-on-bad-exit=1 \
    env PATH="${run_path}" python3 "${runner}" \
    --source "${source_dir}" --binary "${binary}" \
    --expected-binary-sha256 "${APXCHOL_DAINT_BINARY_SHA256}" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}" --timeout 540

srun --uenv="${uenv_spec}" --view=default -N1 -n1 -c1 \
    --exclusive --exact --hint=nomultithread --cpu-bind=cores \
    env PATH="${run_path}" python3 "${runner}" --merge \
    --source "${source_dir}" --binary "${binary}" \
    --expected-binary-sha256 "${APXCHOL_DAINT_BINARY_SHA256}" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}"

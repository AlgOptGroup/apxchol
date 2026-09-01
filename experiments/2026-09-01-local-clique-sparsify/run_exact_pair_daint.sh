#!/bin/bash -l
# One-node repeated exact candidate/control bracket.
#SBATCH --job-name=apx-local-exact
#SBATCH --account=prep34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --output=apx-local-exact-%j.out
#SBATCH --error=apx-local-exact-%j.err

set -euo pipefail

for name in CONTROL_SOURCE CONTROL_BINARY CANDIDATE_SOURCE DATA OUTPUT \
            EXPECTED_CONTROL_HEAD EXPECTED_CANDIDATE_HEAD; do
    variable=APXCHOL_DAINT_${name}
    if test -z "${!variable:-}"; then
        echo "missing ${variable}" >&2
        exit 2
    fi
done

control_source=$(realpath "${APXCHOL_DAINT_CONTROL_SOURCE}")
control_binary=$(realpath "${APXCHOL_DAINT_CONTROL_BINARY}")
candidate_source=$(realpath "${APXCHOL_DAINT_CANDIDATE_SOURCE}")
candidate_build=${APXCHOL_DAINT_CANDIDATE_BUILD:-"${candidate_source}/build-daint"}
uenv_spec=prgenv-gnu/26.3:v1
llvm_root=/capstor/scratch/cscs/okulkov/llvm/LLVM-22.1.8-Linux-ARM64
run_path=${llvm_root}/bin:/user-environment/env/default/bin:/usr/bin:/bin

hostname -f
test "$(git -C "${control_source}" rev-parse HEAD)" = \
    "${APXCHOL_DAINT_EXPECTED_CONTROL_HEAD}"
test "$(git -C "${candidate_source}" rev-parse HEAD)" = \
    "${APXCHOL_DAINT_EXPECTED_CANDIDATE_HEAD}"
for source in "${control_source}" "${candidate_source}"; do
    git -C "${source}" diff --quiet HEAD --
    git -C "${source}" diff --cached --quiet
done
test -x "${control_binary}"

srun --uenv="${uenv_spec}" --view=default -N1 -n1 -c72 \
    --exclusive --exact --hint=nomultithread --cpu-bind=cores \
    "${candidate_source}/experiments/2026-09-01-local-clique-sparsify/daint_build.sh" \
    "${candidate_source}" "${candidate_build}"

if test -e "${APXCHOL_DAINT_OUTPUT}"; then
    echo "refusing to overwrite ${APXCHOL_DAINT_OUTPUT}" >&2
    exit 2
fi
mkdir -p "${APXCHOL_DAINT_OUTPUT}"

runner=${candidate_source}/experiments/2026-09-01-local-clique-sparsify/exact_pair_screen.py
candidate_binary=${candidate_build}/tests/analyze_factor
srun --uenv="${uenv_spec}" --view=default -n4 -c72 \
    --distribution=block:block --cpu-bind=cores --mem-bind=local \
    --kill-on-bad-exit=1 \
    env PATH="${run_path}" python3 "${runner}" \
    --control-source "${control_source}" --control-binary "${control_binary}" \
    --candidate-source "${candidate_source}" \
    --candidate-binary "${candidate_binary}" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}" --threads 72 --timeout 600

srun --uenv="${uenv_spec}" --view=default -N1 -n1 -c1 \
    --exclusive --exact --hint=nomultithread --cpu-bind=cores \
    env PATH="${run_path}" python3 "${runner}" --merge \
    --control-source "${control_source}" --control-binary "${control_binary}" \
    --candidate-source "${candidate_source}" \
    --candidate-binary "${candidate_binary}" \
    --data-root "${APXCHOL_DAINT_DATA}" \
    --output "${APXCHOL_DAINT_OUTPUT}" --threads 72

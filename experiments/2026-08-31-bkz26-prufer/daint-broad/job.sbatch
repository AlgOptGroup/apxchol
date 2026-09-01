#!/usr/bin/env bash
#SBATCH -A prep34
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --mem=450G
#SBATCH -t 00:30:00
#SBATCH -J apx_bkz26
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
ulimit -c 0

: "${APXCHOL_CAMPAIGN:?}"
export APXCHOL_CAMPAIGN
readonly uenv=prgenv-gnu/26.3:v1
mkdir -p "${APXCHOL_CAMPAIGN}/logs" "${APXCHOL_CAMPAIGN}/results"
[[ ! -e ${APXCHOL_CAMPAIGN}/CAMPAIGN_OK ]]
(cd "${APXCHOL_CAMPAIGN}" && sha256sum -c PACKAGE.sha256)
printf 'node=%s job=%s start=%s\n' "$(hostname)" "${SLURM_JOB_ID}" "$(date -Is)"

srun --uenv="${uenv}" --view=default -N1 -n1 -c72 --exclusive \
    --hint=nomultithread --cpu-bind=cores \
    "${APXCHOL_CAMPAIGN}/build.sh"
test -s "${APXCHOL_CAMPAIGN}/BUILD_OK"

srun --uenv="${uenv}" --view=default -N1 -n4 -c72 --exclusive \
    --kill-on-bad-exit=1 --hint=nomultithread --cpu-bind=cores --mem-bind=local \
    --output="${APXCHOL_CAMPAIGN}/logs/run-%j-%t.out" \
    --error="${APXCHOL_CAMPAIGN}/logs/run-%j-%t.err" \
    "${APXCHOL_CAMPAIGN}/run-rank.sh"

for rank in 0 1 2 3; do
    test -s "${APXCHOL_CAMPAIGN}/WORKER_${rank}_OK"
done
[[ $(find "${APXCHOL_CAMPAIGN}/results" -name DONE -type f | wc -l) -eq 8 ]]
[[ $(find "${APXCHOL_CAMPAIGN}/results" -name FAILED -type f | wc -l) -eq 0 ]]

python3 "${APXCHOL_CAMPAIGN}/summarize.py" "${APXCHOL_CAMPAIGN}"
grep -Fqx 'checked matrices 8/8; seeds 5/5; records 120/120; exact GKS brackets 40/40; converged 120/120' \
    "${APXCHOL_CAMPAIGN}/summary.txt"
find "${APXCHOL_CAMPAIGN}/results" -type f -print0 | sort -z | \
    xargs -0 sha256sum > "${APXCHOL_CAMPAIGN}/RESULTS.sha256"
printf 'BKZ26_BROAD_OK matrices=8/8 seeds=5/5 records=120/120 brackets=40/40\n' \
    > "${APXCHOL_CAMPAIGN}/CAMPAIGN_OK"
printf 'end=%s\n' "$(date -Is)"

#!/bin/bash -l
#SBATCH --job-name=cmg-native-bench
#SBATCH --account=prep34
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
ulimit -c 0
readonly cmg_package=$(pwd -P)
readonly cmg_stage=${1:?smoke or production}
readonly cmg_run=$cmg_package/runs/$cmg_stage-$SLURM_JOB_ID
readonly cmg_python=/capstor/scratch/cscs/okulkov/apxchol-grouped-stars-20260905/venv/bin/python
readonly cmg_recovered=/capstor/scratch/cscs/okulkov/cmg-general-20260905-astra-0947/cmg-general-20260905
sha256sum -c MANIFEST.sha256
(cd "$cmg_recovered" && sha256sum -c "$cmg_package/GENERATED.sha256")
mkdir -p "$cmg_run"
scontrol show job "$SLURM_JOB_ID" > "$cmg_run/job.txt"
uname -a > "$cmg_run/uname.txt"
/usr/bin/g++-14 --version > "$cmg_run/compiler.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export PYTHONDONTWRITEBYTECODE=1
export CMG_SOURCE_COMMIT=$(cat "$cmg_package/SOURCE_COMMIT")
if [[ "$cmg_stage" == smoke ]]; then
  test "$SLURM_JOB_NUM_NODES" = 1
  cmake -S "$cmg_package/benchmarks/cmg/native" -B "$cmg_run/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
    -DCMG_GENERATED_DIR="$cmg_recovered/cmg-coder-arm64-cb23c3b/generated" \
    -DFMM_SOURCE_DIR="$cmg_recovered/vendor/fast_matrix_market" \
    -DEIGEN_INCLUDE_DIR="$cmg_recovered/vendor/eigen3" > "$cmg_run/configure.log" 2>&1
  cmake --build "$cmg_run/build" -j4 > "$cmg_run/build.log" 2>&1
  readonly cmg_binary=$cmg_run/build/cmg_driver
  sha256sum "$cmg_binary" > "$cmg_run/binary.sha256"
  ldd "$cmg_binary" > "$cmg_run/ldd.txt"
  srun --nodes=1 --ntasks=1 --cpus-per-task=72 --exact --hint=nomultithread --cpu-bind=cores \
    "$cmg_python" "$cmg_package/benchmarks/cmg_native_runner.py" \
    --manifest "$cmg_package/experiments/2026-09-05-cmg-native-daint/matrices.json" \
    --binary "$cmg_binary" --output "$cmg_run/smoke" --smoke --repetitions 1 --timeout 150
else
  test "$cmg_stage" = production
  test "$SLURM_JOB_NUM_NODES" = 4
  readonly cmg_binary=${2:?verified smoke binary path}
  readonly cmg_binary_hash=${3:?verified smoke binary SHA256}
  test "$(sha256sum "$cmg_binary" | cut -d' ' -f1)" = "$cmg_binary_hash"
  export CMG_RUN_ROOT=$cmg_run CMG_PACKAGE_ROOT=$cmg_package CMG_BINARY=$cmg_binary
  srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=72 --exact --hint=nomultithread --cpu-bind=cores \
    bash -c 'exec /capstor/scratch/cscs/okulkov/apxchol-grouped-stars-20260905/venv/bin/python "$CMG_PACKAGE_ROOT/benchmarks/cmg_native_runner.py" --manifest "$CMG_PACKAGE_ROOT/experiments/2026-09-05-cmg-native-daint/matrices.json" --binary "$CMG_BINARY" --output "$CMG_RUN_ROOT/shard-$SLURM_PROCID" --shards 4 --shard "$SLURM_PROCID" --repetitions 3 --timeout 210'
fi

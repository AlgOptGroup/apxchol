#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

: "${APXCHOL_CAMPAIGN:?}"
source_commit=$(<"${APXCHOL_CAMPAIGN}/SOURCE_COMMIT")
readonly source_commit
readonly source_bundle="${APXCHOL_CAMPAIGN}/source.bundle"
readonly source_dir="${APXCHOL_CAMPAIGN}/source"
readonly build_dir="${source_dir}/build"
readonly llvm=/capstor/scratch/cscs/okulkov/llvm/LLVM-22.1.8-Linux-ARM64
readonly deps=/capstor/scratch/cscs/okulkov/apxchol/build/_deps

cd "${APXCHOL_CAMPAIGN}"
sha256sum -c PACKAGE.sha256
if [[ ! -d ${source_dir}/.git ]]; then
    git clone "${source_bundle}" "${source_dir}"
fi
git -C "${source_dir}" checkout --detach "${source_commit}"
[[ $(git -C "${source_dir}" rev-parse HEAD) == "${source_commit}" ]]
git -C "${source_dir}" diff --quiet
git -C "${source_dir}" diff --cached --quiet
"${llvm}/bin/clang++" --version | head -n 1 | grep -q '22\.1\.8'

cmake --fresh -S "${source_dir}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${llvm}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${llvm}/bin/clang++" \
    -DCMAKE_BUILD_RPATH="${llvm}/lib/aarch64-unknown-linux-gnu" \
    -DAPXCHOL_BUILD_TESTS=ON \
    -DAPXCHOL_BUILD_TOOLS=ON \
    -DAPXCHOL_BUILD_EXAMPLES=OFF \
    -DFETCHCONTENT_SOURCE_DIR_CLI11="${deps}/cli11-src" \
    -DFETCHCONTENT_SOURCE_DIR_SPDLOG="${deps}/spdlog-src" \
    -DFETCHCONTENT_SOURCE_DIR_FAST_MATRIX_MARKET="${deps}/fast_matrix_market-src" \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="${deps}/googletest-src"
cmake --build "${build_dir}" -j72 --target \
    bkz26_quality_probe unit_tests apxchol
OMP_NUM_THREADS=4 ctest --test-dir "${build_dir}" --output-on-failure \
    > "${APXCHOL_CAMPAIGN}/logs/ctest.log"

readonly binary="${build_dir}/tests/bkz26_quality_probe"
test -x "${binary}"
"${binary}" >/dev/null 2>"${APXCHOL_CAMPAIGN}/logs/usage.txt" || \
    [[ $? -eq 2 ]]
sha256sum "${binary}" > BINARY.sha256
test_count=$(ctest --test-dir "${build_dir}" -N | awk '/Total Tests:/ {print $3}')
readonly test_count
printf 'BUILD_OK commit=%s compiler=clang-22.1.8 tests=%s/%s\n' \
    "${source_commit}" "${test_count}" "${test_count}" > BUILD_OK

#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: daint_build.sh SOURCE BUILD" >&2
    exit 2
fi

source_dir=$1
build_dir=$2
llvm_root=/capstor/scratch/cscs/okulkov/llvm/LLVM-22.1.8-Linux-ARM64
llvm_runtime=${llvm_root}/lib/aarch64-unknown-linux-gnu
deps_root=/capstor/scratch/cscs/okulkov/apxchol/build/_deps

env CC="$llvm_root/bin/clang" CXX="$llvm_root/bin/clang++" \
    cmake -S "$source_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_BUILD_RPATH="$llvm_runtime" \
        -DAPXCHOL_BUILD_EXAMPLES=OFF \
        -DAPXCHOL_BUILD_TESTS=ON \
        -DAPXCHOL_BUILD_TOOLS=ON \
        -DFETCHCONTENT_SOURCE_DIR_CLI11="$deps_root/cli11-src" \
        -DFETCHCONTENT_SOURCE_DIR_FAST_MATRIX_MARKET="$deps_root/fast_matrix_market-src" \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$deps_root/googletest-src"
cmake --build "$build_dir" -j72 --target apxchol analyze_factor unit_tests
ctest --test-dir "$build_dir" --output-on-failure

binary=${build_dir}/tests/analyze_factor
readelf -d "$binary" | grep -F "$llvm_runtime" >/dev/null
if ldd "$binary" | grep -q 'not found'; then
    ldd "$binary" >&2
    exit 2
fi

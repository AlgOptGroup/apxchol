#!/usr/bin/env bash
set -euo pipefail

readonly LLVM_VERSION="22.1.8"
readonly LLVM_SHA256="922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888"
readonly LLVM_ARCHIVE="llvm-project-${LLVM_VERSION}.src.tar.xz"
readonly LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${LLVM_ARCHIVE}"

if [[ $# -ne 1 || -z "$1" || "$1" != /* ]]; then
    echo "usage: $0 ABSOLUTE_INSTALL_PREFIX" >&2
    exit 2
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "LLVM OpenMP wheels must be built natively on macOS arm64" >&2
    exit 1
fi
# libc++ std::pmr (used by the factorization workspace) ships with macOS 14.
if [[ "${MACOSX_DEPLOYMENT_TARGET:-}" != "14.0" ]]; then
    echo "MACOSX_DEPLOYMENT_TARGET must be 14.0" >&2
    exit 1
fi

readonly install_prefix="$1"
readonly work_dir="$(mktemp -d "${TMPDIR:-/tmp}/apxchol-libomp.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

archive="$work_dir/$LLVM_ARCHIVE"
curl --fail --location --retry 5 --retry-all-errors \
    --output "$archive" "$LLVM_URL"

actual_sha256="$(shasum -a 256 "$archive")"
actual_sha256="${actual_sha256%% *}"
if [[ "$actual_sha256" != "$LLVM_SHA256" ]]; then
    echo "LLVM source checksum mismatch: $actual_sha256" >&2
    exit 1
fi

tar -xf "$archive" -C "$work_dir"
source_dir="$work_dir/llvm-project-${LLVM_VERSION}.src/openmp"
build_dir="$work_dir/build"

cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$install_prefix" \
    -DCMAKE_INSTALL_NAME_DIR="$install_prefix/lib" \
    -DCMAKE_MACOSX_RPATH=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
    -DOPENMP_ENABLE_LIBOMP_PROFILING=OFF \
    -DLIBOMP_ENABLE_SHARED=ON \
    -DLIBOMP_INSTALL_ALIASES=OFF
cmake --build "$build_dir" --parallel
cmake --install "$build_dir"

libomp="$install_prefix/lib/libomp.dylib"
[[ -f "$libomp" ]] || { echo "libomp was not installed" >&2; exit 1; }
[[ "$(lipo -archs "$libomp")" == "arm64" ]] || {
    echo "libomp is not arm64" >&2
    exit 1
}
otool -D "$libomp" | grep -Fqx "$install_prefix/lib/libomp.dylib" || {
    echo "libomp has an unexpected install name" >&2
    exit 1
}
vtool -show-build "$libomp" | grep -Eq 'minos[[:space:]]+14\.0$' || {
    echo "libomp does not target macOS 14.0" >&2
    exit 1
}

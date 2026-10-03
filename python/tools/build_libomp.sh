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
platform_args=()
case "$(uname -s)" in
    Darwin)
        if [[ "$(uname -m)" != arm64 || "${MACOSX_DEPLOYMENT_TARGET:-}" != 14.0 ]]; then
            echo "macOS wheels require arm64 and MACOSX_DEPLOYMENT_TARGET=14.0" >&2
            exit 1
        fi
        platform_args=(-DCMAKE_OSX_ARCHITECTURES=arm64
                       -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET"
                       -DCMAKE_INSTALL_NAME_DIR="$1/lib" -DCMAKE_MACOSX_RPATH=OFF)
        ;;
    Linux)
        # PyPA's toolchain preserves the manylinux ABI and baseline ISA.
        manylinux-install-clang -v v22.1.8.1
        ;;
    *) echo "Unsupported wheel build platform" >&2; exit 1 ;;
esac

readonly install_prefix="$1"
readonly work_dir="$(mktemp -d "${TMPDIR:-/tmp}/apxchol-libomp.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

archive="$work_dir/$LLVM_ARCHIVE"
curl --fail --location --retry 5 \
    --output "$archive" "$LLVM_URL"

actual_sha256="$(cmake -E sha256sum "$archive")"
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
    "${platform_args[@]}" \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
    -DOPENMP_ENABLE_LIBOMP_PROFILING=OFF \
    -DLIBOMP_ENABLE_SHARED=ON \
    -DLIBOMP_OMPD_SUPPORT=OFF \
    -DLIBOMP_INSTALL_ALIASES=OFF
cmake --build "$build_dir" --parallel
cmake --install "$build_dir"

#!/usr/bin/env bash
# Install the native SDK and package that prefix without rewriting its exports.
set -euo pipefail
device=${1:?usage: build-sdk.sh CPU|CUDA|METAL [CMake arguments...]}
shift
case "$device" in CPU|CUDA|METAL) ;; *) echo "unsupported backend: $device" >&2; exit 2 ;; esac
extra=()
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    test "$device" != METAL
    platform=linux-x86_64
    checksum=(sha256sum)
    ;;
  Darwin-arm64)
    test "$device" != CUDA
    platform=macos-arm64
    checksum=(shasum -a 256)
    extra+=(-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0)
    ;;
  *) echo "unsupported SDK platform" >&2; exit 2 ;;
esac
if [[ $device == CUDA ]]; then
  # Match the portable architecture selection used by the Python wheels.
  extra+=(-DCMAKE_CUDA_ARCHITECTURES=all-major)
fi
backend=$(printf '%s' "$device" | tr '[:upper:]' '[:lower:]')
version=$(sed -n 's/^set(APXCHOL_VERSION \([0-9.]*\))$/\1/p' cmake/apxchol_version.cmake)
name="apxchol-$version-sdk-$platform-$backend"
build="build-sdk-$backend"
prefix="$PWD/stage/$name"
test ! -e "$prefix"
test ! -e "dist/$name.tar.gz"
cmake -S cmake/core -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DAPXCHOL_DEVICE="$device" -DAPXCHOL_NATIVE_ARCH=OFF -DAPXCHOL_INDEX_WIDTH=32 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX="$prefix" \
  "${extra[@]}" "$@"
cmake --build "$build" --parallel 2
cmake --install "$build"
cp docs/sdk.md "$prefix/README.md"
if [[ $device == CUDA ]]; then
  # Same CUDA 13.4 release-image notices retained by the Python wheel build.
  mkdir -p "$prefix/share/licenses/cuda" "$prefix/share/licenses/cccl"
  cp /usr/share/licenses/cuda-cudart-13-4/LICENSE "$prefix/share/licenses/cuda/"
  cp /usr/share/licenses/cccl-13-4/LICENSE "$prefix/share/licenses/cccl/"
fi
mkdir -p dist
tar -czf "dist/$name.tar.gz" -C stage "$name"
(cd dist && "${checksum[@]}" "$name.tar.gz" > "$name.tar.gz.sha256")

#!/usr/bin/env bash
set -euo pipefail
export CC=clang CXX=clang++
if [[ $(uname -s) == Linux ]]; then
  platform=linux-x86_64-cpu; device=CPU; rpath="\$ORIGIN/../lib"
  extra=(); strip_flags=(--strip-unneeded); checksum=(sha256sum)
else
  test "$(uname -m)" = arm64
  platform=macos-arm64-metal; device=METAL; rpath='@executable_path/../lib'
  extra=(-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0)
  strip_flags=(-x); checksum=(shasum -a 256)
fi
cmake -S . -B build-cli -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release \
  -DAPXCHOL_DEVICE="$device" -DAPXCHOL_NATIVE_ARCH=OFF -DBUILD_SHARED_LIBS=OFF \
  -DAPXCHOL_BUILD_TESTS=OFF -DAPXCHOL_BUILD_EXAMPLES=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=ON -DCMAKE_DISABLE_FIND_PACKAGE_spdlog=ON \
  -DCMAKE_INSTALL_RPATH="$rpath" "${extra[@]}"
cmake --build build-cli --target apxchol --parallel 2
version=$(sed -n 's/^set(APXCHOL_VERSION \([0-9.]*\))$/\1/p' cmake/apxchol_version.cmake)
name="apxchol-$version-$platform"
prefix="$PWD/stage/$name"
cmake --install build-cli --component cli --prefix "$prefix"
mkdir -p "$prefix/lib" "$prefix/share/licenses" dist
cp LICENSE "$prefix/"
# These sources are fetched by the build above; preserve all bundled-code notices.
for dep in cli11 spdlog fast_matrix_market; do
  mkdir -p "$prefix/share/licenses/$dep"
  cp build-cli/_deps/"$dep"-src/LICENSE* "$prefix/share/licenses/$dep/"
done
for dep in fast_float dragonbox ryu; do
  mkdir -p "$prefix/share/licenses/$dep"
  cp build-cli/_deps/fast_matrix_market-src/dependencies/"$dep"/LICENSE* "$prefix/share/licenses/$dep/"
done
cp build-cli/_deps/spdlog-src/include/spdlog/fmt/bundled/fmt.license.rst "$prefix/share/licenses/"
mkdir -p "$prefix/share/licenses/eigen" "$prefix/share/licenses/OpenMP"
cp build-cli/_deps/eigen-src/COPYING.* "$prefix/share/licenses/eigen/"
eigen_version=$(sed -n 's/.*set(Eigen3_VERSION \([0-9.]*\)).*/\1/p' cmake/apxchol_eigen.cmake)
sed -e "s/@Eigen3_VERSION@/$eigen_version/g" -e 's/wheel/archive/g;s/Wheels/Archives/g' \
  python/Eigen-NOTICE.txt.in > "$prefix/share/licenses/eigen/NOTICE"
strip "${strip_flags[@]}" "$prefix/bin/apxchol"
if [[ $device == CPU ]]; then
  cp -L /usr/lib64/libomp.so "$prefix/lib/"
  cp /usr/share/licenses/libomp/LICENSE.TXT "$prefix/share/licenses/OpenMP/"
  cp -R /usr/share/licenses/gcc /usr/share/licenses/glibc "$prefix/share/licenses/"
  ldd "$prefix/bin/apxchol"
else
  omp=$(brew --prefix libomp)
  cp -L "$omp/lib/libomp.dylib" "$prefix/lib/"
  omp_version=$(basename "$(cd "$omp" && pwd -P)")
  curl -fsSL --max-time 60 "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-${omp_version%%_*}/openmp/LICENSE.TXT" \
    -o "$prefix/share/licenses/OpenMP/LICENSE.TXT"
  old=$(otool -L "$prefix/bin/apxchol" | awk '$1 ~ /libomp.dylib$/ {print $1}')
  install_name_tool -change "$old" '@executable_path/../lib/libomp.dylib' "$prefix/bin/apxchol"
  install_name_tool -id '@rpath/libomp.dylib' "$prefix/lib/libomp.dylib"
  for file in "$prefix/lib/libomp.dylib" "$prefix/bin/apxchol"; do
    codesign --force --sign - "$file"
    codesign --verify --strict "$file"
    otool -L "$file"
  done
fi
tar -czf "dist/$name.tar.gz" -C stage "$name"
(cd dist && "${checksum[@]}" "$name.tar.gz" > "$name.tar.gz.sha256")

#!/usr/bin/env bash
# Build the apxchol Octave MEX extension (CPU only).
#
# Uses the same CMake core and archive dependencies as native and Python builds.
# Needs CMake, mkoctfile (octave-devel), and an OpenMP-capable C++23 compiler.
set -euo pipefail
cd "$(dirname "$0")"

# mkoctfile builds our TUs with $CXX, but it also compiles its own little
# SOVERSION stub with $CC, under -flto=auto. A gcc-compiled LTO object fed into
# a clang link silently drops __octave_mex_soversion__ and the resulting .mex
# then fails to load — so $CC has to come from the same family as $CXX.
# The OpenMP runtime follows the same rule from the other side: libomp and
# libgomp loaded into one Octave process SEGFAULT it, so link exactly one, the
# one that belongs to $CXX.
compiler_family() {
    if "$1" --version 2>/dev/null | head -n1 | grep -qi clang; then
        echo clang
    else
        echo gcc
    fi
}

CXX="${CXX:-g++}"
CXX_FAMILY=$(compiler_family "$CXX")
if [ "$CXX_FAMILY" = clang ]; then
    CC="${CC:-clang}"
else
    CC="${CC:-gcc}"
fi
CC_FAMILY=$(compiler_family "$CC")
if [ "$CC_FAMILY" != "$CXX_FAMILY" ]; then
    echo "error: CC=$CC ($CC_FAMILY) and CXX=$CXX ($CXX_FAMILY) are different compilers." >&2
    echo "       mkoctfile compiles its SOVERSION stub with \$CC under -flto=auto; a" >&2
    echo "       cross-family object drops __octave_mex_soversion__ and the .mex will" >&2
    echo "       not load. Set both, or neither." >&2
    exit 1
fi
export CC CXX

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build --target apxchol_mex --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

echo "built: $(pwd)/apxchol_mex.mex"
echo "use:   addpath $(pwd)   then   s = apxchol_solver(A); res = s.solve(b);"

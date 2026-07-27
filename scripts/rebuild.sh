#!/usr/bin/env bash
# Reliable rebuild that always picks up header edits.
#
# Why this exists: the library is almost entirely header-only under
# include/apxchol/, compiled into a few TUs (src/factorization.cpp,
# src/solve.cpp, benchmarks/src/benchmark.cpp). CMake DOES track the header
# dependencies in the .o.d depfiles, but `make` only recompiles when a
# prerequisite is *strictly newer* than the object. Editing a header and
# rebuilding within the same wall-clock second leaves header mtime == object
# mtime, so make skips the recompile and the binary links a STALE object.
# This silently produced byte-identical "no-op" benchmark results repeatedly.
#
# Fix: touch the TUs that include the mutable headers before building, so they
# are unconditionally newer than their objects and always recompile.
#
# Usage:
#   scripts/rebuild.sh            # build core (build/) + benchmark (benchmarks/build/)
#   scripts/rebuild.sh core       # only the root project (build/)
#   scripts/rebuild.sh bench      # only the benchmark (benchmarks/build/)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# TUs whose objects depend on the header-only library; touch to defeat the
# same-second mtime race described above.
TUS=(src/factorization.cpp src/solve.cpp benchmarks/src/benchmark.cpp)

what="${1:-all}"
touch "${TUS[@]}"

if [[ "$what" == "core" || "$what" == "all" ]]; then
    [[ -d build ]] && cmake --build build -j"$(nproc)"
fi
if [[ "$what" == "bench" || "$what" == "all" ]]; then
    [[ -d benchmarks/build ]] && cmake --build benchmarks/build -j"$(nproc)"
fi
echo "rebuild.sh: done ($what)"

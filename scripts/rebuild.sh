#!/usr/bin/env bash
# Incrementally build configured projects; CMake tracks header dependencies.
# Usage: scripts/rebuild.sh [all|core|bench]
# Set CMAKE_BUILD_PARALLEL_LEVEL to override the bounded default of six jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-all}" in
    all) builds=("$ROOT/build" "$ROOT/benchmarks/build") ;;
    core) builds=("$ROOT/build") ;;
    bench) builds=("$ROOT/benchmarks/build") ;;
    *) echo "Usage: $0 [all|core|bench]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Usage: $0 [all|core|bench]" >&2
    exit 2
fi
for build in "${builds[@]}"; do
    if [[ ! -f "$build/CMakeCache.txt" ]]; then
        echo "Not configured: $build; run cmake -S <source> -B $build first." >&2
        exit 1
    fi
done
for build in "${builds[@]}"; do
    cmake --build "$build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-6}"
done

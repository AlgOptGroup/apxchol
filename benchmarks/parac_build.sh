#!/usr/bin/env bash
# Build ParAC's CPU experiment driver (factorization + PCG solve) for the benchmark.
#
# This lives HERE, not in the ParAC checkout: the machine-local MKL and
# fast_matrix_market paths and the missing-transitive-include flags are ours, and
# keeping them out of their tree lets that checkout stay at upstream plus the one
# patch in benchmarks/patches/parac/.
#
#   PARAC_CHECKOUT  ParAC working tree (default: $APXCHOL_PARAC_CHECKOUT, else ~/parac)
#   MKLROOT         oneAPI MKL root      (default /opt/intel/oneapi/mkl/2026.0)
#   IOMPDIR         libiomp5 directory   (default /opt/intel/oneapi/compiler/2026.0/lib)
#   FMMINC          fast_matrix_market include dir (default ~/fast_matrix_market/include)
#
# ParAC's committed experiment/makefile points at the authors' NERSC scratch paths
# and cannot build anywhere else; -include climits/cstdint replace transitive
# includes that GCC 15 no longer pulls in.
set -euo pipefail

PARAC_CHECKOUT="${PARAC_CHECKOUT:-${APXCHOL_PARAC_CHECKOUT:-$HOME/parac}}"
MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/2026.0}"
IOMPDIR="${IOMPDIR:-/opt/intel/oneapi/compiler/2026.0/lib}"
FMMINC="${FMMINC:-$HOME/fast_matrix_market/include}"

for d in "$PARAC_CHECKOUT/experiment" "$MKLROOT/include" "$FMMINC"; do
    [ -d "$d" ] || { echo "missing: $d" >&2; exit 1; }
done

# The runner requires both benchmark-only patches.  Refuse to build a binary
# whose output would silently fall back to the old partial timing contract.
grep -Fq 'PARAC_REL_TOL' "$PARAC_CHECKOUT/experiment/driver_local.cpp" || {
    echo "missing ParAC patch 0001 (configurable tolerance)" >&2; exit 1;
}
grep -Fq 'APX factor setup time:' "$PARAC_CHECKOUT/experiment/driver_local.cpp" || {
    echo "missing ParAC patch 0002 (complete factor setup timing)" >&2; exit 1;
}
grep -Fq 'APX adapter preprocessing time:' "$PARAC_CHECKOUT/experiment/pre_process.hpp" || {
    echo "missing ParAC patch 0002 (adapter timing)" >&2; exit 1;
}

cd "$PARAC_CHECKOUT/experiment"
g++ -std=c++20 -O3 \
    -include climits -include cstdint \
    -I"$FMMINC" -I"$MKLROOT/include" \
    -o driver driver_local.cpp \
    -fopenmp \
    -L"$MKLROOT/lib" -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
    -L"$IOMPDIR" -liomp5 \
    -lpthread -lm -ldl

echo "built: $PWD/driver"
echo "run:   LD_LIBRARY_PATH=$MKLROOT/lib:$IOMPDIR MKL_NUM_THREADS=1 \\"
echo "       PARAC_REL_TOL=<calibrated> taskset -c 0-15 ./driver <matrix-amd.mtx> 16 \"\" [1]"

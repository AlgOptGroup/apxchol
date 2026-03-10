# ApxChol v0 — Archived Implementation

This directory contains the original (v0) approximate Cholesky
preconditioner implementation, fully self-contained and buildable.

## Build

```bash
cd archive/v0
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure   # 12 tests
./apxchol_v0_demo           # demo driver
```

## Algorithm

Greedy independent-set elimination with random clique sampling
(Kyng & Sachdeva 2016), implemented as a PCG preconditioner via Eigen.

## Files

- `simple_solver.h` / `simple_solver.cpp` — Core solver (IS elimination + factor build)
- `solver.h` — Abstract `lap_solver` interface
- `graphs.h` / `graphs.cpp` — Graph generators (grid, checkerboard, Erdős-Rényi)
- `mmio.h` / `mmio.cpp` — Matrix Market loader
- `main.cpp` — Demo driver (PCG with ApxChol preconditioner)
- `test_sanity.cpp` / `test_solver.cpp` — GoogleTest suite (12 tests)
- `CMakeLists.txt` — Standalone build (FetchContent for Eigen + GTest)

## Status

**Frozen** — this implementation is preserved for reference and
reproducibility. Active development has moved to a new implementation.

### Known Limitations

- Greedy IS selection (no degree ordering → weaker preconditioner)
- O(d²) clique sampling per eliminated vertex
- No parallelism
- ~37 PCG iterations at n=90K, κ=1000 (vs ~24 for Laplacians.jl AC)

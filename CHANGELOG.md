# Changelog

## Unreleased (since v0.2.2)

### Solver and factorization
- New trace-cycle clique sampler (`--sampler trace_cycle`) on CPU and GPU; GKS stays the default.
- Faster setup: one AoS graph throughout factorization, incremental degrees, and parallel residual sparsification.
- Inputs are validated by operator class; nearly-SDDM inputs are repaired by M-matrix lumping.
- Batched solves take any number of right-hand sides on CPU, CUDA and Metal, processed in panels of up to eight columns on CPU and four on GPU.
- `apxchol::solver` owns a reusable factorization for repeated solves and can be saved to and loaded from portable archives.

### GPU backends
- CUDA setup, factorization and PCG run entirely on the device, with a sync-free dataflow triangular solve and optional fp16 factor storage.
- New Apple Metal backend: setup, factorization and PCG run on the device, for single and batched solves.
- `solve_options.backend` (`--backend auto|cpu|gpu`) selects a complete CPU or GPU route and never falls back silently.

### API and CLI changes
- Breaking: runtime environment variables are replaced by explicit solver options.
- Breaking: the CUDA build no longer uses cuSPARSE or cuBLAS, and the level-set GPU triangular solve is removed.
- Breaking: forward-star and indexed pooled graph storage and the standalone analysis/setup tools are retired.
- Breaking: `APXCHOL_INDEX_WIDTH=32|64_EDGES|64` replaces the `APXCHOL_64BIT_*_INDICES` build options.
- Breaking: the CLI and bindings no longer accept an adjacency matrix as if it were the operator. Previously both solved adjacency input as the operator and failed without a clear error; now validation rejects it and the error suggests the conversion.
- New `apxchol.as_operator(M)`, `apxchol_as_operator(M)` and CLI `--convert-input`: a valid operator is used unchanged, a matrix with no negative off-diagonal entry becomes its graph Laplacian `L = D - M` (self-loops cancel), and anything else is an error.
- The CLI solves an `n` by `k` `--rhs` file as one batch, writes an `n` by `k` solution with `-o`, and logs iterations and residuals per column.
- The v1 API compatibility scope is documented.
- The partitioner and eliminator extension interfaces are experimental in 1.x and may change in minor releases.
- Python wheels for CPython 3.11+ on Linux and macOS arm64 (GPU wheels select the backend at runtime), standalone CLI archives, and MATLAB/Octave batched solves with save/load.
- Wheels declare one `License-Expression` shared by all wheel variants (BSD-3-Clause, MPL-2.0, Apache-2.0 WITH LLVM-exception, Zlib and the NVIDIA CUDA EULA; every wheel ships each license text, so macOS and CPU-only wheels also carry the CUDA EULA) and take the root LICENSE at build time; minimum scikit-build-core is 0.11.
- Relicensed from BSD-4-Clause to BSD-3-Clause.

### Benchmarks
- The benchmark harness checks every result against the original system's residual and reports setup, solve and memory separately.
- Published results now come from CSCS Daint; laptop results are archived.

### Fixes
- Reported residuals are now correct on early solve exits.
- Fixed SDDM misclassification of exact Laplacians.
- CPU solves rescale right-hand sides of extreme magnitude; operators whose scale is not representable in FP32 factor storage are still rejected.
- Parallel factorization handles a shrinking OpenMP team, and parallel vertex selection on the same graph is reproducible at a fixed thread count.

Full history: https://github.com/AlgOptGroup/apxchol/compare/v0.2.2...main

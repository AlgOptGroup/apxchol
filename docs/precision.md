# Precision and factor storage

Precision is chosen separately for graph storage, the assembled factor, the
installed triangular-solve arrays, and the outer iteration. A build with
`APXCHOL_POOL_FP32=OFF` is not an all-FP64 solver.

| Stage | Representation | Reason |
|---|---|---|
| Mutable residual graph | FP32 weights by default; optional FP64 weights | Storage/bandwidth versus rounding during factor construction |
| Assembled factor | FP32 values | Shared factor/export representation |
| CPU triangular solves | FP32 storage by default; optional scaled FP16 off-diagonals | FP16 requires efficient conversion; CPU arithmetic remains FP64 |
| GPU triangular solves | Scaled FP16 off-diagonals by default; FP32 alternative | Reduced value traffic; GPU triangular-solve arithmetic is FP32 |
| FP16 scales and diagonals | FP32 | Retain scale range and diagonal quality |
| Outer CPU/GPU PCG | FP64 vectors and reductions | Preserve the original-system iteration and residual accuracy |

`factor_options::factor_storage` selects triangular-solve storage per solver (`automatic`, `fp16`, or `fp32`). Python exposes this as `factor_storage_dtype=None`, `np.float16`, or `np.float32`. Explicit choices override the environment; C++ owns parsing and default selection.

The environment setting is `APXCHOL_SPTRSV_STORAGE=auto|float16|float32`; `auto` uses the backend default (CPU FP32, GPU FP16). When the named setting is unset or empty, the legacy `APXCHOL_SPTRSV_FP16=0|1` is accepted. Invalid values raise an error. Construction precision is independent of triangular-solve storage; Python factor exports preserve the C++ factor's FP32 dtype, while CPU solve inputs and results use FP64 arithmetic.

CPU FP16 is available on x86 CPUs with AVX/F16C, including portable wheels. The CPU is
checked at setup; an unsupported explicit FP16 request raises an error. Fat levels use the
AVX2/F16C/FMA kernel where available, with scalar F16C conversion otherwise.
CPU setup converts the CSC values once and copies them into CSR. Consuming
setup releases the input before transposing; non-consuming FP16 setup retains
that input and the stored CSC alongside the parallel transpose's scratch.
The GPU operator may use FP32 storage when the original values are exactly
representable; this is separate from narrowing a preconditioner.

## Scaled FP16 contract

For each column, let `s_j` be the maximum absolute off-diagonal factor value,
or one for an empty/zero off-diagonal column. The factor-drop threshold uses
this scale before dropping. Drop compensation runs on FP32 values before
narrowing; the retained values preserve the column sum up to rounding.

An off-diagonal is stored by native conversion of `L_ij / s_j` to FP16.
Under the normal floating-point environment this rounds to nearest, ties to even.
FP16 subnormals are flushed to signed zero. The scaled diagonal remains FP32
and absorbs the off-diagonal rounding residual. Degenerate scales fall back
to one before dropping; invalid final diagonals/scales are rejected.

Writing the scaled factor as `L D^-1`, the forward sweep returns `D y` and
the backward sweep applies the reciprocal scale squared to its input. CPU
arithmetic widens to FP64, GPU triangular-solve arithmetic to FP32. A stored
FP16 factor is not an FP16 outer solve. CPU and GPU-host preparation share the
narrowing/flush rules in `lowprec.h`; device finalization has its own CUDA
implementation and is checked by the GPU finalization tests.

## Accuracy and configuration choice

FP16 applies to the installed triangular-solve arrays, not the mutable graph
or assembled/exported factor. Graph weights use FP32 by default; FP64 pool
storage is an optional reference. Scaled FP16 graph storage is not supported.

Low-precision preconditioning can retain final solution accuracy while changing
iteration count. Select precision using original-system residuals, total solve
cost and memory on the intended inputs; smaller storage alone is not a speed
guarantee. Bitwise factor repeatability is a separate property from convergence.

A fixed factor seed determines the per-vertex random streams, but does not
promise byte-identical independently rebuilt parallel factors. Thread arrival
order can change floating-point sums, which can change sampled edges as well as
factor values. FP32 rounding may hide some differences that FP64 retains; neither
precision makes floating-point addition independent of order. FP16 factor storage
is applied later and does not make factor construction deterministic.

For the same graph, candidates, selector state and actual thread team, selection
must preserve its selected set and insertion order. Tests also retain exact
single-thread factor and same-owned-factor repeated-solve checks. Parallel FP64
rebuilds are checked for valid structure and the requested original-system
residual, rather than equal factor bytes or iteration counts between builds.
This also applies to explicit host factor construction in CUDA-enabled builds.
Low-level import of that factor onto the GPU does not make its earlier
construction repeatable; public CPU solvers keep the factor and solves on CPU.

Historical rejected precision variants and their limits are recorded in
[implementation history](implementation-history.md). They are not universal
precision guarantees. The [benchmark protocol](../benchmarks/README.md) describes
how to compare supported configurations without changing timing boundaries.

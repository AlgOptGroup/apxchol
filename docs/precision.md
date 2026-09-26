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

`APXCHOL_SPTRSV_FP16=0|1` selects triangular-solve storage at setup. CPU FP16
is available only on targets with F16C; it is not promised by portable wheels.
The GPU operator may use FP32 storage when the original values are exactly
representable; this is separate from narrowing a preconditioner.

## Scaled FP16 contract

For each column, let `s_j` be the maximum absolute off-diagonal factor value,
or one for an empty/zero off-diagonal column. The factor-drop threshold uses
this scale before dropping. Drop compensation runs on FP32 values before
narrowing; the retained values preserve the column sum up to rounding.

An off-diagonal is stored as round-to-nearest-even FP16 of `L_ij / s_j`.
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

Historical rejected precision variants and their limits are recorded in
[implementation history](implementation-history.md). They are not universal
precision guarantees. The [benchmark protocol](../benchmarks/README.md) describes
how to compare supported configurations without changing timing boundaries.

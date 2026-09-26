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

## What has and has not been established

Scaled FP16 factor storage is already implemented and tested. The temporary
assembled factor is still FP32; the large installed CSR/CSC arrays are the
ones narrowed. Keeping that temporary factor or its diagonal in FP16 would
be a different change, with different construction, export and compensation
requirements.

There is no scaled FP16 mutable graph representation in the current source.
A graph column changes as fill edges arrive, duplicates coalesce and vertices
are eliminated. Per-column compression would need a stable scale/update rule,
consistent weights at both endpoints, and tests for small weights and excess
mass. Factor-storage results do not establish graph-storage accuracy or speed.
Such a format is a numerical experiment, not a cleanup substitution.

Historical tests rejected BF16/FP24 storage alternatives, FP16 diagonals,
uncompensated dropping/rounding, and packed FP16 arithmetic for their measured
quality/performance tradeoffs. Those alternatives are already absent; see
[implementation history](implementation-history.md). These historical findings
are not universal precision guarantees or fresh measurements.

## Acceptance policy

Numerical validity means meeting the requested original-system residual;
bitwise repeatability is a separate debugging or experimental contract.
Neither equal iteration counts nor a fixed random seed proves factor identity.
Lower precision may change iterations even when the final accuracy is retained.

Remove a runtime alternative when matched evidence shows it is dominated on
its intended tested inputs, considering Setup, Solve, Total, convergence and
memory. Retain necessary portability/fallback behavior and independent numerical
references for a stated reason. Use all observed inputs and slow runs; do not
correct ordinary timings using traces. Prefer lower Solve at roughly flat Total.
A source-only cleanup can instead establish unchanged execution by comparing
complete baseline/candidate binaries under the same build configuration.

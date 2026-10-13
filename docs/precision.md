# Precision and factor storage

Precision is chosen separately for graph storage, the assembled factor, the
installed triangular-solve arrays, and the outer iteration. A build with
`APXCHOL_POOL_FP32=OFF` is not an all-FP64 solver.

| Stage | Representation | Reason |
|---|---|---|
| Mutable residual graph | FP32 weights by default; optional FP64 weights | Storage/bandwidth versus rounding during factor construction |
| Assembled factor | FP32 values | Shared factor/export representation |
| CPU triangular solves | FP32 storage by default; optional scaled FP16 off-diagonals | FP16 requires efficient conversion; CPU arithmetic remains FP64 |
| CUDA triangular solves | Scaled FP16 off-diagonals by default; FP32 alternative | Reduced value traffic; GPU triangular-solve arithmetic is FP32 |
| FP16 scales and diagonals | FP32 | Retain scale range and diagonal quality |
| Outer CPU/CUDA PCG | FP64 vectors and reductions | Preserve the original-system iteration and residual accuracy |
| Metal triangular solves | FP32 storage by default; optional scaled FP16 | Shared CUDA/Metal dataflow worker; FP32 arithmetic |
| Metal PCG | Mixed FP32 and double-float; host FP64 exit residual | Metal has no native FP64 arithmetic; see below |

`factor_options::factor_storage` selects triangular-solve storage per solver (`automatic`, `fp16`, or `fp32`). Python exposes this as `factor_storage_dtype=None`, `np.float16`, or `np.float32`. Explicit choices override the environment; C++ owns parsing and default selection.

The environment setting is `APXCHOL_FACTOR_STORAGE=auto|float16|float32`; `auto` uses the backend default (CPU/Metal FP32, CUDA FP16). Unset or empty selects `auto`. Invalid values raise an error. Construction precision is independent of triangular-solve storage; Python factor exports preserve the C++ factor's FP32 dtype, while CPU solve inputs and results use FP64 arithmetic.

CPU FP16 is available on x86 CPUs with AVX/F16C and on Apple ARM64, including portable wheels. Unsupported explicit requests fail at setup.
On x86, fat levels use the AVX2/F16C/FMA kernel where available, with scalar F16C conversion otherwise; Apple ARM64 uses native conversions.
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

CPU and CUDA solve with the scaled factor `L D^-1`: the forward sweep returns `D y` and
the backward sweep applies the reciprocal scale squared to its input. CPU
arithmetic widens to FP64, GPU triangular-solve arithmetic to FP32. A stored
FP16 factor is not an FP16 outer solve. CPU and GPU-host preparation share the
narrowing/flush rules in `lowprec.h`; device finalization has its own CUDA
implementation and is checked by the GPU finalization tests.
Metal reconstructs FP32 coefficients before multiplication; each column stores an FP32 scale and compensated diagonal.

## Apple Metal PCG

Metal uses host factor preparation with compensated dropping and optional
scaled FP16 storage. [The solver](../README.md#interfaces) handles one RHS
per call, reusing its factor and workspace.

Double-float represents a value as the sum of two FP32 numbers, giving about
48 significant bits but not IEEE FP64 precision or range. Metal uses it for
x, r, A p and fixed-tree reductions. The operator uses FP32 when all values
are exactly representable there, and double-float otherwise. Directions p,
preconditioned residuals z, alpha and beta are FP32.

Each RHS (or initial residual b - A x0) is scaled by an exact power of two,
targeting a largest magnitude in [1, 2) within FP64's exponent range. The host
computes `(tol ||b|| s)^2` in FP64; the device stops when its recursive squared
residual is strictly below that threshold. Breakdown is not convergence.
The returned residual is recomputed in FP64 against the original
operator and unmodified RHS; it can exceed tolerance even after the recursive
test passed, especially on ill-conditioned problems or at tight tolerances.

Laplacian iterations use grounded coordinates. Host FP64 centering applies to
the initial residual and returned solution, and to both sides of `apply`.
Repeatability covers the same factor and execution configuration, not different
host thread counts, devices, OS/compiler versions or independent factorizations.

Runtime compilation uses safe math and disables contraction where supported;
`metal_solver::available()` checks rounding, intermediate precision and FMA.
Nonzero operator values and unscaled factor diagonals need magnitudes in [2^-100, 2^100];
factor off-diagonals must be finite (`std::domain_error` otherwise). Factor and
operator offsets must fit signed 32-bit integers; operator inputs must have
fewer than 2^30 stored entries (`std::length_error` otherwise).

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

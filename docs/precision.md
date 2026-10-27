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

`factor_options::factor_storage` selects triangular-solve storage per solver (`automatic`, `fp16`, or `fp32`). Python exposes this as `factor_storage_dtype=None`, `np.float16`, or `np.float32`. The default is CPU/Metal FP32 and CUDA FP16; C++ owns parsing and default selection.

Construction precision is independent of triangular-solve storage; Python factor exports preserve the C++ factor's FP32 dtype, while CPU solve inputs and results use FP64 arithmetic.

CPU FP16 is available on x86 CPUs with AVX/F16C and on Apple ARM64, including
portable wheels. Unsupported explicit requests fail at setup. The CPU path
selects available conversion and SIMD instructions at runtime.

The GPU operator may use FP32 storage when the original values are exactly
representable; this is separate from narrowing a preconditioner.
CUDA rescales extreme RHS and warm-start vectors by a common power of two and
restores the solution scale on return. Intermediate triangular-solve values
must still fit FP32.

Built-in solves verify the final relative residual in FP64: CUDA checks on
the device, Metal on the host. Verification uses the original symmetric
operator (canonical lower-triangle coefficients) and unmodified RHS, before
any preconditioner lumping.

If a recursive tolerance stop fails this check, the solver can continue from
the current solution within the remaining iteration budget. Numerical breakdown
does not restart. Check the returned `converged` flag. CUDA retains one extra
FP64 RHS vector per allocated panel column for verification.

## Scaled FP16 contract

`drop_tolerance` must be in [0, 1]; zero disables threshold dropping. For each
column, let `s_j` be the maximum absolute off-diagonal factor value, or one
for an empty/zero off-diagonal column. The threshold uses this scale before
dropping. Compensation redistributes dropped mass among retained
off-diagonals before narrowing, when any remain.

An off-diagonal is stored by native conversion of `L_ij / s_j` to FP16.
Under the normal floating-point environment this rounds to nearest, ties to even.
FP16 subnormals are flushed to signed zero. The scaled diagonal remains FP32
and absorbs the off-diagonal rounding residual. Degenerate scales fall back
to one before dropping; invalid final diagonals/scales are rejected.

Let `S` be the diagonal matrix of column scales `s_j`. CPU and CUDA solve with
the scaled factor `L S^-1`: the forward sweep returns `S y`, and the backward
sweep applies the reciprocal scale squared to its input. CPU
arithmetic widens to FP64, GPU triangular-solve arithmetic to FP32. A stored
FP16 factor is not an FP16 outer solve. CPU and GPU-host preparation share the
narrowing/flush rules in `lowprec.h`; CUDA and Metal finalize device-owned factors on the device.
Metal reconstructs FP32 coefficients before multiplication; each column stores an FP32 scale and compensated diagonal.

## Apple Metal PCG

Metal factorizes [supported inputs](usage.md#native-gpu-input-and-options) on the device and prepares FP32 or scaled FP16 solve storage with compensated dropping. [The solver](usage.md#reuse-batches-and-initial-guesses) accepts vector or matrix right-hand sides, reusing its factor and workspace.

Double-float represents a value as the sum of two FP32 numbers, giving about
48 significant bits but not IEEE FP64 precision or range. Metal uses it for
x, r, A p and fixed-tree reductions. The operator uses FP32 when all values
are exactly representable there, and double-float otherwise. Directions p,
preconditioned residuals z, alpha and beta are FP32. Setup accumulates degrees and diagonal excess in double-float; graph weights and exported factor values are FP32.

Each RHS (or initial residual b - A x0) is scaled by an exact power of two,
targeting a largest magnitude in [1, 2) within FP64's exponent range. The host
computes `(tol ||b|| s)^2` in FP64; the device stops when its recursive squared
residual is strictly below that threshold. Breakdown is not convergence.
The FP64 exit check described above also covers early exits and breakdown.

Laplacian iterations use grounded coordinates. Host FP64 centering applies to
the initial residual and returned solution, and to both sides of `apply`.
Repeatability covers the same factor and execution configuration, not different
host thread counts, devices, OS/compiler versions or independent factorizations.

Runtime compilation uses safe math and disables contraction where supported;
`metal_solver::available()` checks rounding, intermediate precision and FMA.
Metal rejects inputs outside these ranges:

- Nonzero operator values and unscaled factor diagonals: magnitude in [2^-100, 2^100].
- Factor off-diagonals: finite values.
- Factor/operator offsets: signed 32-bit integers; the operator must have fewer than 2^30 stored entries.
- Positive regularization: in [2^-100, 2^100], with normal positive FP32 products when multiplied by the operator diagonal.

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

Selection is deterministic for the same graph, candidates, selector state and
actual thread team. Reusing one factor avoids the variability of rebuilding
it; moving a host factor to a GPU does not change how it was constructed.

The [benchmark protocol](../benchmarks/README.md) describes how to compare
supported configurations without changing timing boundaries.

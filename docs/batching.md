# Batching and factor reuse

When several systems share the same matrix `A`, construct one solver and reuse
it. Pass RHS vectors as columns of a matrix to solve them in one native batch:

```python
solver = apxchol.factorize(A)       # setup once
B = np.column_stack((b1, b2))
batch = solver.solve(B, rtol=1e-8)
assert batch.converged.all()        # solutions: batch.x
```

This extends the [Python quick start](../README.md#quick-start); each column
has its own convergence result. See the [Python batch interface](../python/README.md#reuse-batches-and-initial-guesses)
and [C++ matrix solve](reference/cpp.md#reusable-solver) for shapes, output storage
and initial guesses. [MATLAB/Octave](../octave/README.md) also accepts matrix RHSs.

## Accounting for setup and reuse

Setup is paid once per solver. For a workload with several calls, measure setup
plus all solves; dividing that total by the number of RHSs gives the amortized
cost per system. Measure the first batch separately from later calls because
it can allocate reusable workspace. Batch width, matrix structure and backend
affect the benefit, so compare the same RHS columns with separate scalar calls
using the same prepared solver.

Native batching is available on CPU, CUDA and Metal. The dated measurements
below cover CPU and Metal; a qualified CUDA comparison is pending.
For reuse across processes, see [saving and loading](serialization.md).

## Larger graph: YouTube

![YouTube batching speedups on Apple M1 Max](performance/youtube.svg)

This comparison uses the unweighted Laplacian of
[SNAP/com-Youtube](https://sparse.tamu.edu/SNAP/com-Youtube): 1,134,890 rows and
7,110,138 stored entries. It ran on an Apple M1 Max with a ceiling of eight CPU
workers, FP32 factor storage and tolerance `1e-8`. RHSs were normalized `L*x`
vectors from seeded random `x`.

CPU and Metal settings were chosen separately before three held-out seeds were
measured. Both selected degree quantile `0.2`. Each plotted point is the ratio
between sequential scalar calls and one matrix call at the same width, reusing
that route's factor. The bar shows the median of the three points.

| Median time | CPU | Metal |
|---|---:|---:|
| Setup + first scalar solve | 0.614 s | 0.617 s |
| First scalar solve | 0.214 s | 0.240 s |
| Reused four-column solve | 0.397 s | 0.512 s |
| Reused sixteen-column solve | 1.284 s | 2.054 s |

There was no consistent setup-plus-one winner across these seeds. Batching
helped both routes, while CPU finished the four- and sixteen-column solves
sooner on this graph.

All three YouTube blocks passed the preregistered 5% control-variation gate and
all 768 RHS checks passed. The full campaign also included Amazon and Grid5000;
only zero and one of their three blocks passed that gate, respectively. They
are omitted from the charts rather than presented as reliable speedups.

Measured on **2026-10-25**, at
[revision `871e7ee`](https://github.com/AlgOptGroup/apxchol/commit/871e7eeedfc70d51f3ea7fa043a5f7ca6eddeeb0).
[Download timings, settings and provenance](performance/youtube-20261025.json).
These are native C++ API timings.

See [benchmarking and reproducibility](../benchmarks/README.md) for the timing
contract and larger comparisons.

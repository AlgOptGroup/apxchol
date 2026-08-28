# Huge generated-grid scaling on Grace

Status: **size helps strongly up to 16M vertices, then scaling plateaus; the
missing high-thread work is no longer IS selection alone**.

This memo recovers three completed but previously unconsolidated Daint
campaigns.  The exact source is `522514c9df61324407ec959da786dce4e4a17531`
(an ancestor of current main): Clang 22 CPU and the supported CSCS CUDA/GCC
build, `bg+tree[vec_pool_aos]`.  Generated 2D grids have side 4000 (16M
vertices, 79.984M operator nnz) and side 8000 (64M vertices, 319.968M nnz).

Denominator and integrity:

- calibration: 4/4 records;
- component profile: 4/4 records;
- production setup sweep: 60/60 records (2 sizes x CPU/GPU x 5 thread counts
  x 3 seeds), plus 4/4 converged full solves;
- all 72 record-local manifests checked, 144/144 payload hashes matched.

The setup sweep used one exclusive 288-core GH200 node and four NUMA-local
ranks.  Values below are medians of three seeds.  The `maxiter=1` setup runs
are for timing only; the separate full solves establish convergence.

## CPU scaling

| threads | 16M setup (s) | speedup | 64M setup (s) | speedup |
|---:|---:|---:|---:|---:|
| 1 | 16.578 | 1.00x | 67.386 | 1.00x |
| 8 | 3.180 | 5.21x | 12.877 | 5.23x |
| 16 | 2.223 | 7.46x | 8.817 | 7.64x |
| 36 | 1.635 | 10.14x | 6.555 | 10.28x |
| 72 | 1.501 | **11.05x** | 6.089 | **11.07x** |

Yves's size hypothesis is partly right: these grids scale much better than the
250k-vertex grid_500 campaign (about 5x at T=72, on an earlier source).  But
the same-source 16M-to-64M comparison is decisive: multiplying the matrix by
four changes T=72 speedup only 11.05x to 11.07x.  The T=36-to-T=72 increment is
only 1.09x / 1.08x.  More regular-grid work no longer exposes additional CPU
parallelism beyond this point.

The one-run component profile explains the plateau:

| component, T=1 / T=72 | 16M speedup | 64M speedup |
|---|---:|---:|
| complete setup | 11.96x | 11.49x |
| make graph | 7.25x | 6.77x |
| find partition | **26.81x** | **26.17x** |
| prune | 50.65x | 50.53x |
| select | 21.62x | 19.90x |
| collect | 0.98x | 1.11x |
| elimination | **16.76x** | **15.60x** |
| assembly | 7.03x | 6.87x |
| SpTRSV setup | 4.71x | 4.11x |

On the 64M T=72 run, elimination and SpTRSV setup each consume about 26% of
setup; make-graph and partitioning are about 14% each and assembly 11%.
Partitioning is therefore not the main scaling failure on sufficiently large
regular graphs.  Its serial collect portion is visible but only 3.8% of total
setup.  The largest remaining targets are elimination memory traffic and the
factor transpose/schedule build, followed by graph construction and assembly.

## CPU/GPU build comparison

This is not a compiler-controlled frontend A/B: the CPU build uses Clang and
the CUDA build uses the CSCS-supported GCC host compiler.  It is nevertheless
operationally useful.  The CUDA build's setup is faster at T=1 (64M: 55.22 vs
67.39 s), but slower by T=8 (18.46 vs 12.88 s) and remains slower at T=72
(7.02 vs 6.09 s).  This agrees with the current warning that the GPU block
frontend can lose on regular grids; graph size alone does not make it win.

The full T=72 solves converge below `1e-8`:

| size | CPU setup / solve / total | GPU setup / solve / total | iterations |
|---|---:|---:|---:|
| 16M | 1.526 / 1.184 / 2.709 s | 1.597 / 0.433 / 2.030 s | 53 / 54 |
| 64M | 5.925 / 4.831 / 10.757 s | 6.686 / 1.623 / 8.309 s | 53 / 53 |

GPU solve throughput wins enough to overcome the slower high-thread setup for
one RHS on these two grids.

## Decision

Do not use “our matrices are too small” as the general explanation for poor
scaling.  Larger inputs improve partition/elimination scaling substantially,
but 16M and 64M regular grids have the same overall curve and both saturate
after 36 threads.  A current-main rerun is warranted after production defaults
settle, especially on large irregular graphs; it should profile the same named
components rather than report setup alone.

Verified local copies:

- `/tmp/huge-grid-v3-results`;
- `/tmp/huge-grid-profile-v1-results`;
- `/tmp/huge-grid-production-v1-results`;
- `/tmp/current-grid500-timing-results` (separate earlier-source size anchor).

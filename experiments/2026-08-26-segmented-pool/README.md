# Segmented adjacency pool: accepted

Stable mmap-backed segments replaced a relocating adjacency vector. Each
vertex slab stays within one segment; compact offsets identify storage.
Workers claim and copy their own growth, followed by publication, avoiding a
collective relocation of the whole pool. Large aggregate claims are split
across segments. Only needed segments are mapped; the design does not require
an enormous reserved virtual address range.

Historical Daint job 4545868 compared nine matrices at T=36/72, covering
18 cells. The report records matching factors and solve-quality checks in
18/18 cells, with relative residuals below $10^{-8}$. Candidate/baseline
geometric-mean ratios were:

| Setup | Elimination | Total |
|---:|---:|---:|
| 0.9592 | 0.8592 | 0.9732 |

Setup speedup from 36 to 72 threads rose from 1.0831 to 1.1047, while
elimination self-scaling was essentially unchanged. The reported RSS ratio
0.8012 was dominated by small graphs and is not a universal 20% memory-saving
claim.

A giant virtual reservation failed under address-space limits. Pointer-based
arenas increased RSS, and cached/direct pointer variants had mixed timings or
setup regressions. These alternatives were not retained.

The later allocation policy distinguishes lazy slabs from fully sized output
buffers: lazy storage requests transparent huge pages only when reported PMD
granularity is at most 2 MiB. This avoids the first-touch inflation associated
with 512 MiB pages on Daint. See current [AGENTS.md](../../AGENTS.md) for the
allocation contract.

Use the root build and pooled-graph tests for current correctness validation.
No standalone campaign harness is tracked here. The [full historical
record](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/experiments/2026-08-26-segmented-pool/README.md) preserves source identity, measurements, and rejected designs;
it does not establish current integrated performance.

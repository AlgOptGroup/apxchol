# Weighted Prüfer, cycles, and GKS

**The original eight-matrix benchmark favored GKS over weighted Prüfer.
Cycle-based sampling reduces iterations, with extra fill and setup cost.** The current
[CPU/GPU benchmarks](https://github.com/AlgOptGroup/apxchol/tree/main/benchmarks/daint)
therefore retain alternatives rather than declaring one universal winner.

Here $a_i$ denotes the weight from the eliminated vertex to neighbor $i$;
the table assumes positive weights.

| Rule | Construction on a degree-$d$ elimination star |
|---|---|
| GKS | Order neighbors by weight; each independently chooses one later parent, proportional to that parent's weight. Emits $d-1$ edges. |
| Weighted Prüfer / CAST-1 | Weight-proportional Prüfer symbols generate a dependent tree; inverse-inclusion edge weights preserve the exact clique in expectation. |
| CAST-2 | Generate one tree on two half-weight copies of each neighbor, then contract the copies. Contraction can create cycles and merge parallel edges. This changes the law; it is neither averaging two trees nor AC(2). Compare measured fill as well as iterations. |
| Trace-cycle (`trace_cycle`) | Choose a heavy suffix by an exact local Frobenius objective, sample a uniform cycle on it, and attach lighter vertices with probabilities proportional to $a_i+a_j$. |
| Heavy-core K2 (`heavy_core_k2`) | Use a heavy-core cycle and coordinate attachments through two heavy receivers. Its dependence differs from trace-cycle's independent light attachments. |

The cycle rules emit at most $d$ edges; degrees below three use GKS. Both are
explicit options in current CPU and GPU-owned setup; GKS remains the default.

The [actual-family comparison](RECONCILIATION.md) joins stored fill, iterations,
and local variance from the **same 16 factors**, covering a grid, IPM, a social
graph, and Chimera. Prüfer's lower summed Frobenius variance on the grid and
social graph still accompanies more iterations. Common-star spectral estimates
also favor Prüfer there: these local aggregates do not yet predict the solve.

In the six-matrix timing screen, CPU trace-cycle used 35% fewer iterations and
27% less solve time than GKS, but 20% more setup time and 12% more stored entries;
its one-RHS total was 4% slower. Heavy-core K2 was similar. The current GPU-owned
cycle ports had a much larger setup penalty. See the
[route-specific measurements and qualifications](https://github.com/AlgOptGroup/apxchol/blob/main/benchmarks/daint/SAMPLERS.md).
These are measured tradeoffs, not an argument to replace every GKS row.

Why can CAST's positive examples and our negative results coexist? Sampling
changes both the current clique error and later elimination stars; ordering and
matrix family matter. The original eight-matrix/five-seed experiment raised
mean iterations from 44.93 to 59.90 with Prüfer. The small Spielman construction
was a useful positive diagnostic from the CAST study, dominated by degree-three
eliminations; it does not represent the larger stars in IPM and social graphs.
[Original campaign](appendix/historical-benchmarks.md) ·
[Controlled CAST comparisons](appendix/reconciliation-details.md).

[The model and gaps to optimum](SAMPLING-MODEL.md) distinguish Frobenius variance,
population spectral variance, and per-draw spectral error. Numerical degree-five
optima show that better topology-dependent weights can substantially improve
GKS, while closing the larger cycle-rule gaps requires changing its topology
law. These local certificates do not establish a PCG-optimal sampler.

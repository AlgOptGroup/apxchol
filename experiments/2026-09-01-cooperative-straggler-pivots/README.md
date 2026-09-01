# Cooperative straggler pivots

Status: research probe; no production default or public option change.

## Question

The outer elimination loop owns one selected pivot at a time.  Near the end of
a round, a work-sized OpenMP team can have only one large pivot left while its
other workers wait at the fused round barrier.  Can those otherwise-idle
workers draw independent GKS tree sources for that pivot without changing the
factor?

## Exact mechanism

The probe is compiled with `APXCHOL_COOPERATIVE_PIVOTS_PROBE`.  Whole-pivot
ownership and the outer dynamic schedule stay unchanged.  It is considered
only when a round has at most one more selected pivot than workers.  Every
pivot first gathers and deduplicates its live neighbours.  The owner that
observes the final prepared pivot compares its accumulated gathered work with
the exact round target `ceil(total gathered neighbours / workers)`.

If that owner is overloaded, the GKS plan has one valid draw for every source,
and there are at least as many sources as workers, it exposes contiguous source
ranges as OpenMP tasks.  Source `i` writes fixed output slot `i`; its splitmix
stream is jumped to the state reached after exactly `i` serial draws.  Task
order therefore cannot change a bit of the sampled clique.  A taskgroup keeps
the neighbour ordering, inverse-CDF directory and output buffer alive until
all helpers finish.

`APXCHOL_COOPERATIVE_PIVOT_TRACE=1` prints only successful assists.  It is a
probe-only diagnostic and, unlike `APXCHOL_ROUND_TRACE`, does not run the
expensive per-round LPT census.

## Correctness gate

`TreeSampler.CooperativeSourceRangesAreByteIdentical` compares every emitted
edge byte against the serial GKS sampler over three degree/weight families and
three seeds.  `CooperativePivot.SharedNeighborRoundProducesExactFactorBytes`
forces a two-worker/three-pivot assist, then compares the complete permutation
and CSC factor bytes with cooperation disabled.

## Daint decision protocol

Build the same fixed source twice with Clang 22: macro off for the baseline and
macro on for the candidate.  Run baseline/candidate/baseline brackets at 36 and
72 physical Grace cores, seeds 1/42/97, over com-Orkut, com-LiveJournal,
grid_2000, iter0040 and com-Amazon: 90 records and 30 exact-output cells.

Advance only if the trace proves real helper activation and bracketed
elimination/setup improves beyond the measured baseline drift.  Equal factor
nnz, stored nnz, iterations and residual text are required in every cell; the
unit tests provide the byte-level gate.  If no representative input activates,
use the structural trace to redesign the readiness rule rather than adding a
matrix or degree threshold.


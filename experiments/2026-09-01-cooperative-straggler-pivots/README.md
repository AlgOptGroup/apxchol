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

## Daint result: close the final-prepared-owner rule

Daint job `4573421` completed in 10:02, including two clean builds and both
test suites.  The macro-off build passed 321/321 tests and the candidate passed
322/322; three platform-dependent low-precision/memory tests were skipped in
each build.  The campaign checked 90/90 records and 30/30 exact-output cells;
all 90 solves converged.  Independent download verification passed 90/90 raw
log SHA-256 checks under `/tmp/apxchol-cooperative-pivot-daint-r1`.

The rule activated only on com-Orkut: 6/30 cells, covering every Orkut seed and
thread count.  It did real work there, but far too little of the factorization:

| threads | assisted rounds | sampled sources | max planned peer helpers | sources / raw factor nnz | setup | elimination | fused compute+apply | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 500 | 846,064 | 2 | 0.075--0.089% per seed | 0.9999x | 1.0047x | 1.0042x | 1.0000x |
| 72 | 476 | 802,664 | 2 | 0.076--0.084% per seed | 1.0068x | 0.9985x | 0.9955x | 1.0061x |

No activated stage improved consistently over the three seeds.  The two
baseline setup arms themselves differed by 1.0034x at T=36 and 1.0069x at
T=72, placing all candidate differences inside the null bracket.  The
non-activating cells also show why their apparent aggregate gains must not be
credited to this mechanism: short grid/IPM/Amazon setups had baseline drift as
large as 1.10x despite exact output and zero cooperative rounds.

**Verdict:** do not merge or broaden this implementation.  Waiting until the
last prepared owner is safe and exact, but it exposes at most a two- or
three-worker tail in the rounds that pass the existing work-sized-team rule.
On the only activating graph, that tail contains less than 0.1% of the total
sample-source work, so even a free perfect split cannot repair setup scaling.

A successor needs a different scheduling boundary, not a looser graph or
degree gate: workers should continue claiming whole pivots while any remain,
then be able to help unfinished source streams of *any* live pivot.  Before
implementing that larger state machine, use the existing
`APXCHOL_ROUND_TRACE`/`analyze_elimination_stragglers.py` census to measure the
all-round divisible-work upper bound.  If that bound is small, close
intra-pivot sampling entirely; if it is material, prototype a custom
whole-pivot-first shared work queue rather than another last-pivot heuristic.

The trace field historically named `max_helpers` is only
`planned participants - 1`; OpenMP may execute every generated task on the
owner.  It is therefore an upper bound on possible peer participation, not an
executor census.  A read-only call-path audit also found a reachable miss: an
earlier owner can start a large serial sample before the final small pivot is
prepared, and can never reconsider cooperation.  This reinforces the closure;
neither successful peer execution nor selection of the true straggler is
established by the current trace.

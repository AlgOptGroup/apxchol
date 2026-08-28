# Sparse-batched incremental degree experiment

Status: **shipped default candidate; Grace T=36/72 gate passed**.

## Mechanism

The ordinary block-greedy degree prepass scans and compacts every active
adjacency slab in every round. A June prototype replaced those scans with
exact incremental degrees, but lost at high thread counts because every
eliminated incidence performed a contended atomic decrement.

This branch changes that failure mechanism:

- the first ordinary prepass initializes exact vertex-indexed degrees;
- selected vertices receive prefix-summed slices of one exact-size endpoint
  stream, so dynamically scheduled elimination fills it without atomics or
  append-vector over-allocation;
- a deterministic collective byte-radix sorts the entire stream, then each
  surviving endpoint receives one non-atomic run-length decrement globally;
- fill increments reuse the directed-AoS apply histogram and require no extra
  atomics;
- candidate filtering reads cached degrees in O(active vertices), skipping
  the all-adjacency scan.

Skipping pruning indefinitely retains dead incidences inside survivor slabs,
eventually causing slab growth and excessive RSS. The production-shaped path
therefore tracks that debt exactly. A normal prune refresh is performed when

```text
dead survivor incidences / live incidences >= 0.10.
```

Dead incidences retired when a vertex itself is selected are subtracted from
the debt. Thus the rule bounds traversal amplification and resident pool bloat
using residual-observed traffic, rather than a round count or input-size
heuristic. Local threshold probes on coPapersDBLP found 0.10 to be the Pareto
point: 0.05 saved only about 25 MB more peak RSS while adding about 150 ms;
0.25 retained about 112 MB more while saving only about 27 ms.

AUTO activates on the first host-resident completed round that satisfies:

```text
(active vertices * average live degree) / selected-pivot degree work >= 16.
```

The numerator estimates the next full-prune traffic; the denominator is the
incremental update stream. The measured crossover was near 10, so 16 keeps a
margin. The measured break-even ratio 10 is a lower hysteresis boundary: a
sample below 10 classifies the run OFF permanently, while a ratio in `[10,16)`
remains pending as the residual evolves. AUTO additionally requires at least
256 endpoint items per member of
the actual elimination team. This is one item per bin of each byte-radix
histogram; the four passes multiply both the endpoint scans and fixed histogram
work, so they do not multiply the per-pass amortization floor. This prevents a
high traffic ratio from selecting the collective when there is too little
parallel update work to amortize its barriers and histogram prefix. The lower
boundary preserves no-op behavior on grid_2000, Amazon, YouTube and iter0040
(first ratios 2.02, 8.96, 9.92 and 9.30), while coAuthorsDBLP at 15.17 remains
pending and later activates. A sample that passes the traffic gate but misses
the work-per-worker gate is also retried, which is the Kron shape. If the CUDA
block frontend owns earlier rounds, maintenance remains off until its permanent
CPU handoff. Once both activation gates pass, updates keep the cache exact and
the choice is one-way for the remaining main-selector phase.

Production controls:

- unset or `APXCHOL_INCREMENTAL_DEGREE_SPARSE=auto` uses AUTO;
- `=0` disables the mechanism and restores full pruning every round;
- `=1` forces the mechanism for diagnostics;
- `APXCHOL_INCREMENTAL_DEGREE_TRACE=1` prints gate, refresh and reduction data.

## Correctness

- Clang CPU 316/316, GCC CPU 316/316 and CUDA 323/323 tests pass.  The two
  incremental-factor equivalence tests also pass in a no-OpenMP build, and the
  three applicable tests pass with 64-bit node indices.
- `VecPoolAos.AutoIncrementalDegreesPreserveTheFactorByteForByte` compares the
  complete permutation, CSC offsets, row indices and values on a parallel
  128-clique construction. It compares rollback, unset-default and explicitly
  named AUTO. At 4,096 candidates the forced CUDA block frontend owns a real
  first round (128 pivots), then hands 2,013 candidates to the CPU; AUTO
  initializes its host cache only at that handoff. The full factor remains
  byte-identical.
- `SetupDiagnostics.EndpointRadixSortMatchesComparisonSort` checks the
  eight-worker collective radix path on 20,000 deterministic IDs including
  duplicates and both extrema.
- `VecPoolAos.IncrementalDegreeCacheDoesNotLeakIntoBkResidualLoop` activates
  incremental block-greedy on 64 small clique components, then hands a larger
  clique to the BK residual loop and compares the complete baseline/AUTO factor
  byte for byte. BK bypasses the shared degree-prepass by contract; the test
  pins that integration boundary.
- The final local bracket checked 81/81 runs over nine matrices at T=16. Every
  AUTO `nnz(L)` matched both adjacent baseline arms.

## Local timing

Clang 22, LLVM OpenMP, directed AoS pool, block-greedy + tree sampling, seed 42.
Each AUTO run is normalized to the geometric mean of adjacent unchanged runs;
three repetitions per matrix, with matrix order rotated between repetitions.

| matrix | first-round ratio | gate | setup ratio |
|---|---:|---:|---:|
| grid_500 | 2.03 | off | 1.035 |
| grid_2000 | 2.02 | off | 1.000 |
| com-Amazon | 8.96 | off | 0.956 |
| com-Youtube | 9.92 | off | 1.002 |
| as-Skitter | 32.33 | on | **0.859** |
| kron_g500-logn16 | 777.11 | on | **0.900** |
| coPapersDBLP | 141.73 | on | **0.541** |
| com-LiveJournal | 44.23 | on | **0.829** |
| com-Orkut | 61.99 | on | **0.815** |

- all-nine setup geomean: **0.868**;
- activated-cell geomean: **0.776**;
- no-op-cell geomean: **0.998**;
- every activated cell won in every repetition; weakest ratio **0.940**.

The no-op spread (0.892 to 1.089 across individual brackets) is the measured
local noise floor: when the gate stays off, the factorization path is unchanged.

### Scaling interpretation

A fixed-thread-count bracket at T=1/2/4/8/16 separates absolute work reduction
from self-speedup:

| matrix | AUTO/base T=1 | AUTO/base T=16 | baseline speedup | AUTO speedup |
|---|---:|---:|---:|---:|
| as-Skitter | 0.800 | 0.794 | 3.54x | 3.56x |
| coPapersDBLP | 0.492 | 0.549 | 4.20x | 3.77x |
| com-LiveJournal | 0.731 | 0.941 | 4.87x | 3.79x |

The candidate is faster in every cell, but does not generally improve the
parallel scaling factor itself: it removes the full prune, which is one of the
better-scaling stages, and exposes selection/elimination as the new limit. The
36/72-core Grace gate is therefore load-bearing rather than ceremonial. Raw
logs: `/tmp/apxchol-incremental-degree-scaling-20260827`.

Orkut is the important larger-input exception. A separate 15/15 exact bracket
used one seed at T=1/2/4/8/16, with each AUTO run normalized to the geometric
mean of adjacent unchanged runs:

| threads | AUTO / baseline setup | baseline self-speedup | AUTO self-speedup |
|---:|---:|---:|---:|
| 1 | 0.7974 | 1.000x | 1.000x |
| 2 | 0.8120 | 1.519x | 1.492x |
| 4 | 0.8001 | 2.439x | 2.431x |
| 8 | 0.7051 | 3.516x | 3.977x |
| 16 | 0.7137 | 4.265x | 4.765x |

Every arm at a given thread count produced the same factor size. AUTO peak RSS
was 4.6--5.7% above its adjacent baselines. This supports the narrower version
of the large-matrix hypothesis: enough repeated all-adjacency traffic lets the
incremental path improve both absolute work and the T=16 curve. It does not
establish asymptotic scaling, and the 36/72-core Grace gate remains required.
Raw evidence: `/tmp/apxchol-incremental-orkut-scaling-20260827` (results TSV
SHA-256 `cd3e7d270dcbc881e77d6cfe31c1e122fbe728604a6ab22227b7f5ca50d65989`).

Two follow-ups were measured and rejected:

- sizing elimination teams by the tree sampler's `sum(d log d)` sort estimate
  instead of `sum(d)` was exact in 9/9, but setup ratios were 0.985 / 0.972 /
  1.009 on Skitter / coPapers / LiveJournal;
- accumulating a dirty endpoint set and refreshing only those slabs was exact
  in 27/27, but its irregular traversal made the refresh stage **1.071x** and
  setup only 0.990x overall versus the simpler contiguous full-active refresh
  (LiveJournal 1.010x). Neither mechanism remains in the branch.

A third follow-up replaced the four-byte global radix with an exact
owner-bucket reduction. Endpoint IDs were Fibonacci-hashed to one worker;
count/scatter gave that worker exclusive ownership of every corresponding
degree counter, avoiding both atomics and the global sort. It was exact in
27/27 bracketed factors but only noise-level on setup (geomeans 0.997 / 0.977 /
0.982 on Skitter / coPapers / LiveJournal, with the apparent gains coming from
one drifting repetition). The isolated LiveJournal reduction was slightly
slower, 131.8 versus 123.8 ms summed maximum-worker time: global sorting folds
80.5M incidences to 23.1M runs, while owner bucketing preserves 73.1M local
runs and trades barriers for random degree writes. The implementation was
removed; do not retry owner buckets without an aggregation mechanism cheaper
than the radix they replace. Raw logs:
`/tmp/apxchol-owner-bucket-screen-20260827`.

Balancing block-greedy's contiguous candidate blocks by the exact sum of
candidate degrees was also tested. A collective two-pass partition placed
approximately equal adjacency work on each worker before the ordinary greedy,
conflict and maximality-repair passes. It remained deterministic, but an 18/18
T=16 screen found no selection win: grid_500 moved 3.9 to 4.2 ms, Skitter 96.7
to 107.9 ms, and LiveJournal 719.9 to 715.2 ms. Full degree is not the executed
greedy work because scans often stop at their first chosen neighbour; the extra
degree pass and three barriers therefore buy the wrong balance model. Factors
also change because block boundaries define the greedy order. The code was
removed. Raw logs: `/tmp/apxchol-bg-work-balanced-screen-20260827`.

The cached-degree front end was then decomposed before attempting a wider
quantile radix. At T=16, exact quantile selection cost only 3.4 / 8.3 / 11.9 ms
and ordered filtering 7.2 / 7.8 / 23.4 ms on Skitter / coPapers / LiveJournal.
Degree sourcing cost 82.8 / 105.7 / 385.4 ms: the initial and periodic full
adjacency refreshes plus copying cached vertex-indexed degrees into candidate
order. Replacing the four byte-wise quantile passes by a larger radix therefore
has only single-digit-millisecond headroom and is not justified. The next
meaningful lever, if any, is the refresh/source representation rather than the
order-statistic algorithm.

An 11-bit decrement radix was nevertheless isolated because that endpoint
stream, unlike the quantile, has measurable standalone cost. It reduced
unsigned-32 sorting from four count/scatter passes to three while retaining a
globally sorted stream and exact duplicate folding. On LiveJournal it was
slower: summed maximum-worker reduction time rose from 128.0 to 161.9 ms on the
same 80.49M endpoints and 23.12M runs. Clearing and prefixing 2,048 bins per
worker costs more than the removed stream pass. The implementation was removed;
the current 8-bit radix is the measured histogram/traffic sweet spot.

Peak-RSS `/usr/bin/time` sentinels at T=16:

| matrix | baseline | AUTO, no refresh | AUTO, 0.10 refresh |
|---|---:|---:|---:|
| coPapersDBLP | 1.387 GB | 1.757 GB | 1.444 GB |
| com-Orkut | 16.602 GB | 21.161 GB | 17.341 GB |

The 0.10 refresh reduces the original 27% RSS penalty to about 4% while
retaining material setup wins (coPapersDBLP about 33%; Orkut about 12% in the
memory sentinels).

Raw logs:

- `/tmp/apxchol-incremental-degree-refresh-bracket-20260827`
- `/tmp/apxchol-incremental-degree-memory-20260827`
- `/tmp/apxchol-incremental-degree-gate-trace-20260827`

## Decision gate

Run the same baseline/AUTO/baseline protocol on Grace at T=36 and T=72. Ship
only if factors remain exact, activated-cell setup geomean is below 0.95 at
both thread counts, every no-op cell remains within the bracket noise band, and
the memory headroom is acceptable for the 450 GB node allocation.

## Grace gate and delayed-activation closure

Job `4550697` checked 576/576 records over as-Skitter, coAuthorsDBLP,
coPapersDBLP, com-LiveJournal, com-Orkut and kron_g500-logn16 at T=36/72.  The
counterbalanced candidate was the dynamic rule that kept reconsidering every
initially rejected sample.  All 12 cells preserved raw/stored factor nnz,
iterations and residual exactly; maximum residual was `8.56e-9`.

| subset | setup | find partition | elimination | SpTRSV setup | PCG | total |
|---|---:|---:|---:|---:|---:|---:|
| all 12 cells | **0.8963x** | 0.5482x | 1.1253x | 0.9941x | 0.9997x | **0.9228x** |
| T=36 | **0.8770x** | 0.5344x | 1.1232x | 0.9943x | 1.0014x | **0.9066x** |
| T=72 | **0.9161x** | 0.5622x | 1.1274x | 0.9940x | 0.9980x | **0.9393x** |

Kron's underfilled first update no longer poisons the decision: setup is
0.8917x / 0.9148x at T=36/72 because AUTO skips round 0 and activates when
work grows.  More importantly, coAuthorsDBLP starts just below the activation
margin (ratio 15.17) but later activates and reaches setup 0.8429x / 0.8369x.
This refutes a first-round-only gate.

Job `4550798` then checked the missing denominator: 384/384 records over
grid_2000, com-Amazon, com-Youtube and iter0040 at T=36/72.  Unrestricted
delayed activation is not a general default there: setup/total geomeans are
1.0089x/1.0099x over 8 cells; YouTube T=72 reaches 1.0323x setup and 1.2591x
elimination.  Again every factor and solve result is exact (maximum residual
`8.78e-9`).  Their first ratios are 2.02 / 8.96 / 9.92 / 9.30, all below the
independently measured traffic break-even near 10.

The final gate therefore uses hysteresis rather than a matrix exception:

- ratio below 10: permanently OFF;
- ratio in `[10,16)`: pending, re-evaluate on the next exact prepass;
- ratio at least 16 but fewer than 256 endpoints per actual worker: pending;
- both activation gates pass: permanently ON.

Job `4550862` ran that final exact rule from commit `4f0c766`: 960/960 records,
ten matrices, T=36/72, four baseline and four AUTO stages filling all 288 Grace
cores.  All 20 cells preserved raw/stored factor nnz, iterations and residual
exactly; maximum residual was `8.7823e-9`.  The six activated families pass the
predeclared gate:

| activated subset | setup | find partition | elimination | total |
|---|---:|---:|---:|---:|
| T=36 | **0.8975x** | 0.5335x | 1.1215x | **0.9254x** |
| T=72 | **0.9391x** | 0.5674x | 1.1599x | **0.9561x** |

The complete 20-cell setup/PCG/total geomeans were 0.9549x/0.9999x/0.9678x.
One logically disabled grid_2000 T=72 stage was systemically slow and inflated
that cell to 1.1909x.  The dedicated exact-binary sentinel job `4550961`
therefore ran 48/48 alternating records on a fresh node: setup was 1.0090x,
with identical raw/stored nnz, 56 iterations and residual
`8.20022655953e-9`.  Together with job 4550798's 1.0061x grid result, this
classifies the outlier as campaign noise rather than no-op path overhead.

Job `4550862` used 16:04 of its 25-minute one-node ceiling (0.268 node-hours);
the sentinel used 28 seconds (0.0078 node-hours).  This closes the ship gate:
AUTO is the default for block-greedy with directed AoS storage, with `=0` as a
single rollback.  Other selectors and storage layouts are exact no-ops.

Verified local copies:

- `/tmp/apxchol-daint-4550697-results` (2,304/2,304 files checked);
- `/tmp/apxchol-daint-4550798-results` (1,536/1,536 files checked);
- `/tmp/apxchol-daint-4550862-results` (960/960 manifests and 1,920/1,920
  payload hashes checked);
- `/tmp/apxchol-daint-4550961-results` (48/48 manifests and 96/96 payload
  hashes checked).

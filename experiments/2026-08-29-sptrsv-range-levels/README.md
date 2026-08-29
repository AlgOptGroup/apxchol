# CPU SpTRSV implicit round-level ranges

Status: **correctness prototype implemented, fully unit-tested, and locally
screened**. The x86 screen passes; the decisive Grace 36/72-thread check is
still pending.

## Question

The default CPU SpTRSV schedule already receives cumulative elimination-round
boundaries. Factor rows/columns are appended round-major, so each round is a
contiguous interval. The previous setup nevertheless expanded that metadata
into two `vector<vector<node_index>>` schedules, storing every factor row once
for forward and once for backward.

This prototype asks whether the round-derived base schedule can use the
boundaries directly without splitting the level executor or changing any row
arithmetic, ordering, hybrid behavior, or fallback.

## Representation

`detail::round_level_ranges` interprets valid metadata as follows:

- Recorded round `r` is the implicit range
  `[round_bounds[r], round_bounds[r + 1])`.
- Every factor row after `round_bounds.back()` is a singleton residual-peel
  range.
- Forward visits ranges in increasing order. Backward visits the ranges in
  reverse order but keeps row ids ascending inside each range. This is the
  exact order produced by the old stable bucket fill; backward is deliberately
  not a natural descending traversal inside a parallel round.
- Repeated boundaries remain empty level/barrier slots. If no peel follows,
  trailing empty rounds are omitted, matching the old
  `max(row_round) + 1` sizing. The historical empty-factor case retains one
  empty level.
- Metadata must start at zero, be nondecreasing, and end at or before the
  factor dimension. Missing, disabled, or malformed metadata takes the
  retained materialized topological fallback.

The round and materialized schedules expose the same `size()` / `operator[]`
shape to one templated level-loop implementation. Representation is selected
once per team invocation; there is no per-row representation branch and no
duplicated thin/fat executor.

Setup consumes the pending bounds into an immutable active snapshot. Calling
`set_round_bounds()` after setup therefore prepares metadata for the next
setup without mutating the current schedule, hybrid split, or critical plan.
This restores the snapshot semantics of the old materialized level lists and
prevents a shorter post-setup vector from invalidating a non-owning range view.

The hybrid split is still a level index. Its tail contiguity check now skips
empty ranges when finding the first real tail row, while retaining empty
prefix levels and all existing prefix/tail phase barriers. A setup/runtime
OpenMP-team mismatch still touches no scheduled row, then serially replays the
active base levels with each level's original fat/thin arithmetic and row
order. A nested-team test forces this branch; merely changing
`omp_get_max_threads()` does not, because the hybrid requests its setup-sized
team explicitly.

## Critical-tail row storage

The optional second reduction was separable. A backward
`(processor, step)` slot is exactly the forward slot
`(processor, steps - 1 - step)` traversed in reverse. The critical plan now
stores one forward pointer table and one forward row array. The backward
executor maps the step and walks that same slot backward, reproducing the old
row order without the duplicate row or pointer arrays.

## Storage model

For a factor dimension `m`, the default round-derived base schedule removes
the logical payload of `2 * m * sizeof(node_index)` plus the per-level nested
vector allocations. It retains one active round-bound snapshot and two scalar
state fields. Pending bounds are moved into that snapshot during setup and
release their capacity. The topological fallback keeps its materialized lists.

For a critical tail of `t` rows, `P` processors, and `S` supersteps, the second
change removes the logical duplicate payload
`t * sizeof(node_index) + (P * S + 1) * sizeof(size_t)`.

These are representation formulas, not measured RSS or performance results.

## Local x86 performance screen

The initially screened candidate and baseline were built with Clang from base
commit `96f460296d678ab96f19b0a2fc24f7c2fae5be1d`. Their binary SHA-256 hashes
were:

```text
baseline   8c43ea2d04779b11eb8a28f1b7bf57521b6e5531041b13473b3b0ced73fdd8f2
candidate  24497f8c7550eb3927be4274db703a021e1bef46711ae01dc35cb24fb8c99f1a
```

The laptop was in its explicitly authorized performance state (performance
governor and EPP, boost enabled). Each cell used an exclusive timing lock and
an interleaved baseline/candidate/baseline bracket. Ratios below are
candidate/baseline, so lower is better. The aggregate is the geometric mean of
11 complete brackets: three seeds each for grid_2000, G3_circuit and iter0040,
and one seed each for com-Amazon and com-LiveJournal.

| matrix/group | SpTRSV setup | total setup | PCG | total |
|---|---:|---:|---:|---:|
| G3_circuit | 0.8961 | 0.9696 | — | 0.9657 |
| grid_2000 | 0.9137 | 0.9785 | — | 0.9794 |
| iter0040 | 0.9852 | 0.9995 | — | 1.0040 |
| com-Amazon | 0.9458 | 0.9776 | — | 0.9536 |
| com-LiveJournal | 0.9788 | 0.9889 | — | 0.9872 |
| **all 11 brackets** | **0.9365** | **0.9826** | **0.9780** | **0.9806** |

Factor nnz, PCG iterations and reported residual matched exactly in 11/11
brackets. The short com-Amazon total is noisy; the intended signal is the
isolated SpTRSV-setup reduction and the aggregate.

An independent review after this screen found the mutable-view lifetime issue
and a false team-mismatch test. The follow-up fixes affect only post-setup
metadata mutation and the exceptional failed-team fallback; the successful
team's setup and solve loops measured above are unchanged. The Daint gate uses
the reviewed follow-up commit, not the pre-review candidate binary hash above.

A separate `/usr/bin/time -v` grid_2000 run measured maximum RSS of 1,631,544
KiB for the baseline and 1,600,604 KiB for the candidate: **30,940 KiB less**,
consistent with removing two four-million-entry row-id payloads. This is one
measurement, not a distribution.

Local verdict: retain the candidate. Before merge, require a current-source
Grace campaign on the 16M/64M grids at 36 and 72 threads, exact solve quality,
no solve-time regression above noise, and no RSS increase. The 64M case should
remove 512 MiB of logical row-id payload, making it the decisive storage test.

## Validation

Build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

Focused CPU SpTRSV validation:

```bash
./build/tests/unit_tests \
  --gtest_filter=SpTRSVRoundRanges.*:SpTRSVLevelset.*:CriticalSchedule.*
```

Result: **9/9 passed**. The new tests directly check:

- forward/back range endpoints and flattened row order across leading,
  interior, and trailing empty boundaries;
- residual singleton ranges and historical trailing-empty/empty-factor level
  counts;
- byte equality between implicit ranges and the materialized topological
  fallback for forward, backward, and in-place solve pairs in fp32 and fp16;
- lower retained-memory accounting for the implicit representation;
- immutability of the active schedule after a post-setup setter call;
- an empty range exactly at the hybrid prefix/tail split; and
- byte equality after a genuinely forced setup/runtime OpenMP-team mismatch,
  including a mixed fat-prefix/thin-tail factor.

Complete registered suite:

```bash
ctest --test-dir build -N
ctest --test-dir build --output-on-failure
```

Denominator: **320 tests**. Result: **320/320 passed**.

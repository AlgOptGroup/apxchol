# Exact GKS neighbor-order crossover

Status: **accepted at degree 512** after a strict Grace 36/72-core gate. The
production change is one constant; no new sorter or runtime knob remains.

## Mechanism

The GKS tree sampler canonically orders a pivot's neighbors by
`(weight, vertex)`. Production already reproduced that order exactly with a
six-pass 11-bit stable weight radix, followed by a vertex sort inside exact
equal-weight runs, for degrees at least 2048. Lower degrees used `std::sort`.

This experiment asks only where that exact radix should replace comparison
sort. It does not change the estimator, random stream, factor, fill, or solve
quality. The production candidate lowers the existing crossover; it adds no
runtime knob and no second radix implementation.

## Rejected approximate ordering

An initial probe ordered by only the top 16 or 20 weight bits. GKS remains
unbiased for any fixed order, but changing the order changes the sampled tree
and the later residual. On Orkut, B=16 ranged from setup 0.933x to 1.099x
across three seeds; B=20 repaired one seed and regressed another. Iteration
counts happened to remain unchanged in those runs, but the setup/fill path was
not stable enough to ship. Coarse ordering is closed.

## Rejected three-range design

The research branch next tried a small 256-bin byte radix for degrees
512--2047 while retaining the 11-bit radix above 2048. It was exact and
positive, but it did not justify a third algorithm:

| Orkut, T=16, 3 seeds | elimination median | setup median |
|---|---:|---:|
| byte 512--2047, 11-bit above | 0.904x | 0.971x |
| byte for every degree >=512 | 0.913x | 0.969x |
| existing 11-bit for every degree >=512 | 0.918x | **0.966x** |

Every arm matched current main's stored nnz and one-step residual in 3/3
seeds. The simplest rule was at least as good: keep only comparison sort and
the existing 11-bit radix.

## Local threshold sweep

Each row is candidate between two untouched-current-main controls on Orkut,
T=16, seed 42. The controls and candidate use Clang Release with libomp.

| radix from degree | elimination | setup |
|---:|---:|---:|
| 128 | 0.913x | 0.970x |
| **256** | **0.872x** | **0.957x** |
| 512 | 0.899x | 0.966x |
| 1024 | 0.960x | 0.989x |
| 2048 | 1.000x | 1.000x |

The neighboring powers of two bracketed 256 as the apparent laptop crossover.
The clean 256 source matched main's stored nnz and one-step residual for every
run. A later three-seed repetition was contaminated by a host-load excursion:
seed 42 flipped from setup 0.957x to 1.100x while seeds 43/44 were 0.981x and
0.962x; load average had risen above 10. Its median (setup 0.981x,
elimination 0.936x) is recorded but is not the acceptance measurement.

Supporting 512-threshold checks were setup/elimination 0.982x/0.941x on
LiveJournal (3/3 exact seeds). A ten-seed Amazon null control was 0.999x total,
1.003x setup and 1.008x elimination median, with exact stored nnz and residual
in 10/10. The final threshold decision is delegated to a rank-rotated Grace
campaign over both 36 and 72 cores rather than inferred from the noisy laptop.

## Grace acceptance gate

Daint job `4552345` completed in 21:17 on one four-module GH200 node. It
checked 900/900 rank-rotated records: five matrices, seeds 42--44, thresholds
128/256/512/1024 plus current-main 2048, at T=36 and T=72. Each T=36 arm has
24 raw repetitions per matrix and each T=72 arm has 12. Every one of the 30
matrix/thread/seed groups matched factor nnz, stored nnz, iterations, and the
printed residual across all five arms.

The predeclared equal-matrix geomean was useful but sensitive to very short
cells: one Amazon T=72 group had unrelated 151--188 ms setup stalls across four
different arms against a normal 68--74 ms. Pooling all balanced rank/seed
repetitions removes that four-sample-median artifact. Time-weighted pooled
medians versus current main are:

| threshold | T=36 eliminate | T=36 setup | T=36 total | T=72 eliminate | T=72 setup | T=72 total |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.974x | 0.994x | 0.993x | 0.982x | 0.991x | 0.991x |
| 256 | 0.961x | 0.991x | 0.992x | 0.966x | 0.989x | 0.990x |
| **512** | **0.952x** | **0.987x** | **0.987x** | **0.949x** | **0.986x** | **0.986x** |
| 1024 | 0.965x | 0.991x | 0.991x | 0.974x | 0.988x | 0.988x |

Threshold 512 is also the consistent hub result: pooled Orkut setup is
0.985x/0.979x and elimination 0.938x/0.938x at T=36/72; LiveJournal setup is
0.994x/0.997x. Smaller graphs are mostly neutral/noisy (the largest pooled
single-cell total regression is 1.023x). Relative T=36-to-T=72 setup scaling
is essentially unchanged: this is a useful elimination/setup constant, not the
missing scaling breakthrough.

Job `4552262` was the failed first attempt. Its binaries/tests passed, but a
post-run grep incorrectly required indented timing labels at column zero; it
produced no accepted records and consumed 2:32. The corrected retry reused the
checksummed binaries and preserved the failed partial output separately.

`daint-summary.csv` is the predeclared matrix/thread/seed median table,
`daint-summary.txt` its equal-matrix geomean, and `daint-pooled.csv` the robust
all-seed/rank pooled medians used for the acceptance table above.

Inputs used locally:

- `com-Orkut.mtx`: `2208f4e6fcfe0cf5f215e701727fe795e67ad175d045a73d993af4c5c5fd1ee7`
- `com-LiveJournal.mtx`: `bd22675dd3c5aa05a4c8e521c8ef665c606da954184738ddea36bdbee38f3750`
- `com-Amazon.mtx`: `a42f2ced834c89955ffd575af75cb043e44bce2cd249d88bfa2d4583985bc09e`

`bench.py` runs control/candidate/control and divides the candidate by the
geometric mean of its two controls. `--candidate-binary` permits immutable
threshold binaries to be compared without environment-dependent production
code.

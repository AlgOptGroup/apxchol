# Late residual sparsification probe

Status: **promising research branch; do not merge as a default yet**.

## Estimator

At one late BK residual rebuild:

1. combine parallel edges exactly;
2. retain a deterministic spanning forest, so every residual component remains
   connected in every realization;
3. retain each off-forest edge independently with probability `p` and multiply
   its weight by `1/p`.

Each off-forest edge is therefore an unbiased estimator of its original weight;
forest edges are exact. The useful backbone is a one-pass, 12-high-bit fp32
weight bucket order followed by union-find. It approximates a maximum-weight
forest while avoiding four full radix passes.

## Why a rebuild can amortize

Residual density rises even as absolute edge count falls. At the main-selector
handoff, coPapersDBLP had 3.72M physical / 2.41M distinct edges on 21.5k active
vertices; LiveJournal had 18.9M / 16.2M on 74.8k. Their later BK phases perform
tens to hundreds of millions of edge visits, so one O(m) rebuild has genuine
headroom.

## Isolated laptop A/B/A, T=16, p=0.5

Three seeds. Each candidate is divided by the geometric mean of its immediately
adjacent baseline runs; the table reports the geomean of those three ratios.

| matrix | setup | PCG | total | stored nnz | iteration effect |
|---|---:|---:|---:|---:|---:|
| coPapersDBLP | 0.967x | 0.839x | **0.927x** | 0.916x | unchanged, 3/3 |
| com-LiveJournal | 1.010x | 0.875x | **0.961x** | 0.906x | unchanged, 3/3 |
| as-Skitter | 1.035x | 1.012x | **1.028x** | 0.945x | +2 to +3 |

Logs: `/tmp/apxchol-residual-bucket-bracket-isolated-20260826`.

The bucket forest carries 2.67% of residual weight on LiveJournal, 4.29% on
coPapers, and 11.64-11.67% on Skitter. Arbitrary first-found forests double
Skitter's iterations (22-24 to 38-47); exact/coarse heavy forests avoid that.

Exact coalescing without sampling does not rescue Skitter: three bracketed
seeds give about +2-3% total and one +1 iteration. A default therefore needs a
cheap pre-mutation traffic/weight-concentration gate that skips this residual,
not a matrix-name rule and not a post-rebuild abort.

## Breadth/no-op control

With the current `active <= 25000` research trigger, grid_500, com-Amazon,
coAuthorsDBLP, thermal2, ecology1, apache2 and iter0040 never reached the
eligible BK state. The factor nnz and iterations were identical to baseline in
all 7/7 one-seed controls. Logs:
`/tmp/apxchol-residual-bucket-breadth-20260826`.

## Validation and remaining gate

- Fresh Release build: 301/301 tests pass.
- Connectivity is deterministic by construction; off-tree expectation is exact.
- Current env controls are diagnostic only:
  `APXCHOL_RESIDUAL_SPARSIFY_ACTIVE`, `APXCHOL_RESIDUAL_SPARSIFY_P`, and
  `APXCHOL_RESIDUAL_SPARSIFY_TREE=first|max|bucket`.
- Before integration, replace the active-count trigger with a residual-observed
  gate based on absolute live-edge traffic, estimated distinct-pair count,
  expected remaining BK work, and a cheap weight-concentration proxy. Re-run a
  wider social/IPM seed campaign and add direct graph-level estimator tests.

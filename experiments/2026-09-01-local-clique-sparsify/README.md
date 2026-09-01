# Local connectivity-safe two-tree sparsification

Status: research branch; no production default or public option change.

## Estimator

For every eliminated pivot of distinct degree `d`:

1. draw two independent GKS suffix trees and assign each half of the ordinary
   source weight;
2. coalesce duplicate edges in their `O(d)` union;
3. retain a deterministic maximum-weight spanning tree of that union;
4. assign every remaining cycle edge an inclusion probability proportional to
   `sqrt(weight)`, normalized to a requested mean keep rate; and
5. emit a retained cycle edge with weight divided by its own probability.

Conditioned on the two sampled trees, step 4--5 preserves every union edge in
expectation.  Taking expectation over the two trees therefore gives the exact
Schur clique.  The retained tree makes every realization connected.  This is
not the previously rejected "keep one half-tree and HT-thin the other" rule:
the exact backbone is chosen from both trees after coalescing.

If coalescing leaves exactly `d-1` edges, the union is already a spanning tree
and is emitted directly.  This removes all forest/sampling overhead for every
degree-2 pivot and many other small cliques without changing the estimator.

The research control is
`APXCHOL_RESEARCH_LOCAL_TREE_KEEP=<q>` with `q` in `(0,1]`.  When absent, the
production GKS path and random stream are byte-identical.  Expected output is
roughly `(1+q)(d-1)` before duplicate effects.  `q=1` is the coalesced full
two-tree average.

## Decision protocol

First use IPM iter0010/20/40, where fractional oversampling previously bought
the largest solve reductions, with one-tree controls bracketing `q=0.15, 0.25,
0.5, 0.75, 1`.  Record cumulative emitted edges and elimination work in
addition to final factor nnz: final fill hides transient residual amplification.
Every arm must converge below `1e-8`; a single-RHS candidate must win total,
while a solve-only candidate may survive only with a small measured repeated-
RHS break-even.

Then test one grid and two social controls.  Do not expand the campaign unless
an IPM arm improves on the existing fractional-oversampling frontier.

## First local screen

The laptop was not held in a timing-controlled state, so this is a quality and
work diagnostic.  On iter0010 seed 42, baseline used 43 PCG iterations and
8.91M raw factor entries.  The new arms were:

| keep | iterations | raw factor nnz | stored nnz |
|---:|---:|---:|---:|
| 0.15 | 34 | 9.27M | 5.46M |
| 0.20 | 29 | 9.40M | 5.59M |
| 0.25 | 31 | 9.54M | 5.73M |
| 0.50 | 24 | 10.15M | 6.33M |
| 1.00 | 24 | 11.26M | 6.67M |

At `q=0.15`, cumulative emitted edges rose 13.4%, raw adjacency visits 9.1%,
and unique-neighbor work 4.3%, while final raw fill rose only 4.0%.  This
demonstrates why final nnz understates setup cost.  After adding the exact
already-a-tree fast path, one rollback/candidate/rollback timing bracket gave
setup `1.080x`, PCG `0.820x`, and total `0.975x`; this single drifting cell is
only the reason to run the bounded Daint screen, not an acceptance result.

## Daint stage-one result

Daint job `4569078` completed in 1:15.  Checked **20/20 records**: the two
baseline arms matched exactly in all 4/4 matrix cells and all 20/20 solves
converged.  On the three IPM matrices only, candidate divided by its adjacent
controls was:

| keep | setup | PCG | total | iterations | raw factor nnz |
|---:|---:|---:|---:|---:|---:|
| 0.15 | 1.1154x | 0.7875x | 1.0209x | 0.7607x | 1.0613x |
| 0.20 | 1.0490x | 0.7301x | **0.9570x** | 0.6932x | 1.0794x |
| 0.25 | 1.0879x | 0.6536x | **0.9645x** | 0.6119x | 1.0962x |

`q=0.20` won single-RHS total on every IPM cell: iter0010 `0.9582x`, iter0020
`0.9850x`, and iter0040 `0.9286x`.  At `q=0.25`, iter0010/20 require about two
right-hand sides to repay setup, while iter0040 already wins one-RHS total.

The grid control rejects a global fixed-q default: iteration counts changed
45 -> 76/63/53 for q=0.15/0.20/0.25, and q=0.20/0.25 totals were 1.405x and
1.283x.  The q=0.15 setup cell was additionally pathological (16.4x despite
only 1.10x raw-neighbor traffic), consistent with a variance/topology outlier,
not ordinary linear work.

The next candidate should replace fixed q by a per-clique variance budget.
For GKS source mass `s_i` and suffix probabilities `p_ij`, one sample's target
diagonal variance is `s_i^2 * (1 - sum_j p_ij^2)`; two half samples save half
of that.  Conditional HT thinning adds exactly
`2 * sum_e h_e^2 * (1/q_e - 1)` to the squared diagonal-error proxy.  Choose
`q_e` to keep the added term within the saved half, with the minimum expected
edge count (`q_e` is proportional to `h_e`, capped at one).  This is a
graph-independent rule tied directly to the approximation error; if its edge
cost is too high, fall back to one GKS tree for that clique.

Downloaded evidence: `/tmp/apxchol-local-clique-daint-r1`; all 20 raw-log
SHA-256 checks pass.

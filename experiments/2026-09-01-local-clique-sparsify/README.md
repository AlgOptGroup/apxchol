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

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

## Weight-spread activation follow-up

The fixed `q=0.20` estimator is the only local oversample/sparsify arm that
has improved single-RHS total on IPM, but applying it to every pivot is bad on
the grid.  The next branch tests the smallest graph-independent rule suggested
by that split:

```
activate two-tree q=0.20 iff min(star weight) < tau * max(star weight).
```

The comparison occurs after the canonical weight sort, so it adds no scan or
data structure.  A rejected pivot takes the byte-identical ordinary GKS path.
Degree-2 pivots also use GKS unconditionally: two half-trees can only coalesce
to that same single edge, so oversampling them has no quality benefit.

This is not a matrix-name or input-size gate.  It targets the local source of
GKS variance: scale separation inside the pivot star.  The independent eager
light/light census found that `tau=1e-4` classified 524181/524286 iter0010
pivots but 0 grid_500 pivots; `tau=1e-5` and `1e-3` bracket the boundary.

The first Daint screen uses seed 42 on 8/8 matrices: four IPM iterates,
grid_500, G3_circuit, thermal2 and com-Amazon.  It brackets ungated `q=0.20`
and the three thresholds between exact baselines, for 48/48 planned records.
Advance to five seeds only if one threshold preserves the IPM solve/total win,
is an exact or near-exact no-op on controls, and does not add setup work beyond
the ungated estimator on activated trajectories.

### One-seed Daint gate

Daint job `4571971` completed in 1:58.  Checked **48/48 records** over 8/8
matrices; exact baselines matched in 8/8 cells, all true residuals were at most
`1e-8`, and 325/325 tests passed (three platform-dependent skips).  The
downloaded raw-log hashes verify 48/48 under
`/tmp/local-clique-spread-gate-results-20260901-r1`.

The decisive structural comparison is:

| scope / arm | emitted edges | raw-neighbor work | raw factor | iterations |
|---|---:|---:|---:|---:|
| IPM / ungated q=0.20 | 1.2320x | 1.1589x | 1.0822x | 0.6729x |
| IPM / gated at 1e-3 | **1.2316x** | **1.1587x** | **1.0820x** | **0.6781x** |
| controls / ungated q=0.20 | 1.2429x | 1.1495x | 1.0861x | 1.6592x |
| controls / gated at 1e-3 | **1.0005x** | **1.0003x** | **1.0001x** | **1.0112x** |

Thus the local rule preserves essentially the entire IPM estimator while
removing essentially the entire control-graph regression.  `grid_500` is
byte-identical to baseline; com-Amazon is exact through `1e-4` and differs by
only 0.01% of emitted work at `1e-3`; G3_circuit and thermal2 also change only
about 0.2% and 0.01% of emitted work respectively.

The one-pass timing order is not an acceptance result: individual no-op setup
ratios ranged widely (for example, byte-identical com-Amazon arms appeared
about 20% faster, while a byte-identical grid arm appeared 20% slower).
Across the four IPM cells, the 1e-3 gate measured setup 1.0286x, PCG 0.7128x,
and total 0.9311x; controls measured total 1.0644x despite their near-exact
structure.  This is sufficient to advance quality, not to claim speed.

**Next gate:** five seeds, with adjacent baselines around each of ungated
q=0.20 and the `1e-3` spread gate.  The candidate survives only if it keeps
the IPM iteration/fill benefit, remains structurally neutral on controls, and
wins bracketed one-RHS total beyond the null-control timing variation.

### Five-seed verdict: keep the gate and the solve lead; tune setup/q

Daint job `4571995` completed in 2:13.  Checked **200/200 records**, exact
three-baseline signatures in 40/40 matrix/seed cells, and 200/200 converged
true residuals.  The same source passed 325/325 tests (three skips), merge
rechecked the exact source/binary metadata and every raw-log hash, and the
downloaded evidence independently verifies 200/200 hashes under
`/tmp/local-clique-spread-gate-results-20260901-r2`.

Across 20 IPM matrix/seed cells:

| arm | setup | PCG | one-RHS total | iterations | stored factor | emitted edges |
|---|---:|---:|---:|---:|---:|---:|
| ungated q=0.20 | 1.2099x | 0.7012x | 1.0526x | 0.6621x | 1.0578x | 1.2317x |
| spread-gated q=0.20 | **1.1796x** | **0.6977x** | **1.0317x** | **0.6585x** | **1.0576x** | **1.2314x** |

Across 20 control cells, the gate changes only 0.06% of emitted edges and
0.02% of raw factor entries: iterations are 1.0057x and stored factor is
1.0001x.  Ungated q=0.20 instead raises control iterations to 1.6821x and
stored factor to 1.0853x.  The activation rule therefore solves the
IPM-versus-everything-else failure without giving up the IPM quality gain.

The setup/solve trade is real.  Summing the bracketed IPM times (rather than
geometrically weighting tiny and large matrices equally) gives setup 1.1837x,
PCG 0.6924x, and one-RHS total 1.0350x.  The aggregate break-even is **1.38
right-hand sides**: two RHS give 0.9554x total, four give 0.8720x, and eight
give 0.8023x.  Per matrix, one-RHS total is 0.9410x on iter0030 and 0.9983x on
iter0040, but 1.0617x on iter0010 and 1.1448x on iter0020.

Short control timings still contain isolated system outliers (one grid
baseline and one grid candidate each took about 4--5x their neighboring
24-ms setups), so the 1.03x control total is not evidence of a structural
regression; the structural counters above are the stable control result.

**Decision:** retain `min/max < 1e-3` as the successful universal activation
rule and retain local two-tree sparsification as a repeated-RHS/solve win.  It
is not yet a single-RHS default.  The next contained campaign should tune q
around 0.20 on the IPM ladder under this fixed gate, with each candidate in
its own adjacent baseline bracket.  In parallel, reduce setup work in the
connectivity-backbone/union pass; a quality-equivalent implementation needs
roughly 27% less *incremental* setup cost to move the aggregate break-even
from 1.38 RHS to one.

The q-tuning successor fixes the spread gate at `1e-3` and evaluates
`q = 0.10, 0.15, 0.20, 0.25, 0.30` over the four IPM iterates and five seeds.
Every candidate sits between its own two exact baseline arms: 220/220 planned
records and 20/20 six-baseline matrix/seed checks.  Report one-, two-, four-
and eight-RHS totals separately; do not choose q from iterations alone.

### Gated-q trade curve

Daint job `4572100` completed in 1:44.  Checked **220/220 records**, 20/20
exact six-baseline matrix/seed cells and 220/220 converged solves; 325/325
tests passed.  Merge and the downloaded evidence verify all 220 raw hashes
under `/tmp/local-clique-spread-gate-results-20260901-r3-qtune`.

Geometric means over the 20 IPM matrix/seed cells are:

| q | setup | PCG | one-RHS total | iterations | stored factor | emitted edges |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 1.0543x | 0.9529x | 1.0218x | 0.9320x | 0.9996x | 1.1153x |
| 0.15 | 1.1707x | 0.7603x | 1.0416x | 0.7439x | 1.0288x | 1.1734x |
| 0.20 | 1.1935x | 0.6897x | 1.0373x | 0.6585x | 1.0576x | 1.2314x |
| **0.25** | **1.1740x** | **0.6440x** | **1.0136x** | **0.6068x** | 1.0854x | 1.2889x |
| 0.30 | 1.2124x | 0.6269x | 1.0316x | 0.5822x | 1.1127x | 1.3466x |

Time-weighted sums tell the same story.  q=0.25 is 1.0134x for one RHS,
0.9270x for two, 0.8360x for four and 0.7598x for eight; its aggregate
break-even is 1.12 RHS.  q=0.30 wins only by eight RHS (0.7528x versus
0.7598x) and is effectively tied at four.  q=0.10 minimizes setup but saves
too little solve work.

**Parameter decision:** q=0.25 is the retained repeated-RHS setting; q=0.20
is no longer the frontier.  No tested q establishes a single-RHS default,
although q=0.25 misses by only 1.3%.  Because q=0.25 was selected from this
five-value sweep and factor-seed totals remain variable, confirm it on a wider
seed set before exposing even an opt-in production knob.  The implementation
work target is now concrete: remove roughly 11% of q=0.25's incremental setup
cost, or an equivalent absolute amount, to cross the one-RHS line.

The confirmatory successor fixes q=0.25 and the `1e-3` gate over 20 factor
seeds on all four IPM iterates.  Each candidate has adjacent exact baselines:
240/240 planned records and 80/80 matrix/seed baseline checks.  This wider
seed set decides whether the near-one-RHS result is stable enough for an
opt-in mode; it is not another parameter-selection pass.

### Twenty-seed confirmation

Daint job `4572164` completed in 1:53.  Checked **240/240 records**, 80/80
exact matrix/seed baseline cells and 240/240 converged solves; 325/325 tests
passed.  Merge and the downloaded evidence verify all 240 raw hashes under
`/tmp/local-clique-spread-gate-results-20260901-r4-confirm`.

Over the 80 IPM matrix/seed cells, q=0.25 has setup 1.1856x, PCG 0.6464x,
one-RHS total 1.0203x, iterations 0.6060x, stored factor 1.0853x and emitted
edges 1.2890x.  Time-weighted sums give nearly the same conclusion: one RHS
1.0209x, two RHS **0.9323x**, four **0.8394x**, eight **0.7619x**; aggregate
break-even is 1.19 RHS.

The matrix split is stable and explains the remaining variance:

| matrix | setup | PCG | one-RHS total | break-even RHS |
|---|---:|---:|---:|---:|
| iter0010 | 1.2076x | 0.6970x | 1.0657x | 1.78 |
| iter0020 | 1.2393x | 0.7390x | 1.1047x | 2.49 |
| iter0030 | 1.1751x | 0.5995x | 0.9917x | 0.93 |
| iter0040 | 1.1262x | 0.5658x | 0.9351x | 0.56 |

**Final decision for this phase:** the `min/max < 1e-3`, q=0.25 estimator is
a validated repeated-RHS/solve mode, not a single-RHS default.  Keep the
research branch and its exact rollback; do not add default heuristics or more
q values.  Further work is justified only if it reduces the existing union /
backbone setup cost without sacrificing the confirmed factor distribution.
For the project's current single-RHS priority, return attention to setup
parallelism and larger algorithmic changes.

## Output-identical union/backbone optimization gate

Branch `codex/local-clique-fast-union-r1` keeps the confirmed q=0.25 estimator
and random streams exactly.  It exploits the two-tree union's source-group
structure: source `i` contributes either one coalesced edge or two equal-weight
edges, and no duplicate can cross source groups.  Sorting the `d-1` groups and
expanding each group by ascending partner therefore reproduces the existing
Kruskal order while sorting at most half as many items.  The same patch scans
only off-tree edges in the six probability-normalization passes, reuses an
inverse-CDF result when both partner draws land in the same prefix bin, and
stops exact fixed-point or DSU work once its result cannot change.

An independent old/new synthetic replay covered degrees 3 through 1024 and 80
seeds per degree family; its sequence-sensitive 64-bit hash was
`2687c875e4c55b78` for both implementations.  The local Clang Release build
passed 325/325 tests.  These are correctness gates, not timing evidence.

The Daint screen uses the previously checksummed `8cc1f0e` q=0.25 binary as
both sides of an `old-q / new-q / old-q` bracket, with production controls
outside it.  It covers four IPM iterates and five seeds at T=72 (100 planned
records).  Every old/new q arm must match factor nnz, stored nnz, iterations,
printed residual and cumulative clique-work counters exactly before timing is
reported.  The optimization survives only if its setup gain exceeds the
old-q bracket drift; the one-RHS target is about 1.9% of baseline setup.

### First exact-optimization result

Daint job `4572695` completed in 1:30 (0.025 node-hours).  Checked 100/100
records and 100/100 downloaded raw hashes; all solves converged, 20/20
production brackets matched exactly, and all three q=0.25 arms matched exactly
in 20/20 cells.  The optimized q arm divided by its adjacent old-q controls
was:

| setup | PCG | total | factor / stored / iterations / emitted |
|---:|---:|---:|---:|
| **0.9916x** | 1.0170x | 0.9962x | exactly 1.0x |

Against production in the same job, q=0.25 remained 1.1855x setup, 0.6567x
PCG and 1.0228x one-RHS total.  The setup reduction is about 0.8% of the q arm,
roughly half the remaining one-RHS gap.  Treat its exact magnitude cautiously:
the old-q right/left setup ratio ranged from 0.805 to 1.250 even though its
aggregate geometric mean was 1.0002, and only 11/20 optimized cells won.  A
fixed-seed repeated bracket is needed before calling a sub-percent gain stable.

The next exact arm fixes a separate accumulation bug: `edge_emitter::reserve`
was called once per pivot with exactly `buffer.size()+expected_output`, forcing
repeated prefix copies instead of geometric vector growth.  Amortized doubling
does not change any emitted edge; the next Daint sequence brackets
`old / grouped / amortized / grouped / old` repeatedly and records peak RSS as
the memory-side gate.

### Amortized-buffer result

Daint job `4572751` completed in 1:39 (0.0275 node-hours).  Checked 160/160
measured records, 160/160 downloaded raw hashes, 32/32 exact five-way factor
signatures and 160/160 converged solves; the new source passed 326/326 tests.
Each matrix used seed 42 and eight warmed symmetric repetitions.

| comparison | setup | total | peak RSS | all structural/quality fields |
|---|---:|---:|---:|---:|
| amortized / grouped | **0.9846x** | 0.9886x | **0.9239x** | exactly 1.0x |
| amortized / original q | **0.9839x** | 0.9892x | 0.9648x | exactly 1.0x |
| grouped / original q | 0.9993x | 1.0006x | 1.0442x | exactly 1.0x |

The grouped code's earlier nominal 0.9916 setup result does not reproduce in
this tighter protocol, so it is not credited as a speedup.  Amortized reserve
does: it wins 18/32 individual repetitions and its time-weighted setup ratios
are 0.9867 versus grouped and 0.9859 versus the original q implementation.
Per-matrix setup ratios versus grouped are 0.9883 / 0.9413 / 1.0287 / 0.9819
on iter0010/20/30/40.  The lower peak is plausible despite geometric retained
capacity: repeated exact reservations transiently overlap each accumulated old
buffer with its replacement, while doubling usually avoids that copy.

Applying the time-weighted 0.9859 setup ratio to the 20-seed q=0.25 campaign
models setup at 1.1689x production and one-RHS total at about 1.0086x; the
break-even drops from 1.19 to about 1.08 RHS.  This remains a model until a
wide production bracket uses the final implementation.

The last strict implementation candidate stores one 32-byte record per GKS
source plus one double per cycle edge, instead of a 32-byte record and index
scratch per logical union edge.  Its synthetic old/new output hash remains
`2687c875e4c55b78`.  A repeated Daint bracket against the amortized arm decides
whether this lower scratch traffic closes the remaining single-RHS gap; if it
does not, further progress requires changing the estimator rather than more
container variants.

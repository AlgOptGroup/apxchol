# Controlled comparisons

The [main findings](../README.md) summarize the decision. All numbers here
come from saved experiments; [reproducers and source bindings](../reconciliation/README.md)
separate exact arithmetic, saved observations, and native reruns.

## Same local law, different solver protocols

Weighted Prüfer/BKZ26 and CAST-1 implement the same ideal local distribution.
Their inverse-CDF and alias implementations map seeds to trees differently.
CAST means *Canonical Approximate Schur Tree*. VAC is the broader
[volume-sampling elimination framework](https://rasmuskyng.com/papers/BKZ26.pdf);
this branch substitutes its local rule into apxchol, not an entire VAC solver.
CAST-2 samples one tree over two half-weight copies of each terminal, then
contracts copies, discarding loops and merging parallel edges. Contraction can
create a cycle: the copy-tree edges $A_1B_1,A_2C_1,B_2B_1,B_1C_1,C_1C_2$
leave $AB,AC,BC$. It is neither two averaged trees nor AC2's persistent
split/merge mechanism.

The [original campaign](historical-benchmarks.md) held apxchol's surrounding
algorithm fixed. The later Julia adapter used public Laplacians.jl 1.4.1,
fp64 coefficients/PCG, common RHSs, and replayable pivot orders. These are
separate comparisons; their timings must not be pooled. The supplied anonymous
manuscript, *CAST: Canonical Approximate Schur Tree for Approximate Cholesky
on Graphs*, uses the [SDDM2023 Chimera/Spielman collection](https://rjkyng.github.io/SDDM2023/Tutorial.html)
in sections C.2/C.3. It is distinct from the linked VAC theory paper; no public
CAST manuscript URL is verified here.

A CAST-style star-k50 sentinel gave GKS/CAST-1/CAST-2 median mean iterations
50.612/38.528/24.548: nine factors, 250 RHSs each, all maximum original-system
residuals below $10^{-8}$. CAST-2 mean stored fill was 31930 versus GKS's
31901, about 0.09% more. Extra emitted edges therefore do not imply doubled
stored fill. This is historical C++ evidence, not a matched Julia/AC2 study.
[Sentinel data](../reconciliation/star-k50.tsv).

## Full-matrix crossed-order comparisons

In the table, the first letter denotes GKS/CAST-1 sampling; the second denotes
the source of the adaptive pivot order. Crossed arms replay that order.

| Input | GG | CC | CG | GC |
|---|---:|---:|---:|---:|
| Chimera i1, step1 |17.000|17.912|19.020|19.000|
| Chimera i1, step6 |17.188|16.980|19.088|19.116|
| Spielman k100, step1 |7.892|7.004|7.000|7.904|
| Spielman k100, step10 |5.000|5.000|5.000|5.000|

These are means over 250 common RHSs, seed 42, with mirrored repeats:
32 factors and 8,000 solves, all passing original and solver residual checks.
Two Chimera setup controls failed, so the table supports quality only.
Sampler and ordering interact on Chimera. Fixed order still permits different
later star weights. [Four-input evidence](../reconciliation/four-inputs.json).

## Larger real neighborhoods

A *trajectory* means the sequence of pivot neighborhoods in one factorization,
not a matrix family or sampler. The observer retained 288 neighborhoods from
nine factorizations: 106 on Yves iter0010, 170 on Chimera step1, and twelve
old Spielman triangles. The first two families include much larger stars. Here $J$ is the
whole-clique relative variance defined in the [model](sampling-model.md):

| Retained family | Degree range | CAST improves $J$ | Summed CAST/GKS $J$: GG / CC / CG |
|---|---:|---:|---:|
| Yves iter0010 |3–68|3/106|1.2213 / 1.2216 / 1.2012|
| Chimera step1 |3–350|78/170|1.0012 / 1.0775 / 1.0804|

These are analytically evaluated scores on identical saved stars, not noisy
16-draw estimates. Strongly uneven Yves weights mostly favor GKS; Chimera's
early and late neighborhoods behave differently. **The samples are stratified:
these sums are not estimates of population frequency or total factor error.**
[All scalar rows and source hashes](../reconciliation/real-star-scores.json)
and the standard-library reproducer retain that distinction.

One diagnostic case explains the earlier Spielman discrepancy: its complete
sequence has 49 nontrivial triangles. Three nearly uniform ones contribute
92.8% of summed GKS variance and favor CAST; the earlier observer missed them.
Fixed-order CAST averaged 7.000 versus GKS's 7.892 iterations, surviving
arithmetic/RNG controls. Exact updates at the 49 triangles gave one iteration
on all 250 RHSs at unchanged stored fill. This is a narrow diagnostic, not
the main matrix evidence. [Eight factors / 2,000 solves](../reconciliation/spielman.json).

## Quick cycle and parent-probability screens

Both screens used grid_2000, iter0040, and as-Skitter, T72, seeds 42/314159,
one common RHS per matrix, and GKS-before/after controls. Each checked
30/30 solves and 42/42 declared controls. Ratios are six-block geometric means.

| Screen / comparison | Iterations | Stored fill | Setup | Solve | Total |
|---|---:|---:|---:|---:|---:|
| q tree / GKS |1.0201|1.0021|1.0199|1.0160|1.0164|
| cycle-q / cycle-GKS |0.9668|1.0015|0.9969|0.9704|0.9905|
| heavy-GKS / GKS, older cutoff |0.6436|—|—|—|—|
| heavy-K2 / heavy-GKS |0.9560|1.0063|—|0.9558|0.9932|
| threshold / heavy-GKS |1.3729|0.9443|0.9155|1.2997|1.0310|

The q screen uses each cycle rule's own relative-variance cutoff. The separate
K2 screen uses one common older degree-normalized cutoff. Do not pool their
ratios. Threshold means a full cycle if weight ratio is at most two, otherwise
GKS; it never fired on either iter0040 trajectory, missing useful heavy
suffixes. K2 improved five seed cases and worsened one.

Set aside the pure q tree; retain cycle-q as a modestly preferred research
candidate. The large cycle benefit already appears without q or K2, but
includes topology, the extra edge, and changed later stars. These screens
establish neither timing significance nor production readiness.
[Q rows](../reconciliation/q-screen.json) and
[cycle rows](../reconciliation/cycle-screen.json) preserve individual cases.

## Limits

The [local-model comparison](sampling-model.md) reports absolute variance,
spectral error, and budget-specific bounds. None is a general PCG predictor.
A negative sampler campaign does not reject a paper. Remaining gaps include
broader full-solve prediction from representative stars and matched CAST-2/AC2
quality evidence. No exact author-executable or universal superiority claim
follows from the public reconstruction.

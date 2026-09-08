# Weighted Prüfer, CAST and GKS: what the follow-up explains

The [original negative apxchol result](README.md#broad-matrix-evidence-bkz26-versus-gks) stands: replacing its GKS update with weighted Prüfer increased average iterations on the tested matrices. This does not contradict a CAST advantage on other inputs. A controlled follow-up found a concrete positive case, separated ordering and numerical details, and identified why its local sampled-tree errors behave differently.

On Spielman `k100, step1`, fixed-order CAST uses **7.000 versus GKS's 7.892 mean PCG iterations**. Making all **49 nontrivial triangle updates exact** reduces every tested RHS to **one iteration**, with **unchanged stored factor fill**. A full census explains the local comparison: three nearly uniform triangles contribute **92.8%** of GKS's summed local variance, and CAST improves those three. A previous four-star sample missed all three.

This document defines the methods and metrics, reports the controlled comparisons, and separates these empirical explanations from claims that remain unproved. The accompanying [evidence and executable source](reconciliation/README.md) require no private filesystem or conversation context.

## 1. Names and the sampled update

CAST expands to *Canonical Approximate Schur Tree*. The CAST-1/CAST-2 labels below identify the one-copy and two-copy constructions; they do not identify a separate author executable.

An eliminated vertex has positive incident weights `a_1,...,a_d`, total `W=Σa_i`, and pivot diagonal `D`. Exact elimination adds a clique with conductances

\[
c_{ij}=\frac{a_i a_j}{D},\qquad i\ne j.
\]

A sampled update `X` is the Laplacian of the emitted weighted graph. All methods here target `E[X]=C`, the exact clique Laplacian. For a Laplacian pivot `D=W`; the factor `W/D` extends the formulas to an SDDM pivot.

| Name | Meaning in these experiments |
|---|---|
| **GKS** | Sort weights increasingly; source `i` chooses one later parent `j` with probability `a_j/S_i`, where `S_i=Σ_{j>i}a_j`, and emits conductance `a_i S_i/D`. Independent choices produce a connected `d−1`-edge tree. |
| **Weighted Prüfer / BKZ26** | Draw `d−2` independent Prüfer symbols with probabilities `a_i/W`, decode the tree, and give selected edge `ij` conductance `(W/D)a_i a_j/(a_i+a_j)`. |
| **CAST-1** | The same one-copy weighted-Prüfer ideal law. Our CAST implementation uses an alias sampler; the original BKZ26 implementation used inverse-CDF lookup. Their mappings from seed to tree differ, but their intended distributions agree. |
| **CAST-2** | Split each neighbor into two copies of weight `a_i/2`, draw **one tree on all `2d` copies**, then contract copies. Discard loops and combine repeated terminal pairs. It is not an average of two independent CAST-1 trees. |
| **VAC** | The volume-sampling elimination framework of [Baumann, Kyng and Zöcklein](https://rasmuskyng.com/papers/BKZ26.pdf). This branch implements its weighted-Prüfer local rule inside apxchol's existing factorization; it is not a reproduction of an entire different solver. |

The same expansion/contraction idea can be applied to GKS: linearity preserves unbiasedness. It does not automatically improve variance. For two equal original terminals, ordinary GKS is exact; a recorded exact duplicated-GKS example produces conductance `1/4` with probability `1/3` or `5/8` with probability `2/3`, preserving mean `1/2` but adding variance `1/32`. CAST-2 is also distinct from public AC2's persistent multiedge split/merge mechanism.

For Prüfer sampling, `Pr(ij is selected)=(a_i+a_j)/W`, so the stated conductance is exactly `c_ij/Pr(ij)`. The distinction from GKS is dependence and how selected edges are weighted, not unbiasedness or connectivity. Either sampler can change subsequent degrees, pivot priorities and residual weights.

## 2. What the local metric measures

Define

\[
T=\frac WD\operatorname{diag}(a_i),\qquad R=T^{-1/2},\qquad
J=\mathbb E\|R(X-C)R\|_F^2.
\]

This is the expected **relative Frobenius error of the entire sampled clique Laplacian**, including diagonal and off-diagonal errors. It is different from the degree-only RMS metric in the original README.

Write `v_i=√(a_i/W)`. Then `RCR=I−vvᵀ`, the identity on the clique's non-null subspace. This normalization expresses errors relative to the exact local operator rather than favoring stars simply because their conductances are small. If `δw_ij` is a sampled edge's conductance error and `δs_i=Σ_{j≠i}δw_ij`, the objective expands to

\[
J=\mathbb E\left[
 \sum_i\frac{\delta s_i^2}{T_{ii}^2}
 +2\sum_{i<j}\frac{\delta w_{ij}^2}{T_{ii}T_{jj}}
\right].
\]

A selected edge of conductance `w` contributes `w(1/T_ii+1/T_jj)` to normalized trace. This is its endpoint-scaled matrix trace, not a vertex's raw incident weight or only its outgoing GKS contribution. Every CAST-1 edge contributes exactly one; every sampled tree therefore has normalized trace `d−1`.

That property also explains the local minimax statement: among unbiased inverse-inclusion-probability one-tree estimators, the largest possible normalized edge contribution cannot be below one, and CAST-1 attains one on every edge. This optimizes a **worst single-edge contribution**, not `J`, spectral norm, or whole-matrix PCG iterations. Edge covariances and the accumulation of errors across elimination still matter.

The relative Frobenius objective is useful because it includes dependence between incident edges and is exactly calculable on small stars. It is not a proved ordering of PCG performance. Later stars depend on earlier samples, and elimination transforms local errors before they affect the complete factor.

## 3. Matching protocols changes the question

The original broad apxchol campaign changes only its local sampler. The follow-up Julia reconstruction instead uses public [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) `1.4.1`, fp64 factor coefficients and PCG, matched RHSs and explicit replayed pivot orders. These are different surrounding algorithms and must not have their timings pooled.

The small star sentinel used to check the CAST-style protocol produced these medians of three per-factor means, each based on 250 RHSs:

| Star `k50` | GKS | CAST-1 | CAST-2 |
|---|---:|---:|---:|
| Median mean iterations |50.612|38.528|24.548|

All nine recorded factors' maximum original-system residuals are below `1e−8`. Sequential elimination, degree updates, tie handling, precision and grounding were relevant to matching this sentinel. It does not imply that ordinary apxchol's GKS baseline was broken, or that every matrix should show the same advantage. [Recorded table](reconciliation/star-k50.tsv).

To separate sampler from ordering, let the first letter in `GG`, `CC`, `CG`, `GC` denote GKS or CAST-1 sampling, and the second denote the adaptive GKS or CAST pivot order. A crossed arm replays its source order exactly. On four public IPM inputs:

| Input | GG | CC | CG | GC |
|---|---:|---:|---:|---:|
| Chimera `i1, eps0.1, step1` |17.000|17.912|19.020|19.000|
| Chimera `i1, eps0.1, step6` |17.188|16.980|19.088|19.116|
| Spielman `k100, step1` |7.892|7.004|7.000|7.904|
| Spielman `k100, step10` |5.000|5.000|5.000|5.000|

These are mean iterations over 250 common RHSs, factor seed42. Each of the four arms has an identical mirrored repeat: **32 factors and 8,000 solves**, all passing original and solver residual checks at `1e−8`. Two Chimera setup timing controls failed their declared bounds, so this table makes **numerical-quality claims only**, not speed claims. [Source-bound summary](reconciliation/four-inputs.json).

On Chimera, sampling and order interact. Under the GKS order CAST is worse, while under the CAST order it is better than GKS. Fixing pivot order does not fix subsequent stars: earlier sampled edges still alter their weights. On Spielman step1, the advantage survives either order; step10 has no iteration benefit. There is no uniform CAST improvement even across these four paper-corpus inputs.

## 4. Spielman's complete nontrivial tail

Every native trajectory on step1 contains **338351 degree-two pivots, 49 degree-three pivots and one degree-one pivot**. Degree two has one possible clique edge, so ideal support-sampling variance is zero. However, the GKS recurrence and CAST's direct weight formula round differently, and GKS consumes a random draw that CAST does not. The intervention below controls these effects.

All arms use the same saved GKS order, 250 normalized Gaussian RHSs, fp64 factor/application/PCG, tolerance `1e−8` and true-residual grading. The native controls reproduce historical factor, RHS and solution hashes.

| Intervention | Mean iterations | Stored factor entries |
|---|---:|---:|
| Original GKS |7.892|676850|
| Original CAST-1 |7.000|676850|
| CAST with GKS degree-two arithmetic |7.000|676850|
| Previous arm plus GKS degree-two random-draw consumption |6.804|676850|
| GKS with a separate stream at each nontrivial pivot |7.536|676850|
| CAST with matched degree-two arithmetic and per-pivot streams |6.960|676850|
| Exact clique updates at all nontrivial pivots |**1.000**|**676850**|
| Original GKS repeat |7.892|676850|

The degree-two arithmetic substitution changes solution bits but preserves all 250 iteration counts. CAST's advantage also survives the tested stream alignments. These arms are sequential interventions, not independent factor-seed replicates, and common seeds do not make GKS and CAST select identical outcomes.

The exact arm adds the third clique edge at every triangle. All **250/250** RHSs converge in one iteration, with maximum original residual `3.376e−11`. It still has the same 49 triangles and the same stored factor count: extra emitted residual edges need not translate into extra stored factor entries after later graph updates. This is a measured benefit on this input, not a general fill guarantee for cycles.

The full denominator is **8 factors, 2,000 converged solves**, eight excluded warmups, five actual-input identity builds and separately run native correctness fixtures. All **37/37** result files were collected. The executable adapter and source bindings are included with the [complete projected evidence](reconciliation/spielman.json).

## 5. Why the earlier small sample pointed the wrong way

The complete observer retained all 49 triangles per trajectory. An earlier stratified observer had retained only four. Those four all lie late in the tail and have extremely unequal weights.

For sorted triangle weights `a≤b≤c`, `A=a+b+c`, the exact ideal expectations are

\[
J_G=\frac{2a}{A}+\left(\frac aA\right)^2\frac{(c-b)^2}{bc},\qquad
J_C=\frac{4abc}{(a+b)(a+c)(b+c)}.
\]

The accompanying reproducer independently enumerates GKS's two outcomes and CAST's three outcomes, verifies their expected edge weights and computes the full Laplacian errors using exact rational arithmetic from the saved binary64 weights.

| Complete trajectory used as common input stars | Sum `J_G` | Sum `J_C` | CAST/GKS | Stars CAST improves |
|---|---:|---:|---:|---:|
| Native GKS |2.026269|1.740114|**0.858777**|3/49|
| Native CAST, fixed GKS order |2.036292|1.757291|**0.862986**|3/49|

The three improved triangles are the first three nontrivial pivots. Their weights are comparatively even; the first is essentially `(0.5,0.5,0.5)`, with CAST/GKS error ratio `3/4`. They account for **92.8%** of summed GKS local error. The last 39 contribute only **0.0252%**.

The earlier four retained ordinals were `338335,338341,338385,338361`; they omitted all three dominant ordinals `338301,338303,338305`. On the retained skewed triangles, exact CAST expectation is about twice GKS's. A separate 16-draw replay had even understated that expectation because very rare weak-vertex Prüfer events were usually absent. Both statements are correct for those sampled triangles, but neither described the complete tail.

With the full denominator, local variance favors CAST by about 14%, agreeing in direction with its observed 11.3% iteration reduction. This reconciles the apparent local-metric conflict for this input. It does not prove that summing normalized local variances predicts PCG in general.

For the original application IPM inputs, the retained stars instead have strongly concentrated weights, and the evaluated local CAST expectation is typically worse. Chimera's stars show an early-to-late transition. Those observations support an input-dependent explanation, but their stratified samples are not complete trajectory error budgets like the Spielman census above.

## 6. How far are the methods from an optimum?

“Optimal” must specify the metric, edge budget and admissible distributions. Here the comparison uses the same `J` and **all connected simple graphs with exactly `m` edges**, allowing both probabilities and support-dependent positive edge weights to vary, subject only to `E[X]=C`. The optimum is an infimum because the best limiting distribution may place zero weight on some edges; positive connected approximations provide an upper bound.

The four tiny profiles below have exact rational lower/upper certificates with gaps below `5.1e−9`. Tree budgets have `m=d−1`; the current relative-trace cycle rule has `m=d` and needs a separate optimum. These are stronger comparisons than optimizing within the Prüfer family or over a fixed list of marginal probabilities.

| Weights | GKS `J`, tree | CAST-1 `J`, tree | Global tree infimum | Relative-trace cycle `J` | Global `d`-edge infimum |
|---|---:|---:|---:|---:|---:|
| `1,2,3,4` |0.790833|0.907937|0.675088|0.420000|0.306638|
| `1,2,3,4,5` |1.297852|1.460883|1.034674|0.786667|0.643742|
| `1,1,1,1,8` |1.255208|0.948045|0.656387|0.861111|0.352991|
| `1,1,1,8,8` |0.733380|0.815174|0.509902|0.537396|0.294610|

CAST-1 beats GKS on the one-hub profile and loses on the other three, while remaining **34–60% above the global tree infimum**. The relative-trace cycle rule remains **22–144% above its matched `d`-edge infimum**. Thus neither the CAST edge-minimax property nor the current cycle rule makes it globally optimal for Frobenius variance.

The relative-trace rule in this table is a newer comparator, not the branch's original GKS default. It chooses a heavy suffix, samples a uniform cycle there, and independently attaches each lighter vertex to a later parent with probability `(a_i+a_j)/(m_i a_i+S_i)`. It selects the suffix by its exact relative-Frobenius score. The optimum certificates allow dependencies and outcome-dependent weights beyond this family. These are small-star model comparisons, not matrix timing results.

The [sampling-model note](SAMPLING-MODEL.md) gives the full importance-sampling derivation, explains why weighted Prüfer already uses the same plus-weight edge marginals, and proves two additional statements: exact uniform weights make a cycle globally Frobenius-optimal under an at-most-`d`-edge budget; and weight ratio at most two is sufficient for the full cycle to win within the current suffix family. Neither statement establishes spectral or PCG optimality.

## 7. Quick screen: q alone versus q inside a cycle rule

A subsequent direct comparison separated the parent probability
`q(i,j)=(a_i+a_j)/Σ_{l>i}(a_l+a_i)` from the larger cycle construction.
**No arm uses K2 coordination.** Ordinary GKS and the pure q tree each emit
`d−1` edges. The two cycle rules use a heavy-suffix cycle and independent
light-parent choices, emitting `d` edges for degree at least three. Each
cycle rule selects its **own** cutoff using its exact relative-Frobenius
objective; this compares complete rules, not probability changes at a fixed
common core.

The screen used `grid_2000`, `iter0040` and `as-Skitter`, 72 CPU threads,
factor seeds `42` and `314159`, and one common RHS per matrix. Each of the six blocks
ran GKS before, three candidates in rotated order, and GKS after. All
**30/30 factors and solves** pass original-system residual tolerance `1e−8`;
all **42/42** per-metric GKS bracket controls pass the declared 15% span limit.
Ratios below are geometric means over all six blocks; the GKS reference is
the geometric mean of its two brackets. Total means setup plus one solve.

| Comparison | Iterations | Stored fill | Setup | Solve | Total |
|---|---:|---:|---:|---:|---:|
| Pure q tree / GKS |1.020083|1.002088|1.019891|1.016034|1.016443|
| Cycle + independent GKS / GKS |0.637475|1.119867|1.134930|0.685687|0.947171|
| Cycle + independent q / GKS |0.616340|1.121536|1.131368|0.665421|0.938216|
| Cycle + q / cycle + independent GKS |0.966846|1.001491|0.996861|0.970444|0.990546|

| Matrix | Seed | GKS brackets | Pure q tree | Cycle GKS | Cycle q |
|---|---:|---:|---:|---:|---:|
| grid_2000 |42|57 / 57|57|36|34|
| grid_2000 |314159|55 / 55|57|36|35|
| iter0040 |42|58 / 58|57|36|35|
| iter0040 |314159|60 / 60|59|36|35|
| as-Skitter |42|24 / 24|27|16|16|
| as-Skitter |314159|26 / 26|26|17|16|

The **pure q tree is set aside**: it adds about 2% iterations overall without
a useful fill reduction. The **cycle-q rule remains the modestly preferred
cycle candidate**: about 3.3% fewer iterations and 0.15% more stored fill than
cycle-GKS. The larger gain over a GKS tree is already present with cycle-GKS;
it cannot be attributed to q alone. This screen does not separate the cycle
topology from its extra edge or retuned cutoff, establish statistical timing
significance, or justify production promotion.

All four arms share one executable and the frozen `40f6953e` core's graph
storage, ordering policy, drop settings, input interpretation and RHS
generation. The component-compatible RHS for `as-Skitter` is generated by
the driver; it is not the invalid historical RHS excluded from the original
broad campaign. [Thirty projected raw rows and source/input hashes](reconciliation/q-screen.json)
plus a [standard-library reproducer](reconciliation/reproduce_q_screen.py)
recover all six blocks, 42 controls and 28 aggregate ratios. This is saved-data
reproduction; the full native packet and matrix inputs are not bundled here,
and no native rerun or new residual matvec is claimed.

## 8. What is established, and what remains

Established: the original weighted-Prüfer losses reproduce in their apxchol setting; CAST-1 has the same ideal local law; sampler and adaptive ordering interact; the positive Spielman case survives numerical/RNG controls; its complete tail has lower aggregate CAST local variance; and exact triangle updates remove its approximation cost at unchanged stored fill.

Not established: a theorem or reliable general predictor mapping local `J` to complete-factor PCG, uniform CAST superiority, a broad optimum sampler, or a matching implementation/timing reproduction of every CAST result. The reconstruction used known public factor and PCG code; the exact author implementation, all seeds, reduction/RHS details and timing policy were not available for identical-code comparison. A separate shallow-water study also found that augmented-system and original-system residual criteria can grade the same returned vector differently; their convergence criteria must be matched before comparing performance.

The practical research direction is to use complete or appropriately weighted star data, preserve edge-budget distinctions, and evaluate candidate laws by both local mathematics and converged matrix solves. A negative solver campaign is evidence about a measured configuration, not a rejection of a paper or a reason to abandon its explanation.

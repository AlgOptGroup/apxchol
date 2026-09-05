# Heavy-core sampler comparison

Research-only executable on the existing custom-eliminator and `cpu_solver`
interfaces. Baseline: `fd77fc67` (reviewed cleanup plus S1 first-touch port).
There are no production defaults or public API changes.

The seven arms are ordinary GKS, two-receiver coordinated GKS, its corrected
degree-at-most-six cycle gate, a heavy cycle with ordinary light GKS, and the
same heavy cycle with two-receiver, entire-core, or cycle-ordered coordination.
The reference K2 formulas come from `30a8230d`; canonical sorting uses the same
weight/vertex order with `std::sort`. Every heavy arm freezes the cutoff chosen
by the independent-parent clique-normalized Frobenius variance score. Thus
the comparison isolates coordination rather than retuning the core per arm.

## Model

For sorted positive neighbor weights a[0] <= ... <= a[d-1] and eliminated
pivot D, the exact Schur clique has c[i,j] = a[i]*a[j]/D. Ordinary GKS gives
source i one parent j>i, with probability a[j]/S[i], S[i]=sum(a[j],j>i), and
edge weight a[i]*S[i]/D. Each pair therefore has expected weight c[i,j].
The d-1 upward edges form a connected tree.

K2 means two coordinated **receivers**, the two heaviest vertices. For a
receiver r, each unresolved source has conditional probability a[r] divided
by its still-eligible parent mass. Place these probability intervals end to
end, draw one uniform phase U in [0,1), and select intervals intersecting
U plus an integer. Every source keeps its conditional probability, while
the selected source count is floor(sum p) or ceil(sum p). A fresh phase is
drawn for the next receiver. This balances counts; unequal emitted edge
weights mean it does not exactly balance weighted degrees.

For a heavy suffix H of size h>=3, sample a uniform Hamiltonian cycle. A
pair in H occurs with probability 2/(h-1), so weight each included pair by
c[i,j]*(h-1)/2. Every light source keeps its upward-parent law, including
possible parents in the light suffix ahead of it. There are h cycle edges
plus d-h parent edges: d edges total, connected and unbiased.

Choosing the cutoff evaluates all d-2 feasible suffixes in O(d) time after
sorting, using eight suffix moments. The score is E||R(Lhat-L)R||_F^2 with
R[i,i] equal to inverse square root of the exact clique degree. It includes
both diagonal variance and off-diagonal error. Positive rescaling cancels;
the pivot scales every candidate equally. The implementation normalizes
weights by their maximum and falls back to ordinary GKS when its floating
point moments are not representable. It does not choose the largest ratio
gap or fit a clustering threshold.

The structure-aware candidates protect all h core receivers, either in
descending weight order or in the sampled cycle's order starting at the
heaviest vertex. Their conditional marginals remain correct for every
sampled root cycle. This is a useful generalization of K2, but its work can
be O(h*(d-h)); receiver passes and source visits are measured explicitly.

There is also a mathematical limit to this construction: light-parent error
has conditional mean zero given the cycle. Its covariance with cycle error
therefore vanishes. Cycle-aware ordering can reshape attachment noise, but
cannot automatically cancel the cycle's weighted-degree fluctuations. A
joint cycle/attachment law with biased conditional classes could do more,
provided the overall law stays unbiased. That is a separate research step;
the tiny-star global mixture oracles show why it deserves consideration.

Minimum Frobenius variance is a proxy, not the same objective as expected
spectral norm or final PCG time. The Daint comparison measures fill, setup,
actual one-RHS and five-RHS totals, iterations, residuals, memory and sampler
work. No production adoption follows from finite-star variance alone.

## Validation and execution

`heavy_core_check frozen-cutoffs.txt` checks 1,680,000 outcomes across seven
arms and six fixtures: positive unique edges, connectedness, edge budgets,
input-order determinism, and 630 empirical pair means. It also verifies all
180 frozen cutoff choices from the finite-star reference. Mean checks use
seven standard errors plus rounding allowance; they complement, rather
than prove, the marginal derivation above.

`heavy_core_bench matrix arm threads seed rhs_count` builds one factor and
solves all requested right-hand sides. Each residual is independently
recomputed against the original assembled operator. Parsing and RHS creation
are outside setup timing. The same generated RHS vectors are used across
arms and factor seeds. OpenMP thread count is set before parsing or graph
construction. Each experiment invocation is a fresh process.

The frozen Daint package and raw evidence live in
`results/astra-review-20260905/heavy-core-matrix/` in the main checkout. Its
pilot gates production on all 14 matrix/arm cases plus the ARM star check.
Production plans 216 factors and 1,080 solves, with every arm bracketed as a
block by K2+cycle controls, across six matrices, two thread counts and two
factor seeds. Four nodes for at most 30 minutes reserve at most two node-hours.
Timed out and unattempted cells remain in the denominator.

Original GKS algorithm context: https://arxiv.org/abs/2303.00709.

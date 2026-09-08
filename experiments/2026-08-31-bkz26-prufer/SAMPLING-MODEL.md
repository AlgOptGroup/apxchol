# Relative clique variance, trace and dependent tree laws

This note supplies the mathematics behind the [experimental reconciliation](RECONCILIATION.md). It distinguishes optimization of one sampled edge, a structured cycle/tree family, and unrestricted distributions. Statements below use ideal arithmetic; native floating-point and RNG checks are separate implementation evidence.

## Relative normalization and the role of trace

Take `d≥2`; degree-zero/one pivots have trivial clique updates and are excluded from the trace decomposition. Let positive star weights be `a_1,...,a_d`, `A=Σa_i`, and eliminated pivot `D>0`. The target clique is

\[
C=\frac AD\operatorname{diag}(a)-\frac{aa^T}{D},\quad
T=\frac AD\operatorname{diag}(a),\quad R=T^{-1/2}.
\]

For `D>A`, the full SDDM Schur update also retains diagonal terms; this model concerns its sampled clique contribution. With `u_i=√(a_i/A)` and `P=I−uuᵀ`, we have `RCR=P`. Every sampled Laplacian `X` gives `Y=RXR` with `Yu=0`. Thus

\[
J=\mathbb E\|Y-P\|_F^2
 =\left(\frac DA\right)^2\mathbb E\sum_{i,j}\frac{(X-C)_{ij}^2}{a_i a_j}.
\]

The exact local operator is the identity on `u⊥`. The normalized size of the target edge atom `ij` is

\[
t_{ij}=c_{ij}R_{ij}=\frac{a_i+a_j}{A},\qquad
R_{ij}=\frac DA\left(\frac1{a_i}+\frac1{a_j}\right).
\]

Here `R_ij` denotes a scalar endpoint factor, not an entry of the diagonal matrix `R`. An emitted edge of conductance `w` contributes `w R_ij` to `tr(Y)`. For its positive rank-one matrix atom, trace, Frobenius norm and spectral norm coincide. A vertex's total raw incident conductance is a different quantity.

Unbiasedness implies `E tr(Y)=r=d−1`. The exact orthogonal decomposition

\[
\|Y-P\|_F^2
=\left\|Y-\frac{\operatorname{tr}Y}{r}P\right\|_F^2
 +\frac{(\operatorname{tr}Y-r)^2}{r}
\]

shows which scalar error term disappears when every outcome has trace `r`. Fixing trace can nevertheless constrain other improvements; it does not establish global optimality. Scaling a positive preconditioner changes its trace without changing its condition number. Also, `ρ=‖Y−P‖₂<1` gives the bound `κ(Y|u⊥)≤(1+ρ)/(1−ρ)`, but ordering `ρ` need not order actual condition numbers or PCG iterations.

## Why the later-parent probability uses a sum

Fix source `i`, let `m_i=d−i`, `S_i=Σ_{j>i}a_j`, and require exactly one later-parent edge. Define its exact normalized atom

\[
M_{ij}=c_{ij}R(e_i-e_j)(e_i-e_j)^TR.
\]

Choose `j` with probability `q_ij>0` and emit `c_ij/q_ij`. Its expected contribution is fixed, while its variance is

\[
\sum_{j>i}\frac{\|M_{ij}\|_F^2}{q_{ij}}
 -\left\|\sum_{j>i}M_{ij}\right\|_F^2.
\]

Cauchy–Schwarz gives `Σ t_ij²/q_ij ≥ (Σ t_ij)²`, with unique equality at

\[
\boxed{q_{ij}=\frac{a_i+a_j}{\sum_{\ell>i}(a_\ell+a_i)}=\frac{a_i+a_j}{m_i a_i+S_i}}.
\]

This is optimal for this **one-parent component**. Independent centered components have additive variances, making the formula useful in a larger construction. It is not an optimum over arbitrary correlated trees. Its emitted edge has constant normalized trace `(m_i a_i+S_i)/A` for that source, regardless of the parent chosen.

The exact improvement over the ordinary GKS parent law is also calculable.
Let `U_i=Σ_{j>i}1/a_j`. For the same source and later-neighbor set,

\[
J_{\mathrm{GKS},i}-J_{q,i}
=\frac{a_i^2}{A^2}\left(S_iU_i-m_i^2\right)\ge0.
\]

The inequality follows from Cauchy–Schwarz; equality holds exactly when all
later weights are equal, including the trivial one-parent case. Independent
source errors add, so using this rule for every source gives a tree whose
ideal local `J` is no larger than GKS's. This optimizes probabilities over all
positive categorical parent laws, not only uniform draws.

For every fixed cycle core, the same inequality applies to the light
attachments. Optimizing each method's cutoff over the same heavy-suffix family
preserves it: evaluate the q-rule at the GKS rule's best cutoff, then minimize.
Thus both tree-q and cycle-q have local Frobenius guarantees against their
respective independent-GKS comparators. This does not guarantee better
conditioning, PCG iteration counts, finite-precision behavior or total time.

## A heavy cycle and all possible suffix cuts

For a sorted heavy suffix `H` of size `h≥3`, draw a uniform Hamiltonian cycle. Every core edge has inclusion `2/(h−1)` and receives conductance `(h−1)c_ij/2`. Attach every lighter vertex to one later parent using the probability above. The cycle is internally dependent; the light parent choices and cycle draw are mutually independent. Every realization is connected, has `d` edges, and has normalized trace `d−1`.

Writing `Q_H=Σ_{i∈H}a_i²`, its actual relative-Frobenius objective is

\[
J_H=\frac{
 \sum_{i\notin H}(m_i-1)a_i(2S_i+m_i a_i)
 +\frac{(h-3)(h-1)}2 Q_H}{A^2}.
\]

Evaluating this expression for every heavy suffix selects the best cut **within this family**. Two suffix moments and one prefix accumulation make the scan linear after sorting. It is not a largest adjacent-ratio heuristic, nor a search over all graph distributions.

A useful sufficient condition is provable. Suppose a current core has `h≥4` vertices, smallest weight `x`, and remaining weight sum/square-sum `S,Q`. Peeling `x` into a parent attachment changes the objective by

\[
\Delta J=\frac{4(h-2)xS+(h-1)^2x^2-(2h-5)Q}{2A^2}.
\]

If all current weights are at most `2x`, then `Q≤2xS`, so the numerator is at least `2xS+(h−1)²x²>0`. Every later suffix inherits this range bound. Therefore **`max(a)/min(a)≤2` guarantees that the full cycle minimizes this family's objective**. The condition is sufficient, not necessary; it is not a global spectral-optimality theorem or a fitted runtime threshold.

For a uniform Hamiltonian **path** on an `h`-vertex core, edge inclusion is `2/h`, so the inverse-probability conductance is `h c_ij/2`. Its core variance is

\[
\frac{(h-2)\,[h(h-2)Q_H+S_H^2]}{2(h-1)A^2},
\]

which is zero at `h=2`. A path saves one edge relative to a cycle and therefore requires a different budget comparison.

## Exact uniform weights: a global result with a precise scope

For equal weights and `d≥3`, a uniform Hamiltonian cycle globally minimizes expected relative-Frobenius error among **all unbiased random nonnegative weighted Laplacians with at most `d` distinct edges per outcome**. Probabilities and support-dependent weights may vary freely.

In normalized coordinates `P=I−11ᵀ/d`, write the outcome edge weights as `z_e` and weighted degrees as `δ_i`. Unbiasedness gives `Eδ_i=(d−1)/d` and `EΣ_e z_e=(d−1)/2`. Jensen and Cauchy–Schwarz imply

\[
\mathbb E\|Y\|_F^2
=\mathbb E\sum_i\delta_i^2+2\mathbb E\sum_e z_e^2
\ge\frac{(d-1)^2}{d}+\frac{(d-1)^2}{2d}.
\]

Since `EY=P`, subtracting `‖P‖²_F=d−1` gives

\[
\boxed{J\ge\frac{(d-1)(d-3)}{2d}}.
\]

A uniform Hamiltonian cycle attains equality with every normalized selected edge weight `(d−1)/(2d)`. All degrees and total weight are then constant. This proves one optimum, not uniqueness. It does not cover nonuniform inputs, the `d−1`-edge tree budget, spectral norm or conditioning.

## Weighted Prüfer already has the same plus-weight edge sizes

For iid Prüfer symbol probabilities `θ_i>0`, `Σθ_i=1`, the tree law is

\[
\Pr(T)=\prod_i\theta_i^{\deg_T(i)-1}.
\]

Its edge marginals and adjacent joint marginals are

\[
\pi_{ij}=\theta_i+\theta_j,\qquad
\pi_{ij,ik}=\theta_i(\theta_i+\theta_j+\theta_k),
\]

and disjoint edges have zero covariance. Standard weighted Prüfer uses `θ_i=a_i/A`, hence `π_ij=t_ij` and selected conductance `c_ij/π_ij=1/R_ij`. Every selected edge has unit normalized trace and every tree has trace `d−1`.

Thus the **sum-of-endpoint-weights idea is already present in weighted Prüfer**. The later-parent rule normalizes those sizes within each source's eligible parents; weighted Prüfer couples all edges into a different joint tree distribution.

For fixed inverse-marginal weights, full variance includes

\[
J=\sum_e(1/\pi_e-1)\|M_e\|_F^2
 +2\sum_{e<f}\left(\frac{\pi_{ef}}{\pi_e\pi_f}-1\right)
 \langle M_e,M_f\rangle.
\]

Marginals alone do not optimize the second term. Disjoint-edge inner products vanish; incident edges `ij,ik` have inner product `a_j a_k/A²`. Their correlations matter.

Even within the Prüfer family, standard symbols need not minimize `J`. For weights `(1,1,8)`, standard `θ=(1/10,1/10,4/5)` gives `J=16/81`. Using `θ=(2/21,2/21,17/21)`, still assigning `c_ij/(θ_i+θ_j)`, gives `J=7117/36100`, smaller by `1123/2924100`. All three outcomes remain positive, connected and unbiased. This exact counterexample establishes possible improvement; it does not establish that the alternative is optimal or useful in PCG.

## What can be optimized while preserving unbiasedness?

For fixed nonzero matrix atoms, one-draw importance sampling proportional to their Frobenius norms minimizes the inverse-probability estimator's second moment. It does not enforce a tree topology. For a whole correlated tree law `p_T` with normalized outcomes `Y_T`, choosing

\[
q_T\propto p_T\|Y_T\|_F,\qquad \widetilde Y_T=(p_T/q_T)Y_T
\]

preserves the mean and minimizes the second moment within this resampling/scaling family. The resulting `q_T` generally is not representable by iid Prüfer symbols and may be expensive to sample.

For fixed support probabilities `p_H`, optimizing nonnegative outcome-dependent weights under `Σ_{H∋e}p_H w_{H,e}=c_e` is a convex quadratic problem. Freeing the probabilities and using `z_{H,e}=p_H w_{H,e}` gives the convex perspective objective

\[
\sum_H\frac{z_H^T G_H z_H}{p_H}-(d-1),
\quad \sum_Hp_H=1,\quad\sum_Hz_{H,e}=c_e,
\]

where `G_H` is the Gram matrix of normalized edge directions. This is the model behind the included [tiny-star certificates](reconciliation/optimality/comparison.json). Global claims require enumerating or otherwise certifying the declared support population. Strict positivity may yield only an infimum.

An outcome-dependent correction must satisfy `E[δ_e | e included]=0` to preserve an edge mean. Conditional moments belong to the actual law: changing GKS parent probabilities, adding K2 correlations, or switching to Prüfer changes them. Reusing stale conditional means, sequentially changing the degrees used by a simultaneous correction, or repeating a pass can introduce bias. Input-dependent decisions are allowed when unbiasedness holds conditional on the current star/history. This local martingale property does not make the nonlinear factor, inverse or PCG iteration count globally unbiased.

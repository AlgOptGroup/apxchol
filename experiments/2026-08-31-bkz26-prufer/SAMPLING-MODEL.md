# What the local metrics explain

For positive neighbor weights $a_1,\ldots,a_d$, let $A=\sum_i a_i$ and $D$ be
the eliminated diagonal. Exact elimination adds the clique
$C=(A\operatorname{diag}(a)-aa^T)/D$. Every compared sampler emits a random
Laplacian $X$ with $\mathbb E X=C$.

To compare errors relative to the star's weights, use

$$
R=[(A/D)\operatorname{diag}(a)]^{-1/2},\quad
P=RCR=I-uu^T,\quad u_i=\sqrt{a_i/A},\quad Z=R(X-C)R.
$$

$P$ is the identity on the non-null subspace. Three different objectives are

$$
J=\mathbb E\|Z\|_F^2=\operatorname{tr}V,\qquad
V=\mathbb E[Z^2],\qquad \nu=\|V\|_2,\qquad \rho=\|Z\|_2.
$$

$J$ adds variance across directions; $\nu$ measures the worst population
variance direction; $\rho$ is the error of an individual draw. In particular,
$\nu$ is not $\mathbb E\rho$. None alone is a proved matrix-level PCG predictor:
later stars, their directions, ordering, rounding and dropping also matter.
These are sampled-clique metrics; deterministic SDDM excess stays outside $C$.
Their normalization also excludes the pre-existing graph between neighbors,
and their sums exclude residual sparsification noise.

## Why trace-cycle uses $a_i+a_j$

Sort weights increasingly. A light vertex $i$ chooses a later parent $j$ and
emits $c_{ij}/q_{ij}$, where $c_{ij}=a_i a_j/D$. This preserves each edge's mean.
Its normalized exact edge atom has trace $t_{ij}=(a_i+a_j)/A$. Minimizing that
independent source's second moment $\sum_j t_{ij}^2/q_{ij}$ gives

$$
q_{ij}=\frac{a_i+a_j}{\sum_{\ell>i}(a_i+a_\ell)}.
$$

A heavy suffix instead receives a uniform Hamiltonian cycle with inverse
inclusion weights. Trace-cycle evaluates the exact $J$ of **every suffix plus
its independent light attachments**, and chooses the best within that family;
after sorting this takes linear time. It does not optimize every possible
correlation or topology. For equal weights, the full cycle is globally optimal
for both $J$ and $\nu$ at the $d$-edge budget, but not necessarily for per-draw
conditioning. [Derivation and precise contracts](appendix/sampling-model.md).

## How far are the rules from optimum?

The table below uses four **degree-five stress/control profiles**, not whole
matrix families. “Reweighted” optimizes all topology-dependent positive edge
weights while freezing the original topology probabilities. “Global” also
optimizes those probabilities over every connected support at the same edge
budget. Entries round the certified bounds to six decimals; reported numerical
bounds, input weights and source hashes are in [the eight-case extract](conditional-optima.csv).

| Profile | Baseline / edges | Baseline $\nu$ | Reweighted optimum $\nu$ | Global optimum $\nu$ |
|---|---|---:|---:|---:|
| IPM-derived | GKS / 4 | 0.230935 | 0.152179 | 0.139801 |
| IPM-derived | Trace-cycle / 5 | 0.103308 | 0.103069 | 0.040399 |
| Chimera-derived | GKS / 4 | 0.480447 | 0.367111 | 0.289079 |
| Chimera-derived | Trace-cycle / 5 | 0.292476 | 0.288123 | 0.177064 |
| $(1,1,1,1,8)$ | GKS / 4 | 0.431506 | 0.250000 | 0.199607 |
| $(1,1,1,1,8)$ | Trace-cycle / 5 | 0.250000 | 0.250000 | 0.091787 |
| Uniform | GKS / 4 | 0.600000 | 0.600000 | 0.357143 |
| Uniform | Trace-cycle / 5 | 0.200000 | 0.200000 | 0.200000 |

All eight fixed-law problems were certified. Conditional weights improve GKS
by 24–42% on the three nonuniform profiles, but improve trace-cycle by at most
1.5%. Therefore weight corrections alone cannot close its large nonuniform
gaps: its topology probabilities must change too. Uniform trace-cycle already
attains its global optimum. Optimizing the objectives can conflict: the
IPM-derived GKS case reduces $\nu$ by 34% while increasing $J$ by 7.5%.

Compare **absolute errors before relative gaps**. On $(1,1,1,1,8)$, GKS has
$\nu=0.431506$, Prüfer $0.246914$, and trace-cycle $0.25$. Their gaps to the
matching-budget optima are respectively 116%, 24%, and 172%: the larger cycle
gap does not mean its absolute error exceeds GKS's. Tree rules have four edges;
the cycle rule has five. [Earlier four-rule scores](appendix/sampling-model.md).

Inspecting optimal supports provides a concrete direction. The one-heavy-vertex
five-edge optimum mixes hub-spokes plus a chord with four-cycle-plus-leaf
structures, using weights that depend on the complete outcome. In contrast,
trace-cycle permanently leaves one particular light vertex as a leaf. Its
fixed incident probabilities impose $\nu\ge1/4$; correlations elsewhere and
conditional reweighting cannot remove that bound. Changing which vertices
receive leaf/core roles is therefore a useful direction. This does not revive
earlier unsuccessful hub/chord heuristics unchanged, or establish a new
matrix-solve improvement.

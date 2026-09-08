# Local sampling model

These are ideal-arithmetic statements about one star; implementation and
matrix-solve checks are [separate evidence](reconciliation-details.md).
Take $d\ge2$ positive neighbor weights $a_i$, total $A=\sum_i a_i$, and pivot
$D\gt0$. The exact clique and normalization are

$$
C=(A/D)\mathrm{diag}(a)-aa^T/D,\qquad
R=[(A/D)\mathrm{diag}(a)]^{-1/2}.
$$

For $u_i=\sqrt{a_i/A}$, $P=I-uu^T=RCR$ is the identity on $u^\perp$.
A sampled Laplacian $X$ gives $Y=RXR$ with $Yu=0$. We measure

$$
J=\mathbb E\lVert Y-P\rVert_F^2
=(D/A)^2\mathbb E\sum_{i,j}\frac{(X-C)_{ij}^2}{a_i a_j},\qquad
\rho=\lVert Y-P\rVert_2.
$$

This includes diagonal and off-diagonal error. Extra SDDM diagonal terms are
outside the sampled-clique model. For an emitted edge of weight $w$, normalized
trace is $w(D/A)(1/a_i+1/a_j)$, not raw vertex degree. Its exact edge atom has
trace and Frobenius norm $t_{ij}=(a_i+a_j)/A$.

Unbiasedness requires $\mathbb EX=C$ and hence expected trace $r=d-1$.
The decomposition

$$
\lVert Y-P\rVert_F^2
=\lVert Y-(\mathrm{tr}(Y)/r)P\rVert_F^2
 +(\mathrm{tr}(Y)-r)^2/r
$$

explains one advantage of fixed trace, not global optimality. Scaling a
preconditioner changes trace but not condition number. When $\rho\lt1$, its
condition number on $u^\perp$ is bounded by $(1+\rho)/(1-\rho)$; smaller error
does not universally order actual conditioning or PCG iterations.

## Parent sampling and cycles

For increasing weights, source $i$ has $m_i=d-i$ later neighbors with sum
$S_i=\sum_{j\gt i}a_j$. Selecting edge $ij$ with probability $q_{ij}$ and
weight $c_{ij}/q_{ij}$ preserves its mean. Its variable second moment is
$\sum_j t_{ij}^2/q_{ij}$, minimized by Cauchy–Schwarz at

$$
q_{ij}=\frac{a_i+a_j}{\sum_{\ell\gt i}(a_i+a_\ell)}
      =\frac{a_i+a_j}{m_i a_i+S_i}.
$$

Ordinary GKS instead uses $a_j/S_i$ and emits $a_iS_i/D$.
Writing $U_i=\sum_{j\gt i}1/a_j$, the exact variance improvement is

$$
J_{G,i}-J_{q,i}=\frac{a_i^2}{A^2}(S_iU_i-m_i^2)\ge0.
$$

Independent source variances add. Thus tree-q has no larger local $J$ than
GKS, despite its worse iterations in the matrix screen. This is a one-parent
probability optimum, not an optimum over dependent tree laws.

For heavy suffix $H$ of size $h\ge3$, a uniform Hamiltonian cycle includes
each pair with probability $2/(h-1)$; emit $(h-1)c_{ij}/2$. Independently attach
lighter sources using q. Outcomes have $d$ edges and trace $d-1$. With
$Q_H=\sum_{i\in H}a_i^2$,

$$
J_H=\frac{\sum_{i\notin H}(m_i-1)a_i(2S_i+m_i a_i)
 +(h-3)(h-1)Q_H/2}{A^2}.
$$

Two suffix moments and one prefix accumulation evaluate every cutoff in
linear time after sorting. The same parent improvement guarantees cycle-q's
$J$ is no worse than independently attached cycle-GKS when both optimize
this same family. K2 uses correlated attachments and is outside that guarantee.

Weight ratio at most two is sufficient for a full cycle to minimize this
family's objective: peeling smallest weight $x$ from an $h\ge4$ core with
remaining moments $S,Q$ changes the numerator by
$[4(h-2)xS+(h-1)^2x^2-(2h-5)Q]/2$, positive when $Q\le2xS$.
This says nothing about choosing GKS outside that range. A uniform Hamiltonian
path instead has edge inclusion $2/h$ and weight $hc_{ij}/2$; it saves one
edge and requires a tree-budget comparison.

For $d\ge3$ and exactly equal weights, a stronger result holds: a uniform cycle globally
minimizes $J$ over unbiased nonnegative Laplacians with at most $d$ edges.
Jensen on weighted degrees and Cauchy–Schwarz on edge weights give
$J\ge(d-1)(d-3)/(2d)$, attained by equal-weight cycles. Define the population
variance matrix $V=\mathbb E[(Y-P)^2]$ and its norm $\nu=\lVert V\rVert_2$.
Symmetry gives $V=(J/(d-1))P$ for the uniform cycle. Since every unbiased law
satisfies $\nu\ge J/(d-1)$, this cycle also globally minimizes $\nu$, attaining
$(d-3)/(2d)$ at the same $d$-edge budget. These claims do not extend to
nonuniform inputs or tree budgets. They do not assert optimal per-draw error
$\rho$, expected error $\mathbb E\rho$, conditioning, or PCG iterations.

## Dependence and optimum comparisons

Weighted Prüfer draws iid symbols $\theta_i=a_i/A$. Its tree probability is
$\prod_i\theta_i^{\deg_T(i)-1}$, edge inclusion
$\pi_{ij}=\theta_i+\theta_j$, and adjacent joint inclusion
$\pi_{ij,ik}=\theta_i(\theta_i+\theta_j+\theta_k)$; disjoint edges have zero
covariance. Its inverse-inclusion weights give every edge unit normalized
trace. Thus plus-weight edge sizes already occur in Prüfer, but its joint
law differs from q's independent later-parent choices.

For normalized exact atoms $M_e$, variance depends on joint probabilities:

$$
J=\sum_e(1/\pi_e-1)\lVert M_e\rVert_F^2
+2\sum_{e\lt f}(\pi_{ef}/(\pi_e\pi_f)-1)\langle M_e,M_f\rangle.
$$

Incident-edge correlations matter; fixing marginals alone is not optimal.
An unbiased conditional correction must use the actual law's conditional
means. Reusing GKS corrections after changing q, K2, or Prüfer can introduce
bias. Local conditional unbiasedness does not make a nonlinear inverse unbiased.

Global certificates optimize support probabilities and outcome-dependent
positive weights over every connected simple support with a specified edge
budget. Using $z_{H,e}=p_Hw_{H,e}$ gives a convex perspective objective
$\sum_H z_H^TG_Hz_H/p_H-(d-1)$ under unit total probability and correct edge
means. Strict positivity can leave only an infimum. Eight exact certificates
cover 1072 supports; they certify $J$, not spectral error. Absolute comparisons
and spectral observations below concern four synthetic stars, not
representative matrix trajectories.

Each cell below is $(J,\mathbb E\rho,\mathbb E\kappa)$; $\kappa$ is the
condition number on the non-null subspace. Optimum columns bound $J$ only.
Tree methods use $d-1$ edges; K2 and cycle-q use $d$. Compare absolute scores,
not percentages measured from different-budget optima.

| Weights | GKS tree | Prüfer tree | Heavy-K2 | Cycle-q | $J^*_{d-1}$ / $J^*_d$ |
|---|---:|---:|---:|---:|---:|
| $1,2,3,4$ |(.790833,.685972,4.013829)|(.907937,—,—)|(.427500,.491604,2.493868)|(.420000,.494788,2.517766)|.675088 / .306638|
| $1,2,3,4,5$ |(1.297852,.838285,6.073073)|(1.460883,.867922,8.000236)|(.795407,.668342,3.341955)|(.786667,.658000,3.346350)|1.034674 / .643742|
| $1,1,1,1,8$ |(1.255208,.785266,4.607262)|(.948045,.648854,5.481470)|(1.044705,.732912,3.858142)|(.861111,.646008,4.405510)|.656387 / .352991|
| $1,1,1,8,8$ |(.733380,.607821,3.225138)|(.815174,.617482,6.391297)|(.624127,.541607,2.822966)|(.537396,.504731,3.140868)|.509902 / .294610|

Missing Prüfer spectral entries were not saved in this evidence set. [Spectral scores](../reconciliation/tiny-spectra.json) reaggregate recorded
eigenvalues; no spectral optimum is claimed.
K2 and q core sizes coincide here (3/4/3/3), despite different cutoff criteria.
For the one-hub star, q improves absolute $J$ over GKS (.861 versus 1.255),
yet sits farther above its stronger extra-edge optimum. The near-optimal
$d$-edge law mostly uses hub-star-plus-chord supports, sometimes placing the
hub as a leaf; the two-band law often puts only one heavy vertex on its cycle.
Their outcome-dependent weights suggest possibilities beyond a fixed heavy
core. Lower $J$ or mean $\rho$ need not lower mean conditioning, as K2/q show.

[Certificates and exact-law reproducer](../reconciliation/optimality/comparison.json) ·
[Saved heavy/K2 outcomes](../reconciliation/k2-local/index.json) ·
[Reproduction instructions](../reconciliation/README.md)

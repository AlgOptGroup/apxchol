# Degree moments on a fixed star

This historical diagnostic measures degree error, distinct from the whole-clique
relative variance in the [sampling model](appendix/sampling-model.md).
Neither metric determines complete-factor PCG convergence.

For weights $w_i$, total $W$, and pivot $D$, exact clique edges and degrees are
$c_{ij}=w_iw_j/D$ and $t_i=w_i(W-w_i)/D$. In increasing order, GKS source $r$
chooses a later parent with probability $w_j/S_r$, where
$S_r=\sum_{j\gt r}w_j$, and emits constant weight $h_r=w_rS_r/D$.
Only incoming choices vary the degree of vertex $i$:

$$
\mathrm{Var}(\widehat t_i)=\sum_{r\lt i}h_r^2(w_i/S_r)(1-w_i/S_r).
$$

With $Q_r=\sum_{j\gt r}w_j^2$, total variance is
$D^{-2}\sum_r w_r^2(S_r^2-Q_r)$. Swapping adjacent $x,y$ before a suffix
of weight $S$ changes it by $2xyS(x-y)/D^2$, so increasing order minimizes
this degree objective within the independent later-parent family.

Prüfer symbol probabilities $q_i$ give edge inclusion $p_{ij}=q_i+q_j$,
weight $h_{ij}=c_{ij}/p_{ij}$, and adjacent indicator covariance $-q_jq_k$.
Therefore

$$
\mathrm{Var}(\widehat t_i)=\sum_{j\ne i}h_{ij}^2p_{ij}(1-p_{ij})
-2\sum_{j\lt k;\ j,k\ne i}h_{ij}h_{ik}q_jq_k.
$$

The diagnostic is $[\sum_i\mathrm{Var}(\widehat t_i)/\sum_i t_i^2]^{1/2}$.
For four neighbors of weight $\epsilon$ and two of weight one, GKS error is
$2\epsilon+O(\epsilon^2)$; ordinary weighted Prüfer gives
$\sqrt{2\epsilon}+O(\epsilon)$. Prüfer sometimes omits the heavy-heavy edge,
whereas GKS always includes it. This is a local skew-weight failure mode.

[small_star_error.py](small_star_error.py) enumerates six-vertex examples.
[tiny_star_spectral.py](tiny_star_spectral.py) separately measures expected
relative spectral error on four-vertex examples, normalizing by the clique
pseudoinverse. Its fixed-seed multistart all-tree search is **not a global
certificate**. Changing the symbol exponent can improve degree RMS while
worsening spectral error. The later [exact-certificate comparison](reconciliation/optimality/comparison.json)
uses a different, explicitly certified Frobenius objective.

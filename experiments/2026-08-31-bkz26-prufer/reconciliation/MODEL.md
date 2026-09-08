# Factor representation used by the Julia adapter

For a pure Laplacian pivot with sorted positive weights $a_1,\ldots,a_d$, let $W=\sum_i a_i$, $r_j=W-\sum_{k\lt j}a_k$, $f_j=a_j/r_j$ for $j\lt d$, and $s=a_d/W$.

Public `LDLinv` stores the $f_j$ values and a final sentinel 1, with diagonal $\delta=a_d^2/W$. Its elimination column has coefficients

$$
\alpha_j=f_j\prod_{k\lt j}(1-f_k)=a_j/W,\qquad
\alpha_d=s,\qquad \delta/s^2=W.
$$

Thus the compressed representation gives the usual elimination column $[1;-a_1/W;\ldots;-a_d/W]$ with pivot $W$. The factor coefficients depend on the current star, not on the particular sampled residual tree. The same representation therefore works for GKS, CAST and exact-clique residual updates. Different residual updates change subsequent stars and factor columns.

The CAST adapter roots its decoded tree at neighbor $d$. Each $j\lt d$ reuses its old pivot-edge pair for the new parent edge; the parent gains one incidence, while the source replaces its pivot incidence. The root's old pivot incidence is removed. Because a CAST parent may occur earlier in the sorted column, the original neighbor IDs and weights are saved before any pairs are changed.

The exact-tail intervention first reuses the same $d-1$ edge pairs as a root star, then adds each missing non-root pair once with weight $a_i a_j/W$. Both new endpoints receive the matching priority-queue degree increment. It preserves the public factor recurrence. The experiment only observed degree-three nontrivial updates; a degree 16 guard bounds unexpected fill growth.

Native tests compare the default adapter with unchanged public GKS and its saved-order replay, check an independent direct factor application, verify exact weighted-Prüfer means, and check exact-tail solves against small dense operators. Weighted-cycle tests exercise degree-two arithmetic and random-stream matching. Floating-point implementation equality is tested separately from these real-arithmetic identities.

This is a reconstruction using public Laplacians.jl, not the unavailable CAST authors' executable. The broader metric and experimental conclusions are in [the reconciliation](../appendix/reconciliation-details.md).

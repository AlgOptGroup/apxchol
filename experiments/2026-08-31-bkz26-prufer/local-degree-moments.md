# Fixed-star degree and spectral metrics

This note analyzes one fixed pivot star. It explains the diagnostics in the
experiment README; it does **not** claim that a local degree or spectral metric
determines global PCG convergence.

Let the neighbor weights be `w_i`, let `W=sum_i w_i`, and let `D` be the pivot
diagonal. The exact Schur-clique edge and exact clique degree at neighbor `i`
are

```text
c_ij = w_i w_j / D,
t_i  = w_i (W-w_i) / D.
```

## What GKS fixes in every sample

Order the weights increasingly. For source `r`, define
`S_r=sum_{j>r} w_j`. GKS independently chooses one parent `J_r>r` with
probability `w_j/S_r` and emits an edge of weight

```text
h_r = w_r S_r / D.
```

The exact mass from source `r` to its later neighbors is also
`sum_{j>r} c_rj = h_r`. GKS therefore preserves that source contribution in
every sample. The destination is random, and vertex `i` also receives random
edges from sources `r<i`; its total sampled clique degree is not fixed.

Only incoming choices contribute to its variance:

```text
Var(t_hat_i) = sum_{r<i} h_r^2 (w_i/S_r) (1-w_i/S_r).
```

Writing `Q_r=sum_{j>r} w_j^2`, summing over vertices gives

```text
sum_i Var(t_hat_i) = (1/D^2) sum_r w_r^2 (S_r^2-Q_r).
```

The increasing order minimizes this objective within the independent
parent-to-suffix family. Swapping adjacent weights `x,y` before a suffix of
total weight `S` changes it by

```text
V(x,y)-V(y,x) = 2 x y S (x-y) / D^2.
```

Thus placing `x<=y` first never increases the total degree variance.

## Prüfer p-trees

Let `q_i>0`, `sum_i q_i=1`, and draw the iid Prüfer symbols from `q`. Its edge
marginal and the corresponding unbiased edge weight are

```text
p_ij = P({i,j} in T) = q_i+q_j,
h_ij = c_ij / p_ij.
```

BKZ26 uses `q_i=w_i/W`. For two distinct incident edges, the product-tree edge
indicators satisfy

```text
Cov(I_ij,I_ik) = -q_j q_k.
```

Consequently,

```text
Var(t_hat_i)
  = sum_{j!=i} h_ij^2 p_ij(1-p_ij)
    - 2 sum_{j<k; j,k!=i} h_ij h_ik q_j q_k.
```

The degree metric used in the exact model is

```text
RMS degree error = sqrt(sum_i Var(t_hat_i) / sum_i t_i^2).
```

Changing to `q_i proportional to w_i^alpha` remains unbiased after division by
`q_i+q_j`. It can nevertheless create large weights when a pair marginal is
small; the largest multiplier relative to `c_ij` is
`1/min_{i<j}(q_i+q_j)`.

## Normalized spectral metric

Let `C` be the exact clique Laplacian and `C_hat` one sampled weighted-tree
Laplacian. [`tiny_star_spectral.py`](tiny_star_spectral.py) also computes

```text
E || C^{+1/2} (C_hat-C) C^{+1/2} ||_2,
```

where `C^{+1/2}` is the square root of the Moore-Penrose pseudoinverse. This
normalizes away the exact clique's scale and measures the largest relative
quadratic-form error on its range. It is an expected one-pivot error, not a
bound on the accumulated factorization.

For each four-neighbor profile, the script enumerates all `4^(4-2)=16` labeled
trees. GKS and every p-tree point are exact finite sums. The “best-found
all-tree” comparator parameterizes a probability for every tree and uses a
fixed-seed, 15-start numerical search separately for each objective. Every
selected edge is weighted by `c_ij/p_ij`, where `p_ij` is that distribution's
own marginal. The search is deterministic and checks that all marginals remain
positive, but the nonconvex outer problem means the result is **not a
certificate of the global optimum**.

The resulting plot deliberately separates two facts:

- On a uniform star, the BKZ product-tree law beats GKS on both displayed
  metrics.
- On the graded and two-scale stars, GKS beats BKZ. Tilting toward
  `alpha=1.75` improves degree RMS, but it does not necessarily improve the
  spectral metric.
- The numerical all-tree comparator beats both named families in these tiny
  cases, so neither family exhausts the available estimator designs.

These are model results, separate from the matrix-level alpha sweep.

## Why strong skew separates the degree variances

Take four light neighbors of weight `epsilon` and two heavy neighbors of weight
one. As `epsilon` tends to zero, exact calculation gives

```text
GKS                    = 2 epsilon + O(epsilon^2),
BKZ, q proportional w  = sqrt(2 epsilon) + O(epsilon).
```

The BKZ heavy-heavy edge is omitted with probability about `2 epsilon`, yet
its conductance is order one, producing order-`epsilon` variance. GKS always
includes the edge between the two heaviest neighbors; its remaining degree
errors are only order `epsilon` in amplitude and order `epsilon^2` in variance.
Exact enumeration illustrates the growing ratio:

| epsilon | GKS RMS | BKZ RMS | BKZ / GKS |
|---:|---:|---:|---:|
| 1e-2 | .01951 | .13600 | 6.97 |
| 1e-4 | .00019995 | .014136 | 70.70 |
| 1e-6 | .000002000 | .0014142 | 707.11 |

This explains a local failure mode. Across many changing pivot stars, errors
also alter subsequent graph structure, and only the matrix experiments can
measure the resulting PCG behavior.

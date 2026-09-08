# Heavy-core samplers: research harness

This experiment implements custom eliminators through the public factorization
and CPU-solver interfaces. It does not change the production default. Its
historical source baseline is `fd77fc67`; later sampling studies must identify
their own source and objective rather than inherit this label.

A heavy suffix forms a uniform Hamiltonian cycle; light vertices select later
parents, which may themselves be light. For a core of size $h\ge3$, each pair
has inclusion probability $2/(h-1)$ and emitted weight equal to its exact
clique weight times $(h-1)/2$. The construction uses $d$ edges instead of a
GKS tree's $d-1$, so comparisons must report fill as well as iterations.

The seven arms cover GKS, K2, a tiny-star cycle gate, and several heavy-core
attachment/correlation variants. K2 coordinates choices of the two heaviest
receivers with a systematic phase: it preserves conditional marginal
probabilities and bounds selected counts, not weighted degrees.

**The heavy-core arms share the older degree-normalized moment cutoff.**
This is distinct from later relative-Frobenius cutoffs used by the cycle-q
study. A cutoff objective is neither a global spectral optimum nor a
prediction of PCG iterations. Unsupported inputs fall back to ordinary GKS.

Build this optional harness in a separate directory:

```sh
cmake -S experiments/2026-09-05-heavy-core -B build-heavy-core
cmake --build build-heavy-core -j6
./build-heavy-core/heavy_core_check experiments/2026-09-05-heavy-core/frozen-cutoffs.txt
./build-heavy-core/heavy_core_bench matrix.mtx arm threads seed rhs_count
```

The [checker](check.cpp) covers fixture edge means and frozen cutoff decisions;
[driver.cpp](driver.cpp) records setup, fill, iterations, timing, and
original-operator residuals with common generated right-hand sides. Checker
success alone does not establish a matrix performance win. Run timing on
Daint with source/binary/input identities and balanced controls.

The [full historical plan](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/experiments/2026-09-05-heavy-core/README.md) records the arm definitions and original
campaign denominator. These commands describe reproduction; no native
benchmark was rerun for this summary.

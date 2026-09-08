# Residual sparsification: accepted with a work gate

The historical study accepted a late residual rebuild with mean off-tree
sampling probability 0.25. It reduces setup and solve work; it does not by
itself fix parallel scaling. Both pooled layouts use the mechanism, with
`APXCHOL_RESIDUAL_SPARSIFY=0` available for comparison.

The rebuild coalesces parallel edges, retains a deterministic coarse
maximum-weight forest for connectivity, then samples remaining edges
independently with probabilities proportional to square-root weight.
Inverse-probability weighting preserves edge expectations. Indexed incidence
multiplicity uses unbiased stochastic rounding; directed AoS retains its own
distinct-neighbor semantics. A traffic estimate enables rebuilding only when
expected remaining work savings cover its cost. This is a workload gate,
not a matrix-name rule.

Historical Daint job 4545868 checked 18 full-solve cells: nine matrices at
T=36/72, all reported relative residuals below $10^{-8}$. Candidate/baseline
geometric-mean ratios were:

| Setup | PCG | Total | Stored fill |
|---:|---:|---:|---:|
| 0.8950 | 0.7138 | 0.8466 | 0.8487 |

Iterations increased by seven summed across the 18 cells; eleven were
unchanged. Setup speedup from 36 to 72 threads changed only from 1.099 to
1.109. Later job 4547244 checked 108 solves in 36 cells for exact stage
improvements, reporting setup 0.8947 and total 0.9172. RSS differences were
not decision-grade because lazy allocation changed residency tiers.

Probability 0.20 weakened the quality margin. An exact maximum-weight forest
cost more setup than the coarse forest, so neither alternative was retained.

For current validation, use the root build/test instructions and residual
sparsification fixtures. This directory contains a report, not a complete
portable campaign package. The [full historical record](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/experiments/2026-08-26-residual-sparsify/README.md) supplies source,
job, and raw-output provenance; its timings are not fresh measurements of the
integrated solver.

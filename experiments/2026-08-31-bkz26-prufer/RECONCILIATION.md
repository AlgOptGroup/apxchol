# Actual matrix families: fill, iterations, and local error

**16/16 factors passed the original-system residual tolerance $10^{-8}$.**
Each table entry is **stored entries (millions) / PCG iterations / summed $J$
(millions)** from that very factor. All use seed 42, 72 CPU threads, degree
quantile 0.2, fp32 factor storage and drop threshold $10^{-4}$; each matrix has
one common operator and RHS across rules. Capture instrumentation makes these
quality results unsuitable for timing comparisons.

| Family / matrix | GKS | Prüfer | Trace-cycle | Heavy-core K2 |
|---|---:|---:|---:|---:|
| Grid / grid_2000 | 21.865 / 47 / 5.655 | 21.968 / 52 / 5.174 | 26.904 / 28 / 4.569 | 26.955 / 27 / — |
| IPM / iter0040 | 4.623 / 46 / 0.717 | 4.716 / 58 / 0.763 | 5.151 / 27 / 0.621 | 5.184 / 27 / — |
| Social / as-Skitter | 16.749 / 21 / 8.538 | 16.862 / 25 / 7.432 | 17.252 / 15 / 5.202 | 17.203 / 14 / — |
| Chimera | 2.575 / 18 / 1.438 | 2.606 / 18 / 1.342 | 3.014 / 15 / 1.341 | 3.006 / 14 / — |

$J$ sums ideal clique-normalized Frobenius variances over each sampler's **own
complete elimination trajectory**, before rounding and dropping. Residual sparsification noise is not included. Heavy-core K2
has no full-census analytic $J$ implementation, so its entries remain missing.
These are four representative matrices, not averages over whole matrix families.
[All 16 rows, coverage and source/input hashes](actual-families.csv).

The result already rules out ordering these solves by summed $J$ alone: Prüfer
reduces $J$ on the grid and social graph but takes more iterations. The same
mismatch survives common-star comparison: on the GKS trajectories, Prüfer/GKS
$J$ ratios are 0.889 (grid) and 0.865 (Skitter). Thus different trajectories do
not fully explain it. [All common-star census scores](common-star-census.csv).
Trace-cycle
and K2 substantially reduce iterations; their additional emitted edge can also
change fill and all subsequent stars. This is not an isolated edge-count test.

Large stars matter here. Degree at least nine accounts for 100% of GKS's summed
$J$ on IPM, 82% on Skitter and 93% on Chimera, versus 1.4% on the grid. Thus the
larger-degree comparison is distinct from the earlier Spielman diagnostic.

On **common stars from the GKS trajectory**, the spectral ratios below compare
each rule with GKS (1.00). Each cell is **estimated census sum of local $\nu$ /
covered-strata weighted mean $\rho$**; lower is better. These are ratios of local-star aggregates,
not spectral norms of the factorization.

| Family | Prüfer | Trace-cycle | Heavy-core K2 |
|---|---:|---:|---:|
| Grid | 0.82 / 0.90 | 0.41 / 0.62 | — / 0.62 |
| IPM | 1.00 / 1.02 | 0.56 / 0.62 | — / 0.62 |
| Social | 0.84 / 0.90 | 0.53 / 0.53 | — / 0.54 |
| Chimera | 0.98 / 0.96 | 0.79 / 0.73 | — / 0.76 |

Prüfer's spectral estimates also improve on the grid and social graph despite
more PCG iterations. Thus replacing $J$ by these local spectral aggregates
does not resolve the discrepancy. Trace-cycle's larger local gains are
consistent with its iteration gains; this is evidence, not a predictor.

Across all 16 trajectories, replay covers 1,292 saved stars analytically and
1,149 with 16 draws per rule (73,536 draws). Entire degree/weight strata are
admitted under fixed dense-work limits, and inverse sampling probabilities
weight them back to their represented census. On the GKS trajectories, analytic
$\nu$ covers all degree-at-least-three strata; sampled $\rho$ covers 100%, 100%,
99.61% and 96.95% of eligible stars respectively. These coverage percentages
refer to represented populations, not exhaustive spectral evaluation.
The omitted $\rho$ strata contain 5.12% of GKS's full $J$ on Skitter and
28.80% on Chimera: their small star counts do not imply negligible error.
K2 has no analytic $\nu$ here. Sampling uncertainty prevents treating small
differences as rankings. [All 64 trajectory/rule rows, absolute scores,
coverage and estimated standard errors](family-samplers.csv) keep counterfactual
fill/iteration fields empty.

[Metric definitions and numerical optima](SAMPLING-MODEL.md) ·
[Earlier controlled CAST experiments](appendix/reconciliation-details.md).

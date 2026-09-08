# Implicit triangular-solve levels: accepted

Elimination rounds already describe contiguous factor-column ranges. Storing
those boundaries replaces duplicated forward/backward row-ID lists. The
executor and row arithmetic remain shared with ordinary computed levels;
missing or malformed round metadata uses that fallback.

Installed schedules own an immutable bounds snapshot. A pending setter affects
only a later setup. Backward traversal reverses level order while preserving
ascending row order within a level, and empty interior levels preserve
barriers. Critical-tail backward scheduling also reuses forward-stored row
IDs instead of retaining another copy.

The baseline storage reduction is two row-index arrays; critical scheduling
additionally avoids duplicated tail row and step-offset storage. This is an
allocation argument, not a measured peak-RSS guarantee.

Historical accepted Daint job 4555311 used four NUMA ranks in lockstep on the
same arm and matrix. It contained 72 records across 24 brackets: sixteen
setup-only and eight full-solve brackets, with 288 hash files. An earlier
40-record unsynchronized timing attempt was excluded.

| 64M grid, T=72 scope | Setup ratio | Solve ratio | Total ratio |
|---|---:|---:|---:|
| Setup cells | 0.9252 | — | — |
| Converged full solves | 0.9192 | 0.9682 | 0.9454 |

The setup cells reported triangular-solver setup ratio 0.7782; full-solve peak
RSS was 1.0086. Ratios are candidate/baseline. The report records matching
counts, iterations, and residuals, which alone are not factor-identity proof.

For current correctness validation, use the root build instructions and
`SpTRSVRoundRanges.*`, `SpTRSVLevelset.*`, and `CriticalSchedule.*` fixtures.
This directory contains the decision report rather than a portable timing
package. The [full historical record](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/experiments/2026-08-29-sptrsv-range-levels/README.md) preserves source, controls, and
measurement provenance.

# Incremental degrees: accepted behind an automatic traffic gate

The accepted block-greedy/directed-AoS path avoids repeatedly pruning the same
large adjacency lists. It starts from fresh degrees, aggregates eliminated
endpoint updates with a collective radix sort, and applies one non-atomic
update per endpoint. Fill updates reuse the histogram work. Survivor lists
are refreshed when dead-incidence debt reaches 10% of live entries.

Activation compares prune traffic with update traffic: below 10 disables the
candidate, 10–16 remains pending, and at least 16 plus sufficient work per
actual worker activates it. GPU-frontend work must hand back before activation.
This gate matters: indiscriminate incremental maintenance can cost more than
it saves.

Historical final Daint job 4550862 contained 960 records over ten matrices at
T=36/72. The report checked 20 paired cells, including counts, iterations, and
residuals; maximum reported residual was $8.7823\times10^{-9}$.

| Scope | Setup ratio | Total ratio |
|---|---:|---:|
| Six activating families, T=36 | 0.8975 | 0.9254 |
| Six activating families, T=72 | 0.9391 | 0.9561 |
| All twenty cells | 0.9549 | 0.9678 |

Ratios are candidate/baseline. A separate 48-record sentinel investigated an
inactive-grid timing outlier. The refresh policy reduced a historical local
memory penalty from about 27% to 4%; faster setup does not imply improved
self-scaling.

Rejected alternatives included eager atomic decrements, unrestricted delayed
activation, owner-hash buckets, wider radix bins, and irregular dirty-list
refreshes. Balancing selector work by degree also mispredicted cost because
selection scans can exit early.

For current validation, use root build instructions and fixtures covering
`AutoIncrementalDegreesPreserveTheFactorByteForByte`,
`IncrementalDegreeCacheDoesNotLeakIntoBkResidualLoop`, and
`EndpointRadixSortMatchesComparisonSort`. This report is not a portable
campaign package. The [full historical record](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/experiments/2026-08-27-incremental-degree/README.md) contains identities,
intermediate campaigns, and raw-result provenance.

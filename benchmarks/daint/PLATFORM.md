# Platform boundary

Daint GH200 is primary; [laptop](../archive/) data is historical.

The headline denominator is **27 matrices×18 series=486**. Canonical Linux
MATLAB CMG contributes 27 platform-unavailable numerical cells on ARM64.
Packed native CMG is a separate serial series, not a substitute hidden in that row.
[Coverage](coverage.json) and [CSV](results.csv) distinguish absent, unsupported,
failed, nonconverged and timed-out cells.

Requested T72 is a budget; record measured effective threads separately.
CPU/GPU scaling each contain 84 converged points on separately labelled revisions;
the older 189-record one-iteration diagnostic is unrelated. Neither scaling nor
rendering claims pending research changes are integrated.

Setup includes required preparation; CUDA initialization is separate. Whole-cell
deadlines and unknown peak memory are not fabricated solve times or zero memory.

ParAC’s recorded preparation includes avoidable adapter interchange/audit costs;
algorithm-performance comparisons remain provisional pending corrected measurements.

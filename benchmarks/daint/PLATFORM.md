# Platform boundary

Daint GH200 is primary; [laptop data](../archive/) is historical.

The headline declares **27 matrices × 20 series = 540 identities**. Canonical
MATLAB CMG accounts for 27 unavailable ARM64 cells. Packed native CMG remains a
separate serial series. [Coverage](coverage.json) and [values](results.csv)
distinguish missing, unsupported, failed, nonconverged and timed-out outcomes.

The four APX profiles are CPU GKS/trace-cycle at degree quantile 0.2 and
GPU-owned GKS at quantiles 0.8/0.2. GPU rows require actual device-factor adoption;
requested flags alone do not qualify. CPU-setup/GPU-solve and other sampler
variants remain in the [six-matrix comparison](SAMPLERS.md).

T72 is the requested budget; effective thread counts are recorded separately.
Scaling plots retain their own earlier source revisions and timing protocols;
they are not measurements of every refreshed headline profile.

Setup includes required preparation; CUDA initialization is separate. Unknown
memory stays unknown. Whole-cell deadlines are not per-solve lower bounds.
Competitor headline cells are preserved from the previous snapshot. In particular,
ParAC headline preparation still includes older adapter interchange/audit costs;
the corrected representative scaling study is labelled separately.

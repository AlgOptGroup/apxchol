# Platform boundary

Daint GH200 is primary; [laptop data](../archive/) is historical.

The headline declares **27 matrices × 19 series = 513 identities**. Packed serial
CMG replaces unavailable canonical MATLAB CMG in current charts; historical data
remain archived. [Coverage](coverage.json) and [values](results.csv)
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
ParAC Graph CPU preparation is reconciled for all 27 matrices. Its 26 successful
cells combine separately measured required preparation with the original native
setup/solve repetitions after operand/output identity checks. This is a composite
accounting correction; preparation variability and corrected whole-pipeline
peak memory are unknown. Orkut retains its measured 2,027.21-second AMD charge.
G3 Graph CPU/GPU calibration failures are classified as nonconverged.

Current GPU cells have no measured peak VRAM. Host RSS is separate and cannot
substitute for it; GPU memory charts require a dedicated memory-only campaign.

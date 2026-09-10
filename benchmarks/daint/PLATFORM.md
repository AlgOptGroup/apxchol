# Platform boundary

Daint GH200 is primary; [laptop data](../archive/) is historical.

The headline declares **27 matrices × 20 series = 540 identities**. Packed serial
CMG replaces unavailable canonical MATLAB CMG in current charts; historical data
remain archived. [Coverage](coverage.json) and [values](results.csv)
distinguish missing, unsupported, failed, nonconverged and timed-out outcomes.

The five APX profiles are CPU GKS/trace-cycle at degree quantile 0.2,
GPU-owned GKS at quantiles 0.8/0.2, and GPU-owned trace-cycle at 0.8. GPU rows require actual device-factor adoption;
requested flags alone do not qualify. The current trace27 measurements use
source230a65b6 and consumer010b4b1c, separate from both the older GKS campaigns
and the 96-call optimization acceptance study. Earlier CPU-setup/GPU-solve
and sampler variants remain in the [six-matrix comparison](SAMPLERS.md).

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

GPU process memory is measured in a dedicated 54-call study, separately from
headline timings. Polling reports lower bounds on whole-run maxima, with
54/216 GPU identities covered and the other 162 unknown. Host RSS is a separate
measurement; it cannot substitute for GPU memory. See [memory limits](GPU-MEMORY.md).

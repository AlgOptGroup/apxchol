# Daint is the primary benchmark platform

New laptop performance campaigns are retired because the local environment
is unstable. Existing laptop snapshots are historical evidence and are not
substituted into Daint cells or used to claim cross-machine speedups.

The common headline denominator remains **27 matrices × 18 series = 486 cells**.
Canonical MATLAB CMG has 27 explicit platform availability exceptions on this
ARM64 system; these stay missing numerical cells, rather than fabricated n/a
solver outcomes. Native packed CMG is a separate serial implementation and
retains its own row. Remaining missing cells, failures, numerical nonconvergence,
and phase timeouts are enumerated in coverage.json and results.csv.

Scaling uses the declared 12 matrices at T=1/2/4/8/16/36/72 on CPU and GPU:
**168 converged scaling cells**. This is not the historical one-iteration study.
The T72 headline budget includes solvers with lower effective parallelism;
results.csv records effective_threads when the measurement supplied it.

These figures render measured source revisions selected by the cell manifest.
They do not claim that pending experimental optimizations are integrated.
CUDA initialization is separate from setup. Input preparation required by each
solver remains charged to setup. Peak-memory intervals and phase deadlines are
not substituted for solve timing. This snapshot does not imply numerical success for any failed or absent cell.

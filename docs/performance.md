# Performance

Cross-solver results for 1.0 are in preparation and will be published here.
Until then, this site has one dated measurement: the
[batching case study](batching.md#larger-graph-youtube), which compares batched
and one-at-a-time solves on CPU and Metal on the YouTube graph.

To measure apxchol against other solvers on your own hardware, use the
[benchmark harness](../benchmarks/README.md). Its
[measurement contract](../benchmarks/README.md#measurement-contract) fixes the
inputs, checks every result against the original system's residual, and times
setup and solve separately.

CI regression runs compare a baseline and a candidate revision on the same
runner; see [hosted checks](../benchmarks/README.md#hosted-checks).

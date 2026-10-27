# Performance

The [native benchmark suite](../benchmarks/README.md) compares solvers across
graph families. Its [measurement contract](../benchmarks/README.md#measurement-contract)
sets common inputs, original-system residual checks, setup and solve timing
boundaries, and reproducibility requirements.

The [batching guide](batching.md#larger-graph-youtube) contains a dated CPU/Metal
case study, with timings, settings and provenance. It compares batched and
scalar calls reusing each route's factor on the YouTube graph.

CI regression artifacts compare baseline and candidate revisions on the same
runner. Read their numerical checks and paired controls alongside timing
changes; the [measurement contract](../benchmarks/README.md#measurement-contract)
explains the accuracy and timing requirements.

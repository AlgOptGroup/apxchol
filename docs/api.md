# API and compatibility

Starting with v1.0, apxchol preserves documented source-level use of these
interfaces throughout the 1.x series:

- C++ solve and factorize operations, their options and their results.
- [Python](../python/README.md) names exported through `__all__` and their
  documented methods and arguments.
- Documented [MATLAB/Octave wrappers](../octave/README.md) and CLI options.
- `apxchol::core` and the [documented CMake integration](building.md#use-from-cmake).

The [partitioner and eliminator extension contracts](extending.md), and the
graph/context operations they require, are **experimental**: they may change in
minor 1.x releases, and each change is noted in the [changelog](../CHANGELOG.md).

`generate_test_rhs` is a test and diagnostic helper outside this promise.

Numerical results and performance may change within the documented
[precision and convergence contracts](precision.md). This compatibility
policy begins with v1.0; it does not apply to earlier prereleases.

Some implementation headers must be installed for C++ templates. Installation
alone does not make every declaration public. Outside the documented contracts
above, `detail` namespaces, device kernels and capsules, `research_*` members,
internal mutators, benchmark scripts and diagnostic output are implementation
details.

There is no stable C++ binary ABI. Rebuild compiled C++ consumers with
compatible apxchol headers, compiler and standard library, OpenMP runtime,
and architecture options when updating the library.

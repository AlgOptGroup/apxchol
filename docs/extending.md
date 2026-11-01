# Extending the solver

These extension interfaces are experimental in 1.x: they may change in minor
releases, and each change is noted in the [changelog](../CHANGELOG.md).

Factorization alternates between selecting an independent set of vertices and
replacing each selected star by sampled Schur-complement edges. These are the
public **partitioner** and **eliminator** template seams. The resulting factor
can be used by the existing solvers without changing their PCG loop.
See [examples](../examples) for complete programs, the
[README](../README.md#build-and-run) for build commands and
[factor_options.h](../include/apxchol/solver/factor_options.h) for current
defaults and their rationale.

## Custom clique sampling

The contract is in [elimination.h](../include/apxchol/solver/elimination/elimination.h):

```cpp
struct my_eliminator {
    void sample_clique(std::span<apxchol::weighted_neighbor> neighbors,
                       double deg, std::uint64_t seed,
                       apxchol::edge_emitter out) const;
};
static_assert(apxchol::eliminator<my_eliminator>);
auto F = apxchol::factorize(A, my_eliminator{}, opts);
```

`neighbors` contains active neighbors only, with parallel edges merged into
one `(vertex, weight)` entry. Its order is unspecified. The span is mutable
and the library does not reread it after the call. Sort it if your rule needs
an order. `deg` includes SDDM diagonal excess, so it can exceed the sum of the
listed weights: use the supplied value.

For incident weights $w_i$, unbiased clique sampling requires expected emitted
weight $w_iw_j/\mathrm{deg}$ on each pair. This is a local expectation condition;
it does not make the factor or its inverse unbiased. A custom rule must also
preserve the properties needed for a usable preconditioner.

Emit edges in original vertex IDs using `out(u, v, w)`,
`out(deferred_edge{u, v, w})`, or a span of deferred edges. The emitter is
append-only; `out.reserve(n)` reserves room for additional edges. The
orchestrator applies the buffered edges after the parallel elimination phase.
It retains responsibility for factor-column construction.

One const eliminator instance is shared across concurrent calls. Each call
must remain single-threaded; use local or thread-local scratch and a random
generator seeded from `seed`. `apxchol::random_stream{seed}` provides
`next()` and `next_unit()`.

Shared mutable RNG state makes results depend on scheduling. Even with local
random streams, factors can differ across thread counts because parallel
accumulation and the selected order can differ.

`as_eliminator(lambda)` adapts a matching callable. An instance is passed by
const reference, allowing immutable configuration.
[custom_eliminator.cpp](../examples/custom_eliminator.cpp) demonstrates exact
clique elimination, including end-to-end solution checks.

## Custom selection and ordering

The concatenation of selected rounds defines the elimination order. Implement
[partitioner.h](../include/apxchol/solver/partitioner.h):

```cpp
struct my_partitioner {
    static constexpr bool degree_prepass = true;

    void find_partition(const apxchol::partition_graph& G,
                        std::span<const apxchol::node_index> candidates,
                        const apxchol::partition_context& ctx,
                        apxchol::selection& out);
};
static_assert(apxchol::partitioner<my_partitioner>);
```

With `degree_prepass=true`, the orchestrator prunes dead incidences, obtains
degrees, and applies the configured degree cap. `candidates` contains eligible
vertices; **`ctx.degrees[v]` is indexed by vertex ID**, not candidate position.
Without the trait, the orchestrator still prunes dead incidences, candidates
are the ascending active list and the degrees span is empty. The rule then owns any degree estimation or cap application.
See the header for optional `sample_bounded` and
`residual_handoff_threshold` traits.

`ctx.options` contains selection settings, `ctx.seed` supplies randomness,
`ctx.omp_threshold` gates parallel work, and `ctx.degree_tiebreak` asks the rule
to prefer lower-degree vertices in conflicts.
The rule is called once per round and may parallelize internally. This is the
opposite of the eliminator, whose individual calls are already parallelized.

Selected vertices must be pairwise nonadjacent. `out.add(v)` and
`out.remove(v)` permit concurrent operations on distinct vertices;
`out.contains(v)` is an unsynchronized read. Never read a mask location while
another worker writes it. For optimistic selection, use a stable snapshot,
compute conflict decisions, synchronize, and then apply changes. Both the
selected set and insertion order must be deterministic for the same graph,
context, and team size; dynamic arrival order must not choose the result.

`G` is a read-only view of the residual graph. `G.n()` is the vertex count,
including eliminated vertices, and `G.is_active(v)` tells whether `v` is still
live. `G.for_each_neighbor(v, f)` calls `f(u)` for each live neighbor `u`,
once per parallel edge, and returns that count; `G.degree(v)` returns the
count alone. Every query is a pure read, so any threads may query any vertices
concurrently, including from inside `f`.

Pass a partitioner instance to `factorize`, alone or with an eliminator; it is
taken by value and replaces `factor_options::partitioner`. Callable adapters
are `as_partitioner` and `as_prepass_partitioner`.
[custom_order.cpp](../examples/custom_order.cpp) shows complete integration.

## Reusing a custom factor

```cpp
apxchol::solve_options opts;
opts.backend = apxchol::solve_backend::gpu;
apxchol::solver solver(A, std::move(F), opts);
auto result = solver.solve(b);
```

An explicit GPU request uploads the factor for CUDA or Metal solves, without
factorizing again. Automatic adoption stays on CPU. The original matrix `A`
is still needed for PCG and residual checks.

The factor may come from CPU construction or from any solver's
`solver.factor()`, which returns its installed factor; pass it to the
constructor above. The destination rebuilds its own triangular-solve layout.
For a handoff between processes, use [saved solvers](serialization.md).

To precondition another iterative solver with the factor, construct
`solver(A, std::move(F), opts)` and call its `apply`; see
[using another iterative solver](usage.md#using-another-iterative-solver).

A `factorization` holds CSC factor columns
(diagonal first), permutation with `perm[original] = elimination_position`,
operator kind, round metadata and its storage form. Preserve these
conventions. The factor `L` is a read-only view; build factors with
`factorize` or load them from an archive. A custom factor uses
`storage = automatic`, so the solver applies its dropping and storage options.

Custom partitioners and eliminators see the residual graph only through
`partition_graph` and the neighbor span. The PCG recurrence, factor-column
formula, and triangular-solve kernels are not additional template seams.
Preserve the current defaults and [precision contract](precision.md); retired
options are not supported APIs.

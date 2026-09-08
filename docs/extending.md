# Extending the solver

Factorization alternates between selecting an independent set of vertices and
replacing each selected star by sampled Schur-complement edges. These are the
public **partitioner** and **eliminator** template seams. The resulting factor
can be used by the existing solvers without changing their PCG loop.
See [examples](../examples) for complete programs and [AGENTS.md](../AGENTS.md)
for build commands and current defaults.

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
must remain single-threaded; use local or thread-local scratch and a local
random generator seeded from `seed`. `apxchol::random_stream{seed}` provides
`next()` and `next_unit()`. Shared mutable RNG state makes results depend on
scheduling. A per-elimination seed does not guarantee equal factors across
thread counts: parallel accumulation and the selected order can differ.

`as_eliminator(lambda)` adapts a matching callable. An instance is passed by
const reference, allowing immutable configuration. The built-in
`tree_elimination` also supports `exact_clique_max_degree`.
[custom_eliminator.cpp](../examples/custom_eliminator.cpp) demonstrates exact
clique elimination, including end-to-end solution checks.

## Custom selection and ordering

The concatenation of selected rounds defines the elimination order. Implement
[partitioner.h](../include/apxchol/solver/partitioner.h):

```cpp
struct my_partitioner {
    static constexpr std::string_view name = "mine";
    static constexpr bool degree_prepass = true;

    template<apxchol::incidence_storage I>
    void find_partition(apxchol::graph<I>& G,
                        std::span<const apxchol::node_index> candidates,
                        const apxchol::partition_context& ctx,
                        apxchol::selection& out);
};
static_assert(apxchol::partitioner<my_partitioner>);
```

With `degree_prepass=true`, the orchestrator prunes dead incidences, obtains
degrees, and applies the configured degree cap. `candidates` contains eligible
vertices; **`ctx.degrees[v]` is indexed by vertex ID**, not candidate position.
Without the trait, candidates are the ascending active list and the degrees
span is empty. The rule then owns any degree estimation or cap application.
See the header for optional `sample_bounded` and
`residual_handoff_threshold` traits.

`ctx.options` contains selection settings, `ctx.seed` supplies randomness,
`ctx.omp_threshold` gates parallel work, and `ctx.cp` is optional profiling.
The rule is called once per round and may parallelize internally. This is the
opposite of the eliminator, whose individual calls are already parallelized.

Selected vertices must be pairwise nonadjacent. `out.add(v)` and
`out.remove(v)` permit concurrent operations on distinct vertices;
`out.contains(v)` is an unsynchronized read. Never read a mask location while
another worker writes it. For optimistic selection, use a stable snapshot,
compute conflict decisions, synchronize, and then apply changes. Both the
selected set and insertion order must be deterministic for the same graph,
context, and team size; dynamic arrival order must not choose the result.

The graph is mutable only to permit pruning already-dead incidences.
`G.adj(v)`, `G.edge_target(idx, v)`, and `G.is_active(u)` provide traversal;
`G.prune_and_degree(v)` and `G.prune_and_visit(v, f)` remove dead entries.
Do not otherwise mutate the represented graph. `G.m()` counts ever-added
edges, not current live edges.

Partitioner instances are passed by value. Callable adapters are
`as_partitioner` and `as_prepass_partitioner`. Direct template use requires no
registry change; by-name CLI/binding dispatch requires adding the type to
`partitioner_list`. [custom_order.cpp](../examples/custom_order.cpp) shows
complete integration.

## Reusing a custom factor

```cpp
apxchol::cpu_solver solver(A, std::move(F));
auto result = solver.solve(b);
```

Alternatively, install a factor with `apxchol::apx_cholesky::set_factor` and
apply it as a preconditioner. A `factorization` holds CSC factor columns
(diagonal first), permutation with `perm[original] = elimination_position`,
operator kind, and round metadata. Preserve these conventions. Enable
`set_keep_factor(true)` before installation if the preconditioner must retain
exportable factor arrays; consuming solver paths may release them.

Custom graph layouts implement `incidence_storage`. Directed pooled AoS is
the high-level default; storage selection must not silently change the
partitioner. The PCG recurrence, factor-column formula, and triangular-solve
kernels are not additional template seams. Consult
[implementation history](implementation-history.md) before reviving a retired
mechanism; historical options are not supported APIs.

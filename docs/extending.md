# Extending the solver

The factorization `P^T L L^T P ≈ A` is built by repeatedly picking an
independent set of vertices (a *partitioner*), eliminating each selected
vertex independently, and replacing its star by a sparsified Schur-complement
clique (an *eliminator*). Every selected vertex is its own singleton region —
multi-vertex regions existed historically but were removed; independence of
the selected set is what makes the round embarrassingly parallel. Both stages, and through them the elimination order, are
public template seams: you implement a small struct, pass an instance to
`factorize`, and hand the resulting factorization back to the PCG machinery.
Nothing needs to be registered or recompiled inside the library.

Each seam below has a compilable, self-verifying example under
[examples/](../examples).

## 1. Custom star-vertex elimination rule (`eliminator`)

When vertex `v` with active neighbors `(u_i, w_i)` is eliminated, the exact
Schur complement adds the full clique with weights `w_i·w_j / deg(v)`. The
eliminator decides which (sparsified) edges are actually added — the whole
quality/fill trade-off of the method lives here. The contract
([elimination.h](../include/apxchol/solver/elimination/elimination.h)):

```cpp
struct my_eliminator {
    void sample_clique(std::span<apxchol::weighted_neighbor> neighbors,
                       double deg,                    // weighted degree incl. SDDM excess
                       std::uint64_t seed,            // per-elimination random seed
                       apxchol::edge_emitter out) const;
};
static_assert(apxchol::eliminator<my_eliminator>);
```

- `neighbors` is the preprocessed view of v's star: only *active* neighbors,
  parallel edges already merged (each vertex once, weights summed —
  `weighted_neighbor{vertex, weight}`). Order is **unspecified**, the span
  is mutable, and the library never reads it again after your call — permute
  it or overwrite it entirely, as suits your rule. The built-in sorts
  ascending by weight itself. You do not get the raw
  incidence list — the active-filter/dedup pass is what makes multigraph
  elimination correct, and every rule needs it, so it is done once for all.
- `deg` is passed in rather than recomputed from `neighbors` because the two
  differ on SDDM inputs: the diagonal excess (conceptually an edge to ground)
  is part of the degree but has no neighbor entry.
- `out` is an append-only emitter: `out(u, v, w)` (or
  `out(deferred_edge{u, v, w})`) records one sampled clique edge in original
  vertex ids; `out(std::span<const deferred_edge>)` bulk-appends and
  `out.reserve(n)` pre-sizes, so the interface costs nothing over a raw
  vector. "Deferred" because the round runs in parallel: edges land in
  per-thread buffers and are applied to the shared graph in a batched second
  phase. The emitter is the only way to produce output — earlier edges cannot
  be inspected or removed.
- For an unbiased preconditioner, the *expected* sum of emitted edges should
  equal the exact clique (`w_i·w_j/deg` per pair). Deviating is allowed —
  you just change the preconditioner quality, not correctness of PCG.
- `seed` is this elimination's random seed — a pure function of the factor
  seed and the vertex, so the seed/draw sequence is identical across runs,
  schedules, and thread counts. Deterministic rules ignore it; randomized
  rules draw from it via `apxchol::random_stream{seed}` (`.next()`,
  `.next_unit()`) or by seeding any generator locally. This is what makes
  the sampling schedule-independent — there is no library-owned RNG stream.
  The resulting factor is bit-identical at a fixed thread count; at more
  than one thread, merged parallel-edge weights can differ in the final
  ulps (floating-point accumulation order).
- Called concurrently from many threads (each call must itself stay
  single-threaded — the parallelism is outside), and one instance is shared by all
  of them, so the struct must be const-callable and hold no mutable members;
  for per-call working memory use `static thread_local` locals, exactly as
  the built-in rule does for its prefix sums.

The factor column itself (`L(:,v) = w_i/sqrt(deg)`, diagonal `sqrt(deg)`)
is fixed and not the eliminator's concern.

Use it (the matrix overload builds the graph internally; graph-level
overloads exist too):

```cpp
auto F = apxchol::factorize(A, my_eliminator{}, opts);
```

A lambda works via the adapter:

```cpp
auto F = apxchol::factorize(A, apxchol::as_eliminator(
    [](std::span<apxchol::weighted_neighbor> nb, double deg,
       std::uint64_t seed, apxchol::edge_emitter out) {
        for (const auto& [u, wu] : nb)
            for (const auto& [v, wv] : nb) {
                if (u == v) break;                 // each unordered pair once
                out(u, v, wu * wv / deg);
            }
    }));
```

The eliminator is passed as an *instance* (not only a template parameter) so
configured state travels with it — e.g.
`tree_elimination{.exact_clique_max_degree = 8}`; it is taken by `const&` and
never copied, so you can keep and reuse your own instance across factorize
calls. To combine a custom eliminator with a non-default storage backend,
build the graph yourself:
`factorize(make_graph<graph<bstr_incidence>>(A), elim, opts)`.

The built-in rule is
[apxchol::tree_elimination](../include/apxchol/solver/elimination/elimination.h)
(random spanning tree of the clique, with an exact-clique mode at degree at
most `exact_clique_max_degree` — see
[factor_options.h](../include/apxchol/solver/factor_options.h)). Retired
variance-reduction variants (K-averaged trees, heavy-source oversampling)
reduced PCG iterations but never consistently paid for their extra setup
cost; both are reimplementable as custom eliminators.
[examples/custom_eliminator.cpp](../examples/custom_eliminator.cpp) plugs in
exact (zero-variance) clique elimination and shows the whole path end to end.

## 2. Custom independent-set selection and elimination order (`partitioner`)

Each round, the partitioner selects the vertices eliminated in that round.
The global elimination order — and therefore the permutation `P` — is exactly
the concatenation of the rounds, so *the partitioner is also the ordering
seam*. One interface, one shape, for every partitioner
([partitioner.h](../include/apxchol/solver/partitioner.h)):

```cpp
struct my_partitioner {
    static constexpr std::string_view name = "mine";  // dispatch string
    static constexpr bool degree_prepass = true;      // optional trait

    template<apxchol::incidence_storage I>
    void find_partition(apxchol::graph<I>& G,
                        std::span<const apxchol::node_index> candidates,
                        const apxchol::partition_context& ctx,
                        apxchol::selection& out);
};
static_assert(apxchol::partitioner<my_partitioner>);
```

The `degree_prepass` trait (default `false`) decides what the orchestrator
does before the call:

- `true` — the orchestrator prunes dead edges, computes fresh degrees, and
  applies the degree cap (`ctx.options.degree_quantile` /
  `degree_multiplier`): `candidates` is the *eligible* vertex list, and
  `ctx.degrees[v]` gives the current incident-edge count of every active
  vertex. The shipped scan rules (block_greedy, priority_greedy) work this
  way.
- `false` — no O(active + edges) pre-pass: `candidates` is the raw active
  list (ascending original-index order) and `ctx.degrees` is empty.
  baumann_kyng works this way — it estimates degrees from its hash sample
  and applies the cap to that estimate. The pre-pass pieces remain callable
  (`apxchol::prune_and_degrees`, `is_degree_threshold`) if you want them
  à la carte.

Two further optional traits shape how the orchestrator drives the rule:
`sample_bounded` (skip the `min_is_fraction` bailout and retry empty rounds
— for rules whose IS size does not scale with `|active|`) and
`residual_handoff_threshold` (when the main loop bails with more active
vertices than this, parallel BK rounds shrink the residual to the threshold
before the serial peel). Both are documented in
[partitioner.h](../include/apxchol/solver/partitioner.h).

`ctx.options` holds the user's selection preferences
([factor_options.h](../include/apxchol/solver/factor_options.h)) and is
populated identically in both cases — the trait only decides who acts on
the cap knobs. `degree_tiebreak` (prefer the lower-degree vertex when
adjacent candidates compete, biasing each round toward low degrees) is
never applied by the orchestrator: honoring it is always the rule's job.

Degrees come as a *vertex-indexed* span — `ctx.degrees[v]` is indexed by
vertex id, **not** by position in `candidates` — because selection rules
compare priorities of *adjacent* vertices, so they need degree-by-id
lookups for neighbors. They are handed over rather than queried from `G`
because a *current* degree is not an O(1) graph query — incidence lists
accumulate edges to eliminated vertices, so the truth requires the prune
walk the pre-pass already does for the cap. (Maintaining eager degree
counters in the graph instead was tried and lost at high thread counts:
the decrements become atomic contention in the elimination hot loop, and
stale O(1) list sizes overcount without bound. A contention-free
incremental scheme remains an open problem.)

`out` is the round's **selection** — insert/remove/membership in one
structure, so it serves both as the output and as the query side of
independence checking:

- `out.add(v)` / `out.remove(v)` — thread-safe for distinct vertices.
  `remove` exists for greedy rules that pick optimistically and drop
  conflicting picks in a later resolution pass (see block_greedy); it only
  clears the membership mask, and stale entries are dropped when the
  orchestrator collects the round;
- `out.contains(u)` — plain read (during a parallel selection this is
  deliberately racy, exactly like a shared mask — the shipped greedy rules
  rely on it).

Selected vertices **must be pairwise non-adjacent** (eliminated in parallel;
debug builds verify after every round).

Threading contract: `find_partition` is called **once per round** from the
orchestrating thread and is expected to parallelize internally (gate on
`ctx.omp_threshold`) — the opposite of the eliminator seam, where
`sample_clique` is called concurrently and each call must itself stay
single-threaded. The parallelism cannot be hoisted into the orchestrator:
the shipped rules parallelize over different structures (contiguous
candidate blocks, repeated whole-set passes, DAG levels, a hash sample), so
there is no common parallel skeleton to factor out.

The rest of `ctx`: `ctx.seed` is the determinism source — derive all
randomness as pure functions of `(seed, round, vertex)`, either as hash
priorities (like the shipped rules) or by seeding a generator you own; it
is a seed rather than an RNG object so results cannot depend on thread
scheduling. `ctx.omp_threshold` is the minimum work size before engaging
OpenMP, and `ctx.cp` an optional profiling tree (the orchestrator already
brackets `find_partition` with uniform labels, so most rules never touch
it).

`G` is non-const only so the pruning helpers can run. The queries
([graph.h](../include/apxchol/graph/graph.h)):

- `G.n()`, `G.m()` — vertex / edge counts;
- `G.adj(v)` — the incidence slots of `v` (may still contain edges to
  eliminated vertices until pruned);
- `G.edge_target(idx, v)` — the neighbor of `v` across incidence slot
  `idx`;
- `G.is_active(u)` — has `u` not been eliminated yet?
- `G.prune_and_degree(v)` — drop `v`'s dead edges, return its live degree;
  `G.prune_and_visit(v, f)` — same, calling `f(u)` for each live neighbor.

Pruning removes only edges to already-eliminated vertices — it never
changes the graph being represented, and it is the *only* mutation a
partitioner may perform; anything else is out of contract.

What matters for factor quality is *which round* a vertex is eliminated
in — the concatenation of rounds is the fill-reducing ordering, and biasing
early rounds toward low-degree vertices approximates the classic min-degree
heuristic (that is what the cap and tie-break are for). A vertex's position
*within* its round has no effect: the round is an independent set,
eliminated as one parallel batch.

Instances may be stateful (configuration, scratch); pass configured
instances directly. Unlike the eliminator (taken by `const&` and never
copied), the partitioner instance is taken **by value** — state mutated
during the run is not observable in the caller's instance afterwards;
reference it through a member pointer if you need to inspect it post-run.
Lambdas: `as_partitioner(fn)` or
`as_prepass_partitioner(fn)` — identical callable signature, differing only
in the `degree_prepass` trait they declare; zero-overhead template
wrappers, no type erasure.

[examples/custom_order.cpp](../examples/custom_order.cpp) implements a
priority-driven greedy IS that eliminates vertices in a user-supplied order —
the recipe for injecting an ND/AMD/domain-specific ordering. To make a custom
partitioner reachable *by name* (`factor_options::is_select`, the CLI, the
bindings), add its type to `partitioner_list`
([partitioner_list.h](../include/apxchol/solver/partitioner_list.h)) and
rebuild.

The serial tail after the parallel rounds has its own small ordering knob:
`factor_options::residual_peel` (`natural` / `min_degree` / `bk_serial`).

## 3. Using a custom factorization in the solver

A `factorization` produced by any of the above plugs back into the tuned
solve paths:

```cpp
// Full PCG (parallel SpMV + level-set SpTRSV), factor once / solve many:
apxchol::cpu_solver slv(A, std::move(F));
auto res = slv.solve(b);
```

Or, as an alternative to the above, as an Eigen-style preconditioner object
(both snippets consume `F`, so use one or the other):

```cpp
apxchol::apx_cholesky M;
M.set_factor(std::move(F));      // runs the SpTRSV setup
Eigen::VectorXd z = M.solve(r);  // one application of M^{-1}
```

`factorization` itself is an open struct: the factor `L` (CSC, permuted
space, diagonal first per column), the permutation `perm`
(`perm[original] = elimination position`), the SDDM flag, and per-round
statistics — see
[factorization.h](../include/apxchol/solver/factorization.h).

## 4. What you cannot swap (currently)

The factor-column formula, the PCG loop itself, and the SpTRSV backend are
not templated seams. The storage backends behind `graph<>` are — implement
the `incidence_storage` concept
([incidence_list.h](../include/apxchol/graph/incidence_list.h))
if you need a custom adjacency structure, and extend the `graph_storage`
runtime dispatch in `src/factorization.cpp` if it must be reachable from the
enum.

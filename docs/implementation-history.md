# Implementation decisions

This is a concise index of accepted mechanisms and rejected alternatives.
The measurements below are **historical reports**, not a new validation of the
current integrated solver. Ratios are candidate/baseline; below one is faster
or smaller. Current contracts and build instructions live in
[AGENTS.md](../AGENTS.md), [factor_options.h](../include/apxchol/solver/factor_options.h),
and [the extension guide](extending.md).

The [complete earlier record](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/docs/implementation-history.md) preserves campaign details,
obsolete configuration examples, raw-result locations, and longer rationale.
Those historical instructions must not override current source or benchmark
protocols. Both the committed Daint snapshots and laptop timings describe
historical revisions; neither establishes current integrated performance.

## Historical thermal measurement helpers

The scripts in `benchmarks/dev/bench_stable*.sh` and
`bench_reference_table.sh` retain their historical configuration and separate
per-stage summaries. Cooldown, affinity, boost and governor settings do not
prove stable clocks or comparable independent measurements. IQR/median is
observed spread, not an error bar, confidence interval or acceptance threshold.
A small IQR does not establish a causal A/B difference or absence of carry-over.
Current performance decisions require the complete denominator, coherent timing
records and provenance specified in [AGENTS.md](../AGENTS.md).

The indexed `bg+tree[vec_pool]` labels in those helpers and the CUDA context
prewarm measurements describe the configurations originally used. Do not
relabel historical measurements as AoS. Current high-level default storage is
`vec_pool_aos`; that default does not retroactively change old observations.

## Serial residual ordering

The optional `min_degree` peel is a historical lazy degree-heap heuristic.
It revalidates the popped vertex but does not update every neighbor after
elimination. A decreased degree can remain hidden under a stale larger heap
key, so the selected pivot need not have the current minimum degree. Older
comments claiming exact lowest-degree selection, guaranteed best fill, or an
unconditional O(n log n) total bound were too strong. Historical measurements
must retain the behavior of the implementation that produced them.

Natural ordering remains the default; `bk_serial` samples candidates and selects
the smallest live degree it observes. Neither alternative is established as
universally dominated. Changing the heap policy changes pivot order, factor
quality and cost, and requires its own correctness and performance validation.

## GPU solve and storage

**Dataflow triangular solves replaced level launches and proprietary analysis.**
The library uses its own persistent dataflow kernel, with an epoch-tagged
value per row and topologically ordered work tickets. This avoids a kernel
launch for every level and an analysis buffer proportional to factor nnz.
Our library links `cudart`; benchmark competitors may independently require
cuSPARSE/cuBLAS. Earlier records describing a selectable cuSPARSE backend are
obsolete. The design and ordering constraints are documented in
[cuda_dataflow.h](../include/apxchol/solver/sptrsv/cuda_dataflow.h).

**Long rows require segmentation.** On hub-heavy factors, repeatedly loading,
waiting, and accumulating chunks in one row placed substantial work on a
serial dependency chain. Segment tasks expose that work concurrently and a
finalizer combines their partials. Historical laptop measurements reported
com-Orkut dropping from 726 to 136 ms per iteration and kron from 17.28 to
8.97 ms; these are workload-specific observations, not current benchmark
promises. Longer segments hid less latency. Packed fp16 arithmetic failed
accuracy checks, while mixed hub precision and packed-load alternatives added
register, memory, or integer-instruction costs without a useful overall win.

**GPU PCG owns its arithmetic and reductions.** Fused operator application and
fixed reduction structure avoid proprietary solver dependencies. Outer PCG
vectors and reductions are FP64. The triangular-solve dataflow state is FP32;
its tagged representation does not provide an FP64 triangular-solve path.
The former untested FP64-factor-storage build choice was removed rather than
advertised as supported. These are separate choices; see [precision and
storage](precision.md).

**Low-precision storage retains numerical compensation.** CPU and GPU share
column scaling, a separate fp32 diagonal, subnormal flushing, and compensation
for storage rounding. fp16 defaults on for GPU and off for CPU. Keeping the
diagonal in fp16 increased historical iter0040 iterations from 44 to 50;
omitting rounding compensation increased 45 to 64. A single runtime storage
choice replaced separate binaries and GPU-only aliases. Scalar gathers won
over the tested SIMD gather variant for short CPU rows.

**Factor dropping preserves column sums.** The shared compensated drop,
default threshold `1e-4`, reduces stored work while transferring the discarded
sum into the diagonal treatment. Plain removal had a large iteration penalty
on iter0040. Compensated dropping is a numerical mechanism, not interchangeable
with arbitrary truncation.

**CUDA initialization is accounted separately.** Cold context creation once
polluted setup comparisons. Prewarming and consistent solver-complete timing
boundaries fix the measurement protocol; they are not a solver optimization.
Likewise, polling `nvidia-smi` inside a timed invocation perturbs GH200 setup.
Peak-memory diagnostics run separately from timing.

## GPU setup decisions

**Complete CPU/GPU routes (2026-10-02).** The public mixed CPU-setup/GPU-solve
workflow is removed. `cpu_solver` and Eigen preconditioning stay entirely on
CPU in CUDA builds; one-shot GPU solves require device-owned setup and operator
preparation. A single backend option selects the route before setup. Automatic
selection considers configuration only, and a selected GPU route reports errors
rather than retrying on CPU. Explicit low-level factor import remains useful for
independent handoff and validation; no disk format or public session API is added.
This is a contract simplification, not a claim of universal GPU speedup. The
frontend and owned-setup measurements below retain their historical profiles.

**The fixed-priority GPU frontend was removed.** A historical 90-record forced
campaign produced setup ratio 1.102. An independently implemented asynchronous
MIS variant preserved the recorded factors and iterations in 54 bracketed
runs, but selector time was 1.057 and enclosing partition time 1.069; setup
was near noise at 0.989. Keeping another mode was not justified.

**The block frontend is explicit opt-in.** An automatic thread-count governor
was removed: host-thread count must not silently choose a different algorithm.
The historical 270-record GH200 study found single-RHS total ratios
0.711/0.762/0.826/0.927 at T=1/2/4/8, but 1.050 at T=16 and 1.201 at T=72,
with individual matrix losses even at low thread counts. These results support
an explicit research choice, not universal GPU dominance. Batching transfers
and reusing resident topology were useful; a tested 2D-copy alternative did
not justify retention.

**GPU-owned setup remains explicit research functionality.** The internal
consuming path supports numerical rounds on device, direct CSC/operator
preparation, bounded selection, local row batches and device factor/plan
construction. Host metadata and validated public/export/fallback paths remain.
The low-yield residual step preserves the CPU coalescing/one-forest/Bernoulli-HT
law and ordered normalization; a different selector can reach a different
sparsification boundary. Exact activation and contracts are in
[AGENTS.md](../AGENTS.md#architecture-and-contracts).

Ordinary setup preserves host allocation order and temporary lifetimes. PCG
reuses the touched host RHS buffer for the returned solution after the upload
has completed, avoiding a second full-sized staging allocation. GPU-owned
adoption and operator construction complete their queued setup before returning;
common benchmark boundaries synchronize every route.

FP32 validation covers the complete CUDA suite, focused memory/synchronization
checks, and 66 converged original-system solves across two default-route and four
owned-route matrix comparisons. The owned measurements used degree quantile 0.8,
which was the default for both routes at the time; the cap is now chosen by route
(0.2 host, 0.5 GPU-owned) because a later two-machine sweep found 0.8 ties 0.5 on
the mean while costing 2.1-2.5x on kron_g500-logn16. This evidence supports the bounded opt-in
integration scope; FP64 acceptance and parity with Yves remain unestablished.

**The old per-pivot GPU trace sampler is retired.** Small rows retain the
item-based emission path and large rows retain the cooperative path. The old
`APXCHOL_GPU_TRACE_ITEMS` opt-out no longer selects an alternative. Storage
backends, solver object layouts and numerical controls are unchanged by this
focused removal; it is separate from the held broader storage cleanup.

**Large trace-cycle rows use cooperative warps.** A thread per whole star left
late rounds with few large stars poorly parallelized. Range checks, suffix
moments, cutoff evaluation, parent searches and edge writes share a warp.
The shuffle keeps one owner; parent draws use the indexed stream after its
actual rejection count. A monotone suffix repair preserves binary-search CDF
ordering. Floating-point folds and seeded factors can change; the intended
sampling law and input-only numerical fallback remain. Oversized-only rounds
use one warp per block; mixed rounds keep larger blocks.

The cumulative refinement passed the native/sanitizer checks and 96 paired
calls on three matrices: trace-cycle geometric-mean one-RHS total fell 3.1%,
primarily from
LiveJournal gains of 9.5–10.1%; solve time rose 0.8% and iterations 1.8%.
All 108 calls in the subsequent 27-matrix campaign passed original-system
residual checks. This is a setup improvement, not a change to the default
sampler or a universal one-RHS advantage over GKS.

## CPU triangular solves and factor memory

**Critical-tail scheduling addresses narrow late levels.** A persistent team
processes broad levels conventionally and dependency chains in the structural
tail without a barrier per tiny level. CPU schedules share row arithmetic and
storage handling. A smaller actual OpenMP team must enter the fallback before
any worker touches scheduled rows. Counter ordering and immutable schedule
snapshots are correctness requirements.

The strict historical GH200 campaign 4531713 checked 112 records across 28
cells and reported setup 1.0145, PCG 0.5117, and total 0.8091. All 28 cells
improved in that campaign; the result does not establish dominance on every
matrix or machine. Additional public schedule modes were not retained merely
to expose internal experiments.

**Implicit round ranges replace duplicated row lists.** Forward/backward
levels derived from elimination rounds need boundaries rather than two arrays
of every row ID. Backward traversal reverses levels while preserving within-level
row order; empty levels retain their synchronization meaning. Invalid or absent
metadata falls back to computed topological levels. Pending metadata setters
must not alter an already installed factor. The
[range-level experiment](../experiments/2026-08-29-sptrsv-range-levels/README.md)
records the accepted campaign and its limits.

**Allocate outputs for their writers and release transients at last use.**
Fully overwritten nnz-sized transpose/copy buffers need no serial
value-initialization or first touch. This does not permit reading uninitialized
storage: each retained entry must be written first. Public/custom/exportable
factors retain the arrays their contracts require. Temporary graphs and input
copies can be released after their final consumer. Clearing a vector alone
does not release its capacity, and a monotonic allocator must outlive factor
assembly. Exceptions raised inside OpenMP work must be captured and rethrown
outside the region.

**Extra passes need end-to-end evidence.** Shared count/transpose construction
was retained where useful. A parallel row-length collection experiment increased
a historical stage from 12.3 to 20.3 ms, while extra transpose passes could lose
at T=1. Reducing an operation count is insufficient if memory traffic or
synchronization increases.

## Graph construction, elimination, and residual work

**Directed pooled AoS is the high-level default.** It avoids an indexed edge
lookup and uses distinct-neighbor degree semantics; indexed pools retain
multiplicity semantics. A graph layout must not silently switch the selector.
Packed activity masks require coordinated word updates and the existing round
barriers. Full, unique, paired symmetric CSC inputs admit column-owned graph
construction with canonical lower weights; other input patterns retain the
general builder.

**Segmented pools avoid relocating adjacency storage.** Stable mmap-backed
segments let workers grow and copy their own slabs without relocating the
entire pool. Huge virtual reservations failed under address-space limits;
pointer-based alternatives added memory or gave mixed timing results. See the
[segmented-pool decision](../experiments/2026-08-26-segmented-pool/README.md).
Lazy adjacency allocations request huge pages only when the kernel's reported
PMD size is at most 2 MiB. GH200's 512 MiB granularity inflated first-touch work;
fully sized output buffers have a separate allocation policy.

**Incremental degree updates require aggregation and a traffic gate.**
Eager atomic decrements lost at high thread counts. The accepted directed-AoS
path sorts endpoint updates collectively and applies one update per endpoint,
refreshing survivor lists when dead-entry debt warrants it. It activates only
when saved prune traffic can pay for update work. Unrestricted activation,
owner-hash buckets, oversized radix bins, and irregular dirty-list refreshes
were rejected. The [incremental-degree study](../experiments/2026-08-27-incremental-degree/README.md)
records activation rules and the final campaign.

**Connectivity-preserving residual sparsification removes late work.**
Coalescing, a coarse maximum-weight forest, and unbiased off-tree sampling
retain connectivity while reducing fill traffic. The shipped mean sampling
probability is 0.25, applied only after a work-based profitability gate. Lower
probability reduced quality margin; an exact maximum-weight forest cost too
much extra setup. A separate coalescing policy was retired, while coalescing
remains part of this rebuild. See the
[residual-sparsification report](../experiments/2026-08-26-residual-sparsify/README.md).

### Blocked normalization

**Blocked normalization fixes a serial residual stage.** Ordered 16,384-item
partial sums and an ordered fold use one persistent team for normalization
updates. The same input gives the same result across thread counts, but the
arithmetic intentionally differs from a legacy serial fold. A historical
Orkut observation reduced that stage from about 762 to 45 ms and setup from
12.61 to 11.86 seconds. A broader campaign's failed timing controls cannot be
turned into an accepted aggregate. The integration report
[CPU-RESIDUAL-BLOCKED.md](../CPU-RESIDUAL-BLOCKED.md) separates valid observations
from remaining limits.

**Size elimination teams from actual available work.** A small independent set
can contain substantial adjacency work but still cannot use one full team per
pivot. The accepted rule combines pivot count, degree work, and available
threads; singleton elimination stays serial. Historical job 4532834 checked
72 bracketed first-iteration records and reported setup 0.9695. Separate job
4532858 checked 36 converged records across 18 cells. One-iteration timing was
never treated as converged-solve evidence. Splitting one high-degree pivot
requires separate evidence; it is not achieved by this team rule.

**Tail handoff depends on eligible-candidate yield.** Using all active vertices
as the denominator accidentally imposed a much stronger selection threshold.
Candidate-relative yield, protection near the handoff boundary, and retained
large selections avoid premature serial tails. Dense-residual adaptation is a
setup/fill/solve tradeoff; factor reuse can favor a later handoff. Defaults and
exact boundaries belong in factor_options.h, not duplicated instructions.

**BK retries empty sampled rounds.** An empty sample does not establish that
productive parallel elimination has ended. Retrying under the same bounded
empty-round budget reduced premature peeling and fill in the historical
study. Seed average degree from current prepass information, not `G.m()`:
ever-added edge counts drift away from live work. Reducing or scaling the BK
sampling constant with threads traded away quality and was not promoted.

**Canonical sorting remains the simple baseline.** Neighbors are ordered by
weight and vertex and sampled with exact cumulative weights. Approximate
weight buckets changed the law or factor trajectory. Even an exact coarse
radix variant's small total-time benefit did not justify a second algorithm
and crossover; it was retired. The
[coarse-order report](../experiments/2026-08-28-tree-coarse-order/README.md)
preserves evidence. Inverse-CDF lookup directories likewise did not justify
another path.

## Correctness boundaries and rejected shortcuts

**Operator interpretation is shared; CLI guessing is not.** Validation checks
finite values, symmetry, diagonal signs, and M-matrix structure. Diagonal
dominance is reported rather than required for every accepted input. Positive
mass lumping changes the preconditioner only: solves and residuals use the
original operator. A supported non-diagonally-dominant example is not a proof
for arbitrary SPD matrices. Bindings use the operator contract rather than
CLI adjacency detection. Only generated Laplacian right-hand sides are
projected per connected component; explicit right-hand sides are unchanged.

**Deterministic selection needs synchronized snapshots.** Optimistic selection
and conflict repair cannot read concurrently changing mask entries. Stable
snapshots, full tie-breaking orders, and ordered application fix this. A tiny
fixture that never entered the parallel path once concealed the defect.
Randomized priority and root-set variants did not deliver sufficient setup
benefit to replace the accepted selector. Selection must repeat for the same graph, context and team size. This is
not a factor-identity guarantee: pooled adjacency arrival order can change
floating-point sums even at a fixed seed and thread count, particularly with
FP64 pool values. Such differences can propagate into sampled structure, not
only final ulps. Equal fill or iterations do not establish equal factors—compare
structure and value digests. Ownership-path comparisons must reuse one factor
or make an owning copy rather than independently factorizing twice.

**Removed knobs do not constitute supported alternatives.** Retirements include
compile-time low-precision storage, GPU-only fp16 aliases, fp16 diagonal and
rounding-compensation switches, the level-set/cuSPARSE library backends,
arrival-order CPU tails, and standalone residual coalescing. Historical FTZ
experiments found no relevant subnormal work and added a fork/join; they did
not justify changing process floating-point state. Preserve substantial
numerical mechanisms instead of reconstructing every rejected configuration.

## Evidence and follow-up

[Experiment READMEs](../experiments) summarize individual decisions; the
[benchmark guide](../benchmarks/README.md) defines current measurement and
rendering contracts. Large-grid results demonstrated better setup speedups
than several smaller problems, but do not isolate graph structure from size.
Research samplers and GPU residency work remain experiments unless their
integration and full-solve evidence are explicitly reported.

The earlier campaigns also established practical limits: setup-only records
cannot supply converged-solve scaling, successful scheduler exit is not a
numerical acceptance test, and unrelated ranks must survive another arm's
failure. Timing contamination invalidates the affected timing claim, not
necessarily its residual or structural observations. Preserve complete
planned denominators and original verdicts; label later subsets as
retrospective. A repaired parser can reanalyze saved outputs without rerunning
an unchanged numerical campaign. These lessons belong in the current concise
protocol, not in a growing duplicate operational manual.

## Indexed pooled storage retirement

The indexed `vec_pool` backend and its CLI/Python selection spelling are
retired. Use `vec_pool_aos`; the low-level `graph<>` default now also uses
directed AoS. This removes the separate indexed coalescer and multiplicity
sidecar. At that retirement, vector, string and forward-star remained available; custom
incidence/partitioner templates remain supported. Historical benchmark records
keep their original storage labels and are not relabelled as AoS.

That retirement preserved the storage-selection API and forward-star options.
Remaining enum values retain their prior numeric values; the removed value
is rejected rather than silently selecting another backend.


## Forward-star storage retirement

The linked-node `forward_star` backend, its `fs_compact_threshold` and
`fs_filter_append` fields, and their CLI controls are retired. Runtime value 1
and the old `forward_star` / `fwd_star` selections are rejected. The surviving
vec, bstr and directed-AoS enum values remain 0, 2 and 4; custom incidence
templates and storage-independent selector choices remain supported.

The setup-analysis tools now use AoS in place of their former forward-star
default or fixed-storage sweeps and label new output accordingly. Historical
forward-star measurements and their chart labels remain unchanged. This
retirement does not establish that the surviving layouts are universally
dominated, and it does not change the default solver or its numerical controls.

## Apple Metal backend

`metal_solver` ports a private downstream feasibility prototype (a Rust host
with MSL kernels, run on exported apxchol factors). Its measurements belong to
that project and are not reproduced here; the findings below are historical
and qualitative, not a validation of this integration.

**Precision variants.** An all-FP32 solve reached its recursive tolerance while
the true original-system residual stalled far above 1e-8. FP64 iterative
refinement on the CPU around restarted FP32 solves converged but cost several
times the FP64 iteration count at 1e-10. FP32 PCG with reliable updates (FP64 on
the CPU, or double-float on the GPU) kept the search directions but diverged at
1e-10 on the largest case. The FP32 preconditioner with double-float Krylov
recurrences (x, r, A p and the operator) reached 1e-10 true residuals without
restarts at close to FP64 iteration counts; it is the only mode retained.

**Inherited constants.** At most 64 right-hand sides per batch, node-major
blocks; rows with more than 32 dependencies are heavy; runs of narrow levels
(light rows only, at most 2048 row-column items) share one threadgroup of up to
1024 threads; one command buffer per iteration above 200,000 rows and per four
iterations below. None is an environment knob. The port does the host packing,
centring and FP64 exit check in block-wide node-major passes (the prototype
made one strided pass per column) and keeps two command buffers in flight (the
prototype waited for each one); the inherited iterations per command buffer
are unchanged.

**Fixed while porting.** Breakdown reported as convergence (a skipped update
left a zero r.r behind); `<=` stopping (now strict `<`); FP32 underflow of
tol^2 ||b||^2 (now exact power-of-two scaling and a host double-float
threshold); reductions whose order depended on the batch width (now one tree
fixed by n, and 32 virtual lanes for heavy rows); 32-bit overflow of block
indices (n kc < 2^32 is enforced); unchecked command-buffer errors; heavy-row
lanes taken from another pipeline's thread limit; double-float exactness left
to the compiler's defaults (now safe math mode, contraction off and a device
self-test); device lookup without CoreGraphics linked. The prototype also
applied the undropped factor; the port applies the CPU's dropped FP32 factor.
Its other modes and their environment controls were not ported.

No timing study accompanies this integration; performance relative to the CPU
or CUDA solves is not established.

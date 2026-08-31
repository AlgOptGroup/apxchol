#pragma once
/// Per-call scratch for the ELIMINATION phase of factorize(): per-thread
/// gather/sample buffers and round bookkeeping.  One workspace per
/// factorize() call, reused across rounds; owning the scratch here (instead
/// of `static`/`thread_local` buffers) keeps factorize() re-entrant.
/// Internal — partitioners own their scratch as instance members and never
/// see this type.

#include "apxchol/types.h"
#include "apxchol/solver/elimination/elimination.h"
#include <array>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

namespace apxchol {

/// Unweighted residual-topology update consumed by the optional GPU priority
/// front-end. The numerical edge weight remains owned by graph<> on the CPU;
/// selection needs only the endpoints (including multiplicity).
namespace detail {
struct gpu_topology_edge {
    node_index u;
    node_index v;
};

/// A non-owning view of one elimination thread's deferred-edge buffer. The
/// GPU front-end copies only each record's leading {u,v}; the CPU graph keeps
/// sole ownership of the numerical weight.
struct gpu_topology_batch {
    const deferred_edge* data = nullptr;
    std::size_t size = 0;
};
} // namespace detail

struct factorize_workspace {
    struct per_thread {
        std::vector<weighted_neighbor>             neighbors;
        std::vector<detail::deferred_edge>         edge_buffer;
        // Owner-resident GKS prefix/directory storage. Cooperative sample
        // tasks only read it, and the owner waits before reusing the vectors.
        detail::tree_sample_workspace              tree_sample;

        std::vector<std::pair<node_index, double>> excess_buffer;
        // Dedup-bucket scratch for process_vertex / merge_parallel_edges.
        // Indexed by vertex; entries are pool indices or npos.
        std::vector<node_index>     dedup_bucket;
        std::vector<node_index>  dedup_touched;
        // Pooled backends deduplicate each pivot in a compact local hash table
        // instead of probing a vertex-sized table. Epoch tags avoid clearing
        // the retained allocation between pivots.
        struct dedup_hash_entry {
            node_index key = 0;
            node_index value = 0;
            std::uint32_t stamp = 0;
        };
        std::vector<dedup_hash_entry> dedup_hash;
        std::uint32_t dedup_hash_epoch = 0;
        // Touched-vertex list populated during the vec_pool atomic histogram.
        // First-bump (0→1) pushes the vertex here, so the serial reserve_for
        // loop iterates only the touched vertices instead of all G.n().
        std::vector<node_index>     touched_buffer;
        // Directed AoS apply reuses the histogram's atomic return values as
        // endpoint slots, interleaved {u_offset,v_offset} per deferred edge.
        std::vector<node_index> endpoint_offsets;

        // Per-worker diagnostics for the research incremental-degree path.
        std::size_t degree_decrement_unique = 0;
        double degree_decrement_ms = 0.0;

        // Exact-size factor-column allocations share one monotonic resource.
        // The owning pointer is moved out before the rest of the workspace is
        // released, keeping the allocations alive through CSC assembly.
        std::unique_ptr<std::pmr::monotonic_buffer_resource> factor_entries;
    };
    std::vector<per_thread> threads;

    // Exact-size endpoint stream for incremental degree maintenance. Selected
    // pivots own disjoint prefix-summed slices, so dynamic elimination can fill
    // this array without atomics or append-vector over-allocation. The second
    // exact-size array is the global parallel-radix destination.
    std::vector<node_index> degree_decrements;
    std::vector<node_index> degree_decrement_scratch;
    std::vector<std::size_t> degree_decrement_offsets;
    std::vector<std::array<std::size_t, 256>> degree_decrement_histograms;
    std::size_t degree_removed_incidence = 0;
    std::size_t degree_retired_dead_incidence = 0;
    std::size_t degree_fill_edges = 0;

    /// Scratch for the vec_pool bulk-reserve path: concatenated per-thread
    /// touched_buffers, reused across rounds.
    std::vector<node_index> touched_concat;

    /// Per-vertex incoming-edge histogram for the vec_pool fused path
    /// (sized G.n() once, lazily). Invariant: all-zero between rounds -- the
    /// round that fills it resets exactly the touched entries afterwards, so
    /// no O(n) allocate+zero is paid per round (that per-round memset was
    /// 16 MB/round on grid_2000 and would dominate a small-IS tail round).
    std::vector<node_index> incoming;

    /// Sampled clique endpoints created by the current elimination round.
    /// Populated only when the guarded GPU setup front-end is active, so the
    /// default CPU path pays neither the copy nor the retained allocation.
    std::vector<detail::gpu_topology_edge> gpu_topology_updates;
    std::vector<detail::gpu_topology_batch> gpu_topology_batches;

    /// Round counter, incremented by the factorize loop once per round.
    uint64_t round_index = 0;

    /// Clear per-round per-thread buffers (edge_buffer, excess_buffer,
    /// dedup_touched).  Allocations are kept.  Call at the top of each
    /// elimination round.
    void reset_for_round() {
        gpu_topology_updates.clear();
        gpu_topology_batches.clear();
        degree_decrements.clear();
        degree_removed_incidence = 0;
        degree_retired_dead_incidence = 0;
        degree_fill_edges = 0;
        for (auto& t : threads) {
            t.edge_buffer.clear();
            t.excess_buffer.clear();
            t.dedup_touched.clear();
            t.degree_decrement_unique = 0;
            t.degree_decrement_ms = 0.0;
        }
    }
};

} // namespace apxchol

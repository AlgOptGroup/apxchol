#pragma once
/// Per-call scratch for the ELIMINATION phase of factorize(): per-thread
/// gather/sample buffers and round bookkeeping.  One workspace per
/// factorize() call, reused across rounds; owning the scratch here (instead
/// of `static`/`thread_local` buffers) keeps factorize() re-entrant.
/// Internal — partitioners own their scratch as instance members and never
/// see this type.

#include "apxchol/types.h"
#include "apxchol/solver/elimination/elimination.h"
#include <limits>
#include <utility>
#include <vector>

namespace apxchol {

struct factorize_workspace {
    struct per_thread {
        std::vector<weighted_neighbor>             neighbors;
        std::vector<detail::deferred_edge>         edge_buffer;

        std::vector<std::pair<node_index, double>> excess_buffer;
        // Dedup-bucket scratch for process_vertex / merge_parallel_edges.
        // Indexed by vertex; entries are pool indices or npos.
        std::vector<node_index>     dedup_bucket;
        std::vector<node_index>  dedup_touched;
        // Touched-vertex list populated during the vec_pool atomic histogram.
        // First-bump (0→1) pushes the vertex here, so the serial reserve_for
        // loop iterates only the touched vertices instead of all G.n().
        std::vector<node_index>     touched_buffer;
    };
    std::vector<per_thread> threads;

    /// Scratch for the vec_pool bulk-reserve path: concatenated per-thread
    /// touched_buffers, reused across rounds.
    std::vector<node_index> touched_concat;

    /// Round counter, incremented by the factorize loop once per round.
    uint64_t round_index = 0;

    /// Clear per-round per-thread buffers (edge_buffer, excess_buffer,
    /// dedup_touched).  Allocations are kept.  Call at the top of each
    /// elimination round.
    void reset_for_round() {
        for (auto& t : threads) {
            t.edge_buffer.clear();
            t.excess_buffer.clear();
            t.dedup_touched.clear();
        }
    }
};

} // namespace apxchol

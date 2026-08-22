#pragma once
/// Core index types for the apxchol library.
///
/// Since apxchol owns all sparse storage (sparse_csc, vec_pool — no Eigen factor),
/// it is NOT bound by Eigen's signed-`StorageIndex` requirement, so both index
/// types are UNSIGNED. Two roles, deliberately different sizes:
///
///   node_index : vertex ids. n << 4.29B for any realistic graph, so this stays
///                32-bit even under APXCHOL_64BIT_INDICES -- the large row-index
///                arrays (factor inner indices) never double.
///   edge_index : cumulative column offsets and edge ids, which DO reach billions
///                on dense factors / over-allocated pools -> the one widened by
///                APXCHOL_64BIT_EDGE_INDICES once they exceed uint32 capacity.
///
/// Two independent width knobs (CMake options of the same name):
///   APXCHOL_64BIT_EDGE_INDICES -> edge_index = uint64. Needed when the factor
///       nnz / pool size exceeds 2^32-1 even though the vertex count does not.
///       com-Orkut's roughly 2.15e9-entry raw factor exceeds signed int32 but
///       fits the default unsigned edge_index; it does NOT need this option.
///   APXCHOL_64BIT_NODE_INDICES -> node_index = uint64. Needed for graphs with
///       more than ~4.29B vertices. Since #edges >= #vertices, a 64-bit node
///       index implies a 64-bit edge index -- this knob AUTO-WIDENS edges too.
///   APXCHOL_64BIT_INDICES (deprecated) -> alias for the EDGE knob (the realistic
///       overflow case; keeps node_index 32-bit, the memory-optimal default).
///
/// Unsigned-safety rules (the surface is tiny -- audited):
///   * `npos = (T)-1` is a valid sentinel, but ONLY compared with `==`/`!=`
///     (never `< 0`, which is always false for unsigned).
///   * reverse loops use `for (i = n; i-- > 0; )` (or `~i`), never `i >= 0`.
///   * offsets are added as cumulative non-negative sums (asserted not to wrap);
///     never `(lo+hi)/2` (use `lo + (hi-lo)/2`) and never a diff that can go < 0.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// A 64-bit node index forces 64-bit edges; the legacy macro maps to the edge knob.
#if defined(APXCHOL_64BIT_NODE_INDICES) && !defined(APXCHOL_64BIT_EDGE_INDICES)
#  define APXCHOL_64BIT_EDGE_INDICES
#endif
#if defined(APXCHOL_64BIT_INDICES) && !defined(APXCHOL_64BIT_EDGE_INDICES)
#  define APXCHOL_64BIT_EDGE_INDICES
#endif

namespace apxchol {

#ifdef APXCHOL_64BIT_NODE_INDICES
using node_index = std::uint64_t;          // vertices: widened for >4.29B graphs
#else
using node_index = std::uint32_t;          // vertices: 32-bit (n << 4.29B)
#endif

#ifdef APXCHOL_64BIT_EDGE_INDICES
using edge_index = std::uint64_t;          // offsets / edge ids: widened for >4.29B
#else
using edge_index = std::uint32_t;
#endif

// (No `index_t` alias: every use is now explicitly node_index or edge_index so
//  the node/edge width distinction is type-checked, not by-convention.)

/// Abort with a clear, actionable message when an edge_index counter (a factor
/// column offset or an incidence-pool offset) would overflow its capacity.
/// Turns a potential wrap into a clean, fix-it error. The move from signed to
/// unsigned 32-bit already fixed com-Orkut's old signed-int wrap; this guard is
/// for counts that exceed uint32 as well. Always on; the call sites are
/// O(1)-per-grow, never per-edge.
[[noreturn]] inline void edge_index_overflow(const char* where) {
    std::fprintf(stderr,
        "apxchol: %s: an edge_index counter exceeded its %zu-bit capacity. "
        "Rebuild with -DAPXCHOL_64BIT_EDGE_INDICES=ON (or _64BIT_NODE_INDICES).\n",
        where, sizeof(edge_index) * 8);
    std::abort();
}

/// graph_storage enumerates the available incidence list backends
/// for runtime dispatch (CLI, factor_options).
enum class graph_storage { vec, forward_star, bstr, vec_pool };

} // namespace apxchol

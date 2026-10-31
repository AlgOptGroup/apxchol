#pragma once
/// Unsigned vertex IDs (node_index) and stored-entry offsets (edge_index).
/// APXCHOL_INDEX_WIDTH selects 32-bit indices (32, default), 64-bit offsets
/// with 32-bit vertices (64_EDGES), or 64-bit indices throughout (64).
/// Wide offsets support factors and pools exceeding 2^32-1 stored entries
/// without doubling their vertex-index arrays.

#include <cstdint>
#include <stdexcept>
#include <string>

// A 64-bit node index forces 64-bit edges.
#if defined(APXCHOL_64BIT_NODE_INDICES) && !defined(APXCHOL_64BIT_EDGE_INDICES)
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

namespace detail {

struct directed_pool_edge {
    node_index to;
    float w;

    friend bool operator<(const directed_pool_edge& a,
                          const directed_pool_edge& b) {
        return a.to < b.to || (a.to == b.to && a.w < b.w);
    }
};

/// splitmix64 output finalizer (Steele, Lea & Flood 2014), shared by the
/// counter-based random streams and the entry/pair hashes.
inline constexpr std::uint64_t splitmix64_mix(std::uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

} // namespace detail

// (No `index_t` alias: every use is now explicitly node_index or edge_index so
//  the node/edge width distinction is type-checked, not by-convention.)

/// Reject factor or incidence-pool offsets that exceed edge_index capacity.
[[noreturn]] inline void edge_index_overflow(const char* where) {
    throw std::length_error(std::string("apxchol: ") + where
        + ": an edge_index counter exceeded its "
        + std::to_string(sizeof(edge_index) * 8) + "-bit capacity. "
          "Rebuild with -DAPXCHOL_INDEX_WIDTH=64_EDGES or 64.");
}

} // namespace apxchol

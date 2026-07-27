#pragma once
/// Independent-set result from a partitioner.
///
/// One round of factorize() produces a partition_result, listing the
/// vertices to eliminate this round.  Every selected vertex is its own
/// region (a singleton): the IS is pairwise non-adjacent, so each vertex
/// can be eliminated independently of the others.
///
/// Historically this was a CSR layout (data[]/starts[]) supporting
/// multi-vertex regions; those partitioners were removed, leaving every
/// region a single vertex.  `starts` was then always the identity, so it
/// is gone — `data` alone is the flat list of selected vertices.

#include "apxchol/types.h"
#include <vector>

namespace apxchol {

struct partition_result {
    std::vector<node_index> data;        // flat list of selected vertices

    // Each selected vertex is its own region, so region count == vertex count.
    size_t num_regions() const { return data.size(); }
    size_t num_vertices() const { return data.size(); }

    void clear() { data.clear(); }
};

} // namespace apxchol

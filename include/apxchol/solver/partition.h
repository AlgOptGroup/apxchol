#pragma once
/// Independent-set result from a partitioner.
///
/// One round of factorize() produces a partition_result, listing the
/// vertices to eliminate this round.  Every selected vertex is its own
/// region (a singleton): the IS is pairwise non-adjacent, so each vertex
/// can be eliminated independently of the others.

#include "apxchol/types.h"
#include <vector>

namespace apxchol::detail {

struct partition_result {
    std::vector<node_index> data;        // flat list of selected vertices
};

} // namespace apxchol::detail

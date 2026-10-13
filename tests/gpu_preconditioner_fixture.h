#pragma once
#include "apxchol/solver/preconditioner.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"

#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
namespace apxchol::test {
template<class Graph>
std::unique_ptr<detail::gpu_block_frontend> initialize_owned_selector(
        detail::gpu_round_shadow_session& owner, const Graph& graph) {
    std::unique_ptr<detail::gpu_block_frontend> selector;
    owner.initialize_owned_raw_for_test(detail::make_gpu_round_shadow_input(
        graph, std::span<const node_index>{}, 0), selector);
    return selector;
}
}
#endif

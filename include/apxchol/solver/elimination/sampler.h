#pragma once
namespace apxchol {
// Explicit experimental alternatives; ordinary GKS remains the default.
enum class clique_sampler { gks, trace_cycle, heavy_core_k2 };
}

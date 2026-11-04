#pragma once
#include "apxchol/detail/option_names.h"

namespace apxchol {
// Clique-sampling algorithms for factor construction; GKS is the default.
enum class clique_sampler { gks, trace_cycle };

namespace detail {
inline constexpr std::array<option_name<clique_sampler>, 2> clique_sampler_names{{
    {"gks", clique_sampler::gks},
    {"trace_cycle", clique_sampler::trace_cycle},
}};
} // namespace detail
}

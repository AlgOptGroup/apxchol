#pragma once
/// Hybrid IS selection: block_greedy for the bulk, baumann_kyng for the tail.
///
/// Motivation: block_greedy produces large ISs and is fast in early rounds
/// (avg_deg low, |active| large), but its O(|active|/T) prune phase becomes
/// the bottleneck once |IS| shrinks below ~5% of |active| at high thread
/// counts (the per-thread block becomes narrow → cross-block conflicts kill
/// the IS → snowballs into the elim_remaining serial fallback).
///
/// baumann_kyng has O(|sample|·d) per-round work and produces small but
/// non-empty ISs round after round, perfect for the tail where BG falls off
/// a cliff.  It also produces less fill than the serial peeling fallback
/// (BG fallback path) since it still picks vertices with locally-low degree.
///
/// Switch trigger: when BG returns an IS smaller than
/// `opts.min_is_fraction × |active|` — the same threshold that previously
/// triggered the serial fallback.  After switching, we never switch back.
///
/// Marked `has_custom_find_is = true` so `factorize_impl` treats hybrid as
/// sample-bounded and never breaks out into the serial elim_remaining tail.

#include "apxchol/solver/is/independent_set.h"
#include "apxchol/solver/is/block_greedy.h"
#include "apxchol/solver/is/baumann_kyng.h"

namespace apxchol {

struct hybrid_is {
    block_greedy_is bg;
    baumann_kyng_is bk;
    bool switched_to_bk = false;

    static constexpr bool has_custom_find_is = true;

    template<incidence_storage Incidence>
    detail::find_is_result find_is(graph<Incidence>& G,
                                   std::span<const node_index> active,
                                   const factor_options& opts,
                                   checkpoint* cp = nullptr) {
        if (!switched_to_bk) {
            // Run the standard prune+select+collect pipeline with BG.
            // (find_independent_set handles its own cp descend("find_is"),
            // so we delegate without descending here.)
            auto result = detail::find_independent_set(bg, G, active, opts, cp);
            // Switch trigger: same fraction as the legacy fallback.
            if (result.is.size() < active.size() * opts.min_is_fraction)
                switched_to_bk = true;
            return result;
        }
        // BK takes over for the tail.  Its find_is is O(|sample|·d) and
        // also descends cp itself, so we just delegate.
        return bk.find_is(G, active, opts, cp);
    }

    // Stubs — unused with has_custom_find_is = true.
    template<incidence_storage Incidence>
    void select(const graph<Incidence>&,
                std::span<const node_index>,
                std::span<const index_t>,
                double, std::span<char>,
                const factor_options&) {}

    void cleanup(std::span<const node_index>,
                 const factor_options&) {}
};

} // namespace apxchol

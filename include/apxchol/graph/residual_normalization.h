#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol::detail {

// Arithmetic partition, not a runtime scheduling knob. Each block visits at
// most 16384*(sizeof(Item)+sizeof(double)) input bytes; its ordered scalar sum
// is identical regardless of which worker executes it.
inline constexpr std::size_t residual_normalization_block_items = 16384;

struct residual_normalization_result {
    double scale = 0.0;
    unsigned passes = 0;
    std::size_t blocks = 0;
};

template<class Item>
residual_normalization_result normalize_residual_importance(
        std::span<const Item> items, std::span<const double> importance,
        double target) {
    assert(items.size() == importance.size());
    constexpr auto block_size = residual_normalization_block_items;
    const auto blocks = items.size() / block_size +
                        (items.size() % block_size != 0);
    residual_normalization_result result{0.0, 0, blocks};
    const auto block_sum = [&](std::size_t block, double scale, bool initial) {
        const auto begin = block * block_size;
        const auto end = begin + std::min(block_size, items.size() - begin);
        double sum = 0.0;
        if (initial) {
            for (std::size_t i = begin; i < end; ++i)
                if (!items[i].backbone) sum += importance[i];
        } else {
            for (std::size_t i = begin; i < end; ++i)
                if (!items[i].backbone)
                    sum += std::min(1.0, scale * importance[i]);
        }
        return sum;
    };
    // Empty/single-block inputs need neither temporary storage nor a team.
    // This is the same one-block arithmetic partition at every thread count.
    if (blocks <= 1) {
        const double mass = block_sum(0, 0.0, true);
        result.scale = target > 0.0 && mass > 0.0 ? target / mass : 0.0;
        for (unsigned iteration = 0;
             iteration < 6 && result.scale > 0.0; ++iteration) {
            const double expected = block_sum(0, result.scale, false);
            ++result.passes;
            if (expected <= 0.0) break;
            result.scale *= target / expected;
        }
        return result;
    }

    std::vector<double> partials(blocks);
    const auto fold = [&] {
        double sum = 0.0;
        for (double partial : partials) sum += partial;
        return sum;
    };
    int team = 1;
#ifdef _OPENMP
    team = static_cast<int>(std::min(blocks,
        static_cast<std::size_t>(std::max(1, omp_get_max_threads()))));
#endif
    bool advance = false;
    // One team owns the initial scan and up to six updates. Each for's barrier
    // publishes partials; each single's barrier publishes scale/advance.
    // Even workers with no assigned block participate in these barriers.
#pragma omp parallel if(team > 1) num_threads(team) shared(advance, result, partials)
    {
#pragma omp for schedule(static)
        for (std::size_t block = 0; block < blocks; ++block)
            partials[block] = block_sum(block, 0.0, true);
#pragma omp single
        {
            const double mass = fold();
            result.scale = target > 0.0 && mass > 0.0 ? target / mass : 0.0;
            advance = result.scale > 0.0;
        }
        for (unsigned iteration = 0; iteration < 6; ++iteration) {
            if (!advance) break;
            const double scale = result.scale;
#pragma omp for schedule(static)
            for (std::size_t block = 0; block < blocks; ++block)
                partials[block] = block_sum(block, scale, false);
#pragma omp single
            {
                const double expected = fold();
                ++result.passes;
                if (expected <= 0.0) {
                    advance = false;
                } else {
                    result.scale *= target / expected;
                    advance = result.scale > 0.0;
                }
            }
        }
    }
    return result;
}

} // namespace apxchol::detail

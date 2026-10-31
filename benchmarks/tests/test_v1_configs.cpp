#include <iostream>
#include <stdexcept>
#include <string_view>

// Exercise the benchmark's actual presets and resolver without another API.
#define main apxchol_benchmark_cli_main
#include "../src/benchmark.cpp"
#undef main
#include "apxchol/solver/partitioner_helpers.h"

namespace {
const V1Combo& config(std::string_view name) {
    for (const auto& combo : v1_combos)
        if (combo.name == name) return combo;
    throw std::runtime_error("missing benchmark preset");
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        const auto& capped = config("bg+tree/cap4[vec_pool_aos]");
        const auto& standard = config("bg+tree[vec_pool_aos]");
        const std::vector<apxchol::node_index> degrees{1, 2, 20};
        const double average = 23.0 / 3;
        std::vector<apxchol::node_index> scratch;
        for (std::optional<double> quantile : {std::optional<double>{}, {0.0}, {0.8}}) {
            apxchol::factor_options requested;
            requested.partition.degree_quantile = quantile;
            requested.partition.degree_multiplier = 3;
            const auto opts = v1_factor_options(capped, requested);
            require(opts.partition.degree_quantile == 0, "cap4 did not disable quantile selection");
            require(opts.partition.degree_multiplier == 4, "cap4 did not set multiplier 4");
            require(apxchol::detail::candidate_degree_cap(degrees, degrees.size(), average,
                        opts.partition, scratch) == 4 * average,
                    "cap4 did not apply the advertised average-degree threshold");
            const auto unchanged = v1_factor_options(standard, requested);
            require(unchanged.partition.degree_quantile == quantile &&
                        unchanged.partition.degree_multiplier == 3,
                    "default tree preset changed requested degree controls");
        }
        std::cout << "v1 preset contract passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

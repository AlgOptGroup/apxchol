#include <gtest/gtest.h>
#include "apxchol/solver/elimination/elimination.h"

using namespace apxchol;

TEST(EliminatorBounds, TreeReturnsDegMinusOne) {
    EXPECT_EQ(tree_elimination{}.max_clique_edges(0), 0u);
    EXPECT_EQ(tree_elimination{}.max_clique_edges(1), 0u);
    EXPECT_EQ(tree_elimination{}.max_clique_edges(5), 4u);
    // Exact-clique mode: full pair count below the threshold.
    tree_elimination xc{.exact_clique_max_degree = 8};
    EXPECT_EQ(xc.max_clique_edges(5), 10u);
    EXPECT_EQ(xc.max_clique_edges(9), 8u);   // above threshold: sampled, d-1
}

TEST(EliminatorBounds, BoundIsSafeForRandomInput) {
    // Empirical: actual clique edges emitted never exceeds max_clique_edges(deg).
    std::vector<weighted_neighbor> neighbors;
    std::vector<deferred_edge> buf;

    for (node_index deg = 2; deg <= 12; ++deg) {
        neighbors.clear();
        for (node_index i = 0; i < deg; ++i)
            neighbors.emplace_back(i, 1.0 + i);
        const double sum = double(deg) * (deg + 1) / 2.0;

        buf.clear();
        tree_elimination{}.sample_clique(neighbors, sum, /*seed=*/42 + deg,
                                         edge_emitter(buf));
        EXPECT_LE(buf.size(), tree_elimination{}.max_clique_edges(deg))
            << "tree, deg=" << deg;
    }
}

#pragma once
// Serial matrix-to-adjacency oracle; independent of production graph builders.
#include "apxchol/types.h"
#include "csc_test_util.h"
#include <vector>

class reference_graph {
    struct edge { apxchol::node_index to; double w; };
    std::vector<std::vector<edge>> adjacent_;
    std::vector<double> excess_;
    apxchol::edge_index edges_ = 0;
public:
    explicit reference_graph(const apxchol::test::sparse_matrix& A)
        : adjacent_(A.rows), excess_(A.rows) {
        for (std::int64_t col = 0; col < A.cols; ++col) {
            double diagonal = 0, degree = 0;
            for (auto p = A.col_ptr[col]; p < A.col_ptr[col + 1]; ++p) {
                const std::int64_t row = A.row_idx[p];
                const double value = A.values[p];
                if (row == col) diagonal = value;
                else degree -= value;
                if (row <= col) continue;
                const double weight = static_cast<float>(-value);
                adjacent_[col].push_back({static_cast<apxchol::node_index>(row), weight});
                adjacent_[row].push_back({static_cast<apxchol::node_index>(col), weight});
                ++edges_;
            }
            const double excess = diagonal - degree;
            excess_[col] = excess > diagonal * 1e-12 ? excess : 0;
        }
    }
    auto n() const { return adjacent_.size(); }
    auto m() const { return edges_; }
    const auto& neighbors(apxchol::node_index v) const { return adjacent_[v]; }
    double excess(apxchol::node_index v) const { return excess_[v]; }
};

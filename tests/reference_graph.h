#pragma once
// Serial matrix-to-adjacency oracle; independent of production graph builders.
#include "apxchol/types.h"
#include <Eigen/Sparse>
#include <vector>

class reference_graph {
    struct edge { apxchol::node_index to; double w; };
    std::vector<std::vector<edge>> adjacent_;
    std::vector<double> excess_;
    apxchol::edge_index edges_ = 0;
public:
    explicit reference_graph(const Eigen::SparseMatrix<double>& A)
        : adjacent_(A.rows()), excess_(A.rows()) {
        for (int col = 0; col < A.outerSize(); ++col) {
            double diagonal = 0, degree = 0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, col); it; ++it) {
                if (it.row() == col) diagonal = it.value();
                else degree -= it.value();
                if (it.row() <= col) continue;
                const double weight = static_cast<apxchol::pool_value_t>(-it.value());
                adjacent_[col].push_back({static_cast<apxchol::node_index>(it.row()), weight});
                adjacent_[it.row()].push_back({static_cast<apxchol::node_index>(col), weight});
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

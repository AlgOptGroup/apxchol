#include "apxchol.h"
#include "config.h"
#include <fast_matrix_market/app/Eigen.hpp>
#include <spdlog/spdlog.h>
#include <fstream>

using namespace apxchol;

int main(int argc, char* argv[]) {
    auto cfg = parse_args(argc, argv);

    spdlog::debug("loading matrix from {}", cfg.input_path);
    Eigen::SparseMatrix<double> L;
    {
        std::ifstream f(cfg.input_path);
        fast_matrix_market::read_matrix_market_eigen(f, L);
    }
    spdlog::info("n = {},  nnz(L) = {}", L.rows(), L.nonZeros());

    Eigen::VectorXd b;
    if (cfg.rhs_path) {
        spdlog::debug("loading RHS from {}", *cfg.rhs_path);
        std::ifstream f(*cfg.rhs_path);
        fast_matrix_market::read_matrix_market_eigen_dense(f, b);
    } else {
        spdlog::debug("generating random test RHS");
        b = generate_test_rhs(L.rows());
    }

    auto res = solve(L, b, cfg.solve_opts);

    spdlog::info("iterations: {}", res.iterations);
    spdlog::info("residual:   {}", res.residual);
    spdlog::info("timings:\n{}", res.timings.report());

    if (cfg.output_path) {
        std::ofstream f(*cfg.output_path, std::ios_base::binary);
        fast_matrix_market::write_matrix_market_eigen_dense(f, res.x);
        spdlog::info("solution written to {}", *cfg.output_path);
    }

    return 0;
}

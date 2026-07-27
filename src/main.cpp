#include "apxchol.h"
#include "config.h"
#include <fast_matrix_market/app/Eigen.hpp>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <fstream>

using namespace apxchol;

// Exit codes: 0 = converged to --tol, 1 = did not converge (maxiter or
// stagnation), 2 = input/output error. Argument-parse failures exit with
// CLI11's own codes (see parse_args).
int main(int argc, char* argv[]) try {
    auto cfg = parse_args(argc, argv);

    spdlog::debug("loading matrix from {}", cfg.input_path);
    Eigen::SparseMatrix<double> L;
    {
        std::ifstream f(cfg.input_path);
        fast_matrix_market::read_matrix_market_eigen(f, L);
    }
    if (L.rows() != L.cols()) {
        spdlog::error("matrix must be square, got {}x{}", L.rows(), L.cols());
        return 2;
    }
    spdlog::info("n = {},  nnz(L) = {}", L.rows(), L.nonZeros());

    Eigen::VectorXd b;
    if (cfg.rhs_path) {
        spdlog::debug("loading RHS from {}", *cfg.rhs_path);
        std::ifstream f(*cfg.rhs_path);
        fast_matrix_market::read_matrix_market_eigen_dense(f, b);
        if (b.size() != L.rows()) {
            spdlog::error("RHS length {} does not match n = {}", b.size(), L.rows());
            return 2;
        }
    } else if (cfg.random_rhs) {
        spdlog::debug("generating random test RHS");
        std::srand(cfg.solve_opts.factor_opts.seed);  // Eigen::Random uses rand()
        b = generate_test_rhs(L.rows());
    } else {
        spdlog::error("no right-hand side: pass --rhs <file.mtx>, or --random-rhs "
                      "to solve against a random test vector");
        return 2;
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

    const bool converged = res.residual < cfg.solve_opts.tol;
    if (!converged)
        spdlog::error("did not reach tol {} (residual {}); on a disconnected "
                      "Laplacian a globally generated RHS may be inconsistent "
                      "per component", cfg.solve_opts.tol, res.residual);
    return converged ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return 2;
}

#include "apxchol.h"
#include "config.h"
#include "mtx_input.h"
#include <fast_matrix_market/app/Eigen.hpp>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

using namespace apxchol;

// Exit codes: 0 = converged to --tol, 1 = did not converge, 2 = input/output
// error. Argument-parse failures use CLI11's own codes (see parse_args).
int main(int argc, char* argv[]) try {
    auto cfg = parse_args(argc, argv);

    spdlog::debug("loading matrix from {}", cfg.input_path);
    Eigen::SparseMatrix<double> L;
    fast_matrix_market::matrix_market_header hdr;
    {
        std::ifstream f(cfg.input_path);
        fast_matrix_market::read_matrix_market_eigen(f, hdr, L);
    }
    if (L.rows() != L.cols()) {
        spdlog::error("matrix must be square, got {}x{}", L.rows(), L.cols());
        return 2;
    }

    // Interpret the input and assemble adjacency matrices into Laplacians
    // before operator validation (see src/mtx_input.h).
    // An unusable or ambiguous matrix throws here; the handler below logs it
    // and exits 2, as it does for any other input error.
    const auto scan = scan_input(L);
    std::string reason;
    const input_kind kind = resolve_input_kind(
        cfg.input, scan, hdr.field == fast_matrix_market::pattern, reason);
    spdlog::info("{}", describe_input(kind, scan, reason));
    if (kind == input_kind::adjacency)
        adjacency_to_laplacian(L);

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
    } else {
        // parse_args requires exactly one of --rhs / --random-rhs.
        spdlog::debug("generating random test RHS");
        std::srand(cfg.solve_opts.factor_opts.seed);  // Eigen::Random uses rand()
        b = generate_test_rhs(L.rows());
        if (kind == input_kind::adjacency ||
            (scan.excess_rows == 0 && scan.deficient_rows == 0)) {
            const Eigen::Index components =
                project_laplacian_rhs_components(L, b);
            if (components > 1)
                spdlog::info(
                    "projected random RHS independently over {} connected components",
                    components);
        }
    }

    auto res = solve(L, b, cfg.solve_opts);

    spdlog::info("backend:    {}", res.backend == solve_backend::gpu ? "gpu" : "cpu");
    spdlog::info("iterations: {}", res.iterations);
    spdlog::info("residual:   {}", res.residual);
    if (cfg.solve_opts.collect_timings)
        spdlog::info("timings:\n{}", res.timings.report());

    if (cfg.output_path) {
        std::ofstream f(*cfg.output_path, std::ios_base::binary);
        fast_matrix_market::write_matrix_market_eigen_dense(f, res.x);
        f.close();
        if (!f)
            throw std::runtime_error("cannot write solution to " + *cfg.output_path);
        spdlog::info("solution written to {}", *cfg.output_path);
    }

    if (!res.converged)
        spdlog::error("did not reach tol {} in {} iterations (residual {}).{}",
                      cfg.solve_opts.tol, res.iterations, res.residual,
                      diagnose_failure(kind, scan));
    return res.converged ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return 2;
}

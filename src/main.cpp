#include "apxchol.h"
#include "config.h"
#include "mtx_input.h"
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
    fast_matrix_market::matrix_market_header hdr;
    {
        std::ifstream f(cfg.input_path);
        fast_matrix_market::read_matrix_market_eigen(f, hdr, L);
    }
    if (L.rows() != L.cols()) {
        spdlog::error("matrix must be square, got {}x{}", L.rows(), L.cols());
        return 2;
    }

    // Decide what the file actually holds before handing it to the solver:
    // an adjacency matrix fed in as an operator gives negative edge weights,
    // zero fill and an immediate PCG breakdown, with nothing on screen to say
    // why. See src/mtx_input.h.
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
    } else if (cfg.random_rhs) {
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
    if (!converged) {
        // iterations == 0 without convergence means PCG stopped before its
        // first update, i.e. p·Ap <= 0: the operator is not positive
        // semidefinite and NOTHING was solved. That is the symptom a
        // misinterpreted input produces, so name the interpretation.
        const std::string why = diagnose_failure(kind, scan);
        if (res.iterations == 0)
            spdlog::error("PCG broke down before its first update (p·Ap <= 0): the "
                          "operator is not positive semidefinite, so x = 0 was "
                          "returned and the reported residual {} is meaningless.{} "
                          "The input was read as {}.",
                          res.residual, why,
                          kind == input_kind::adjacency
                              ? "an adjacency graph (L = D - A)"
                              : "an assembled Laplacian/SDDM operator");
        else
            spdlog::error("did not reach tol {} in {} iterations (residual {}).{}",
                          cfg.solve_opts.tol, res.iterations, res.residual, why);
    }
    return converged ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return 2;
}

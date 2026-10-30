#include "apxchol.h"
#include "config.h"
#include "mtx_input.h"
#include <fast_matrix_market/app/Eigen.hpp>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

using namespace apxchol;

namespace {

std::string input_kind_reason(input_kind requested, input_kind kind) {
    switch (requested) {
    case input_kind::as_operator: return "--input-kind operator, the default";
    case input_kind::adjacency: return "--input-kind adjacency";
    case input_kind::automatic: break;
    }
    return kind == input_kind::adjacency
        ? "--input-kind auto: not a valid operator, but an adjacency matrix or pattern file"
        : "--input-kind auto: used as given";
}

template<class Dense>
void write_solution(const std::string& path, const Dense& x) {
    std::ofstream f(path, std::ios_base::binary);
    fast_matrix_market::write_matrix_market_eigen_dense(f, x);
    f.close();
    if (!f)
        throw std::runtime_error("cannot write solution to " + path);
    spdlog::info("solution written to {}", path);
}

// An n x k RHS file: one reusable solver, one batched solve over its columns.
int solve_batch(const Eigen::SparseMatrix<double>& L, const Eigen::MatrixXd& B,
                const run_config& cfg, input_kind kind, const input_scan& scan) {
    checkpoint setup;
    const solver owner(L, cfg.solve_opts, cfg.solve_opts.collect_timings ? &setup : nullptr);
    Eigen::MatrixXd X(B.rows(), B.cols());
    const auto res = owner.solve(B, X);

    spdlog::info("backend:    {}", owner.backend() == solve_backend::gpu ? "gpu" : "cpu");
    spdlog::info("columns:    {}", B.cols());
    Eigen::Index failed = 0;
    for (std::size_t c = 0; c < res.columns.size(); ++c) {
        const auto& col = res.columns[c];
        spdlog::info("column {}: iterations {}, residual {}{}", c, col.iterations,
                     col.residual, col.converged ? "" : " (did not converge)");
        failed += !col.converged;
    }
    if (cfg.solve_opts.collect_timings)
        spdlog::info("timings:\n{}\n{}", setup.report(), res.timings.report());

    if (cfg.output_path)
        write_solution(*cfg.output_path, X);

    if (failed)
        spdlog::error("{} of {} columns did not reach tol {} within {} iterations.{}",
                      failed, B.cols(), cfg.solve_opts.tol, cfg.solve_opts.max_iter,
                      diagnose_failure(kind, scan));
    return failed ? 1 : 0;
}

} // namespace

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

    // Read the file as --input-kind says (see src/mtx_input.h); validation
    // errors from the solver are logged by the handler below, exit 2.
    const auto scan = scan_input(L);
    const input_kind kind = detail::resolve_input_kind(
        L, cfg.input, hdr.field == fast_matrix_market::pattern);
    spdlog::info("{}", describe_input(kind, scan, input_kind_reason(cfg.input, kind)));
    if (kind == input_kind::adjacency)
        L = detail::adjacency_laplacian(L);

    spdlog::info("n = {},  nnz(L) = {}", L.rows(), L.nonZeros());

    Eigen::VectorXd b;
    if (cfg.rhs_path) {
        spdlog::debug("loading RHS from {}", *cfg.rhs_path);
        Eigen::MatrixXd B;
        {
            std::ifstream f(*cfg.rhs_path);
            fast_matrix_market::read_matrix_market_eigen_dense(f, B);
        }
        if (B.rows() != L.rows()) {
            spdlog::error("RHS length {} does not match n = {}", B.rows(), L.rows());
            return 2;
        }
        if (B.cols() > 1)
            return solve_batch(L, B, cfg, kind, scan);
        b = B.col(0);
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

    if (cfg.output_path)
        write_solution(*cfg.output_path, res.x);

    if (!res.converged)
        spdlog::error("did not reach tol {} in {} iterations (residual {}).{}",
                      cfg.solve_opts.tol, res.iterations, res.residual,
                      diagnose_failure(kind, scan));
    return res.converged ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return 2;
}

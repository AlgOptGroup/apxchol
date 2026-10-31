#include "apxchol.h"
#include "apxchol/detail/checkpoint.h"
#include "apxchol/operator_class.h"
#include "apxchol/solver/detail/solver_route.h"
#include "config.h"
#include "mtx_input.h"
#include <fast_matrix_market/app/Eigen.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>

using namespace apxchol;

namespace {

// Zero-mean, unit-norm RHS that is the same on every platform for a given seed:
// mt19937_64 output is fully specified, and the sums use explicit fma in order.
Eigen::VectorXd random_rhs(Eigen::Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Eigen::VectorXd b(n);
    double sum = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        b[i] = static_cast<double>(rng() >> 11) * 0x1p-52 - 1.0;  // uniform in [-1, 1)
        sum += b[i];
    }
    const double mean = n > 0 ? sum / static_cast<double>(n) : 0.0;
    double squares = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        b[i] -= mean;
        squares = std::fma(b[i], b[i], squares);
    }
    if (squares > 0.0) {
        const double norm = std::sqrt(squares);
        for (Eigen::Index i = 0; i < n; ++i) b[i] /= norm;
    }
    return b;
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

// -v attaches the internal timing tree, which is printed after the solve.
solver make_solver(const Eigen::SparseMatrix<double>& L, const run_config& cfg,
                   detail::checkpoint& timings) {
    return cfg.solve_opts.factor_opts.verbose
        ? detail::solver_access::with_timings(L, cfg.solve_opts, timings)
        : solver(L, cfg.solve_opts);
}

// An n x k RHS file: one reusable solver, one batched solve over its columns.
int solve_batch(const Eigen::SparseMatrix<double>& L, const Eigen::MatrixXd& B,
                const run_config& cfg, bool graph, const input_scan& scan) {
    detail::checkpoint timings;
    solver owner = make_solver(L, cfg, timings);
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
    if (cfg.solve_opts.factor_opts.verbose)
        spdlog::info("timings:\n{}", timings.report());

    if (cfg.output_path)
        write_solution(*cfg.output_path, X);

    if (failed)
        spdlog::error("{} of {} columns did not reach tol {} within {} iterations.{}",
                      failed, B.cols(), cfg.solve_opts.tol, cfg.solve_opts.max_iter,
                      diagnose_failure(graph, scan));
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
    if (L.rows() == 0) {
        spdlog::error("matrix is empty (0x0)");
        return 2;
    }

    // The file is the operator; --convert-input applies the as_operator rule
    // (see src/mtx_input.h). Validation errors are logged by the handler
    // below, exit 2. A pattern file stores unit weights, so it has no
    // negative off-diagonal and converts like any adjacency matrix.
    const auto scan = scan_input(L);
    const bool graph = cfg.convert_input && detail::is_graph_adjacency(L);
    spdlog::info("{}", describe_input(graph, scan,
                                      !cfg.convert_input ? "the default"
                                      : graph ? "--convert-input: no negative off-diagonal"
                                              : "--convert-input: not a graph adjacency matrix"));
    if (graph)
        L = detail::adjacency_laplacian(L);

    spdlog::info("n = {},  nnz(L) = {}", L.rows(), L.nonZeros());

    Eigen::VectorXd b;
    if (cfg.rhs_path) {
        spdlog::debug("loading RHS from {}", *cfg.rhs_path);
        // Check the shape before reading: the reader divides by the row count.
        fast_matrix_market::matrix_market_header rhs_header;
        {
            std::ifstream f(*cfg.rhs_path);
            fast_matrix_market::read_header(f, rhs_header);
        }
        if (rhs_header.nrows != L.rows()) {
            spdlog::error("RHS length {} does not match n = {}", rhs_header.nrows, L.rows());
            return 2;
        }
        if (rhs_header.ncols == 0) {
            spdlog::error("RHS file has no columns");
            return 2;
        }
        Eigen::MatrixXd B;
        {
            std::ifstream f(*cfg.rhs_path);
            fast_matrix_market::read_matrix_market_eigen_dense(f, B);
        }
        if (B.cols() > 1)
            return solve_batch(L, B, cfg, graph, scan);
        b = B.col(0);
    } else {
        // parse_args requires exactly one of --rhs / --random-rhs.
        spdlog::debug("generating random test RHS");
        b = random_rhs(L.rows(), cfg.solve_opts.factor_opts.seed);
        if (graph ||
            (scan.excess_rows == 0 && scan.deficient_rows == 0)) {
            const Eigen::Index components =
                project_laplacian_rhs_components(L, b);
            if (components > 1)
                spdlog::info(
                    "projected random RHS independently over {} connected components",
                    components);
        }
    }

    detail::checkpoint timings;
    solver owner = make_solver(L, cfg, timings);
    const auto res = owner.solve(b);

    spdlog::info("backend:    {}", res.backend == solve_backend::gpu ? "gpu" : "cpu");
    spdlog::info("iterations: {}", res.iterations);
    spdlog::info("residual:   {}", res.residual);
    if (cfg.solve_opts.factor_opts.verbose)
        spdlog::info("timings:\n{}", timings.report());

    if (cfg.output_path)
        write_solution(*cfg.output_path, res.x);

    if (!res.converged)
        spdlog::error("did not reach tol {} in {} iterations (residual {}).{}",
                      cfg.solve_opts.tol, res.iterations, res.residual,
                      diagnose_failure(graph, scan));
    return res.converged ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return 2;
}

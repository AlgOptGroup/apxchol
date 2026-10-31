#include "apxchol.h"
#include "apxchol/detail/checkpoint.h"
#include "apxchol/operator_class.h"
#include "apxchol/solver/detail/solver_route.h"
#include "config.h"
#include "mtx_input.h"
#include <fast_matrix_market/app/array.hpp>
#include <fast_matrix_market/app/triplet.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

using namespace apxchol;

namespace {

// Zero-mean, unit-norm RHS that is the same on every platform for a given seed:
// mt19937_64 output is fully specified, and the sums use explicit fma in order.
std::vector<double> random_rhs(std::int64_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<double> b(static_cast<std::size_t>(n));
    double sum = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<double>(rng() >> 11) * 0x1p-52 - 1.0;  // uniform in [-1, 1)
        sum += b[i];
    }
    const double mean = n > 0 ? sum / static_cast<double>(n) : 0.0;
    double squares = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] -= mean;
        squares = std::fma(b[i], b[i], squares);
    }
    if (squares > 0.0) {
        const double norm = std::sqrt(squares);
        for (double& v : b) v /= norm;
    }
    return b;
}

// A dense rows x cols column-major array.
struct dense_file {
    std::int64_t rows = 0, cols = 0;
    std::vector<double> values;
};

void write_solution(const std::string& path, const dense_file& x) {
    std::ofstream f(path, std::ios_base::binary);
    fast_matrix_market::matrix_market_header header(x.rows, x.cols);
    fast_matrix_market::write_matrix_market_array(f, header, x.values,
                                                  fast_matrix_market::col_major);
    f.close();
    if (!f)
        throw std::runtime_error("cannot write solution to " + path);
    spdlog::info("solution written to {}", path);
}

// The matrix of an .mtx file. A symmetric file is generalized with an
// explicit zero added to each diagonal entry, a pattern file has unit values,
// and duplicates are summed in file order.
detail::csc_matrix<double> read_matrix_body(std::istream& f,
                                            const fast_matrix_market::matrix_market_header& header) {
    if (static_cast<std::uint64_t>(header.nrows) >= std::numeric_limits<node_index>::max())
        throw std::invalid_argument(fmt::format(
            "matrix has {} rows, more than this build's {}-bit vertex indices allow",
            header.nrows, sizeof(node_index) * 8));
    fast_matrix_market::read_options options;
    options.generalize_coordinate_diagnonal_values =
        fast_matrix_market::read_options::ExtraZeroElement;
    const auto count = static_cast<std::size_t>(fast_matrix_market::get_storage_nnz(header, options));
    std::vector<node_index> rows(count), cols(count);
    std::vector<double> values(count);
    fast_matrix_market::triplet_parse_handler handler(rows.begin(), cols.begin(), values.begin());
    fast_matrix_market::read_matrix_market_body(f, header, handler, 1.0, options);
    std::vector<detail::triplet> entries(count);
    for (std::size_t k = 0; k < count; ++k) entries[k] = {rows[k], cols[k], values[k]};
    std::vector<node_index>().swap(rows);
    std::vector<node_index>().swap(cols);
    std::vector<double>().swap(values);
    return detail::csc_from_triplets(static_cast<node_index>(header.nrows), entries);
}

// A dense matrix body. The header's shape is checked before allocating, so a
// crafted header cannot size the array smaller than the entries it addresses.
dense_file read_dense_body(std::istream& f, const fast_matrix_market::matrix_market_header& header) {
    dense_file out{header.nrows, header.ncols, {}};
    std::int64_t count = 0;
    if (header.nrows < 0 || header.ncols < 0 ||
        __builtin_mul_overflow(header.nrows, header.ncols, &count) ||
        static_cast<std::uint64_t>(count) > out.values.max_size())
        throw std::invalid_argument(fmt::format(
            "apxchol: dense matrix file claims {}x{} entries, more than can be stored",
            header.nrows, header.ncols));
    out.values.resize(static_cast<std::size_t>(count));
    fast_matrix_market::dense_adding_parse_handler handler(
        out.values.begin(), fast_matrix_market::col_major, header.nrows, header.ncols);
    fast_matrix_market::read_matrix_market_body(f, header, handler, 1.0);
    return out;
}

// -v attaches the internal timing tree, which is printed after the solve.
solver make_solver(const detail::csc_matrix<double>& L, const run_config& cfg,
                   detail::checkpoint& timings) {
    return detail::solver_access::make(L, cfg.solve_opts,
                                       cfg.solve_opts.factor_opts.verbose ? &timings : nullptr);
}

// The --x0 file must have the RHS's shape. The header is checked before the
// body: the reader divides by the row count.
std::optional<dense_file> read_initial_guess(const run_config& cfg, std::int64_t rows,
                                             std::int64_t cols) {
    if (!cfg.x0_path) return std::nullopt;
    std::ifstream f(*cfg.x0_path);
    fast_matrix_market::matrix_market_header header;
    fast_matrix_market::read_header(f, header);
    if (header.nrows != rows || header.ncols != cols)
        throw std::invalid_argument(fmt::format(
            "x0 is {}x{}, but the RHS is {}x{}", header.nrows, header.ncols, rows, cols));
    return read_dense_body(f, header);
}

// An n x k RHS file: one reusable solver, one batched solve over its columns.
int solve_batch(const detail::csc_matrix<double>& L, const dense_file& B,
                const run_config& cfg, bool graph, const input_scan& scan) {
    const auto X0 = read_initial_guess(cfg, B.rows, B.cols);
    detail::checkpoint timings;
    solver owner = make_solver(L, cfg, timings);
    dense_file X{B.rows, B.cols, std::vector<double>(B.values.size())};
    const auto res = owner.solve(const_matrix_view{B.values.data(), B.rows, B.cols, B.rows},
                                 matrix_view{X.values.data(), X.rows, X.cols, X.rows},
                                 cfg.tol, cfg.max_iter,
                                 X0 ? const_matrix_view{X0->values.data(), X0->rows, X0->cols, X0->rows}
                                    : const_matrix_view{});

    spdlog::info("backend:    {}", owner.backend() == solve_backend::gpu ? "gpu" : "cpu");
    spdlog::info("columns:    {}", B.cols);
    std::int64_t failed = 0;
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
                      failed, B.cols, cfg.tol, cfg.max_iter,
                      diagnose_failure(graph, scan));
    return failed ? 1 : 0;
}

} // namespace

// Exit codes: 0 = converged to --tol, 1 = did not converge, 2 = input/output
// error. Argument-parse failures use CLI11's own codes (see parse_args).
int main(int argc, char* argv[]) try {
    auto cfg = parse_args(argc, argv);

    spdlog::debug("loading matrix from {}", cfg.input_path);
    std::ifstream input(cfg.input_path);
    fast_matrix_market::matrix_market_header hdr;
    fast_matrix_market::read_header(input, hdr);
    if (hdr.nrows != hdr.ncols) {
        spdlog::error("matrix must be square, got {}x{}", hdr.nrows, hdr.ncols);
        return 2;
    }
    if (hdr.nrows == 0) {
        spdlog::error("matrix is empty (0x0)");
        return 2;
    }
    detail::csc_matrix<double> L = read_matrix_body(input, hdr);
    input.close();

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

    const std::int64_t n = L.rows();
    spdlog::info("n = {},  nnz(L) = {}", n, L.ref().nnz());

    std::vector<double> b;
    if (cfg.rhs_path) {
        spdlog::debug("loading RHS from {}", *cfg.rhs_path);
        // Check the shape before reading: the reader divides by the row count.
        std::ifstream f(*cfg.rhs_path);
        fast_matrix_market::matrix_market_header rhs_header;
        fast_matrix_market::read_header(f, rhs_header);
        if (rhs_header.nrows != n) {
            spdlog::error("RHS length {} does not match n = {}", rhs_header.nrows, n);
            return 2;
        }
        if (rhs_header.ncols == 0) {
            spdlog::error("RHS file has no columns");
            return 2;
        }
        dense_file B = read_dense_body(f, rhs_header);
        if (B.cols > 1)
            return solve_batch(L, B, cfg, graph, scan);
        b = std::move(B.values);
    } else {
        // parse_args requires exactly one of --rhs / --random-rhs.
        spdlog::debug("generating random test RHS");
        b = random_rhs(n, cfg.solve_opts.factor_opts.seed);
        if (graph ||
            (scan.excess_rows == 0 && scan.deficient_rows == 0)) {
            const std::int64_t components =
                project_laplacian_rhs_components(L, b);
            if (components > 1)
                spdlog::info(
                    "projected random RHS independently over {} connected components",
                    components);
        }
    }

    const auto X0 = read_initial_guess(cfg, n, 1);
    detail::checkpoint timings;
    solver owner = make_solver(L, cfg, timings);
    const auto res = owner.solve(b, cfg.tol, cfg.max_iter,
                                 X0 ? std::span<const double>(X0->values) : std::span<const double>());

    spdlog::info("backend:    {}", res.backend == solve_backend::gpu ? "gpu" : "cpu");
    spdlog::info("iterations: {}", res.iterations);
    spdlog::info("residual:   {}", res.residual);
    if (cfg.solve_opts.factor_opts.verbose)
        spdlog::info("timings:\n{}", timings.report());

    if (cfg.output_path)
        write_solution(*cfg.output_path, dense_file{n, 1, res.x});

    if (!res.converged)
        spdlog::error("did not reach tol {} in {} iterations (residual {}).{}",
                      cfg.tol, res.iterations, res.residual,
                      diagnose_failure(graph, scan));
    return res.converged ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return 2;
}

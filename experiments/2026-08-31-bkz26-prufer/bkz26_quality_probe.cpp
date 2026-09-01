#include "apxchol.h"
#include "mtx_input.h"

#include <fast_matrix_market/app/Eigen.hpp>

#include <Eigen/Sparse>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct matrix_input {
    Eigen::SparseMatrix<double> matrix;
    bool laplacian = false;
};

matrix_input read_matrix(const char* path) {
    matrix_input result;
    fast_matrix_market::matrix_market_header header;
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("cannot open ") + path);
    fast_matrix_market::read_matrix_market_eigen(input, header, result.matrix);
    const auto scan = apxchol::scan_input(result.matrix);
    std::string reason;
    const auto kind = apxchol::resolve_input_kind(
        apxchol::input_kind::automatic, scan,
        header.field == fast_matrix_market::pattern, reason);
    if (kind == apxchol::input_kind::adjacency) {
        apxchol::adjacency_to_laplacian(result.matrix);
        result.laplacian = true;
    } else {
        result.laplacian = scan.excess_rows == 0 && scan.deficient_rows == 0;
    }
    return result;
}

struct run_result {
    std::size_t raw_nnz = 0;
    std::uint64_t stored_nnz = 0;
    int iterations = 0;
    double true_residual = std::numeric_limits<double>::infinity();
    double setup_s = 0.0;
    double solve_s = 0.0;
};

run_result run_arm(const Eigen::SparseMatrix<double>& matrix,
                   const Eigen::VectorXd& rhs, unsigned seed,
                   const char* sampler) {
    apxchol::factor_options factor_options;
    factor_options.seed = seed;
    factor_options.clique_sampler = sampler;
    const auto begin = std::chrono::steady_clock::now();
    auto factor = apxchol::factorize(
        matrix, apxchol::graph_storage::vec_pool_aos, factor_options);
    run_result result;
    result.raw_nnz = static_cast<std::size_t>(factor.L.nonZeros());

    apxchol::solve_options solve_options;
    solve_options.tol = 1e-8;
    solve_options.max_iter = 3000;
    solve_options.stagnation_window = 0;
    apxchol::cpu_solver solver(matrix, std::move(factor), solve_options);
    const auto built = std::chrono::steady_clock::now();
    result.stored_nnz = solver.preconditioner().trsv().stored_nnz();
    const auto solved = solver.solve(rhs);
    const auto done = std::chrono::steady_clock::now();
    result.iterations = static_cast<int>(solved.iterations);
    if (solved.x.size() != rhs.size())
        throw std::runtime_error("solver returned a wrong-length solution");
    const Eigen::VectorXd residual = rhs - matrix * solved.x;
    result.true_residual = residual.norm() / rhs.norm();
    result.setup_s = std::chrono::duration<double>(built - begin).count();
    result.solve_s = std::chrono::duration<double>(done - built).count();
    return result;
}

void print_result(const char* label, unsigned seed, const char* arm,
                  const run_result& result) {
    std::printf("%s\t%u\t%s\t%zu\t%llu\t%d\t%.10g\t%.6f\t%.6f\n",
                label, seed, arm, result.raw_nnz,
                static_cast<unsigned long long>(result.stored_nnz),
                result.iterations, result.true_residual,
                result.setup_s, result.solve_s);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s LABEL MATRIX.mtx SEED [SEED ...]\n", argv[0]);
        return 2;
    }
    const char* label = argv[1];
    auto input = read_matrix(argv[2]);

    // RHS generation is intentionally independent of every factor seed and
    // sampler arm. Pure Laplacians are projected component by component.
    std::srand(20260901);
    Eigen::VectorXd rhs = apxchol::generate_test_rhs(input.matrix.rows());
    if (input.laplacian)
        (void)apxchol::project_laplacian_rhs_components(input.matrix, rhs);

    std::puts("matrix\tseed\tarm\traw_nnz\tstored_nnz\titers\ttrue_residual\tsetup_s\tsolve_s");
    for (int arg = 3; arg < argc; ++arg) {
        const unsigned seed =
            static_cast<unsigned>(std::strtoul(argv[arg], nullptr, 10));
        print_result(label, seed, "gks_before",
                     run_arm(input.matrix, rhs, seed, "gks"));
        print_result(label, seed, "bkz26",
                     run_arm(input.matrix, rhs, seed, "bkz26"));
        print_result(label, seed, "gks_after",
                     run_arm(input.matrix, rhs, seed, "gks"));
    }
    return 0;
}

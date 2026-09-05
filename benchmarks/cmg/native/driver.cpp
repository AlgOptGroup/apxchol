#include "cmg_pcg_packed.h"
#include "cmg_setup_packed_initialize.h"
#include "cmg_setup_packed_terminate.h"
#include "sparse1.h"
#include <fast_matrix_market/app/Eigen.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

extern double cmg_setup_seconds;
extern int cmg_setup_calls;
extern bool cmg_hierarchy_valid;
extern unsigned cmg_levels;
using Matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

int main(int argc, char **argv) try {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: cmg_driver operator.mtx rhs.mtx [tol [maxit [solution.mtx]]]\n"
                  << "Inputs are an assembled real operator and an explicit n-by-1 RHS.\n";
        return 2;
    }
    const double tol = argc > 3 ? std::stod(argv[3]) : 1e-8;
    const int maxit = argc > 4 ? std::stoi(argv[4]) : 1000;
    if (!std::isfinite(tol) || tol <= 0.0 || tol >= 1.0 || maxit < 1)
        throw std::runtime_error("tol must be in (0,1), maxit must be positive");
    fast_matrix_market::read_options options;
    options.num_threads = 1;
    Matrix A;
    fast_matrix_market::matrix_market_header header;
    std::ifstream matrix_file(argv[1]);
    if (!matrix_file) throw std::runtime_error("cannot open operator file");
    fast_matrix_market::read_header(matrix_file, header);
    if (header.nrows != header.ncols || header.nrows < 1 ||
        header.nrows >= std::numeric_limits<int>::max() ||
        header.nnz >= std::numeric_limits<int>::max()/2 ||
        (header.field != fast_matrix_market::real &&
         header.field != fast_matrix_market::integer))
        throw std::runtime_error("operator must be square real/integer and fit signed 32-bit Coder CSC (conservative 2*nnz bound)");
    matrix_file.clear(); matrix_file.seekg(0);
    fast_matrix_market::read_matrix_market_eigen(matrix_file, header, A, options);
    A.makeCompressed();
    for (Eigen::Index i = 0; i < A.nonZeros(); ++i)
        if (!std::isfinite(A.valuePtr()[i])) throw std::runtime_error("non-finite operator entry");

    Eigen::MatrixXd rhs_matrix;
    std::ifstream rhs_file(argv[2]);
    if (!rhs_file) throw std::runtime_error("cannot open RHS file");
    fast_matrix_market::read_matrix_market_eigen_dense(rhs_file, rhs_matrix, options);
    if (rhs_matrix.rows() != A.rows() || rhs_matrix.cols() != 1 || !rhs_matrix.allFinite())
        throw std::runtime_error("RHS must be a finite n-by-1 column");
    const Eigen::VectorXd b = rhs_matrix.col(0);
    // Common parsing is excluded; solver-specific one-based CSC conversion is setup.
    const auto adaptation_begin = std::chrono::steady_clock::now();
    coder::array<double, 1U> values, rhs;
    coder::array<int, 1U> row_indices, column_ptr;
    values.set_size(static_cast<int>(A.nonZeros()));
    row_indices.set_size(static_cast<int>(A.nonZeros()));
    column_ptr.set_size(static_cast<int>(A.cols()) + 1);
    rhs.set_size(static_cast<int>(b.size()));
    for (int k = 0; k < values.size(0); ++k) {
        values[k] = A.valuePtr()[k]; row_indices[k] = A.innerIndexPtr()[k] + 1;
    }
    for (int k = 0; k < column_ptr.size(0); ++k) column_ptr[k] = A.outerIndexPtr()[k] + 1;
    for (int k = 0; k < rhs.size(0); ++k) rhs[k] = b[k];
    coder::sparse generated_A;
    generated_A.init(values, column_ptr, row_indices, static_cast<int>(A.rows()), static_cast<int>(A.cols()));
    cmg_setup_packed_initialize();
    const double adaptation_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - adaptation_begin).count();
    coder::array<double, 1U> generated_x;
    int flag = -99, iter = -1, setup_flag = -99;
    double relres = std::numeric_limits<double>::infinity();
    const auto begin = std::chrono::steady_clock::now();
    cmg_pcg_packed(&generated_A, rhs, tol, static_cast<double>(maxit),
                   generated_x, &flag, &relres, &iter, &setup_flag);
    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    if (cmg_setup_calls != 1) throw std::runtime_error("expected exactly one instrumented setup call");
    if (generated_x.size(0) != A.rows()) throw std::runtime_error("unexpected solution length");
    Eigen::VectorXd x(A.rows());
    for (int k = 0; k < generated_x.size(0); ++k) x[k] = generated_x[k];
    const double bnorm = b.stableNorm();
    const double rnorm = (A*x-b).stableNorm();
    const double true_relres = bnorm == 0.0 ? rnorm : rnorm/bnorm;
    const bool converged = cmg_hierarchy_valid && flag == 0 && x.allFinite() &&
                           std::isfinite(true_relres) && true_relres <= tol;
    std::cout << std::setprecision(17) << "CMG n=" << A.rows() << " nnz=" << A.nonZeros()
              << " hierarchy_valid=" << cmg_hierarchy_valid << " levels=" << cmg_levels
              << " setup_flag=" << setup_flag << " pcg_flag=" << flag << " iter=" << iter
              << " reported_relres=" << relres << " true_relres=" << true_relres
              << " adaptation_s=" << adaptation_s
              << " setup_s=" << cmg_setup_seconds+adaptation_s << " solve_s=" << total_s-cmg_setup_seconds
              << " total_s=" << total_s+adaptation_s << " setup_calls=" << cmg_setup_calls
              << " converged=" << converged << '\n';
    if (argc == 6) {
        std::ofstream out(argv[5]);
        if (!out) throw std::runtime_error("cannot open solution file");
        fast_matrix_market::write_matrix_market_eigen_dense(out, x);
    }
    cmg_setup_packed_terminate();
    if (setup_flag == -1)
        std::cerr << "CMG requires n >= 500; upstream recommends A\\b. No fallback solver was timed.\n";
    // Setup flags 1/3 can accompany a valid iterative terminal hierarchy;
    // retain that canonical fallback and decide solve success by true residual.
    return converged ? 0 : 1;
} catch (const std::exception &e) {
    std::cerr << "CMG input/runtime error: " << e.what() << '\n';
    return 2;
}

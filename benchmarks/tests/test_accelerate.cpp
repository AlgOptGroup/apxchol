#include "apxchol/types.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// Exercise the benchmark's Accelerate adapter through its real static helpers.
#define main apxchol_benchmark_cli_main
#include "../src/benchmark.cpp"
#undef main

namespace {

using Matrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

void require_contract(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Laplacian of two weighted paths plus an isolated vertex: three components,
// so the pin must ground each of them.
Matrix disconnected_laplacian() {
    std::vector<Triplet> entries;
    auto edge = [&](int u, int v, double w) {
        entries.emplace_back(u, v, -w);
        entries.emplace_back(v, u, -w);
        entries.emplace_back(u, u, w);
        entries.emplace_back(v, v, w);
    };
    edge(0, 1, 1.0); edge(1, 2, 1e3); edge(2, 3, 1e-2);
    edge(4, 5, 7.0); edge(5, 6, 0.5);
    Matrix L(8, 8);  // vertex 7 is isolated and has no stored diagonal
    L.setFromTriplets(entries.begin(), entries.end());
    L.makeCompressed();
    return L;
}

void set_mode(const Matrix& L, bool laplacian) {
    g_laplacian_mode = laplacian;
    g_laplacian_components.clear();
    g_laplacian_dimension = 0;
    auto components = connected_components(L);
    g_component_count = components.size();
    if (laplacian && components.size() > 1) {
        g_laplacian_dimension = L.rows();
        g_laplacian_components = std::move(components);
    }
}

void check_solve(const Matrix& L, const Eigen::VectorXd& b, const char* label) {
    BenchResult r = run_accelerate_cholesky(L, b, label, 1e-12, 500);
    require_contract(r.solver_name == "Cholesky [Accelerate]", "solver label changed");
    require_contract(std::isfinite(r.rel_residual) && r.rel_residual <= 1e-12,
                     "Accelerate Cholesky missed the original-system tolerance");
    require_contract(r.iterations >= 1 && r.iterations <= 8 && r.solve_passes == r.iterations,
                     "a direct solve or refinement must count as one work unit");
    require_contract(r.total_time == r.setup_time + r.solve_time, "total is not setup + solve");
}

void disconnected_laplacian_is_grounded_per_component() {
    const Matrix L = disconnected_laplacian();
    set_mode(L, true);
    Eigen::VectorXd g(8);
    g << 1, -2, 3, 0.5, -1, 4, 2, 9;
    Eigen::VectorXd b = L * g;
    center_if_laplacian(b);
    check_solve(L, b, "disconnected-laplacian");
}

void sddm_is_solved_unpinned() {
    Matrix A = disconnected_laplacian();
    for (int i = 0; i < A.rows(); ++i) A.coeffRef(i, i) += 1e-3 * (i + 1);
    A.makeCompressed();
    set_mode(A, false);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(8, -1.0, 2.0);
    check_solve(A, b, "sddm");
}

void zero_budget_reports_no_work() {
    Matrix A = disconnected_laplacian();
    for (int i = 0; i < A.rows(); ++i) A.coeffRef(i, i) += 1.0;
    A.makeCompressed();
    set_mode(A, false);
    Eigen::VectorXd b = Eigen::VectorXd::Ones(8);
    BenchResult r = run_accelerate_cholesky(A, b, "zero-budget", 1e-8, 0);
    require_contract(r.iterations == 0 && r.rel_residual == 1.0,
                     "a zero budget fabricated a solve");
}

}  // namespace

int main() try {
    disconnected_laplacian_is_grounded_per_component();
    sddm_is_solved_unpinned();
    zero_budget_reports_no_work();
    std::cout << "accelerate contract tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

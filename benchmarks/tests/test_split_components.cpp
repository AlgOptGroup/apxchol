#include "apxchol/types.h"
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Exercise the benchmark's actual static helpers without exposing a second API.
// Competitor HAVE_* definitions belong to the benchmark executable, not this test.
#define main apxchol_benchmark_cli_main
#include "../src/benchmark.cpp"
#undef main

namespace {

using Matrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
constexpr double requested_tolerance = 2.5e-9;
constexpr int requested_iterations = 37;

void require_contract(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, const char* message) {
    require_contract(std::isfinite(actual) &&
                     std::abs(actual - expected) <= 1e-12 * std::max(1.0, std::abs(expected)),
                     message);
}

Matrix make_matrix(int n, std::initializer_list<Triplet> entries) {
    Matrix matrix(n, n);
    matrix.setFromTriplets(entries.begin(), entries.end());
    matrix.makeCompressed();
    return matrix;
}

template<class Callback>
BenchResult split(const Matrix& matrix, const Eigen::VectorXd& rhs, Callback callback,
                  double discovery_seconds = 0.0, int iteration_budget = requested_iterations) {
    const auto components = connected_components(matrix);
    return run_split(
        [&](const Matrix& block, const Eigen::VectorXd& block_rhs,
            const std::string& name, double tolerance, int maxiter) {
            require_contract(name == "split-contract", "component graph name changed");
            require_contract(tolerance == requested_tolerance, "component tolerance changed");
            require_contract(maxiter == iteration_budget, "component iteration budget changed");
            return callback(block, block_rhs);
        }, matrix, rhs, "split-contract", requested_tolerance, iteration_budget,
        components, components.size(), discovery_seconds);
}

void positive_diagonal_singletons() {
    const auto matrix = make_matrix(3, {{0, 0, 2}, {1, 1, 3}, {2, 2, 5}});
    Eigen::Vector3d rhs(1, 2, 3);
    int calls = 0;
    auto result = split(matrix, rhs, [&](const Matrix& block, const Eigen::VectorXd& b) {
        require_contract(block.rows() == 1 && block.cols() == 1 && b.size() == 1,
                         "positive scalar was not passed as a scalar system");
        require_close(block.coeff(0, 0), matrix.coeff(calls, calls), "scalar diagonal changed");
        require_close(b[0], rhs[calls], "scalar RHS changed");
        ++calls;
        BenchResult part;
        part.rel_residual = 0;
        part.solve_time = 0.25;
        return part;
    });
    require_contract(calls == 3, "all-diagonal SDDM lost a scalar solve");
    require_close(result.solve_time, 0.75, "scalar solve times were not charged");
    require_close(result.rel_residual, 0, "exact scalar solves have nonzero residual");

    // Exercise the real checked Eigen adapter with the benchmark's SDDM mode.
    struct RestoreMode {
        bool previous = g_laplacian_mode;
        ~RestoreMode() { g_laplacian_mode = previous; }
    } restore_mode;
    g_laplacian_mode = false;
    for (int budget : {requested_iterations, 0}) {
        calls = 0;
        result = split(matrix, rhs, [&](const Matrix& block, const Eigen::VectorXd& b) {
            ++calls;
            return run_cg_no_precond(block, b, "split-contract", requested_tolerance, budget);
        }, 0.0, budget);
        require_contract(calls == 3, "real adapter lost a scalar solve");
        if (budget == 0) {
            require_contract(result.iterations == 0 && result.solve_passes == 0,
                             "zero budget performed a native iteration");
            require_close(result.rel_residual, 1, "zero budget fabricated convergence");
        } else {
            require_contract(result.iterations > 0 && result.solve_passes > 0 &&
                             std::isfinite(result.rel_residual) &&
                             result.rel_residual <= requested_tolerance,
                             "real scalar CG solve did not converge");
        }
    }
}

void mixed_scalar_and_pair() {
    const auto matrix = make_matrix(3, {{0, 0, 2}, {1, 1, 3}, {2, 2, 3},
                                        {1, 2, -1}, {2, 1, -1}});
    Eigen::Vector3d rhs(3, 4, 0);
    int calls = 0;
    auto result = split(matrix, rhs, [&](const Matrix& block, const Eigen::VectorXd& b) {
        const bool scalar = calls++ == 0;
        require_contract(block.rows() == (scalar ? 1 : 2), "mixed component size changed");
        require_close(b.norm(), scalar ? 3.0 : 4.0, "mixed component RHS changed");
        if (!scalar) {
            require_close(block.coeff(0, 1), -1, "pair off-diagonal changed");
            require_close(block.coeff(1, 1), 3, "pair diagonal changed");
        }
        BenchResult part;
        part.setup_time = scalar ? 0.25 : 0.5;
        part.solve_time = scalar ? 0.5 : 0.75;
        part.rel_residual = scalar ? 0.1 : 0.2;
        part.iterations = scalar ? 7 : 11;
        part.solve_passes = scalar ? 1 : 2;
        part.stop_check_seconds = scalar ? 0.125 : 0.25;
        return part;
    }, 0.125);
    require_contract(calls == 2, "mixed SDDM lost a component solve");
    require_close(result.rel_residual, std::sqrt(0.3 * 0.3 + 0.8 * 0.8) / 5,
                  "mixed residual did not use component RHS norms");
    require_close(result.solve_time, 1.25, "component solve times were not summed");
    require_contract(result.setup_time >= 0.875, "setup omitted component or discovery time");
    require_close(result.total_time, result.setup_time + result.solve_time, "total timing differs");
    require_contract(result.iterations == 11 && result.solve_passes == 2,
                     "component maximum iteration/pass counts changed");
    require_close(result.stop_check_seconds, 0.375, "included stop-check times were not summed");
}

void compatible_zero_isolates() {
    const auto matrix = make_matrix(4, {{0, 0, 1}, {1, 1, 1}, {0, 1, -1}, {1, 0, -1}});
    Eigen::Vector4d rhs(1, -1, 0, 0);
    int calls = 0;
    auto result = split(matrix, rhs, [&](const Matrix& block, const Eigen::VectorXd& b) {
        ++calls;
        require_contract(block.rows() == 2 && b.size() == 2, "compatible zero isolate was solved");
        require_close(b.sum(), 0, "compatible Laplacian RHS changed");
        return BenchResult{};
    });
    require_contract(calls == 1, "Laplacian block was skipped or zero isolates were solved");
    require_close(result.rel_residual, 0, "zero isolates changed the aggregate residual");
}

void incompatible_zero_scalar() {
    const auto matrix = make_matrix(2, {});
    Eigen::Vector2d rhs(1, 0);
    int calls = 0;
    auto result = split(matrix, rhs, [&](const Matrix& block, const Eigen::VectorXd& b) {
        ++calls;
        require_contract(block.rows() == 1 && block.coeff(0, 0) == 0 && b[0] == 1,
                         "incompatible zero scalar input changed");
        BenchResult part;
        part.rel_residual = 1;
        return part;
    });
    require_contract(calls == 1, "incompatible zero scalar disappeared");
    require_close(result.rel_residual, 1, "incompatible zero scalar fabricated convergence");
}

void explicit_zero_edges() {
    const auto matrix = make_matrix(4, {{0, 0, 1}, {1, 1, 1}, {0, 1, -1}, {1, 0, -1},
                                        {2, 2, 1}, {3, 3, 1}, {2, 3, -1}, {3, 2, -1},
                                        {1, 2, 0}, {2, 1, 0}});
    require_contract(matrix.nonZeros() == 10, "fixture lost its explicit zero entries");
    const auto components = connected_components(matrix);
    require_contract(components == std::vector<std::vector<int>>{{0, 1}, {2, 3}},
                     "explicit zero entries connected separate Laplacian blocks");
    std::vector<int> pins;
    const auto pinned = dirichlet_pin(matrix, pins);
    require_contract(pins.size() == 2, "grounding did not pin both numerical components");
    require_contract((pins[0] < 2) != (pins[1] < 2), "both pins belong to the same block");
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            require_close(pinned.coeff(row, col), row == col ? 1 : 0,
                          "grounded two-block operator is not the expected identity");
    int calls = 0;
    split(matrix, Eigen::Vector4d(1, -1, 2, -2),
          [&](const Matrix& block, const Eigen::VectorXd&) {
              ++calls;
              require_contract(block.rows() == 2, "zero edge changed the split block size");
              return BenchResult{};
          });
    require_contract(calls == 2, "zero-edge split did not solve both blocks");
}

void finite_zero_rhs_residual() {
    const auto matrix = make_matrix(3, {{0, 0, 2}, {1, 1, 3}, {2, 2, 3},
                                        {1, 2, -1}, {2, 1, -1}});
    int calls = 0;
    auto result = split(matrix, Eigen::Vector3d(0, 3, 4),
        [&](const Matrix&, const Eigen::VectorXd& b) {
            ++calls;
            BenchResult part;
            part.rel_residual = b.norm() == 0 ? 2 : 0;
            return part;
        });
    require_contract(calls == 2, "positive scalar with zero RHS was skipped");
    require_close(result.rel_residual, 0.4, "finite zero-RHS component error disappeared");

    // The all-zero global RHS uses the same absolute-residual convention.
    const auto diagonal = make_matrix(2, {{0, 0, 2}, {1, 1, 3}});
    calls = 0;
    result = split(diagonal, Eigen::Vector2d::Zero(),
        [&](const Matrix&, const Eigen::VectorXd&) {
            BenchResult part;
            part.rel_residual = calls++ == 0 ? 0.3 : 0.4;
            return part;
        });
    require_contract(calls == 2, "all-zero RHS skipped positive scalar systems");
    require_close(result.rel_residual, 0.5, "all-zero RHS lost the absolute residual norm");
}

void nan_residual_propagates() {
    const auto matrix = make_matrix(4, {{0, 0, 2}, {1, 1, 2}, {0, 1, -1}, {1, 0, -1},
                                        {2, 2, 2}, {3, 3, 2}, {2, 3, -1}, {3, 2, -1}});
    int calls = 0;
    auto result = split(matrix, Eigen::Vector4d(0, 0, 1, 2),
        [&](const Matrix&, const Eigen::VectorXd& b) {
            ++calls;
            BenchResult part;
            part.rel_residual = b[0] == 0 ? std::numeric_limits<double>::quiet_NaN() : 0;
            return part;
        });
    require_contract(calls == 2 && std::isnan(result.rel_residual), "NaN component error was hidden");
}

void unavailable_sentinel_propagates() {
    const auto matrix = make_matrix(4, {{0, 0, 2}, {1, 1, 2}, {0, 1, -1}, {1, 0, -1},
                                        {2, 2, 2}, {3, 3, 2}, {2, 3, -1}, {3, 2, -1}});
    for (bool negative_iterations : {false, true}) {
        int calls = 0;
        auto result = split(matrix, Eigen::Vector4d::Ones(),
            [&](const Matrix&, const Eigen::VectorXd&) {
                const bool unavailable = calls++ == 0;
                BenchResult part;
                part.iterations = unavailable && negative_iterations ? -1 : 7;
                part.rel_residual = unavailable && !negative_iterations ? -1 : 0;
                return part;
            });
        require_contract(calls == 2 && result.iterations == -1 && result.rel_residual == -1,
                         "later successful component erased an unavailable sentinel");
    }
}

void callback_error_propagates() {
    const auto matrix = make_matrix(1, {{0, 0, 2}});
    int calls = 0;
    bool caught = false;
    try {
        split(matrix, Eigen::VectorXd::Ones(1),
              [&](const Matrix& block, const Eigen::VectorXd& b) -> BenchResult {
                  ++calls;
                  require_contract(block.rows() == 1 && b[0] == 1, "connected scalar input changed");
                  throw std::runtime_error("controlled component failure");
              });
    } catch (const std::runtime_error& error) {
        caught = std::string(error.what()) == "controlled component failure";
    }
    require_contract(calls == 1 && caught, "component exception was swallowed or replaced");
}

} // namespace

int main() {
    const struct { const char* name; void (*run)(); } cases[] = {
        {"positive_diagonal_singletons", positive_diagonal_singletons},
        {"mixed_scalar_and_pair", mixed_scalar_and_pair},
        {"compatible_zero_isolates", compatible_zero_isolates},
        {"incompatible_zero_scalar", incompatible_zero_scalar},
        {"explicit_zero_edges", explicit_zero_edges},
        {"finite_zero_rhs_residual", finite_zero_rhs_residual},
        {"nan_residual_propagates", nan_residual_propagates},
        {"callback_error_propagates", callback_error_propagates},
        {"unavailable_sentinel_propagates", unavailable_sentinel_propagates},
    };
    int passed = 0;
    for (const auto& test : cases) {
        try {
            test.run();
            ++passed;
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    const int planned = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
    std::cout << passed << '/' << planned << " split-component contract cases passed\n";
    return passed == planned ? 0 : 1;
}

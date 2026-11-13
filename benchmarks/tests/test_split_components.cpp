#include "apxchol/types.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
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
        BenchResult part;
        part.execution_route = "cpu";
        part.factor_offdiag = 1;
        return part;
    });
    require_contract(calls == 1, "Laplacian block was skipped or zero isolates were solved");
    require_close(result.rel_residual, 0, "zero isolates changed the aggregate residual");
    require_contract(matrix.nonZeros() == matrix.rows(), "zero-isolate fixture no longer exposes nnz-minus-n error");
    require_contract(stored_offdiagonal_entries(matrix) == 2, "missing diagonals changed adjacency count");
    require_close(result.fillin, 1, "missing isolate diagonals corrupted aggregate fill");
    // Whole v1 and split metrics use this same post-timing diagnostic count.
    const double whole_formula = 2.0 * result.factor_offdiag / stored_offdiagonal_entries(matrix);
    require_close(result.fillin, whole_formula, "whole/split fill denominators disagree");
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

void complete_route_and_fill_propagate() {
    const auto matrix = make_matrix(4, {{0, 0, 2}, {1, 1, 2}, {0, 1, -1}, {1, 0, -1},
                                        {2, 2, 2}, {3, 3, 2}, {2, 3, -1}, {3, 2, -1}});
    for (const char* route : {"cpu", "gpu"}) {
        const auto result = split(matrix, Eigen::Vector4d::Ones(),
            [&](const Matrix&, const Eigen::VectorXd&) {
                BenchResult part;
                part.execution_route = route;
                part.factor_offdiag = 3;
                return part;
            });
        require_contract(result.execution_route == route, "split lost the complete route");
        require_contract(result.factor_offdiag == 6, "split lost measured factor entries");
        require_close(result.fillin, 3.0, "split normalized fill is not from all components");
    }
    int calls = 0;
    bool caught = false;
    try {
        split(matrix, Eigen::Vector4d::Ones(), [&](const Matrix&, const Eigen::VectorXd&) {
            BenchResult part;
            part.execution_route = calls++ == 0 ? "cpu" : "gpu";
            return part;
        });
    } catch (const std::runtime_error& error) {
        caught = std::string(error.what()).find("changed execution route") != std::string::npos;
    }
    require_contract(caught, "split accepted a mixed CPU/GPU result");
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

void auto_split_only_for_disconnected_laplacians() {
    auto plan = [](const char* solver, const char* ground, bool disconnected, bool laplacian) {
        return resolve_desing(solver, "auto", ground, disconnected, laplacian);
    };
    require_contract(plan("boomeramg", "auto", true, true).decompose == decompose_mode::whole,
                     "BoomerAMG split a disconnected Laplacian under auto");
    require_contract(plan("boomeramg", "auto", true, false).decompose == decompose_mode::whole,
                     "a disconnected SDDM operator splits for BoomerAMG");
    require_contract(plan("boomeramg", "auto", false, true).decompose == decompose_mode::whole,
                     "a connected Laplacian splits for BoomerAMG");
    require_contract(plan("amgcl", "coarse", true, true).decompose == decompose_mode::split,
                     "a disconnected Laplacian no longer splits under coarse grounding");
    const auto sddm_coarse = plan("amgcl", "coarse", true, false);
    require_contract(sddm_coarse.supported && sddm_coarse.decompose == decompose_mode::whole,
                     "a disconnected SDDM operator is not solved whole under coarse grounding");
    require_contract(!resolve_desing("amgcl", "whole", "coarse", true, true).supported,
                     "whole + coarse on a disconnected Laplacian became supported");
    require_contract(resolve_desing("boomeramg", "split", "auto", true, false).decompose
                         == decompose_mode::split,
                     "an explicit split request is ignored for SDDM input");
}

void mixed_grounding() {
    // Interleave the Laplacian pair with a grounded pair, a positive scalar
    // and a zero isolate. Only the first pair and the isolate need a gauge.
    const auto A = make_matrix(6, {{0,0,1},{2,2,1},{0,2,-1},{2,0,-1},
                                   {1,1,2},{3,3,1},{1,3,-1},{3,1,-1},{4,4,3}});
    require_contract(prepare_components(A, false), "mixed operator was missed");
    require_contract(g_singular_vertices == std::vector<char>({1,0,1,0,0,1}),
                     "incorrect constant-nullspace mask");
    std::vector<int> pins;
    auto P = dirichlet_pin(A, pins);
    require_contract(pins.size() == 2 && (pins[0] == 0 || pins[0] == 2) && pins[1] == 5,
                     "expected one gauge for each singular block");
    Eigen::VectorXd b(6);
    b << 2,3,-2,4,6,0;
    auto rhs = b;
    for (int pin : pins) rhs[pin] = 0;
    Eigen::VectorXd x = P.toDense().ldlt().solve(rhs);
    center_if_laplacian(x);
    require_close((A*x-b).norm(), 0, "whole solve changed a nonsingular equation");
    const auto result = split(A, b, [&](const Matrix& block, const Eigen::VectorXd& sub_rhs) {
        require_contract(g_laplacian_mode == (sub_rhs[0] == 2), "wrong split block mode");
        return run_cg_no_precond(block, sub_rhs, "mixed", requested_tolerance, 37);
    });
    require_close(result.rel_residual, 0, "split did not solve the original equations");
    require_contract(g_laplacian_mode, "split leaked its last block mode");
    const double shift = std::nextafter(1.0, 2.0);
    const auto shifted = make_matrix(2, {{0,0,shift},{1,1,1},{0,1,-1},{1,0,-1}});
    require_contract(!constant_null_component(shifted, {0,1}), "pinned a one-ulp shift");
}


// The serial implementations before the adaptation work was parallelized.
// The parallel ones must reproduce them exactly: same components, same pins,
// same operators and slices, at every thread count.
namespace reference {

std::vector<std::vector<int>> connected_components(const Matrix& L) {
    const int n = static_cast<int>(L.rows());
    std::vector<int> parent(n); std::iota(parent.begin(), parent.end(), 0);
    auto find=[&](int x){ while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];} return x; };
    for (int k=0;k<L.outerSize();++k)
        for (Matrix::InnerIterator it(L,k);it;++it)
            if (it.row()!=it.col() && it.value()!=0.0){int a=find((int)it.row()),b=find((int)it.col()); if(a!=b)parent[a]=b;}
    std::vector<int> cid(n,-1); std::vector<std::vector<int>> comps;
    for (int v=0;v<n;++v){int r=find(v); if(cid[r]<0){cid[r]=(int)comps.size();comps.emplace_back();} comps[cid[r]].push_back(v);}
    return comps;
}

Matrix dirichlet_pin(const Matrix& L, std::vector<int>& pinned) {
    pinned.clear();
    const int n = static_cast<int>(L.rows());
    const auto* outer = L.outerIndexPtr();
    const auto* inner = L.innerIndexPtr();
    const auto* values = L.valuePtr();
    std::vector<char> is_pin(n, 0), seen(n, 0);
    std::vector<int> stk, cur(n);
    for (int s = 0; s < n; ++s) {
        if (seen[s]) continue;
        seen[s] = 1; cur[s] = outer[s]; stk.push_back(s);
        int leaf = -1;
        while (!stk.empty()) {
            const int u = stk.back();
            int w = -1;
            while (cur[u] < outer[u + 1]) {
                const int edge = cur[u]++;
                const int v = inner[edge];
                if (v != u && values[edge] != 0.0 && !seen[v]) { w = v; break; }
            }
            if (w != -1) { seen[w] = 1; cur[w] = outer[w]; stk.push_back(w); }
            else { stk.pop_back(); if (leaf == -1) leaf = u; }
        }
        if (L.rows() == g_laplacian_dimension && !g_singular_vertices.empty()
            && !g_singular_vertices[s]) continue;
        is_pin[leaf] = 1; pinned.push_back(leaf);
    }
    Matrix P(n, n);
    Eigen::VectorXi cnt(n);
    for (int k = 0; k < n; ++k) {
        if (is_pin[k]) { cnt[k] = 1; continue; }
        int c = 0;
        for (Matrix::InnerIterator it(L, k); it; ++it) if (!is_pin[it.row()]) ++c;
        cnt[k] = c;
    }
    P.reserve(cnt);
    for (int k = 0; k < n; ++k) {
        if (is_pin[k]) { P.insert(k, k) = 1.0; continue; }
        for (Matrix::InnerIterator it(L, k); it; ++it)
            if (!is_pin[it.row()]) P.insert(it.row(), k) = it.value();
    }
    P.makeCompressed();
    return P;
}

Matrix slice(const Matrix& L, const std::vector<int>& nodes) {
    const int sn = static_cast<int>(nodes.size());
    std::vector<int> g2l(L.rows(), -1);
    for (int i = 0; i < sn; ++i) g2l[nodes[i]] = i;
    Matrix subL(sn, sn);
    Eigen::VectorXi cnt(sn);
    for (int i = 0; i < sn; ++i) {
        int c = 0;
        for (Matrix::InnerIterator it(L, nodes[i]); it; ++it) if (g2l[it.row()] >= 0) ++c;
        cnt[i] = c;
    }
    subL.reserve(cnt);
    for (int i = 0; i < sn; ++i)
        for (Matrix::InnerIterator it(L, nodes[i]); it; ++it) {
            const int gl = g2l[it.row()]; if (gl >= 0) subL.insert(gl, i) = it.value();
        }
    subL.makeCompressed();
    return subL;
}

} // namespace reference

// Same shape, same index arrays and bitwise-equal values.
bool identical(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols() || !a.isCompressed() || !b.isCompressed()
        || a.nonZeros() != b.nonZeros())
        return false;
    const auto nnz = static_cast<std::size_t>(a.nonZeros());
    return std::equal(a.outerIndexPtr(), a.outerIndexPtr() + a.cols() + 1, b.outerIndexPtr())
        && std::equal(a.innerIndexPtr(), a.innerIndexPtr() + nnz, b.innerIndexPtr())
        && std::memcmp(a.valuePtr(), b.valuePtr(), nnz * sizeof(double)) == 0;
}

// A weighted Laplacian on n vertices split into `groups` random vertex sets,
// with stored zeros across sets, isolated vertices and shuffled labels. Groups
// listed in `shifted` get a positive diagonal excess (a mixed operator).
Matrix random_operator(int n, int groups, std::uint64_t seed, int shifted = 0) {
    std::mt19937_64 rng(seed);
    std::vector<int> group(n);
    std::vector<std::vector<int>> members(groups);
    for (int v = 0; v < n; ++v) {
        group[v] = static_cast<int>(rng() % groups);
        members[group[v]].push_back(v);
    }
    std::vector<Triplet> entries;
    std::vector<double> degree(n, 0.0);
    std::uniform_real_distribution<double> weight(0.5, 2.0);
    for (long e = 0; e < 3L * n; ++e) {
        const int a = static_cast<int>(rng() % n);
        if (group[a] == 0) continue;                 // group 0 stays isolated
        const bool across = rng() % 8 == 0;          // a stored zero across groups
        const auto& pool = members[group[a]];
        const int b = across ? static_cast<int>(rng() % n)
                             : pool[rng() % pool.size()];
        if (a == b) continue;
        const double w = across ? 0.0 : weight(rng);
        entries.emplace_back(a, b, -w); entries.emplace_back(b, a, -w);
        degree[a] += w; degree[b] += w;
    }
    for (int v = 0; v < n; ++v) {
        const double excess = group[v] > 0 && group[v] <= shifted ? 0.25 : 0.0;
        if (degree[v] != 0.0 || excess != 0.0) entries.emplace_back(v, v, degree[v] + excess);
    }
    Matrix L(n, n);
    L.setFromTriplets(entries.begin(), entries.end());
    L.makeCompressed();
    return L;
}

Matrix grid_laplacian(int side) {
    std::vector<Triplet> entries;
    const int n = side * side;
    std::vector<double> degree(n, 0.0);
    auto edge = [&](int a, int b, double w) {
        entries.emplace_back(a, b, -w); entries.emplace_back(b, a, -w);
        degree[a] += w; degree[b] += w;
    };
    for (int r = 0; r < side; ++r)
        for (int c = 0; c < side; ++c) {
            const int v = r * side + c;
            if (c + 1 < side) edge(v, v + 1, 1.0 + (v % 7));
            if (r + 1 < side) edge(v, v + side, 1.0 + (v % 5));
        }
    for (int v = 0; v < n; ++v) entries.emplace_back(v, v, degree[v]);
    Matrix L(n, n);
    L.setFromTriplets(entries.begin(), entries.end());
    L.makeCompressed();
    return L;
}

void adaptation_matches_serial_reference() {
    struct RestoreGlobals {
        ~RestoreGlobals() {
            g_singular_vertices.clear(); g_laplacian_components.clear();
            g_laplacian_dimension = 0; g_component_count = 0; g_laplacian_mode = true;
        }
    } restore;
    const std::vector<std::pair<const char*, Matrix>> cases = {
        {"many components", random_operator(120000, 4000, 7)},
        {"few components", random_operator(90000, 3, 11)},
        {"grid", grid_laplacian(300)},
        {"mixed", random_operator(80000, 50, 13, 20)},
    };
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    const std::vector<int> thread_counts = {1, 3, 8};
#else
    const std::vector<int> thread_counts = {1};
#endif
    for (const auto& [name, L] : cases) {
        require_contract(L.nonZeros() >= (1 << 16), "fixture too small for the parallel paths");
        const bool mixed = std::string(name) == "mixed";
        const auto expected_components = reference::connected_components(L);
        require_contract(expected_components.size() > 1 || std::string(name) == "grid",
                         "fixture lost its components");
        // Reference pins read the same globals (only a mixed operator sets them).
        g_singular_vertices.clear(); g_laplacian_dimension = 0;
        if (mixed) require_contract(prepare_components(L, false), "mixed fixture is not mixed");
        std::vector<int> expected_pins;
        const Matrix expected_pinned = reference::dirichlet_pin(L, expected_pins);
        for (int threads : thread_counts) {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
            const std::string where = std::string(name) + " at " + std::to_string(threads) + " threads: ";
            require_contract(connected_components(L) == expected_components,
                             (where + "components changed").c_str());
            if (mixed) {
                require_contract(prepare_components(L, false), "mixed fixture is not mixed");
                require_contract(g_laplacian_components == expected_components,
                                 (where + "discovered components changed").c_str());
            }
            std::vector<int> pins;
            const Matrix pinned = dirichlet_pin(L, pins);
            require_contract(pins == expected_pins, (where + "pins changed").c_str());
            require_contract(identical(pinned, expected_pinned),
                             (where + "pinned operator changed").c_str());
            std::size_t block = 0;
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(L.rows());
            run_split([&](const Matrix& sub, const Eigen::VectorXd&, const std::string&, double, int) {
                          while (expected_components[block].size() == 1
                                 && L.coeff(expected_components[block][0],
                                            expected_components[block][0]) == 0.0)
                              ++block;   // zero isolates with a zero RHS are skipped
                          require_contract(identical(sub, reference::slice(L, expected_components[block++])),
                                           (where + "component slice changed").c_str());
                          BenchResult part; part.rel_residual = 0; return part;
                      }, L, rhs, "identity", 1e-8, 1, expected_components,
                      expected_components.size(), 0.0);
        }
        if (mixed) { g_singular_vertices.clear(); g_laplacian_components.clear(); }
    }
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
#endif
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
        {"complete_route_and_fill_propagate", complete_route_and_fill_propagate},
        {"unavailable_sentinel_propagates", unavailable_sentinel_propagates},
        {"mixed_grounding", mixed_grounding},
        {"auto_split_only_for_disconnected_laplacians", auto_split_only_for_disconnected_laplacians},
        {"adaptation_matches_serial_reference", adaptation_matches_serial_reference},
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

// bench_rchol.cpp — Benchmark driver for RCHOL factorization + Eigen PCG
//
// Builds against librchol.a (no MKL needed) and uses Eigen for PCG iteration.
// Output format matches our main C++ benchmark for easy comparison.
//
// Build:
//   cd extern/rchol/c++ && make -f Makefile.gcc
//   g++ -O3 -std=c++17 -I../extern/rchol/c++ -I../include \
//       bench/rchol/bench_rchol.cpp -Lextern/rchol/c++ -lrchol \
//       $(pkg-config --cflags --libs eigen3) -o build/bench_rchol

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

// RCHOL headers
#include "sparse.hpp"
#include "rchol.hpp"
#include "util.hpp"

// ── Helpers ──────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

static SparseCSR eigen_to_csr(const Eigen::SparseMatrix<double, Eigen::RowMajor>& M) {
    int n = static_cast<int>(M.rows());
    std::vector<size_t> rowPtr(n + 1);
    std::vector<size_t> colIdx;
    std::vector<double> val;

    colIdx.reserve(M.nonZeros());
    val.reserve(M.nonZeros());

    for (int i = 0; i <= n; ++i)
        rowPtr[i] = static_cast<size_t>(M.outerIndexPtr()[i]);
    for (int k = 0; k < M.nonZeros(); ++k) {
        colIdx.push_back(static_cast<size_t>(M.innerIndexPtr()[k]));
        val.push_back(M.valuePtr()[k]);
    }
    return SparseCSR(rowPtr, colIdx, val);
}

// Build grid Laplacian (matching our benchmark)
static Eigen::SparseMatrix<double, Eigen::RowMajor>
grid_laplacian(int rows, int cols, double kappa, int tile) {
    int n = rows * cols;
    using T = Eigen::Triplet<double>;
    std::vector<T> trips;
    auto idx = [cols](int r, int c) { return r * cols + c; };
    auto weight = [&](int r1, int c1, int r2, int c2) -> double {
        int t1 = (r1 / tile + c1 / tile) % 2;
        int t2 = (r2 / tile + c2 / tile) % 2;
        return (t1 != t2) ? 1.0 : kappa;
    };

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double deg = 0;
            auto add = [&](int r2, int c2) {
                double w = weight(r, c, r2, c2);
                trips.emplace_back(idx(r, c), idx(r2, c2), -w);
                deg += w;
            };
            if (r > 0) add(r - 1, c);
            if (r < rows - 1) add(r + 1, c);
            if (c > 0) add(r, c - 1);
            if (c < cols - 1) add(r, c + 1);
            trips.emplace_back(idx(r, c), idx(r, c), deg);
        }
    }
    // RCHOL requires strictly SDD (not SPSD). Add small diagonal shift.
    // eps/n ensures the shift is small relative to the smallest edge weight.
    double eps = 1e-4;
    for (int i = 0; i < n; ++i) trips.emplace_back(i, i, eps);
    Eigen::SparseMatrix<double, Eigen::RowMajor> L(n, n);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

// RCHOL factor as Eigen PCG preconditioner
// RCHOL produces an upper triangular G such that A ≈ G^T G
// Preconditioner: M^{-1} b = G^{-1} G^{-T} b
struct RcholPreconditioner {
    using StorageIndex = int;
    Eigen::SparseMatrix<double> G_eigen;  // upper triangular factor
    int n_ = 0;

    void init(const SparseCSR& G) {
        n_ = static_cast<int>(G.size());
        using T = Eigen::Triplet<double>;
        std::vector<T> trips;
        for (int i = 0; i < n_; ++i) {
            for (size_t k = G.rowPtr[i]; k < G.rowPtr[i + 1]; ++k) {
                int j = static_cast<int>(G.colIdx[k]);
                trips.emplace_back(i, j, G.val[k]);
            }
        }
        G_eigen.resize(n_, n_);
        G_eigen.setFromTriplets(trips.begin(), trips.end());
    }

    // M^{-1} b = G^{-1} (G^{-T} b)
    Eigen::VectorXd solve(const Eigen::VectorXd& b) const {
        // Step 1: G^T x = b  (lower triangular solve since G is upper)
        Eigen::VectorXd x = G_eigen.transpose().triangularView<Eigen::Lower>().solve(b);
        // Step 2: G y = x  (upper triangular solve)
        Eigen::VectorXd y = G_eigen.triangularView<Eigen::Upper>().solve(x);
        return y;
    }

    int rows() const { return n_; }
    int cols() const { return n_; }
};

// ── Main ─────────────────────────────────────────────

int main(int argc, char* argv[]) {
    int n = 100;
    double kappa = 1000.0;
    int tile = 4;
    double tol = 1e-8;
    int maxiter = 500;
    bool csv = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kappa") && i + 1 < argc) kappa = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--maxiter") && i + 1 < argc) maxiter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv")) csv = true;
    }

    // Build Laplacian
    auto L = grid_laplacian(n, n, kappa, tile);
    int N = static_cast<int>(L.rows());
    long nnz = L.nonZeros();

    // RHS: b = L * random, projected to zero mean
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0, 1);
    Eigen::VectorXd g(N);
    for (int i = 0; i < N; ++i) g[i] = dist(rng);
    Eigen::VectorXd b = L * g;
    b.array() -= b.mean();
    b /= b.norm();

    // Convert to CSR for RCHOL
    SparseCSR A = eigen_to_csr(L);

    // Time factorization (suppress rchol stdout)
    auto t0 = Clock::now();
    SparseCSR G;
    {
        std::streambuf* old = std::cout.rdbuf();
        std::ostringstream devnull;
        std::cout.rdbuf(devnull.rdbuf());
        rchol(A, G);
        std::cout.rdbuf(old);
    }
    auto t1 = Clock::now();
    double setup_s = std::chrono::duration<double>(t1 - t0).count();

    // Build Eigen preconditioner from G
    RcholPreconditioner P;
    P.init(G);
    double fillin = 2.0 * static_cast<double>(G.nnz()) / static_cast<double>(A.nnz());

    // Manual PCG with RCHOL preconditioner
    auto t2 = Clock::now();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd r = b;
    Eigen::VectorXd z = P.solve(r);
    Eigen::VectorXd p = z;
    double rz = r.dot(z);
    int iters = 0;
    double bnorm = b.norm();

    for (int it = 0; it < maxiter; ++it) {
        Eigen::VectorXd Ap = L * p;
        double pAp = p.dot(Ap);
        if (std::abs(pAp) < 1e-30) break;
        double alpha = rz / pAp;
        x += alpha * p;
        r -= alpha * Ap;
        iters = it + 1;
        if (r.norm() / bnorm < tol) break;

        z = P.solve(r);
        double rz_new = r.dot(z);
        double beta = rz_new / rz;
        p = z + beta * p;
        rz = rz_new;
    }
    auto t3 = Clock::now();
    double solve_s = std::chrono::duration<double>(t3 - t2).count();

    // Remove null-space component
    x.array() -= x.mean();
    Eigen::VectorXd res = b - L * x;
    res.array() -= res.mean();
    double rel_res = res.norm() / bnorm;
    double total_s = setup_s + solve_s;

    std::string graph_name = "checker_" + std::to_string(n) + "_k" + std::to_string((int)kappa) + "_t" + std::to_string(tile);

    if (csv) {
        std::cout << "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz\n";
        std::cout << "RCHOL,"
                  << graph_name << ","
                  << N << ","
                  << nnz << ","
                  << std::scientific << std::setprecision(6) << setup_s << ","
                  << solve_s << ","
                  << total_s << ","
                  << iters << ","
                  << rel_res << ","
                  << std::fixed << std::setprecision(4) << fillin << ","
                  << (total_s * 1e6 / nnz) << "\n";
    } else {
        std::cout << "RCHOL on " << graph_name << "\n"
                  << "  n=" << N << "  nnz=" << nnz << "\n"
                  << "  Setup:   " << std::fixed << std::setprecision(4) << setup_s << " s\n"
                  << "  Solve:   " << solve_s << " s (" << iters << " iters)\n"
                  << "  Total:   " << total_s << " s\n"
                  << "  RelRes:  " << std::scientific << std::setprecision(3) << rel_res << "\n"
                  << "  Fill-in: " << std::fixed << std::setprecision(2) << fillin << "\n";
    }

    return 0;
}

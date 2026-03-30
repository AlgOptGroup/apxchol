/// bench_backends.cpp – Comprehensive benchmark comparing solver backends.
///
/// Measures all combinations of:
///   - Graph types: grid, checkerboard, Matrix Market files
///   - SpTRSV backends: Eigen (sequential), OMP level-set, CUDA (cuSPARSE)
///
/// Usage:
///   bench_backends [--grid] [--checker] [--mtx path ...] [--sizes s1,s2,...] [--csv]

#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Sparse>
#include <fast_matrix_market/app/Eigen.hpp>

#include "apxchol/graph/conversions.h"
#include "apxchol/graph/graph.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"

// ── Graph generators ──────────────────────────────────

static Eigen::SparseMatrix<double> grid_laplacian(int rows, int cols) {
    apxchol::graph<> G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
        }
    return apxchol::laplacian(G);
}

static Eigen::SparseMatrix<double> checkerboard_laplacian(int rows, int cols,
                                                           double kappa, int tile) {
    apxchol::graph<> G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    double w_high = kappa, w_low = 1.0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            int block = (r / tile + c / tile) % 2;
            double w = block ? w_high : w_low;
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), w);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), w);
        }
    return apxchol::laplacian(G);
}

static Eigen::SparseMatrix<double> load_mtx(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);
    Eigen::SparseMatrix<double> M;
    fast_matrix_market::read_matrix_market_eigen(f, M);
    // Ensure symmetric
    if (M.rows() != M.cols())
        throw std::runtime_error("Matrix is not square");
    // Make sure it's the full matrix (symmetrize if needed)
    Eigen::SparseMatrix<double> Mt = M.transpose();
    Eigen::SparseMatrix<double> S = (M + Mt) * 0.5;
    return S;
}

// ── Solver dispatch ──────────────────────────────────

struct bench_result {
    const char* solver;
    const char* graph;
    int n;
    long long nnz;
    double setup_ms;
    double solve_ms;
    double total_ms;
    int iters;
    double residual;
    char label[64];
};

static const char* backend_name() {
#if defined(APXCHOL_USE_CUDA)
    return "CUDA";
#else
    return "OMP";
#endif
}

static bench_result run_eigen_pcg(const Eigen::SparseMatrix<double>& L,
                                   const Eigen::VectorXd& b,
                                   const char* graph_label) {
    apxchol::solve_options opts;
    auto res = apxchol::solve(L, b, opts);

    bench_result r{};
    char name[64];
    std::snprintf(name, sizeof(name), "EigenPCG+%s", backend_name());
    r.solver = strdup(name);
    r.graph = graph_label;
    r.n = static_cast<int>(L.rows());
    r.nnz = L.nonZeros();
    r.setup_ms = res.timings.total("setup") * 1000;
    r.solve_ms = res.timings.total("solve") * 1000;
    r.total_ms = r.setup_ms + r.solve_ms;
    r.iters = static_cast<int>(res.iterations);
    r.residual = res.residual;
    return r;
}

// ── Printing ─────────────────────────────────────────

static void print_header() {
    std::printf("%-22s %-24s %10s %12s %10s %10s %10s %6s %12s\n",
                "Solver", "Graph", "n", "nnz", "setup(ms)", "solve(ms)", "total(ms)", "iters", "residual");
    std::printf("%s\n", std::string(120, '-').c_str());
}

static void print_result(const bench_result& r) {
    std::printf("%-22s %-24s %10d %12lld %10.1f %10.1f %10.1f %6d %12.2e\n",
                r.solver, r.graph, r.n, r.nnz,
                r.setup_ms, r.solve_ms, r.total_ms, r.iters, r.residual);
}

static void print_csv_header() {
    std::printf("solver,graph,n,nnz,setup_ms,solve_ms,total_ms,iters,residual\n");
}

static void print_csv(const bench_result& r) {
    std::printf("%s,%s,%d,%lld,%.2f,%.2f,%.2f,%d,%.6e\n",
                r.solver, r.graph, r.n, r.nnz,
                r.setup_ms, r.solve_ms, r.total_ms, r.iters, r.residual);
}

// ── Main ─────────────────────────────────────────────

int main(int argc, char** argv) {
    bool do_grid = false, do_checker = false, do_csv = false;
    std::vector<std::string> mtx_paths;
    std::vector<int> sizes;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--grid") == 0)     { do_grid = true; }
        else if (std::strcmp(argv[i], "--checker") == 0) { do_checker = true; }
        else if (std::strcmp(argv[i], "--csv") == 0) { do_csv = true; }
        else if (std::strcmp(argv[i], "--mtx") == 0 && i + 1 < argc) {
            mtx_paths.push_back(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t next = s.find(',', pos);
                if (next == std::string::npos) next = s.size();
                sizes.push_back(std::stoi(s.substr(pos, next - pos)));
                pos = next + 1;
            }
        }
    }

    // Defaults
    if (!do_grid && !do_checker && mtx_paths.empty()) {
        do_grid = true;
        do_checker = true;
    }
    if (sizes.empty()) {
        sizes = {100, 200, 500, 1000};
    }

    if (do_csv) print_csv_header();
    else print_header();

    auto emit = [&](const bench_result& r) {
        if (do_csv) print_csv(r); else print_result(r);
    };

    auto run_all_solvers = [&](const Eigen::SparseMatrix<double>& L,
                               const char* label) {
        auto b = apxchol::generate_test_rhs(L.rows());

        // Run Eigen PCG with the compiled-in SpTRSV backend
        emit(run_eigen_pcg(L, b, label));
    };

    // ── Grid graphs ──
    if (do_grid) {
        for (int side : sizes) {
            char label[64];
            std::snprintf(label, sizeof(label), "grid_%dx%d", side, side);
            auto L = grid_laplacian(side, side);
            run_all_solvers(L, label);
        }
    }

    // ── Checkerboard graphs (κ = 1000, tile = 4) ──
    if (do_checker) {
        for (int side : sizes) {
            char label[64];
            std::snprintf(label, sizeof(label), "checker_%dx%d_k1e3", side, side);
            auto L = checkerboard_laplacian(side, side, 1000.0, 4);
            run_all_solvers(L, label);
        }
    }

    // ── Matrix Market files ──
    for (const auto& path : mtx_paths) {
        auto L = load_mtx(path);
        // Extract filename for label
        auto pos = path.rfind('/');
        std::string name = (pos != std::string::npos) ? path.substr(pos + 1) : path;
        // Truncate at 24 chars
        if (name.size() > 24) name = name.substr(0, 21) + "...";
        run_all_solvers(L, name.c_str());
    }

    return 0;
}

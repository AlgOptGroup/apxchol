#include "apxchol.h"
#include <iostream>
#include <string>
#include <random>

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <input.mtx> [options]\n"
        << "\n"
        << "Solve a Laplacian system Lx = b using approximate Cholesky + PCG.\n"
        << "\n"
        << "Options:\n"
        << "  --tol <val>      Relative residual tolerance (default: 1e-8)\n"
        << "  --maxiter <n>    Maximum PCG iterations (default: 200)\n"
        << "  --seed <n>       Random seed for factorization and RHS (default: 42)\n"
        << "  --info           Print matrix info and exit (no solve)\n"
        << "  --help           Show this message\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    std::string mtx_path;
    double tol = 1e-8;
    int max_iter = 200;
    unsigned seed = 42;
    bool info_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        else if (arg == "--tol" && i + 1 < argc)     tol = std::stod(argv[++i]);
        else if (arg == "--maxiter" && i + 1 < argc)  max_iter = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc)     seed = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (arg == "--info")                      info_only = true;
        else if (arg[0] != '-')                        mtx_path = arg;
        else { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
    }

    if (mtx_path.empty()) {
        std::cerr << "Error: no input file specified\n";
        return 1;
    }

    // Load graph
    auto G = apxchol::load_mtx(mtx_path);
    auto L = G.laplacian();

    std::cout << "n = " << G.n() << ",  m = " << G.m()
              << ",  nnz(L) = " << L.nonZeros() << "\n";

    if (info_only) return 0;

    // Generate random RHS: b = L * g  (so the system is consistent)
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> N(0.0, 1.0);
    Eigen::VectorXd g(G.n());
    for (int i = 0; i < G.n(); ++i) g[i] = N(rng);

    Eigen::VectorXd b = L * g;
    b.array() -= b.mean();
    double bnorm = b.norm();
    if (bnorm > 0) b /= bnorm;

    // Solve
    apxchol::solve_options opts;
    opts.tol = tol;
    opts.max_iter = max_iter;
    opts.factor_opts.seed = seed;

    auto res = apxchol::solve(L, b, opts);

    std::cout << "setup:      " << res.setup_seconds << " s\n"
              << "solve:      " << res.solve_seconds << " s\n"
              << "iterations: " << res.iterations << "\n"
              << "residual:   " << res.residual << "\n";

    return 0;
}

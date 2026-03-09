// benchmark.cpp – comprehensive Laplacian solver benchmark
//
// Usage:
//   benchmark [options]
//
// Options:
//   --graph <type>     grid | checkerboard | erdos | mtx   (default: checkerboard)
//   --n <int>          grid side length / vertex count      (default: 500)
//   --kappa <double>   condition parameter (a_high/a_low)   (default: 1000)
//   --tile <int>       checkerboard tile size               (default: 4)
//   --er-p <double>    Erdős-Rényi edge probability         (default: 0.01)
//   --mtx <path>       Matrix Market file path
//   --solver <list>    comma-separated: apxchol,cg,icc,ldlt,all  (default: all)
//   --tol <double>     PCG tolerance                        (default: 1e-8)
//   --maxiter <int>    PCG max iterations                   (default: 500)
//   --csv              output in CSV format
//   --seed <int>       RNG seed                             (default: 42)

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <set>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>

#include "../include/graphs.h"
#include "../include/solver.h"
#include "../include/simple_solver.h"
#include "../include/mmio.h"

// ──────────────────── timer helper ────────────────────
struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double>(clock::now() - t0).count();
    }
};

// ──────────────────── build Laplacian from adjacency ────────────────────
static Eigen::SparseMatrix<double>
laplacian_from_adj(const std::vector<std::vector<Edge>>& adj)
{
    using T = Eigen::Triplet<double>;
    int n = static_cast<int>(adj.size());
    std::vector<T> trips;

    for (int i = 0; i < n; ++i) {
        double deg = 0.0;
        for (auto& e : adj[i]) {
            deg += e.w;
            trips.emplace_back(i, e.to, -e.w / 2);
            trips.emplace_back(e.to, i, -e.w / 2);
        }
        trips.emplace_back(i, i, deg);
    }

    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

// ──────────────────── apxchol preconditioner adaptor ────────────────────
class apx_preconditioner {
public:
    using Scalar = double;
    using RealScalar = double;
    using StorageIndex = int;

    apx_preconditioner() = default;
    explicit apx_preconditioner(lap_solver& s) : s_(&s) {}

    template <class MatrixType>
    apx_preconditioner& compute(const MatrixType& A) {
        n_ = A.rows();
        info_ = (s_ != nullptr) ? Eigen::Success : Eigen::NumericalIssue;
        return *this;
    }

    Eigen::Index rows() const { return n_; }
    Eigen::Index cols() const { return n_; }
    Eigen::ComputationInfo info() const { return info_; }

    template <class Rhs>
    Eigen::VectorXd solve(const Rhs& b) const {
        Eigen::VectorXd bb = b;
        bb.array() -= bb.mean();
        std::vector<double> bv(bb.data(), bb.data() + bb.size());
        auto xv = s_->solve(bv);
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(xv.data(), xv.size());
        x.array() -= x.mean();
        return x;
    }

private:
    lap_solver* s_ = nullptr;
    Eigen::Index n_ = 0;
    Eigen::ComputationInfo info_ = Eigen::Success;
};

namespace Eigen { namespace internal {
    template <> struct traits<apx_preconditioner> : traits<Eigen::SparseMatrix<double>> {};
}}

// ──────────────────── benchmark result ────────────────────
struct BenchResult {
    std::string solver_name;
    std::string graph_name;
    int n = 0;
    int nnz = 0;
    double setup_time = 0;
    double solve_time = 0;
    double total_time = 0;
    int iterations = 0;
    double rel_residual = 0;
    double fillin = 0;       // nnz_factor / nnz_original
    double us_per_nnz = 0;   // total µs / nnz
};

static void print_header_pretty() {
    std::cout << std::left
              << std::setw(16) << "Solver"
              << std::setw(22) << "Graph"
              << std::setw(10) << "n"
              << std::setw(12) << "nnz"
              << std::setw(12) << "Setup(s)"
              << std::setw(12) << "Solve(s)"
              << std::setw(12) << "Total(s)"
              << std::setw(8)  << "Iters"
              << std::setw(14) << "RelRes"
              << std::setw(10) << "Fill-in"
              << std::setw(12) << "µs/nnz"
              << "\n";
    std::cout << std::string(136, '-') << "\n";
}

static void print_result_pretty(const BenchResult& r) {
    std::cout << std::left
              << std::setw(16) << r.solver_name
              << std::setw(22) << r.graph_name
              << std::setw(10) << r.n
              << std::setw(12) << r.nnz
              << std::setw(12) << std::fixed << std::setprecision(4) << r.setup_time
              << std::setw(12) << std::fixed << std::setprecision(4) << r.solve_time
              << std::setw(12) << std::fixed << std::setprecision(4) << r.total_time
              << std::setw(8)  << r.iterations
              << std::setw(14) << std::scientific << std::setprecision(3) << r.rel_residual
              << std::setw(10) << std::fixed << std::setprecision(2) << r.fillin
              << std::setw(12) << std::fixed << std::setprecision(2) << r.us_per_nnz
              << "\n";
}

static void print_csv_header() {
    std::cout << "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz\n";
}

static void print_result_csv(const BenchResult& r) {
    std::cout << r.solver_name << ","
              << r.graph_name << ","
              << r.n << ","
              << r.nnz << ","
              << std::scientific << std::setprecision(6) << r.setup_time << ","
              << r.solve_time << ","
              << r.total_time << ","
              << r.iterations << ","
              << r.rel_residual << ","
              << std::fixed << std::setprecision(4) << r.fillin << ","
              << r.us_per_nnz << "\n";
}

// ──────────────────── generate RHS ────────────────────
static Eigen::VectorXd make_rhs(const Eigen::SparseMatrix<double>& L, unsigned seed) {
    int n = static_cast<int>(L.rows());
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> N(0.0, 1.0);

    Eigen::VectorXd g(n);
    for (int i = 0; i < n; ++i) g[i] = N(rng);

    Eigen::VectorXd b = L * g;
    b.array() -= b.mean();
    double nrm = b.norm();
    if (nrm > 0.0) b /= nrm;
    return b;
}

// ──────────────────── solver runners ────────────────────

static BenchResult run_apxchol(
    const std::vector<std::vector<Edge>>& adj,
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = "ApxChol+PCG";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    Timer t;
    t.start();
    simple_solver solver(adj);
    r.setup_time = t.elapsed();

    apx_preconditioner M(solver);
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                             Eigen::Lower | Eigen::Upper,
                             apx_preconditioner> cg;
    cg.setMaxIterations(maxiter);
    cg.setTolerance(tol);
    cg.preconditioner() = M;

    t.start();
    cg.compute(L);
    Eigen::VectorXd x = cg.solve(b);
    r.solve_time = t.elapsed();

    r.total_time = r.setup_time + r.solve_time;
    r.iterations = static_cast<int>(cg.iterations());

    x.array() -= x.mean();
    Eigen::VectorXd res = b - L * x;
    res.array() -= res.mean();
    double bnorm = b.norm();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);
    r.fillin = solver.num_nonzeros() * 2.0 / r.nnz;
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    return r;
}

static BenchResult run_cg_no_precond(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = "CG(none)";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                             Eigen::Lower | Eigen::Upper,
                             Eigen::DiagonalPreconditioner<double>> cg;
    // DiagonalPreconditioner is Eigen's default "identity-like" preconditioner
    cg.setMaxIterations(maxiter);
    cg.setTolerance(tol);

    Timer t;
    t.start();
    cg.compute(L);
    r.setup_time = t.elapsed();

    t.start();
    Eigen::VectorXd x = cg.solve(b);
    r.solve_time = t.elapsed();

    r.total_time = r.setup_time + r.solve_time;
    r.iterations = static_cast<int>(cg.iterations());

    x.array() -= x.mean();
    Eigen::VectorXd res = b - L * x;
    res.array() -= res.mean();
    double bnorm = b.norm();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);
    r.fillin = 0;
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    return r;
}

static BenchResult run_cg_icc(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = "CG+ICC";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    // Shift diagonal slightly to ensure positive-definiteness for ICC
    Eigen::SparseMatrix<double> Ls = L;
    double shift = 1e-6;
    for (int k = 0; k < Ls.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Ls, k); it; ++it)
            if (it.row() == it.col())
                it.valueRef() += shift;

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                             Eigen::Lower | Eigen::Upper,
                             Eigen::IncompleteCholesky<double>> cg;
    cg.setMaxIterations(maxiter);
    cg.setTolerance(tol);

    Timer t;
    t.start();
    cg.compute(Ls);
    r.setup_time = t.elapsed();

    t.start();
    Eigen::VectorXd x = cg.solve(b);
    r.solve_time = t.elapsed();

    r.total_time = r.setup_time + r.solve_time;
    r.iterations = static_cast<int>(cg.iterations());

    x.array() -= x.mean();
    Eigen::VectorXd res = b - L * x;
    res.array() -= res.mean();
    double bnorm = b.norm();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);
    r.fillin = 0;
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    return r;
}

static BenchResult run_ldlt(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name)
{
    BenchResult r;
    r.solver_name = "SimplicLDLT";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    // Pin one vertex (remove last row/col) to make it PD
    int n = r.n;
    int m = n - 1;
    Eigen::SparseMatrix<double> Lsub = L.topLeftCorner(m, m);
    Eigen::VectorXd bsub = b.head(m);
    bsub.array() -= bsub.mean();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;

    Timer t;
    t.start();
    ldlt.compute(Lsub);
    r.setup_time = t.elapsed();

    if (ldlt.info() != Eigen::Success) {
        r.solver_name += "(FAIL)";
        r.solve_time = 0;
        r.total_time = r.setup_time;
        r.iterations = 0;
        r.rel_residual = std::numeric_limits<double>::quiet_NaN();
        return r;
    }

    t.start();
    Eigen::VectorXd xsub = ldlt.solve(bsub);
    r.solve_time = t.elapsed();

    r.total_time = r.setup_time + r.solve_time;
    r.iterations = 1; // direct

    Eigen::VectorXd x(n);
    x.head(m) = xsub;
    x(m) = 0;
    x.array() -= x.mean();

    Eigen::VectorXd res = b - L * x;
    res.array() -= res.mean();
    double bnorm = b.norm();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);
    r.fillin = 0;
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    return r;
}

// ──────────────────── argument parsing ────────────────────
struct Args {
    std::string graph = "checkerboard";
    int n = 500;
    double kappa = 1000.0;
    int tile = 4;
    double er_p = 0.01;
    std::string mtx_path;
    std::set<std::string> solvers;
    double tol = 1e-8;
    int maxiter = 500;
    bool csv = false;
    unsigned seed = 42;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--graph")    a.graph = next();
        else if (arg == "--n")   a.n = std::stoi(next());
        else if (arg == "--kappa") a.kappa = std::stod(next());
        else if (arg == "--tile")  a.tile = std::stoi(next());
        else if (arg == "--er-p")  a.er_p = std::stod(next());
        else if (arg == "--mtx")   { a.mtx_path = next(); a.graph = "mtx"; }
        else if (arg == "--tol")   a.tol = std::stod(next());
        else if (arg == "--maxiter") a.maxiter = std::stoi(next());
        else if (arg == "--seed")  a.seed = static_cast<unsigned>(std::stoul(next()));
        else if (arg == "--csv")   a.csv = true;
        else if (arg == "--solver") {
            std::string s = next();
            std::istringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) a.solvers.insert(tok);
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::exit(1);
        }
    }
    if (a.solvers.empty() || a.solvers.count("all"))
        a.solvers = {"apxchol", "cg", "icc", "ldlt"};
    return a;
}

// ──────────────────── main ────────────────────
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // Build graph
    std::vector<std::vector<Edge>> adj;
    std::string graph_name;

    if (args.graph == "grid") {
        adj = grid_graph(args.n, args.n);
        graph_name = "grid_" + std::to_string(args.n);
    } else if (args.graph == "checkerboard") {
        adj = grid_graph_checkerboard(args.n, args.n, args.kappa, 1.0, args.tile);
        graph_name = "checker_" + std::to_string(args.n) + "_k" + std::to_string(static_cast<int>(args.kappa)) + "_t" + std::to_string(args.tile);
    } else if (args.graph == "erdos") {
        adj = erdos_renyi_graph(args.n, args.er_p, args.seed);
        graph_name = "erdos_" + std::to_string(args.n) + "_p" + std::to_string(args.er_p).substr(0,4);
    } else if (args.graph == "mtx") {
        auto res = load_mtx_as_adjacency(args.mtx_path);
        adj = std::move(res.adj);
        // extract filename for label
        auto pos = args.mtx_path.rfind('/');
        graph_name = (pos != std::string::npos) ? args.mtx_path.substr(pos + 1) : args.mtx_path;
    } else {
        std::cerr << "Unknown graph type: " << args.graph << "\n";
        return 1;
    }

    int n = static_cast<int>(adj.size());
    std::cerr << "Graph: " << graph_name << ", n=" << n;

    Eigen::SparseMatrix<double> L = laplacian_from_adj(adj);
    int nnz = static_cast<int>(L.nonZeros());
    std::cerr << ", nnz=" << nnz << "\n";

    Eigen::VectorXd b = make_rhs(L, args.seed);

    // Print header
    if (args.csv) print_csv_header();
    else print_header_pretty();

    auto print = [&](const BenchResult& r) {
        if (args.csv) print_result_csv(r);
        else print_result_pretty(r);
    };

    // Run selected solvers
    if (args.solvers.count("apxchol")) {
        // Suppress cout from simple_solver (redirect temporarily)
        std::streambuf* old = std::cout.rdbuf();
        std::ostringstream devnull;
        std::cout.rdbuf(devnull.rdbuf());
        auto r = run_apxchol(adj, L, b, graph_name, args.tol, args.maxiter);
        std::cout.rdbuf(old);
        print(r);
    }

    if (args.solvers.count("cg")) {
        print(run_cg_no_precond(L, b, graph_name, args.tol, args.maxiter));
    }

    if (args.solvers.count("icc")) {
        print(run_cg_icc(L, b, graph_name, args.tol, args.maxiter));
    }

    if (args.solvers.count("ldlt")) {
        print(run_ldlt(L, b, graph_name));
    }

    return 0;
}

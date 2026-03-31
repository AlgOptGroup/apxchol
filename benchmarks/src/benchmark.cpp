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
//   --solver <list>    comma-separated: apxchol,cg,ldlt,rchol,cholmod,all
//   --tol <double>     PCG tolerance                        (default: 1e-8)
//   --maxiter <int>    PCG max iterations                   (default: 500)
//   --csv              output in CSV format
//   --seed <int>       RNG seed                             (default: 42)
//   --repeat <int>     repetitions per solver (median taken)  (default: 1)

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <set>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>

#include "graphs.h"
#include "solver.h"
#include "simple_solver.h"
#include "mmio.h"

#ifdef HAVE_APXCHOL_V1
#include "apxchol/solver/solve.h"
#include "apxchol/solver/factor_options.h"
#endif

#ifdef HAVE_RCHOL
#include "sparse.hpp"
#include "rchol.hpp"
#include "util.hpp"
#ifdef HAVE_MKL
#include <mkl_spblas.h>
#include <mkl.h>
#endif
#ifdef HAVE_METIS
#include "rchol_parallel.hpp"
#endif
#endif

#ifdef HAVE_CHOLMOD
#include <cholmod.h>
#endif

#ifdef HAVE_AMGCL
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/eigen.hpp>
#endif

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

// Helper: run a solver function N times, return the result with median total_time.
// The first run is a warmup when N > 1.
template<typename Fn>
static BenchResult median_run(Fn&& fn, int repeats) {
    std::vector<BenchResult> results;
    results.reserve(repeats);
    for (int i = 0; i < repeats; ++i)
        results.push_back(fn());
    std::sort(results.begin(), results.end(),
              [](const BenchResult& a, const BenchResult& b) {
                  return a.total_time < b.total_time;
              });
    return results[results.size() / 2];
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
    r.solver_name = "ApxChol+PCG [Kyng16]";
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

#ifdef HAVE_APXCHOL_V1
static BenchResult run_apxchol_v1(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    const std::string& combo_name,
    apxchol::is_strategy is,
    apxchol::elimination_strategy elim,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = combo_name;
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    auto res = apxchol::solve(L, b,
        {.tol = tol, .max_iter = maxiter,
         .storage = apxchol::graph_storage::forward_star,
         .factor_opts = {.seed = 42, .is_select = is, .elim = elim}});

    r.setup_time = res.timings.total("setup");
    r.solve_time = res.timings.total("solve");
    r.total_time = r.setup_time + r.solve_time;
    r.iterations = static_cast<int>(res.iterations);
    r.rel_residual = res.residual;
    r.fillin = 0.0;  // not tracked in v1 solve_result
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    return r;
}
#endif

static BenchResult run_cg_no_precond(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = "CG [Eigen]";
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
    r.solver_name = "CG+ICC [Eigen]";
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
    r.solver_name = "LDLT [Eigen]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    int n = r.n;

    // Regularize: L' = L + εI makes the system SPD.
    // ε is tiny relative to the diagonal, so L'x ≈ Lx for x ⊥ 1.
    double eps = 1e-12 * L.diagonal().array().abs().mean();
    Eigen::SparseMatrix<double> Lreg = L;
    for (int k = 0; k < n; ++k)
        Lreg.coeffRef(k, k) += eps;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;

    Timer t;
    t.start();
    ldlt.compute(Lreg);
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
    Eigen::VectorXd x = ldlt.solve(b);
    // Iterative refinement on the original L
    for (int refine = 0; refine < 3; ++refine) {
        Eigen::VectorXd res = b - L * x;
        res.array() -= res.mean();
        x += ldlt.solve(res);
    }
    x.array() -= x.mean();
    r.solve_time = t.elapsed();

    r.total_time = r.setup_time + r.solve_time;
    r.iterations = 1;

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
    int repeat = 1;
    int threads = 0;  // 0 = auto-detect
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
        else if (arg == "--repeat")  a.repeat = std::max(1, std::stoi(next()));
        else if (arg == "--threads") a.threads = std::stoi(next());
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
        a.solvers = {"apxchol", "cg", "ldlt"
#ifdef HAVE_APXCHOL_V1
            , "apxchol_v1"
#endif
#ifdef HAVE_RCHOL
            , "rchol"
#ifdef HAVE_MKL
            , "rchol_mkl"
            , "rchol_mkl1"
#endif
#ifdef HAVE_METIS
            , "rchol_par"
#endif
#endif
#ifdef HAVE_CHOLMOD
            , "cholmod"
#endif
        };
    return a;
}

// ──────────────────── RCHOL solver ────────────────────
#ifdef HAVE_RCHOL
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

struct RcholPreconditioner {
    Eigen::SparseMatrix<double> G_eigen;
    int n_ = 0;

    void init(const SparseCSR& G) {
        n_ = static_cast<int>(G.size());
        using T = Eigen::Triplet<double>;
        std::vector<T> trips;
        for (int i = 0; i < n_; ++i)
            for (size_t k = G.rowPtr[i]; k < G.rowPtr[i + 1]; ++k)
                trips.emplace_back(i, static_cast<int>(G.colIdx[k]), G.val[k]);
        G_eigen.resize(n_, n_);
        G_eigen.setFromTriplets(trips.begin(), trips.end());
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& b) const {
        Eigen::VectorXd x = G_eigen.transpose().triangularView<Eigen::Lower>().solve(b);
        return G_eigen.triangularView<Eigen::Upper>().solve(x);
    }
    int rows() const { return n_; }
    int cols() const { return n_; }
};

static BenchResult run_rchol(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = "RCHOL+PCG [Chen20]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());
    int N = r.n;

    // RCHOL needs strictly SDD — add small diagonal shift (only for factorization)
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm(L);
    double eps = 1e-6;
    for (int k = 0; k < Lrm.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(Lrm, k); it; ++it)
            if (it.row() == it.col()) it.valueRef() += eps;

    SparseCSR A = eigen_to_csr(Lrm);
    SparseCSR G;

    Timer t;
    t.start();
    {
        std::streambuf* old = std::cout.rdbuf();
        std::ostringstream devnull;
        std::cout.rdbuf(devnull.rdbuf());
        rchol(A, G);
        std::cout.rdbuf(old);
    }
    r.setup_time = t.elapsed();

    RcholPreconditioner P;
    P.init(G);
    r.fillin = 2.0 * static_cast<double>(G.nnz()) / static_cast<double>(A.nnz());

    // Manual PCG
    t.start();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd rv = b;
    Eigen::VectorXd z = P.solve(rv);
    Eigen::VectorXd p = z;
    double rz = rv.dot(z);
    double bnorm = b.norm();
    r.iterations = 0;

    for (int it = 0; it < maxiter; ++it) {
        Eigen::VectorXd Ap = L * p;
        double pAp = p.dot(Ap);
        if (std::abs(pAp) < 1e-30) break;
        double alpha = rz / pAp;
        x += alpha * p;
        rv -= alpha * Ap;
        r.iterations = it + 1;
        if (rv.norm() / bnorm < tol) break;
        z = P.solve(rv);
        double rz_new = rv.dot(z);
        p = z + (rz_new / rz) * p;
        rz = rz_new;
    }
    r.solve_time = t.elapsed();
    r.total_time = r.setup_time + r.solve_time;

    x.array() -= x.mean();
    Eigen::VectorXd res = b - L * x;
    res.array() -= res.mean();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    return r;
}

// ──────────────────── RCHOL + MKL PCG solver ────────────────────
#ifdef HAVE_MKL
// MKL CSR wrapper for SparseCSR (converts size_t -> MKL_INT)
struct MklSparse {
    sparse_matrix_t handle = nullptr;
    std::vector<MKL_INT> rowStart, rowEnd, cols;
    std::vector<double> vals;

    void init(const SparseCSR& S) {
        MKL_INT n = static_cast<MKL_INT>(S.size());
        rowStart.resize(n);
        rowEnd.resize(n);
        cols.resize(S.nnz());
        vals.resize(S.nnz());
        for (MKL_INT i = 0; i < n; ++i) {
            rowStart[i] = static_cast<MKL_INT>(S.rowPtr[i]);
            rowEnd[i]   = static_cast<MKL_INT>(S.rowPtr[i + 1]);
        }
        for (size_t k = 0; k < S.nnz(); ++k) {
            cols[k] = static_cast<MKL_INT>(S.colIdx[k]);
            vals[k] = S.val[k];
        }
        mkl_sparse_d_create_csr(&handle, SPARSE_INDEX_BASE_ZERO, n, n,
                                 rowStart.data(), rowEnd.data(),
                                 cols.data(), vals.data());
    }
    ~MklSparse() { if (handle) mkl_sparse_destroy(handle); }
};

static BenchResult run_rchol_mkl(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter, int mkl_threads = 0)
{
    BenchResult r;
    r.solver_name = mkl_threads == 1 ? "RCHOL+MKL1 [Chen20]" : "RCHOL+MKL [Chen20]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());
    int N = r.n;

    // Convert to row-major CSR
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm(L);

    // Original (unshifted) matrix for SpMV in PCG
    SparseCSR A_orig = eigen_to_csr(Lrm);

    // Shift diagonal for RCHOL factorization (needs strictly SDD)
    double eps = 1e-6;
    for (int k = 0; k < Lrm.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(Lrm, k); it; ++it)
            if (it.row() == it.col()) it.valueRef() += eps;

    SparseCSR A_shifted = eigen_to_csr(Lrm);
    SparseCSR G;

    Timer t;
    t.start();
    {
        std::streambuf* old = std::cout.rdbuf();
        std::ostringstream devnull;
        std::cout.rdbuf(devnull.rdbuf());
        rchol(A_shifted, G);
        std::cout.rdbuf(old);
    }
    r.setup_time = t.elapsed();
    r.fillin = 2.0 * static_cast<double>(G.nnz()) / static_cast<double>(A_shifted.nnz());

    // Create MKL sparse handles
    MklSparse mklL, mklG;
    mklL.init(A_orig);   // ORIGINAL matrix for SpMV (not shifted!)
    mklG.init(G);

    matrix_descr desL;
    desL.type = SPARSE_MATRIX_TYPE_GENERAL;

    matrix_descr desG;
    desG.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    desG.mode = SPARSE_FILL_MODE_UPPER;
    desG.diag = SPARSE_DIAG_NON_UNIT;

    // Set MKL thread count for solve phase
    int old_threads = mkl_get_max_threads();
    if (mkl_threads > 0) mkl_set_num_threads(mkl_threads);

    // MKL PCG
    t.start();
    std::vector<double> x(N, 0.0), rv(b.data(), b.data() + N);
    std::vector<double> z(N), p(N), Ap(N), prev_r(N), prev_z(N);
    double bnorm = cblas_dnrm2(N, rv.data(), 1);
    r.iterations = 0;

    for (int it = 0; it < maxiter; ++it) {
        // Apply preconditioner: z = G^{-1} G^{-T} r
        std::vector<double> tmp(N);
        mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1.0, mklG.handle, desG,
                           rv.data(), tmp.data());
        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mklG.handle, desG,
                           tmp.data(), z.data());

        if (it == 0) {
            cblas_dcopy(N, z.data(), 1, p.data(), 1);
        } else {
            double d1 = cblas_ddot(N, rv.data(), 1, z.data(), 1);
            double d2 = cblas_ddot(N, prev_r.data(), 1, prev_z.data(), 1);
            cblas_dscal(N, d1 / d2, p.data(), 1);
            cblas_daxpy(N, 1.0, z.data(), 1, p.data(), 1);
        }

        // Ap = L * p  (using ORIGINAL matrix, not shifted)
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mklL.handle, desL,
                         p.data(), 0.0, Ap.data());

        double d1 = cblas_ddot(N, p.data(), 1, rv.data(), 1);
        double d2 = cblas_ddot(N, p.data(), 1, Ap.data(), 1);
        double alpha = d1 / d2;

        cblas_daxpy(N, alpha, p.data(), 1, x.data(), 1);
        cblas_dcopy(N, rv.data(), 1, prev_r.data(), 1);
        cblas_dcopy(N, z.data(), 1, prev_z.data(), 1);
        cblas_daxpy(N, -alpha, Ap.data(), 1, rv.data(), 1);
        r.iterations = it + 1;

        if (cblas_dnrm2(N, rv.data(), 1) / bnorm < tol) break;
    }
    r.solve_time = t.elapsed();
    r.total_time = r.setup_time + r.solve_time;

    // Restore MKL thread count
    mkl_set_num_threads(old_threads);

    Eigen::VectorXd xe = Eigen::Map<Eigen::VectorXd>(x.data(), N);
    xe.array() -= xe.mean();
    Eigen::VectorXd res = b - L * xe;
    res.array() -= res.mean();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    return r;
}
#endif

// ──────────────────── Parallel RCHOL + PCG solver ────────────────────
#ifdef HAVE_METIS
struct PermutedRcholPreconditioner {
    Eigen::SparseMatrix<double> G_eigen;
    Eigen::VectorXi fwd_perm;  // original -> permuted
    Eigen::VectorXi inv_perm;  // permuted -> original
    int n_ = 0;

    void init(const SparseCSR& G, const std::vector<size_t>& perm) {
        n_ = static_cast<int>(G.size());
        using T = Eigen::Triplet<double>;
        std::vector<T> trips;
        for (int i = 0; i < n_; ++i)
            for (size_t k = G.rowPtr[i]; k < G.rowPtr[i + 1]; ++k)
                trips.emplace_back(i, static_cast<int>(G.colIdx[k]), G.val[k]);
        G_eigen.resize(n_, n_);
        G_eigen.setFromTriplets(trips.begin(), trips.end());

        // perm[new_i] = old_i, so inv_perm[old_i] = new_i
        fwd_perm.resize(n_);
        inv_perm.resize(n_);
        for (int i = 0; i < n_; ++i) {
            fwd_perm[i] = static_cast<int>(perm[i]);
            inv_perm[static_cast<int>(perm[i])] = i;
        }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& b) const {
        // Permute to reordered space: bp[new_i] = b[perm[new_i]]
        Eigen::VectorXd bp(n_);
        for (int i = 0; i < n_; ++i)
            bp[i] = b[fwd_perm[i]];
        // Triangular solves: G^{-T} then G^{-1}
        Eigen::VectorXd x = G_eigen.transpose().triangularView<Eigen::Lower>().solve(bp);
        x = G_eigen.triangularView<Eigen::Upper>().solve(x);
        // Unpermute: result[perm[new_i]] = x[new_i]
        Eigen::VectorXd result(n_);
        for (int i = 0; i < n_; ++i)
            result[fwd_perm[i]] = x[i];
        return result;
    }
    int rows() const { return n_; }
    int cols() const { return n_; }
};

static BenchResult run_rchol_parallel(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter, int forced_threads = 0)
{
    BenchResult r;
    r.solver_name = "pRCHOL+PCG [Chen20;par]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());
    int N = r.n;

    Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm(L);
    double eps = 1e-6;
    for (int k = 0; k < Lrm.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(Lrm, k); it; ++it)
            if (it.row() == it.col()) it.valueRef() += eps;

    SparseCSR A = eigen_to_csr(Lrm);
    SparseCSR G;
    std::vector<size_t> perm;

    // Skip pRCHOL on very dense graphs — METIS segfaults on dense random graphs
    double density = static_cast<double>(r.nnz) / (static_cast<double>(N) * N);
    if (density > 0.01) {
        r.solver_name = "pRCHOL+PCG [Chen20;par] FAIL: too dense for METIS";
        r.setup_time = 0;
        r.solve_time = 0;
        r.total_time = 0;
        r.iterations = 0;
        r.rel_residual = -1;
        return r;
    }

    Timer t;
    t.start();
    {
        std::streambuf* old = std::cout.rdbuf();
        std::ostringstream devnull;
        std::cout.rdbuf(devnull.rdbuf());
        int nthreads = forced_threads > 0 ? forced_threads : std::thread::hardware_concurrency();
        if (nthreads < 1) nthreads = 4;
        // Round down to nearest power of 2 (RCHOL requires it)
        nthreads = 1 << static_cast<int>(std::log2(nthreads));
        // METIS crashes on tiny/very dense graphs when nthreads is too large
        while (nthreads > 1 && nthreads > N / 16)
            nthreads /= 2;
        if (nthreads < 1) nthreads = 1;
        r.solver_name = "pRCHOL+PCG [Chen20;par] t=" + std::to_string(nthreads);
        rchol(A, G, perm, nthreads);
        std::cout.rdbuf(old);
    }
    r.setup_time = t.elapsed();

    PermutedRcholPreconditioner P;
    P.init(G, perm);
    r.fillin = 2.0 * static_cast<double>(G.nnz()) / static_cast<double>(A.nnz());

    // PCG solve
    t.start();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd rv = b;
    Eigen::VectorXd z = P.solve(rv);
    Eigen::VectorXd p = z;
    double rz = rv.dot(z);
    double bnorm = b.norm();
    r.iterations = 0;

    for (int it = 0; it < maxiter; ++it) {
        Eigen::VectorXd Ap = L * p;
        double pAp = p.dot(Ap);
        if (std::abs(pAp) < 1e-30) break;
        double alpha = rz / pAp;
        x += alpha * p;
        rv -= alpha * Ap;
        r.iterations = it + 1;
        if (rv.norm() / bnorm < tol) break;
        z = P.solve(rv);
        double rz_new = rv.dot(z);
        p = z + (rz_new / rz) * p;
        rz = rz_new;
    }
    r.solve_time = t.elapsed();
    r.total_time = r.setup_time + r.solve_time;

    x.array() -= x.mean();
    Eigen::VectorXd res = b - L * x;
    res.array() -= res.mean();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    return r;
}
#endif

#endif

// ──────────────────── CHOLMOD solver ────────────────────
#ifdef HAVE_CHOLMOD
static BenchResult run_cholmod(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name)
{
    BenchResult r;
    r.solver_name = "CHOLMOD [SuiteSparse]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());
    int n = r.n;

    // Regularize: L' = L + εI makes the system SPD
    double eps = 1e-12 * L.diagonal().array().abs().mean();
    Eigen::SparseMatrix<double> Lreg = L;
    for (int k = 0; k < n; ++k)
        Lreg.coeffRef(k, k) += eps;
    Lreg.makeCompressed();

    cholmod_common c;
    cholmod_start(&c);

    cholmod_sparse A_chol;
    A_chol.nrow = n;
    A_chol.ncol = n;
    A_chol.nzmax = Lreg.nonZeros();
    A_chol.p = const_cast<int*>(Lreg.outerIndexPtr());
    A_chol.i = const_cast<int*>(Lreg.innerIndexPtr());
    A_chol.x = const_cast<double*>(Lreg.valuePtr());
    A_chol.z = nullptr;
    A_chol.stype = 1;  // upper triangular (symmetric)
    A_chol.itype = CHOLMOD_INT;
    A_chol.xtype = CHOLMOD_REAL;
    A_chol.dtype = CHOLMOD_DOUBLE;
    A_chol.sorted = 1;
    A_chol.packed = 1;

    Timer t;
    t.start();
    cholmod_factor* factor = cholmod_analyze(&A_chol, &c);
    cholmod_factorize(&A_chol, factor, &c);
    r.setup_time = t.elapsed();

    if (c.status != CHOLMOD_OK) {
        r.solver_name += "(FAIL)";
        r.solve_time = 0;
        r.total_time = r.setup_time;
        r.rel_residual = std::numeric_limits<double>::quiet_NaN();
        cholmod_free_factor(&factor, &c);
        cholmod_finish(&c);
        return r;
    }

    Eigen::VectorXd bvec = b;  // mutable copy
    cholmod_dense b_chol;
    b_chol.nrow = n;
    b_chol.ncol = 1;
    b_chol.nzmax = n;
    b_chol.d = n;
    b_chol.x = bvec.data();
    b_chol.z = nullptr;
    b_chol.xtype = CHOLMOD_REAL;
    b_chol.dtype = CHOLMOD_DOUBLE;

    t.start();
    cholmod_dense* x_chol = cholmod_solve(CHOLMOD_A, factor, &b_chol, &c);
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(
        static_cast<double*>(x_chol->x), n);
    cholmod_free_dense(&x_chol, &c);

    // Iterative refinement on original L
    for (int refine = 0; refine < 3; ++refine) {
        Eigen::VectorXd res = b - L * x;
        res.array() -= res.mean();

        cholmod_dense r_chol;
        r_chol.nrow = n; r_chol.ncol = 1; r_chol.nzmax = n; r_chol.d = n;
        r_chol.x = res.data(); r_chol.z = nullptr;
        r_chol.xtype = CHOLMOD_REAL; r_chol.dtype = CHOLMOD_DOUBLE;

        cholmod_dense* dx_chol = cholmod_solve(CHOLMOD_A, factor, &r_chol, &c);
        x += Eigen::Map<Eigen::VectorXd>(
            static_cast<double*>(dx_chol->x), n);
        cholmod_free_dense(&dx_chol, &c);
    }
    x.array() -= x.mean();
    r.solve_time = t.elapsed();
    r.total_time = r.setup_time + r.solve_time;
    r.iterations = 1;

    Eigen::VectorXd res = b - L * x;
    res.array() -= res.mean();
    double bnorm = b.norm();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);
    r.fillin = 0;
    r.us_per_nnz = r.total_time / r.nnz * 1e6;

    cholmod_free_factor(&factor, &c);
    cholmod_finish(&c);
    return r;
}
#endif

// ──────────────────── AMGCL solver ────────────────────
#ifdef HAVE_AMGCL
static BenchResult run_amgcl(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = "AMG+CG [AMGCL]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());
    int n = r.n;
    int m = n - 1;

    // Pin one vertex
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lsub = L.topLeftCorner(m, m);
    Eigen::VectorXd bsub = b.head(m);
    bsub.array() -= bsub.mean();

    // Add small diagonal shift for strict positive definiteness
    for (int k = 0; k < Lsub.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(Lsub, k); it; ++it)
            if (it.row() == it.col()) it.valueRef() += 1e-6;

    using Backend = amgcl::backend::builtin<double>;
    using Solver = amgcl::make_solver<
        amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
        amgcl::solver::cg<Backend>
    >;

    Solver::params prm;
    prm.solver.tol = tol;
    prm.solver.maxiter = maxiter;

    Timer t;
    t.start();
    Solver solve(Lsub, prm);
    r.setup_time = t.elapsed();

    std::vector<double> rhs(bsub.data(), bsub.data() + m);
    std::vector<double> sol(m, 0.0);

    t.start();
    auto [iters, error] = solve(rhs, sol);
    r.solve_time = t.elapsed();
    r.total_time = r.setup_time + r.solve_time;
    r.iterations = static_cast<int>(iters);

    Eigen::VectorXd x(n);
    x.head(m) = Eigen::Map<Eigen::VectorXd>(sol.data(), m);
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
#endif

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
        char pbuf[32];
        std::snprintf(pbuf, sizeof(pbuf), "%.4g", args.er_p);
        graph_name = "erdos_" + std::to_string(args.n) + "_p" + pbuf;
    } else if (args.graph == "mtx") {
        auto res = load_mtx_as_adjacency(args.mtx_path);
        adj = std::move(res.adj);
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

    if (args.csv) print_csv_header();
    else print_header_pretty();

    auto print = [&](const BenchResult& r) {
        if (args.csv) print_result_csv(r);
        else print_result_pretty(r);
    };

    const int R = args.repeat;

    if (args.solvers.count("apxchol")) {
        print(median_run([&]() {
            std::streambuf* old = std::cout.rdbuf();
            std::ostringstream devnull;
            std::cout.rdbuf(devnull.rdbuf());
            auto r = run_apxchol(adj, L, b, graph_name, args.tol, args.maxiter);
            std::cout.rdbuf(old);
            return r;
        }, R));
    }

#ifdef HAVE_APXCHOL_V1
    if (args.solvers.count("apxchol_v1")) {
        // Best combos from empirical testing on SuiteSparse matrices.
        struct V1Combo {
            const char* name;
            apxchol::is_strategy is;
            apxchol::elimination_strategy elim;
        };
        static const V1Combo v1_combos[] = {
            {"ApxChol-v1 bg+tree",   apxchol::is_strategy::block_greedy, apxchol::elimination_strategy::tree},
            {"ApxChol-v1 bg+star",   apxchol::is_strategy::block_greedy, apxchol::elimination_strategy::star},
            {"ApxChol-v1 luby+tree", apxchol::is_strategy::luby,         apxchol::elimination_strategy::tree},
            {"ApxChol-v1 root+tree", apxchol::is_strategy::rootset,      apxchol::elimination_strategy::tree},
        };
        for (const auto& combo : v1_combos) {
            print(median_run([&]() {
                return run_apxchol_v1(L, b, graph_name, combo.name, combo.is, combo.elim,
                                      args.tol, args.maxiter);
            }, R));
        }
    }
#endif

    if (args.solvers.count("cg"))
        print(median_run([&]() { return run_cg_no_precond(L, b, graph_name, args.tol, args.maxiter); }, R));

    if (args.solvers.count("icc"))
        print(median_run([&]() { return run_cg_icc(L, b, graph_name, args.tol, args.maxiter); }, R));

    if (args.solvers.count("ldlt"))
        print(median_run([&]() { return run_ldlt(L, b, graph_name); }, R));

#ifdef HAVE_RCHOL
    if (args.solvers.count("rchol"))
        print(median_run([&]() { return run_rchol(L, b, graph_name, args.tol, args.maxiter); }, R));
#ifdef HAVE_MKL
    if (args.solvers.count("rchol_mkl"))
        print(median_run([&]() { return run_rchol_mkl(L, b, graph_name, args.tol, args.maxiter); }, R));
    if (args.solvers.count("rchol_mkl1"))
        print(median_run([&]() { return run_rchol_mkl(L, b, graph_name, args.tol, args.maxiter, 1); }, R));
#endif
#ifdef HAVE_METIS
    if (args.solvers.count("rchol_par")) {
        try {
            print(median_run([&]() { return run_rchol_parallel(L, b, graph_name, args.tol, args.maxiter, args.threads); }, R));
        } catch (const std::exception& e) {
            std::cerr << "[skip] pRCHOL failed: " << e.what() << "\n";
        }
    }
#endif
#endif

#ifdef HAVE_CHOLMOD
    if (args.solvers.count("cholmod"))
        print(median_run([&]() { return run_cholmod(L, b, graph_name); }, R));
#endif

#ifdef HAVE_AMGCL
    if (args.solvers.count("amgcl"))
        print(median_run([&]() { return run_amgcl(L, b, graph_name, args.tol, args.maxiter); }, R));
#endif

    return 0;
}

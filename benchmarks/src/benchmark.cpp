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
#include <fstream>
#include <cmath>
#include <cstdio>
#include <set>
#include <thread>

#ifdef APXCHOL_USE_CUDA
#include <cuda_runtime.h>   // cudaMemGetInfo for read_vram_mb() (declared early; the
                            // cuBLAS/cuSPARSE GPU solvers include their headers locally)
#endif

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>

#include "graphs.h"
#include "solver.h"
#include "simple_solver.h"
#include "mmio.h"

#ifdef HAVE_APXCHOL_V1
#include "apxchol/solver/solve.h"
#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/factorization.h"
#ifdef _OPENMP
#include <omp.h>
#endif
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
#include "bench_result.h"

// AMGCL CUDA adapter (compiled separately as amgcl_cuda.cu with nvcc).
#if defined(HAVE_AMGCL) && defined(APXCHOL_USE_CUDA)
extern "C" void run_amgcl_cuda_impl(
    BenchResult* r,
    int m,
    const std::ptrdiff_t* row_ptr,
    const std::ptrdiff_t* col_idx,
    const double*         vals,
    const double*         bsub,
    double                tol,
    int                   maxiter,
    int                   full_n,
    const double*         full_b,
    const std::ptrdiff_t* full_row_ptr,
    const std::ptrdiff_t* full_col_idx,
    const double*         full_vals,
    int                   is_laplacian,
    int                   relax_coarse);
#endif

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

// Current host resident-set size (MB) from /proc/self/status. Read at the end of each
// solver's solve phase (factor+operator+vectors still alive, setup pool already freed)
// to record the SOLVE-held memory; the process peak (setup-dominated) comes from
// /usr/bin/time externally, so peak - solve_held = the setup transient.
static double read_vmrss_mb() {
    std::ifstream f("/proc/self/status"); std::string ln;
    while (std::getline(f, ln))
        if (ln.rfind("VmRSS:", 0) == 0) {
            try { return std::stol(ln.substr(6)) / 1024.0; } catch (...) { return -1.0; }
        }
    return -1.0;
}

// Device memory in use (MB), the GPU analog of read_vmrss_mb: total - free from
// cudaMemGetInfo. Includes the CUDA context (~300-600MB) + caching-allocator footprint,
// so it overcounts the minimal working set the same way VmRSS overcounts host -- read at
// solve end to record the SOLVE-held VRAM (operator + factor + SpSV bufs + PCG vectors).
// Returns -1 when CUDA isn't compiled in or the query fails (CPU solvers stay unmeasured).
static double read_vram_mb() {
#ifdef APXCHOL_USE_CUDA
    size_t mf = 0, mt = 0;
    if (cudaMemGetInfo(&mf, &mt) == cudaSuccess && mt >= mf)
        return (mt - mf) / (1024.0 * 1024.0);
#endif
    return -1.0;
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
    std::cout << "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz,solve_rss_mb,solve_vram_mb\n";
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
              << r.us_per_nnz << ","
              << std::setprecision(1) << r.solve_rss_mb << ","
              << std::setprecision(1) << r.solve_vram_mb << "\n";
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
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)
    r.iterations = static_cast<int>(cg.iterations()) + 1;  // unify w/ apxchol convention

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
    std::string is,
    apxchol::graph_storage storage,
    double tol, int maxiter,
    bool dump_profile = false,
    size_t exact_clique_max_degree = 0,
    double degree_multiplier_override = 0.0)
{
    BenchResult r;
    if (std::getenv("APXCHOL_PROFILE")) dump_profile = true;  // checkpoint breakdown
    r.solver_name = combo_name;
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    apxchol::factor_options fopts{.seed = 42, .is_select = is};
    if (exact_clique_max_degree > 0)
        fopts.exact_clique_max_degree = exact_clique_max_degree;
    if (degree_multiplier_override > 0.0)
        fopts.partition.degree_multiplier = degree_multiplier_override;
    // Experiment knobs to mimic AC's adaptive min-degree elimination: set
    // min_is_fraction=1 so the IS bails immediately and the whole factor is
    // peeled sequentially in residual_peel order (min_degree == AC's :deg).
    if (const char* e = std::getenv("APXCHOL_MIN_IS_FRACTION"))
        fopts.min_is_fraction = std::atof(e);
    if (const char* e = std::getenv("APXCHOL_RESIDUAL_PEEL")) {
        const std::string s = e;
        fopts.residual_peel = (s == "min_degree") ? apxchol::residual_peel_strategy::min_degree
                            : (s == "bk_serial")  ? apxchol::residual_peel_strategy::bk_serial
                            :                        apxchol::residual_peel_strategy::natural;
    }
    // Exact Schur clique up to a degree cap (zero sampling variance).
    if (const char* e = std::getenv("APXCHOL_EXACT_CLIQUE"))
        fopts.exact_clique_max_degree = static_cast<size_t>(std::atol(e));
    if (const char* e = std::getenv("APXCHOL_DEGREE_MULT"))
        fopts.partition.degree_multiplier = std::atof(e);
    if (const char* e = std::getenv("APXCHOL_DEGREE_QUANTILE"))
        fopts.partition.degree_quantile = std::atof(e);
    if (const char* e = std::getenv("APXCHOL_DEGREE_TIEBREAK"))
        fopts.partition.degree_tiebreak = std::atoi(e) != 0;

    const auto t_wall_start = std::chrono::high_resolution_clock::now();
    auto res = apxchol::solve(L, b,
        {.tol = tol, .max_iter = maxiter,
         .storage = storage,
         .factor_opts = fopts});
    const auto t_wall_end = std::chrono::high_resolution_clock::now();
    const double wall_total =
        std::chrono::duration<double>(t_wall_end - t_wall_start).count();

    // Honest accounting:
    //   setup_time   — preconditioner build (from checkpoint, accurate)
    //   solve_time   — wall_total minus setup, i.e. the PCG loop INCLUDING
    //                  the outer SpMV / dot-product / norm cost.
    //   The checkpoint's "solve" entry only tracks `precond.solve()` (the
    //   triangular forward/back), which was UNDER-REPORTING solve time by
    //   the full PCG SpMV cost — making our v1 numbers look ~2× better
    //   than reality vs Eigen-CG-based competitors. Fixed 2026-05-24.
    r.setup_time = res.timings.total("setup");
    r.solve_time = wall_total - r.setup_time;
    if (r.solve_time < 0.0) r.solve_time = 0.0;
    r.total_time = wall_total;
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)
    r.solve_vram_mb = res.solve_vram_mb; // sampled INSIDE apxchol::solve (the GPU-resident
                                         // PCG frees all device state before returning);
                                         // -1 on CPU builds / host-PCG paths.
    r.iterations = static_cast<int>(res.iterations);
    r.rel_residual = res.residual;
    r.fillin = 0.0;  // not tracked in v1 solve_result
    if (std::getenv("APXCHOL_REPORT_FILL")) {
        // Measurement-only: re-factorize to read factor nnz. AC's fill ratio is
        // 2*offdiag(L)/nnz(adj); match it. nnz(adj) = input L.nonZeros() - n.
        auto Fmeas = apxchol::factorize(L, storage, fopts);
        const long long Lnnz   = Fmeas.L.nonZeros();
        const long long n_fac  = Fmeas.L.rows();
        const long long offdiag = Lnnz - n_fac;          // strict lower entries
        const long long adj_nnz = (long long)L.nonZeros() - L.rows();
        std::string stored;
        // What the SpTRSV actually holds after its setup (L11 = the factor minus
        // the Laplacian's grounded last row/col, minus APXCHOL_FACTOR_DROP's
        // compaction): the CSR and the CSC each store stored_nnz entries. Same
        // shared drop on both backends (factor_drop.h); the CUDA backend
        // additionally reports the device bytes of its factor arrays and its
        // runtime storage mode. (Formatted before the FILL line is printed:
        // setup itself prints a line under APXCHOL_VERBOSE.)
        {
#if defined(APXCHOL_USE_CUDA)
            apxchol::cuda_sptrsv trsv;
#else
            apxchol::omp_sptrsv trsv;
#endif
            trsv.setup(Fmeas.L, static_cast<apxchol::node_index>(Fmeas.sddm ? n_fac : n_fac - 1));
            const auto& st = trsv.drop_stats();
            std::ostringstream os;
            os << " stored_nnz=" << st.nnz_stored
               << " (L11_nnz=" << st.nnz_factor << " dropped=" << st.dropped
               << " drop_rel=" << st.rel << " compensate=" << (st.compensate ? 1 : 0) << ")";
#if defined(APXCHOL_USE_CUDA)
            os << " gpu=" << (trsv.levelset() ? "levelset" : "cusparse")
               << "/" << (trsv.fp16() ? "fp16" : apxchol::cuda_sptrsv::value_name)
               << " factor_dev_MB=" << std::fixed << std::setprecision(1) << trsv.factor_device_bytes() / 1e6
               << " dev_delta_MB=" << trsv.device_bytes_delta() / 1e6
               << " cusparse_buf_MB=" << trsv.cusparse_buffer_bytes() / 1e6;
#endif
            stored = os.str();
        }
        std::cerr << "FILL " << combo_name
                  << "  Lnnz=" << Lnnz << " offdiag=" << offdiag
                  << " adj_nnz=" << adj_nnz
                  << " ratio(2*offdiag/adj)=" << (2.0 * offdiag / adj_nnz)
                  << stored << "\n" << std::flush;
    }
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    if (dump_profile) {
        std::cerr << "\n--- profile: " << combo_name << " ---\n";
        res.timings.report(std::cerr);
        std::cerr << std::flush;
    }
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
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)
    // Eigen's CG returns the loop-counter i (incremented after the convergence
    // check), so a 1-iter convergence reports 0. apxchol_v1 / cuda_pcg / AMGCL /
    // Hypre all report "iters completed" (1 for that case). Add 1 to unify.
    r.iterations = static_cast<int>(cg.iterations()) + 1;

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
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)
    r.iterations = static_cast<int>(cg.iterations()) + 1;  // unify w/ apxchol convention

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
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)
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
    int ny = 0;  // second dimension for rectangular 2D grids; 0 = square (use n)
    double kappa = 1000.0;
    int tile = 4;
    double er_p = 0.01;
    std::string mtx_path;
    std::string dump_mtx;          // if set: write the built Laplacian to this path and exit
    bool pin_dump = false;         // with --dump-mtx: write the per-component Dirichlet-PINNED
                                   // matrix (dirichlet_pin) instead of the raw L -- the FAIR
                                   // grounded operator for external solvers (ParAC/CMG), so
                                   // they solve the same singular system everyone else does
    bool giant_dump = false;       // with --dump-mtx: write a connected component's PURE Laplacian
                                   // (relabeled 0..cn-1) -- the comp_rank-th LARGEST (0 = giant).
                                   // = the full pure L for a connected matrix. Lets ParAC physics
                                   // ground a CONNECTED component by its single-node trim, and the
                                   // runner loop the large components (the per-component "split").
    int comp_rank = 0;             // with --giant-dump: which component (by descending size) to dump
                                   // without the --reg-rel eps*I perturbation.
    double reg_rel = 0.0;          // >0: regularize L += reg_rel*mean|diag|*I (unify singular grids to SDDM)
    std::set<std::string> solvers;
    double tol = 1e-8;
    int maxiter = 500;
    bool csv = false;
    unsigned seed = 42;
    int repeat = 1;
    int threads = 0;  // 0 = use current OMP setting
    // Optional subset filter for --solver apxchol_v1.
    // If non-empty, only configs whose label is in this set are run.
    // Match accepts either the bare combo name ("bg+tree") to run all
    // storage variants, or fully-qualified "bg+tree[vec]" / "bg+tree[fwd_star]".
    std::set<std::string> v1_configs;
    bool v1_profile = false;
    // De-singularization is two orthogonal axes (see resolve_desing). decompose:
    // auto|whole|split (auto = split iff disconnected). ground: auto|pin|coarse|native
    // (auto = pin for AMG solvers / native for apxchol). The grounding WORK is timed
    // into setup; the is_laplacian detection SpMV is a benchmark-only classification
    // and stays OUT of setup (deployment knows a priori).
    std::string decompose = "auto";
    std::string ground    = "auto";
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
        else if (arg == "--ny")  a.ny = std::stoi(next());
        else if (arg == "--kappa") a.kappa = std::stod(next());
        else if (arg == "--tile")  a.tile = std::stoi(next());
        else if (arg == "--er-p")  a.er_p = std::stod(next());
        else if (arg == "--mtx")   { a.mtx_path = next(); a.graph = "mtx"; }
        else if (arg == "--dump-mtx") a.dump_mtx = next();
        else if (arg == "--pin-dump") a.pin_dump = true;
        else if (arg == "--giant-dump") a.giant_dump = true;
        else if (arg == "--comp-rank") a.comp_rank = std::stoi(next());
        else if (arg == "--reg-rel") a.reg_rel = std::stod(next());
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
        else if (arg == "--v1-configs") {
            std::string s = next();
            std::istringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) a.v1_configs.insert(tok);
        }
        else if (arg == "--v1-profile") a.v1_profile = true;
        else if (arg == "--decompose") {
            a.decompose = next();
            static const std::set<std::string> ok{"auto","whole","split"};
            if (!ok.count(a.decompose)) {
                std::cerr << "Unknown --decompose: " << a.decompose
                          << " (expected auto|whole|split)\n"; std::exit(1);
            }
        }
        else if (arg == "--ground") {
            a.ground = next();
            static const std::set<std::string> ok{"auto","pin","coarse","native"};
            if (!ok.count(a.ground)) {
                std::cerr << "Unknown --ground: " << a.ground
                          << " (expected auto|pin|coarse|native)\n"; std::exit(1);
            }
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

    // FAIRNESS: AMD fill-reducing reorder before factorization. RCHOL's README
    // recommends reordering (AMD/METIS); the prior code factorized in NATURAL
    // order, inflating fill on irregular graphs. AMD ordering + the P·A·Pᵀ
    // permute + csr are all counted in setup (the cost a fair RCHOL user pays).
    // RCHOL_NO_AMD=1 reproduces the natural-order path for an A/B.
    const bool rchol_amd = std::getenv("RCHOL_NO_AMD") == nullptr;

    Timer t;
    t.start();   // setup: AMD + permute + csr + rchol factorization (all timed)

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm(N);
    if (rchol_amd) {
        Eigen::SparseMatrix<double> Lcol(Lrm);          // column-major for AMD
        Eigen::AMDOrdering<int> amd;
        amd(Lcol, perm);                                 // fill-reducing permutation
    } else {
        perm.setIdentity(N);
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lfac =
        rchol_amd ? Eigen::SparseMatrix<double, Eigen::RowMajor>(perm * Lrm * perm.transpose())
                  : Lrm;
    SparseCSR A = eigen_to_csr(Lfac);
    SparseCSR G;
    {
        std::streambuf* old = std::cout.rdbuf();
        std::ostringstream devnull;
        std::cout.rdbuf(devnull.rdbuf());
        rchol(A, G);
        std::cout.rdbuf(old);
    }
    // PCG runs entirely in permuted space: Lp = P·L·Pᵀ (unshifted, for SpMV),
    // bp = P·b; the RCHOL preconditioner G is the factor of P·(L+εI)·Pᵀ. These
    // permutes are built ONCE and reused every iter -> counted in setup (so the
    // AMD path's full reorder cost, incl. the SpMV-operator permute, is timed).
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lp =
        rchol_amd ? Eigen::SparseMatrix<double, Eigen::RowMajor>(perm * L * perm.transpose())
                  : Eigen::SparseMatrix<double, Eigen::RowMajor>(L);
    Eigen::VectorXd bp = rchol_amd ? Eigen::VectorXd(perm * b) : b;
    r.setup_time = t.elapsed();

    RcholPreconditioner P;
    P.init(G);
    r.fillin = 2.0 * static_cast<double>(G.nnz()) / static_cast<double>(A.nnz());
    double bnorm = b.norm();   // norm is permutation-invariant

    t.start();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd rv = bp;
    Eigen::VectorXd z = P.solve(rv);
    Eigen::VectorXd p = z;
    double rz = rv.dot(z);
    r.iterations = 0;

    for (int it = 0; it < maxiter; ++it) {
        Eigen::VectorXd Ap = Lp * p;
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
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)

    // unpermute the solution back to the original ordering: x = Pᵀ·y
    Eigen::VectorXd x_orig = rchol_amd ? Eigen::VectorXd(perm.transpose() * x) : x;
    x_orig.array() -= x_orig.mean();
    Eigen::VectorXd res = b - L * x_orig;
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

    // MKL inspector-executor: precompute the triangular-solve / SpMV execution
    // plan ONCE. Required by the MKL sparse API for efficient *repeated* calls —
    // without it, mkl_sparse_d_trsv has no level-schedule and re-analyzes on every
    // PCG iteration, which thrashes catastrophically under threading.
    {
        Timer topt; topt.start();
        mkl_sparse_set_mv_hint(mklL.handle, SPARSE_OPERATION_NON_TRANSPOSE, desL,
                               static_cast<MKL_INT>(maxiter));
        mkl_sparse_set_sv_hint(mklG.handle, SPARSE_OPERATION_TRANSPOSE, desG,
                               static_cast<MKL_INT>(maxiter));
        mkl_sparse_set_sv_hint(mklG.handle, SPARSE_OPERATION_NON_TRANSPOSE, desG,
                               static_cast<MKL_INT>(maxiter));
        mkl_sparse_optimize(mklL.handle);
        mkl_sparse_optimize(mklG.handle);
        r.setup_time += topt.elapsed();
    }

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
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)

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
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)

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
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)
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

// ────────── de-singularization helpers (shared: AMGCL, Hypre, apxchol) ──────────
// De-singularization of a singular Laplacian is TWO orthogonal axes (the grounding
// WORK is timed into setup; the is_laplacian detection is not -- that's a
// benchmark-only classification):
//   decompose: whole  -- hand the (whole) operator to the solver
//              split   -- decompose into connected components, solve each, recombine
//   ground:    pin     -- symmetric Dirichlet pin (1 node/component) -> SPD -> the
//                         solver runs with its DEFAULT config (no coarsener tweak)
//              coarse  -- (AMGCL only) keep the singular operator, relax the coarsest
//                         grid (direct_coarse=false, NON-default) to avoid the
//                         skyline_lu "Zero sum" crash on a singular coarse operator
//              native  -- (apxchol only) intrinsic rank-aware mean-centering
enum class decompose_mode { whole, split };
enum class ground_mode    { pin, coarse, native };

// Validity matrix + `auto` defaults in ONE place (invalid cells -> n/a, never silent).
// solver key: "apxchol" | "amgcl" | "boomeramg".
struct desing_plan { bool supported; decompose_mode decompose; ground_mode ground; };
static desing_plan resolve_desing(const std::string& solver, const std::string& dec_req,
                                  const std::string& gnd_req, bool disconnected) {
    // ground: auto = the config-clean default (pin for the AMG solvers -> default
    // config; native for apxchol).
    ground_mode gnd = (gnd_req=="pin")    ? ground_mode::pin
                    : (gnd_req=="coarse") ? ground_mode::coarse
                    : (gnd_req=="native") ? ground_mode::native
                    : (solver=="apxchol") ? ground_mode::native : ground_mode::pin;  // auto
    // decompose: auto = whole, EXCEPT split when disconnected AND either the ground is
    // `coarse` (whole+coarse is invalid -- single near-null vector can't span a k-dim
    // null space) or the solver is BoomerAMG (measured: split avoids dragging the dust
    // components through one AMG hierarchy, ~3-5% faster). apxchol and AMGCL+pin handle
    // the whole disconnected operator faster than paying per-component build overhead.
    bool auto_split = disconnected && (gnd==ground_mode::coarse || solver=="boomeramg");
    decompose_mode dec = (dec_req=="whole") ? decompose_mode::whole
                       : (dec_req=="split") ? decompose_mode::split
                       : (auto_split ? decompose_mode::split : decompose_mode::whole);
    bool ok = true;
    if      (solver=="apxchol")                       ok = (gnd==ground_mode::native);
    else if (solver=="boomeramg")                     ok = (gnd==ground_mode::pin);
    else if (solver=="amgcl")                         ok = (gnd==ground_mode::pin || gnd==ground_mode::coarse);
    // The invalid cell: whole + coarse on a DISCONNECTED graph -- the unpinned singular
    // operator has a k-dim constant null space the single near-null vector can't span
    // (AMGCL CG breaks down / NaN). split makes every piece connected (k=1), so it's ok.
    if (gnd==ground_mode::coarse && dec==decompose_mode::whole && disconnected) ok = false;
    return {ok, dec, gnd};
}

// Symmetric Dirichlet pin grounding the FULL null space: one node per CONNECTED
// COMPONENT. For each grounded node p: zero row p's off-diagonals, set L(p,p)=1,
// zero column p. Keeps L SYMMETRIC, full n x n, and SPD without perturbing the
// operator we SCORE against -- and unlike dropping a row/col + mean-centering the
// RHS it controls the residual at EVERY row, so the true residual (b - L_orig*x)
// reaches 1e-8 instead of flooring at ~1e-6. A connected Laplacian has one null
// vector (the all-ones), so this pins a single node (the last one); a graph with
// isolated nodes / multiple components (e.g. as-Skitter) has one null vector PER
// component and each must be grounded, else the residual block stays singular
// (zero diagonal on isolated nodes -> AMGCL NaN / BoomerAMG floors at ~1e-5). The
// grounded node ids are returned in `pinned` (their RHS entry is set to 0). It is
// exactly the multi-component generalization of a single stencil-diagonal pin.
static Eigen::SparseMatrix<double> dirichlet_pin(const Eigen::SparseMatrix<double>& L,
                                                 std::vector<int>& pinned) {
    pinned.clear();
    const int n = static_cast<int>(L.rows());
    // One PROVABLY-SAFE pin per connected component, found by a single STACK-based DFS
    // (iterative -- the giant component has >1.6M nodes, so recursion would overflow).
    // The FIRST node finished in each component (popped with no unvisited neighbor) is a
    // DFS-tree LEAF, which is never an articulation point -- so zeroing its row+column
    // can never split its component into an ungrounded piece. Grounding a cut-vertex
    // would: it disconnects the component into blocks that no longer contain a pin,
    // leaving Lsub singular there and flooring the residual. This replaces the old
    // min-degree heuristic, which only guaranteed safety for degree-1 leaves (a min-
    // degree vertex of degree >=2 can be a cut-vertex, e.g. a barbell's degree-2 bridge).
    const auto* outer = L.outerIndexPtr();
    const auto* inner = L.innerIndexPtr();
    std::vector<char> is_pin(n, 0);
    std::vector<char> seen(n, 0);
    std::vector<int>  stk; stk.reserve(256);
    std::vector<int>  cur(n);   // per-node DFS edge cursor (into the CSC column)
    for (int s = 0; s < n; ++s) {
        if (seen[s]) continue;
        seen[s] = 1; cur[s] = outer[s]; stk.push_back(s);
        int leaf = -1;
        while (!stk.empty()) {
            const int u = stk.back();
            int w = -1;
            while (cur[u] < outer[u + 1]) {
                const int v = inner[cur[u]++];
                if (v != u && !seen[v]) { w = v; break; }
            }
            if (w != -1) { seen[w] = 1; cur[w] = outer[w]; stk.push_back(w); }
            else { stk.pop_back(); if (leaf == -1) leaf = u; }  // first finished = DFS leaf
        }
        is_pin[leaf] = 1; pinned.push_back(leaf);
    }
    // Build P's CSC directly in column order. L is column-major and already sorted, so
    // dropping pinned rows/cols preserves order -- insert in lexicographic order (cheap,
    // no per-column search) into a reserve()'d matrix, skipping setFromTriplets' O(nnz
    // log nnz) sort. (Tried a parallel manual-CSC build: NEGATIVE -- the op is
    // bandwidth-bound, so parallelism doesn't help and the Map->SparseMatrix copy makes
    // it net slower.) A pinned column p collapses to just P(p,p)=1 (also handles isolated
    // nodes, which have no diagonal entry to edit).
    Eigen::SparseMatrix<double> P(n, n);
    Eigen::VectorXi cnt(n);
    for (int k = 0; k < n; ++k) {
        if (is_pin[k]) { cnt[k] = 1; continue; }
        int c = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            if (!is_pin[it.row()]) ++c;
        cnt[k] = c;
    }
    P.reserve(cnt);
    for (int k = 0; k < n; ++k) {
        if (is_pin[k]) { P.insert(k, k) = 1.0; continue; }
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            if (!is_pin[it.row()]) P.insert(it.row(), k) = it.value();
    }
    P.makeCompressed();
    return P;
}

// Connected components of L via union-find over the off-diagonal structure.
static std::vector<std::vector<int>> connected_components(const Eigen::SparseMatrix<double>& L) {
    const int n = static_cast<int>(L.rows());
    std::vector<int> parent(n); std::iota(parent.begin(), parent.end(), 0);
    auto find=[&](int x){ while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];} return x; };
    for (int k=0;k<L.outerSize();++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k);it;++it)
            if (it.row()!=it.col()){int a=find((int)it.row()),b=find((int)it.col()); if(a!=b)parent[a]=b;}
    std::vector<int> cid(n,-1); std::vector<std::vector<int>> comps;
    for (int v=0;v<n;++v){int r=find(v); if(cid[r]<0){cid[r]=(int)comps.size();comps.emplace_back();} comps[cid[r]].push_back(v);}
    return comps;
}

// Connected-component SPLIT (the `decompose = split` axis): decompose L into its
// connected components and solve each as an INDEPENDENT connected Laplacian (with the
// chosen `ground`), then recombine. Each sub-solve is connected (k=1), so the per-
// component null space is one constant -- no multi-vector near-null-space B (which
// would blow SA up into a k-coarse-DOF-per-aggregate explosion at many components).
// Combined metrics: setup/solve SUM over components, iters = MAX over components
// (the bottleneck block; summing would let 100s of trivial specks swamp the count),
// residual = sqrt(sum ||sub_res||^2)/||b|| (L is block-diagonal across components, so
// per-component residuals are independent). A no-op (single per_solver call) when L is
// connected. Used by all solvers' run_desing dispatch for decompose = split.
template<typename Fn>
static BenchResult run_split(Fn per_solver, const Eigen::SparseMatrix<double>& L,
                             const Eigen::VectorXd& b, const std::string& name,
                             double tol, int maxiter) {
    Timer wall; wall.start();   // whole-split wall, to charge decomposition+extraction to setup
    auto comps = connected_components(L);
    if (comps.size() <= 1) return per_solver(L, b, name, tol, maxiter);
    BenchResult r; r.graph_name=name; r.n=(int)L.rows(); r.nnz=(int)L.nonZeros(); r.fillin=0;
    double setup=0, solve=0, res2=0; int it=0; const double bn=b.norm();
    double rss=0, vram=-1;   // combined solve-held RSS/VRAM = MAX over components (peak
                             // resident during the largest sub-solve); else the combined
                             // cell loses these and shows blank on the memory heatmaps.
    const bool dbg = std::getenv("SPLIT_DEBUG") != nullptr;
    // (size, rel_c, ||res_c||, iters) per non-singleton component, for SPLIT_DEBUG.
    std::vector<std::tuple<int,double,double,int>> dbg_rows;
    for (auto& nodes : comps) {
        const int sn=(int)nodes.size();
        if (sn==1) continue;  // singleton component: trivially x=0 (b~0), no solve
        std::vector<int> g2l(L.rows(),-1);
        for (int i=0;i<sn;++i) g2l[nodes[i]]=i;
        // Build subL's CSC directly. nodes[] is in increasing global order so g2l is
        // monotonic; iterating column nodes[i] (rows globally sorted) yields local rows
        // already sorted -> insert in order, skipping setFromTriplets' sort. (Parallel
        // manual-CSC build tried: NEGATIVE, see dirichlet_pin -- bandwidth-bound.)
        Eigen::SparseMatrix<double> subL(sn,sn);
        Eigen::VectorXi cnt(sn);
        for (int i=0;i<sn;++i) {
            int c=0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(L,nodes[i]);it;++it)
                if (g2l[it.row()]>=0) ++c;
            cnt[i]=c;
        }
        subL.reserve(cnt);
        for (int i=0;i<sn;++i)
            for (Eigen::SparseMatrix<double>::InnerIterator it(L,nodes[i]);it;++it) {
                const int gl=g2l[it.row()]; if (gl>=0) subL.insert(gl,i)=it.value();
            }
        subL.makeCompressed();
        Eigen::VectorXd subb(sn); for (int i=0;i<sn;++i) subb[i]=b[nodes[i]];
        BenchResult rc = per_solver(subL, subb, name, tol, maxiter);
        setup += rc.setup_time; solve += rc.solve_time; it=std::max(it,rc.iterations);
        if (rc.solve_rss_mb > rss) rss = rc.solve_rss_mb;       // peak over components
        if (rc.solve_vram_mb > vram) vram = rc.solve_vram_mb;
        const double sbn=subb.norm(); const double rnc = rc.rel_residual*sbn;
        res2 += rnc*rnc;
        r.solver_name = rc.solver_name;
        if (dbg) dbg_rows.emplace_back(sn, rc.rel_residual, rnc, rc.iterations);
    }
    if (dbg) {
        // Report the components that dominate the combined residual: sort by
        // ||res_c|| desc, show the top offenders + how many exceed tol.
        std::sort(dbg_rows.begin(), dbg_rows.end(),
                  [](auto&a,auto&b){ return std::get<2>(a) > std::get<2>(b); });
        int over=0; for (auto&t:dbg_rows) if (std::get<1>(t) > tol) over++;
        std::fprintf(stderr,
            "[split-debug] %s  comps=%zu (non-singleton=%zu)  combined_rel=%.3e  "
            "comps_over_tol=%d\n", name.c_str(), comps.size(), dbg_rows.size(),
            std::sqrt(res2)/(bn>0?bn:1.0), over);
        std::fprintf(stderr, "[split-debug]   top-%d by ||res_c||:  (size, rel_c, ||res_c||, iters)\n",
                     (int)std::min<size_t>(10, dbg_rows.size()));
        for (size_t i=0;i<dbg_rows.size() && i<10;++i)
            std::fprintf(stderr, "[split-debug]     sz=%-9d rel=%.3e  ||res||=%.3e  it=%d\n",
                std::get<0>(dbg_rows[i]), std::get<1>(dbg_rows[i]),
                std::get<2>(dbg_rows[i]), std::get<3>(dbg_rows[i]));
    }
    // The split PREPROCESSING -- union-find decomposition + per-component triplet
    // extraction + recombine -- is wall minus the summed per-component solve work; it
    // belongs in setup (apples-to-apples with apxchol's graph build / a whole-matrix
    // pin), not free. Charge it to setup_time.
    const double overhead = std::max(0.0, wall.elapsed() - (setup + solve));
    if (std::getenv("PREP_TIME"))
        std::fprintf(stderr, "[prep] %s split: decomp+extract overhead = %.3fs "
            "(of setup %.3fs incl per-comp builds %.3fs; solve %.3fs)\n",
            name.c_str(), overhead, setup+overhead, setup, solve);
    r.setup_time=setup+overhead; r.solve_time=solve; r.total_time=r.setup_time+solve;
    r.iterations=it; r.rel_residual = std::sqrt(res2)/(bn>0?bn:1.0);
    r.us_per_nnz = r.total_time/std::max(1,r.nnz)*1e6;
    r.solve_rss_mb = rss; r.solve_vram_mb = vram;   // carry peak over components
    return r;
}

// ──────────────────── AMGCL solver ────────────────────
#ifdef HAVE_AMGCL
static BenchResult run_amgcl(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter, ground_mode ground = ground_mode::pin)
{
    BenchResult r;
    r.solver_name = "AMG+CG [AMGCL]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());
    int n = r.n;
    // is_laplacian classification -- NOT timed (benchmark-only: deployment knows a
    // priori whether L is a singular Laplacian or already SDDM). If L is strictly
    // SDDM (--reg-rel / IPM) we solve the full operator unchanged regardless of method.
    Eigen::VectorXd row_sum = L * Eigen::VectorXd::Ones(n);
    const bool is_laplacian = row_sum.cwiseAbs().maxCoeff()
        / std::max(1e-30, L.diagonal().cwiseAbs().maxCoeff()) < 1e-10;
    using Backend = amgcl::backend::builtin<double>;
    using Solver = amgcl::make_solver<
        amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
        amgcl::solver::cg<Backend>
    >;
    Solver::params prm;
    prm.solver.tol = tol;
    prm.solver.maxiter = maxiter;

    Timer t;
    t.start();   // setup timer: de-sing prep + matrix conversion + AMG build (all counted)
    // Build the solve operator for the singular Laplacian per `ground` (split, if any,
    // is applied one level up by run_split, so L here is whole or a single component):
    //  pin    : symmetric Dirichlet pin (dirichlet_pin) -> SPD -> AMGCL's DEFAULT config
    //           (direct skyline_lu coarse solve). No coarsener deviation.
    //  coarse : keep the singular operator + relax the coarsest grid
    //           (direct_coarse=false, NON-default) to avoid the skyline_lu "Zero sum"
    //           crash. Valid only on a single connected component (whole+coarse on a
    //           disconnected graph is gated out by resolve_desing).
    Eigen::VectorXd rhs_v = b;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
    if (is_laplacian && ground == ground_mode::pin) {
        Timer tp; tp.start();
        std::vector<int> pinned;
        A = dirichlet_pin(L, pinned);
        for (int p : pinned) rhs_v(p) = 0.0;
        if (std::getenv("PREP_TIME"))
            std::fprintf(stderr, "[prep] %s amgcl dirichlet_pin = %.3fs (of setup; n=%d nnz=%d)\n",
                graph_name.c_str(), tp.elapsed(), n, r.nnz);
    } else {
        A = L;
        if (is_laplacian) prm.precond.direct_coarse = false;   // ground == coarse
    }
    Solver solve(A, prm);
    r.setup_time = t.elapsed();
    // AMGCL_PRINT=1 -> dump the SA hierarchy for the BoomerAMG-vs-AMGCL comparison.
    if (std::getenv("AMGCL_PRINT")) std::cerr << "[amgcl-hierarchy]\n" << solve.precond() << std::endl;

    std::vector<double> rhs(rhs_v.data(), rhs_v.data() + n);
    std::vector<double> sol(n, 0.0);

    t.start();
    auto [iters, error] = solve(rhs, sol); (void)error;
    r.solve_time = t.elapsed();
    r.total_time = r.setup_time + r.solve_time;
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)
    r.iterations = static_cast<int>(iters);
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(sol.data(), n);
    Eigen::VectorXd res = b - L * x;
    if (is_laplacian) res.array() -= res.mean();
    const double bnorm = b.norm() > 0 ? b.norm() : 1.0;
    r.rel_residual = res.norm() / bnorm;
    r.fillin = 0;
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    return r;
}

#ifdef APXCHOL_USE_CUDA
// AMGCL with CUDA backend (cuSPARSE + thrust). Builds amgcl::backend::cuda<double>
// in a separate .cu translation unit (amgcl_cuda.cu) compiled with nvcc.
static BenchResult run_amgcl_cuda(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter, ground_mode ground = ground_mode::pin)
{
    BenchResult r;
    r.solver_name = "AMG+CG [AMGCL;cuda]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());
    const int n = r.n;

    // Mirror the CPU run_amgcl 2-axis. ground = pin: Dirichlet-pin -> SPD -> AMGCL's
    // DEFAULT direct coarse solve. ground = coarse: keep the singular operator + relax
    // the coarsest grid (relax_coarse=1, set in amgcl_cuda.cu). split is applied
    // upstream by run_split, so L here is whole or a single component. The residual is
    // always scored against the ORIGINAL L (full_*), mean-centered when is_laplacian.
    Eigen::VectorXd amgcl_row_sum = L * Eigen::VectorXd::Ones(n);
    const double amgcl_row_sum_rel = amgcl_row_sum.cwiseAbs().maxCoeff()
                                   / std::max(1e-30, L.diagonal().cwiseAbs().maxCoeff());
    const bool is_laplacian = amgcl_row_sum_rel < 1e-10;

    // Solve operator (pinned for ground=pin, else original L) + relax-coarse flag.
    Eigen::VectorXd rhs_v = b;
    Eigen::SparseMatrix<double, Eigen::RowMajor> Asolve;
    int relax_coarse;
    if (is_laplacian && ground == ground_mode::pin) {
        std::vector<int> pinned;
        Asolve = dirichlet_pin(L, pinned);   // SPD
        for (int p : pinned) rhs_v(p) = 0.0;
        relax_coarse = 0;                    // default direct coarse
    } else {
        Asolve = L;
        relax_coarse = is_laplacian ? 1 : 0; // ground=coarse relaxes; SDDM keeps default
    }
    const int m = n;  // Asolve is full-size (pin keeps n x n)

    // ptrdiff_t CSR for the solve operator (Asolve) and the residual operator (L).
    std::vector<std::ptrdiff_t> sub_row(n + 1), full_row(n + 1);
    for (int i = 0; i <= n; ++i) { sub_row[i] = Asolve.outerIndexPtr()[i]; }
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm = L;
    for (int i = 0; i <= n; ++i) { full_row[i] = Lrm.outerIndexPtr()[i]; }
    const std::ptrdiff_t snnz = sub_row[n], fnnz = full_row[n];
    std::vector<std::ptrdiff_t> sub_col(snnz), full_col(fnnz);
    std::vector<double>          sub_val(snnz), full_val(fnnz);
    for (std::ptrdiff_t p = 0; p < snnz; ++p) { sub_col[p]=Asolve.innerIndexPtr()[p]; sub_val[p]=Asolve.valuePtr()[p]; }
    for (std::ptrdiff_t p = 0; p < fnnz; ++p) { full_col[p]=Lrm.innerIndexPtr()[p];    full_val[p]=Lrm.valuePtr()[p]; }

    run_amgcl_cuda_impl(
        &r, m,
        sub_row.data(), sub_col.data(), sub_val.data(),
        rhs_v.data(),
        tol, maxiter,
        n, b.data(),
        full_row.data(), full_col.data(), full_val.data(),
        is_laplacian ? 1 : 0, relax_coarse);
    return r;
}
#endif
#endif

// ──────────────────── Hypre BoomerAMG solver ────────────────────
#ifdef HAVE_HYPRE
#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_mv.h>
#include <_hypre_parcsr_mv.h>  // for hypre_ParVectorLocalVector / hypre_VectorData
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_krylov.h>

// Sequential-build shim: Hypre defines MPI_Comm/MPI_Op etc. via macros in
// SEQUENTIAL mode, but does NOT expose MPI_COMM_WORLD as a public constant.
// All Hypre internal sequential code uses 0 for the world communicator.
#if defined(HYPRE_SEQUENTIAL) && !defined(MPI_COMM_WORLD)
#define MPI_COMM_WORLD 0
#endif

// BoomerAMG coarsening config, selected by env BOOMERAMG_CFG, plotted as separate
// series so the comparison is transparent rather than us silently picking one:
//   "default" -> bare Hypre defaults (classical Ruge-Stuben C/F + classical interp).
//                Excellent on structured PDEs, but OPERATOR COMPLEXITY explodes on
//                high-degree / scale-free graphs (com-Youtube: 210s setup vs AMGCL
//                2.9s -- a coarsening artifact, not a real AMG limitation).
//   "cut"     -> defaults + CoarsenCutFactor (hub -> F-point): the fix for the
//                mega-hub social giants; a no-op on grids/IPM. The sweep runs
//                "default" and "cut" as the two charted series.
//   "agg"     -> Hypre's OWN documented recipe for hard/unstructured problems (the
//                BoomerAMG user manual + Li-Yang perf study): strong threshold 0.7
//                (the manual calls it "by far the most important option"), HMIS (CPU)
//                / PMIS (GPU) coarsening, a few aggressive-coarsening levels + extra
//                paths, and capped+truncated extended+i interpolation. Lost to
//                default/cut in the probe; kept env-selectable, NOT swept.
// Default when BOOMERAMG_CFG is unset = "default" (bare Hypre defaults — the
// probe showed it's best on grids/IPM and most social graphs).
static void configure_boomeramg(HYPRE_Solver amg, bool gpu) {
    // BOOMERAMG_PRINT=N -> Hypre AMG setup print level (2/3 dumps the per-level grid +
    // operator complexity, the direct evidence for the hub-fill hypothesis).
    if (const char* pv = std::getenv("BOOMERAMG_PRINT")) HYPRE_BoomerAMGSetPrintLevel(amg, std::atoi(pv));
    const char* e = std::getenv("BOOMERAMG_CFG");
    std::string cfg = e ? e : "default";                // default = bare Hypre defaults (probe showed
    if (cfg == "default") return;                       //   it's best on grids/IPM and most social graphs)
    if (cfg == "cut") {
        // CoarsenCutFactor: any row with nnz > cut*avg_nnz is forced to a FINE point
        // (not coarsened, no interpolation weights). This makes the mega-hub an
        // F-point instead of a dense C-point, so the O(deg) coarse-row fill never
        // forms -- Hypre's closest analog to how SA absorbs the hub into an aggregate.
        // Everything else stays at Hypre defaults (HMIS, ext+i, P_max 4).
        int cut = 10; if (const char* cv = std::getenv("BOOMERAMG_CUT")) cut = std::atoi(cv);
        HYPRE_BoomerAMGSetCoarsenCutFactor(amg, cut);
        return;
    }
    if (cfg == "graph") {
        // For UNWEIGHTED graph Laplacians with mega-hubs (com-Youtube: max degree
        // 28754, half the nodes degree-1): strength-of-connection is inert (every edge
        // weight 1 -> strength 1, so the threshold prunes nothing), the hub stays a
        // dense C-point, and classical Galerkin coarsening fills O(deg^2) in the coarse
        // operator (-> 207s setup). NON-GALERKIN sparsification of R*A*P drops that
        // fill directly; PMIS + 2 aggressive levels coarsen the degree-1 pendants fast.
        HYPRE_BoomerAMGSetCoarsenType(amg, 8);          // PMIS (fully parallel)
        HYPRE_BoomerAMGSetAggNumLevels(amg, 2);
        HYPRE_BoomerAMGSetInterpType(amg, 6);           // extended+i
        HYPRE_BoomerAMGSetPMaxElmts(amg, 4);
        HYPRE_BoomerAMGSetNonGalerkinTol(amg, 0.1);     // sparsify coarse operators
        HYPRE_BoomerAMGSetMaxLevels(amg, 25);
        return;
    }
    if (cfg == "gpu_equiv") {
        // The GPU axis runs cfg=default under HYPRE_EXEC_DEVICE, where Hypre's own
        // device defaults silently switch the ALGORITHM: PMIS coarsening, ext+i
        // interpolation, l1-Jacobi smoothing (hybrid-GS is sequential). This config
        // mirrors those three knobs on the CPU so the CPU-vs-GPU gap can be split
        // into algorithm-difference vs raw-bandwidth: if CPU/gpu_equiv reproduces
        // the GPU iteration counts, the remaining time ratio is pure hardware
        // (as already proven for AMGCL, whose iters match across devices).
        HYPRE_BoomerAMGSetCoarsenType(amg, 8);          // PMIS
        HYPRE_BoomerAMGSetInterpType(amg, 6);           // extended+i
        HYPRE_BoomerAMGSetRelaxType(amg, 18);           // l1-Jacobi
        return;
    }
    // "pmis" forces PMIS coarsening (fully parallel; the manual's escalation when
    // HMIS's sequential pass is too slow on high-degree hubs); "agg" uses HMIS on CPU.
    // NOTE both "pmis" and "agg" bundle the 0.7-threshold/aggressive knobs below --
    // neither is the GPU-equivalent config; use "gpu_equiv" for that comparison.
    int coarsen = (cfg == "pmis") ? 8 : (gpu ? 8 : 10);
    HYPRE_BoomerAMGSetCoarsenType(amg, coarsen);        // PMIS (8) / HMIS (10)
    HYPRE_BoomerAMGSetStrongThreshold(amg, 0.7);        // most important knob (unstructured)
    HYPRE_BoomerAMGSetAggNumLevels(amg, 4);             // aggressive coarsening levels
    HYPRE_BoomerAMGSetNumPaths(amg, 5);                 // aggressive coarsening paths
    HYPRE_BoomerAMGSetInterpType(amg, 6);               // extended+i
    HYPRE_BoomerAMGSetPMaxElmts(amg, 2);                // cap interpolation stencil
    HYPRE_BoomerAMGSetTruncFactor(amg, 0.3);            // interpolation truncation
    HYPRE_BoomerAMGSetMaxLevels(amg, 25);
}

// Build a Hypre IJ matrix from Eigen sparse matrix. Caller must
// HYPRE_IJMatrixDestroy after use.
static HYPRE_IJMatrix eigen_to_hypre_ij(const Eigen::SparseMatrix<double>& L) {
    HYPRE_Int n = static_cast<HYPRE_Int>(L.rows());
    HYPRE_IJMatrix A;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, n - 1, 0, n - 1, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    // Bulk SetValues in ONE call. Previously this looped per-row, which on
    // GPU triggered n host->device transfers (524k for IPM, 4M for grid_2000).
    // Hypre's IJMatrixSetValues accepts multi-row format:
    //   nrows, ncols[nrows], rows[nrows], cols[sum(ncols)], values[sum(ncols)]
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm = L;
    const HYPRE_Int nnz = static_cast<HYPRE_Int>(Lrm.nonZeros());
    std::vector<HYPRE_Int> rows(n);
    std::vector<HYPRE_Int> ncols(n);
    std::vector<HYPRE_Int> cols(nnz);
    std::vector<double>    vals(nnz);
    HYPRE_Int off = 0;
    for (HYPRE_Int row = 0; row < n; ++row) {
        rows[row] = row;
        const HYPRE_Int begin = static_cast<HYPRE_Int>(Lrm.outerIndexPtr()[row]);
        const HYPRE_Int end   = static_cast<HYPRE_Int>(Lrm.outerIndexPtr()[row + 1]);
        ncols[row] = end - begin;
        for (HYPRE_Int p = begin; p < end; ++p) {
            cols[off] = static_cast<HYPRE_Int>(Lrm.innerIndexPtr()[p]);
            vals[off] = Lrm.valuePtr()[p];
            ++off;
        }
    }
    HYPRE_IJMatrixSetValues(A, n, ncols.data(), rows.data(), cols.data(), vals.data());
    HYPRE_IJMatrixAssemble(A);
    return A;
}

#ifdef APXCHOL_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

// GPU-resident PCG using cuSPARSE SpMV + cuBLAS axpy/dot/nrm2.
// Preconditioner (apx_cholesky) stays as-is; we bounce r,z through
// host per iter because the precond's solve_LLt API is host-pointer-based.
// Net solve cost per iter still ~5x lower than CPU-PCG with cuSPARSE
// sptrsv (which we already use in CUDA build) because SpMV + vector ops
// now stay on device.
//
// Same vertex-pinning convention as run_hypre_boomeramg.
static BenchResult run_apxchol_gpu_pcg(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    const std::string& combo_label,
    const apxchol::factor_options& fopts,
    apxchol::graph_storage storage,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = "apxchol+GPU-PCG (" + combo_label + ")";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    // Detect Laplacian (zero row sums) vs SDDM (positive row sums).
    // For SDDM (full-rank), no pinning needed — pinning is a perturbation
    // that on IPM iter10 blew up iter count 53 -> 83. For Laplacian,
    // pin one vertex to make L full-rank (avoids CG breakdown).
    int n = r.n;
    Eigen::VectorXd row_sum = L * Eigen::VectorXd::Ones(n);
    const double row_sum_rel = row_sum.cwiseAbs().maxCoeff()
                             / std::max(1e-30, L.diagonal().cwiseAbs().maxCoeff());
    const bool is_laplacian = row_sum_rel < 1e-10;
    // Symmetric Dirichlet pin (full n x n, SPD) for a singular Laplacian -- see
    // dirichlet_pin / run_amgcl. Reaches 1e-8 vs the original L.
    const int m = n;
    std::vector<int> pinned;
    Eigen::SparseMatrix<double> Lsub = is_laplacian ? dirichlet_pin(L, pinned) : L;
    Eigen::VectorXd bsub = b;
    if (is_laplacian) for (int p : pinned) bsub(p) = 0.0;

    const auto t_wall_start = std::chrono::high_resolution_clock::now();

    apxchol::apx_cholesky pc;
    pc.set_options(fopts);
    pc.set_storage(storage);
    pc.compute(Lsub);

    // Build CSR full-storage of Lsub for GPU SpMV.
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm = Lsub.template selfadjointView<Eigen::Lower>();
    Lrm.makeCompressed();

    cublasHandle_t cublas; cublasCreate(&cublas);
    cusparseHandle_t cusparse; cusparseCreate(&cusparse);

    // Upload matrix.
    int *d_Arow, *d_Acol; double *d_Aval;
    cudaMalloc(&d_Arow, (m+1)*sizeof(int));
    cudaMalloc(&d_Acol, Lrm.nonZeros()*sizeof(int));
    cudaMalloc(&d_Aval, Lrm.nonZeros()*sizeof(double));
    cudaMemcpy(d_Arow, Lrm.outerIndexPtr(), (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Acol, Lrm.innerIndexPtr(), Lrm.nonZeros()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Aval, Lrm.valuePtr(), Lrm.nonZeros()*sizeof(double), cudaMemcpyHostToDevice);
    cusparseSpMatDescr_t A;
    cusparseCreateCsr(&A, m, m, Lrm.nonZeros(), d_Arow, d_Acol, d_Aval,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    double *d_x, *d_r, *d_z, *d_p, *d_Ap, *d_b;
    cudaMalloc(&d_x, m*sizeof(double));
    cudaMalloc(&d_r, m*sizeof(double));
    cudaMalloc(&d_z, m*sizeof(double));
    cudaMalloc(&d_p, m*sizeof(double));
    cudaMalloc(&d_Ap, m*sizeof(double));
    cudaMalloc(&d_b, m*sizeof(double));
    cusparseDnVecDescr_t vec_p, vec_Ap;
    cusparseCreateDnVec(&vec_p, m, d_p, CUDA_R_64F);
    cusparseCreateDnVec(&vec_Ap, m, d_Ap, CUDA_R_64F);

    double one = 1.0, zero = 0.0;
    size_t spmv_buf_sz; void* spmv_buf;
    cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, vec_p, &zero, vec_Ap,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_sz);
    cudaMalloc(&spmv_buf, spmv_buf_sz);

    cudaMemcpy(d_b, bsub.data(), m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, m*sizeof(double));
    cudaMemcpy(d_r, d_b, m*sizeof(double), cudaMemcpyDeviceToDevice);

    const auto t_setup_done = std::chrono::high_resolution_clock::now();
    r.setup_time = std::chrono::duration<double>(t_setup_done - t_wall_start).count();

    // PCG loop (cuBLAS for vector ops, cuSPARSE for SpMV; precond bounces).
    Eigen::VectorXd r_host(m), z_host(m);
    double rz, rz_new, alpha, beta, pAp, rnorm, bnorm;
    cublasDnrm2(cublas, m, d_b, 1, &bnorm);

    cudaMemcpy(r_host.data(), d_r, m*sizeof(double), cudaMemcpyDeviceToHost);
    z_host = pc.solve(r_host);
    cudaMemcpy(d_z, z_host.data(), m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, d_z, m*sizeof(double), cudaMemcpyDeviceToDevice);
    cublasDdot(cublas, m, d_r, 1, d_z, 1, &rz);

    int iters = 0;
    for (; iters < maxiter; ++iters) {
        cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, vec_p, &zero, vec_Ap,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buf);
        cublasDdot(cublas, m, d_p, 1, d_Ap, 1, &pAp);
        if (pAp <= 0) break;
        alpha = rz / pAp;
        double neg_alpha = -alpha;
        cublasDaxpy(cublas, m, &alpha, d_p, 1, d_x, 1);
        cublasDaxpy(cublas, m, &neg_alpha, d_Ap, 1, d_r, 1);
        cublasDnrm2(cublas, m, d_r, 1, &rnorm);
        if (rnorm / bnorm < tol) { iters++; break; }
        cudaMemcpy(r_host.data(), d_r, m*sizeof(double), cudaMemcpyDeviceToHost);
        z_host = pc.solve(r_host);
        cudaMemcpy(d_z, z_host.data(), m*sizeof(double), cudaMemcpyHostToDevice);
        cublasDdot(cublas, m, d_r, 1, d_z, 1, &rz_new);
        beta = rz_new / rz;
        cublasDscal(cublas, m, &beta, d_p, 1);
        cublasDaxpy(cublas, m, &one, d_z, 1, d_p, 1);
        rz = rz_new;
    }
    cudaDeviceSynchronize();

    r.iterations = iters;
    r.solve_vram_mb = read_vram_mb();   // device VRAM held at solve end (operator+factor+
                                        // SpMV buf+PCG vectors), before the frees below.

    // D2H x; reconstruct full x and recompute true residual.
    Eigen::VectorXd x_full(n);
    cudaMemcpy(x_full.data(), d_x, m*sizeof(double), cudaMemcpyDeviceToHost);
    if (is_laplacian) {
        x_full(m) = 0;
        x_full.array() -= x_full.mean();
    }
    Eigen::VectorXd res = b - L * x_full;
    if (is_laplacian) res.array() -= res.mean();
    double bnorm_full = b.norm();
    r.rel_residual = res.norm() / (bnorm_full > 0 ? bnorm_full : 1.0);

    const auto t_wall_end = std::chrono::high_resolution_clock::now();
    r.total_time = std::chrono::duration<double>(t_wall_end - t_wall_start).count();
    r.solve_time = r.total_time - r.setup_time;
    r.fillin = 0;
    r.us_per_nnz = r.total_time / r.nnz * 1e6;

    cudaFree(d_Arow); cudaFree(d_Acol); cudaFree(d_Aval);
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap); cudaFree(d_b);
    cudaFree(spmv_buf);
    cusparseDestroyDnVec(vec_p); cusparseDestroyDnVec(vec_Ap);
    cusparseDestroySpMat(A);
    cublasDestroy(cublas); cusparseDestroy(cusparse);
    return r;
}
#endif // APXCHOL_USE_CUDA

static BenchResult run_hypre_boomeramg(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = "BoomerAMG+PCG [Hypre]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    // If L is already strictly SDDM (full-rank, e.g. IPM or --reg-rel
    // regularized grid), solve the FULL operator with NO pinning so the result
    // is measured against the same matrix every other solver sees. Only pin
    // when L is a singular Laplacian (row sums ~ 0) AND not regularized.
    int n = r.n;
    Eigen::VectorXd row_sum = L * Eigen::VectorXd::Ones(n);
    const double row_sum_rel = row_sum.cwiseAbs().maxCoeff()
                             / std::max(1e-30, L.diagonal().cwiseAbs().maxCoeff());
    const bool is_laplacian = row_sum_rel < 1e-10;
    // Setup timer STARTS here: the de-sing grounding WORK (dirichlet_pin) + Hypre
    // format conversion + AMG/PCG build all count toward setup, apples-to-apples with
    // apxchol's graph-build-inclusive setup. The is_laplacian SpMV above is a
    // benchmark-only classification and is deliberately left out of the timed region.
    Timer t;
    t.start();
    // Symmetric Dirichlet pin (one node per connected component) for a singular
    // Laplacian (full n x n, SPD), so the true residual vs the ORIGINAL L reaches 1e-8.
    // SDDM/regularized L solves the full operator unchanged. (Under --desing split this
    // wrapper is called per connected component, so dirichlet_pin grounds a single node.)
    const int m = n;
    std::vector<int> pinned;
    Eigen::SparseMatrix<double> Lsub = is_laplacian ? dirichlet_pin(L, pinned) : L;
    Eigen::VectorXd bsub = b;
    if (is_laplacian) for (int p : pinned) bsub(p) = 0.0;

    HYPRE_IJMatrix A = eigen_to_hypre_ij(Lsub);
    HYPRE_ParCSRMatrix parA;
    HYPRE_IJMatrixGetObject(A, (void**)&parA);

    HYPRE_IJVector bij, xij;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, m - 1, &bij);
    HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(bij);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, m - 1, &xij);
    HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(xij);

    std::vector<HYPRE_Int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    HYPRE_IJVectorSetValues(bij, m, idx.data(), bsub.data());
    std::vector<double> zero(m, 0.0);
    HYPRE_IJVectorSetValues(xij, m, idx.data(), zero.data());
    HYPRE_IJVectorAssemble(bij);
    HYPRE_IJVectorAssemble(xij);

    HYPRE_ParVector parB, parX;
    HYPRE_IJVectorGetObject(bij, (void**)&parB);
    HYPRE_IJVectorGetObject(xij, (void**)&parX);

    HYPRE_Solver amg;
    HYPRE_BoomerAMGCreate(&amg);
    HYPRE_BoomerAMGSetPrintLevel(amg, 0);
    HYPRE_BoomerAMGSetTol(amg, 0.0);
    HYPRE_BoomerAMGSetMaxIter(amg, 1);
    configure_boomeramg(amg, /*gpu=*/false);  // scalable coarsening (see helper)
    HYPRE_BoomerAMGSetup(amg, parA, parB, parX);

    HYPRE_Solver pcg;
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg);
    HYPRE_PCGSetTol(pcg, tol);
    HYPRE_PCGSetTwoNorm(pcg, 1);  // 2-norm matches our bench recompute (else preconditioned norm)
    // RELATIVE stop ||r||_2 < tol*||b||_2 (StopCrit=0). NOTE: StopCrit=1 is HYPRE's
    // ABSOLUTE criterion ||r|| < tol -- it only happened to behave correctly because
    // make_rhs normalizes ||b||=1, but under --desing split each component is solved
    // against its OWN ||b_c|| (sum ||b_c||^2 = 1, so specks have ||b_c|| ~ 1e-4) where
    // absolute 1e-8 is a LOOSE relative ~1e-4 -> specks undersolve and the combined
    // residual floors above tol. Relative stop converges every component to tol; on the
    // whole matrix (||b||=1) it's identical to the old absolute behavior.
    HYPRE_PCGSetStopCrit(pcg, 0);
    HYPRE_PCGSetMaxIter(pcg, maxiter);
    HYPRE_PCGSetPrintLevel(pcg, 0);
    HYPRE_PCGSetPrecond(pcg,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
        amg);
    HYPRE_ParCSRPCGSetup(pcg, parA, parB, parX);
    r.setup_time = t.elapsed();

    t.start();
    HYPRE_ParCSRPCGSolve(pcg, parA, parB, parX);
    r.solve_time = t.elapsed();
    r.total_time = r.setup_time + r.solve_time;
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)

    HYPRE_Int iters_out;
    double final_res;
    HYPRE_PCGGetNumIterations(pcg, &iters_out);
    HYPRE_PCGGetFinalRelativeResidualNorm(pcg, &final_res);
    r.iterations = static_cast<int>(iters_out);
    r.fillin = 0;
    r.us_per_nnz = r.total_time / r.nnz * 1e6;

    // Extract solution; compute true residual against the original full system.
    std::vector<double> xvals(m);
    HYPRE_IJVectorGetValues(xij, m, idx.data(), xvals.data());
    Eigen::VectorXd x(n);
    x.head(m) = Eigen::Map<Eigen::VectorXd>(xvals.data(), m);
    if (is_laplacian)
        x.array() -= x.mean();   // sol[pinned]=0 already from the solve; no x(n-1) hack
    Eigen::VectorXd res = b - L * x;
    if (is_laplacian) res.array() -= res.mean();
    double bnorm = b.norm();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);

    HYPRE_BoomerAMGDestroy(amg);
    HYPRE_ParCSRPCGDestroy(pcg);
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(bij);
    HYPRE_IJVectorDestroy(xij);
    return r;
}

#ifdef APXCHOL_USE_CUDA
static BenchResult run_hypre_boomeramg_gpu(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& b,
    const std::string& graph_name,
    double tol, int maxiter)
{
    BenchResult r;
    r.solver_name = "BoomerAMG+PCG [Hypre;cuda]";
    r.graph_name = graph_name;
    r.n = static_cast<int>(L.rows());
    r.nnz = static_cast<int>(L.nonZeros());

    // Match the CPU path (run_hypre_boomeramg): only pin one node + add the
    // 1e-6 diagonal shift when L is a SINGULAR Laplacian (row sums ~ 0). If L is
    // already strictly SDDM (--reg-rel regularized, or IPM), solve the FULL
    // operator with no pinning — otherwise we'd solve a differently-regularized
    // system than the residual is checked against (b - L*x), which floors the
    // true residual at ~1e-6 on ill-conditioned problems and never reaches tol.
    int n = r.n;
    Eigen::VectorXd row_sum = L * Eigen::VectorXd::Ones(n);
    const double row_sum_rel = row_sum.cwiseAbs().maxCoeff()
                             / std::max(1e-30, L.diagonal().cwiseAbs().maxCoeff());
    const bool is_laplacian = row_sum_rel < 1e-10;
    // Setup timer STARTS here: de-sing grounding (dirichlet_pin) + GPU-mode switch +
    // Hypre conversion + AMG/PCG build all count (apples-to-apples with apxchol). The
    // is_laplacian SpMV above is a benchmark-only classification, left out of setup.
    Timer t;
    t.start();
    // Symmetric Dirichlet pin (one node per connected component) for a singular
    // Laplacian (full n x n, SPD); the true residual vs the ORIGINAL L reaches 1e-8.
    // SDDM/regularized L solves the full operator unchanged. (Under --decompose split
    // this wrapper sees one connected component, so dirichlet_pin grounds a single node.)
    const int m = n;
    std::vector<int> pinned;
    Eigen::SparseMatrix<double> Lsub = is_laplacian ? dirichlet_pin(L, pinned) : L;
    Eigen::VectorXd bsub = b;
    if (is_laplacian) for (int p : pinned) bsub(p) = 0.0;

    // Switch Hypre to GPU execution. HYPRE_Init was already called in main().
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

    HYPRE_IJMatrix A = eigen_to_hypre_ij(Lsub);
    HYPRE_ParCSRMatrix parA;
    HYPRE_IJMatrixGetObject(A, (void**)&parA);

    HYPRE_IJVector bij, xij;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, m - 1, &bij);
    HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(bij);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, m - 1, &xij);
    HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(xij);

    std::vector<HYPRE_Int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    HYPRE_IJVectorSetValues(bij, m, idx.data(), bsub.data());
    std::vector<double> zero(m, 0.0);
    HYPRE_IJVectorSetValues(xij, m, idx.data(), zero.data());
    HYPRE_IJVectorAssemble(bij);
    HYPRE_IJVectorAssemble(xij);

    HYPRE_ParVector parB, parX;
    HYPRE_IJVectorGetObject(bij, (void**)&parB);
    HYPRE_IJVectorGetObject(xij, (void**)&parX);

    HYPRE_Solver amg;
    HYPRE_BoomerAMGCreate(&amg);
    HYPRE_BoomerAMGSetPrintLevel(amg, 0);
    HYPRE_BoomerAMGSetTol(amg, 0.0);
    HYPRE_BoomerAMGSetMaxIter(amg, 1);
    configure_boomeramg(amg, /*gpu=*/true);  // scalable coarsening (see helper)
    HYPRE_BoomerAMGSetup(amg, parA, parB, parX);

    HYPRE_Solver pcg;
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg);
    HYPRE_PCGSetTol(pcg, tol);
    HYPRE_PCGSetTwoNorm(pcg, 1);  // 2-norm matches our bench recompute (else preconditioned norm)
    // RELATIVE stop ||r||_2 < tol*||b||_2 (StopCrit=0). NOTE: StopCrit=1 is HYPRE's
    // ABSOLUTE criterion ||r|| < tol -- it only happened to behave correctly because
    // make_rhs normalizes ||b||=1, but under --desing split each component is solved
    // against its OWN ||b_c|| (sum ||b_c||^2 = 1, so specks have ||b_c|| ~ 1e-4) where
    // absolute 1e-8 is a LOOSE relative ~1e-4 -> specks undersolve and the combined
    // residual floors above tol. Relative stop converges every component to tol; on the
    // whole matrix (||b||=1) it's identical to the old absolute behavior.
    HYPRE_PCGSetStopCrit(pcg, 0);
    HYPRE_PCGSetMaxIter(pcg, maxiter);
    HYPRE_PCGSetPrintLevel(pcg, 0);
    HYPRE_PCGSetPrecond(pcg,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
        amg);
    HYPRE_ParCSRPCGSetup(pcg, parA, parB, parX);
    r.setup_time = t.elapsed();

    t.start();
    HYPRE_ParCSRPCGSolve(pcg, parA, parB, parX);
    r.solve_time = t.elapsed();
    r.total_time = r.setup_time + r.solve_time;
    r.solve_rss_mb = read_vmrss_mb();   // solve-held host RSS (peak from /usr/bin/time)

    HYPRE_Int iters_out;
    double final_res;
    HYPRE_PCGGetNumIterations(pcg, &iters_out);
    HYPRE_PCGGetFinalRelativeResidualNorm(pcg, &final_res);
    r.iterations = static_cast<int>(iters_out);
    r.fillin = 0;
    r.us_per_nnz = r.total_time / r.nnz * 1e6;
    r.solve_vram_mb = read_vram_mb();   // device VRAM held at solve end (GPU AMG hierarchy
                                        // + ParCSR operator + PCG vectors), before destroy.

    std::vector<double> xvals(m);
    HYPRE_IJVectorGetValues(xij, m, idx.data(), xvals.data());
    Eigen::VectorXd x(n);
    x.head(m) = Eigen::Map<Eigen::VectorXd>(xvals.data(), m);
    if (is_laplacian)
        x.array() -= x.mean();   // sol[pinned]=0 already from the solve; no x(n-1) hack
    Eigen::VectorXd res = b - L * x;
    if (is_laplacian) res.array() -= res.mean();
    double bnorm = b.norm();
    r.rel_residual = res.norm() / (bnorm > 0 ? bnorm : 1.0);

    HYPRE_BoomerAMGDestroy(amg);
    HYPRE_ParCSRPCGDestroy(pcg);
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(bij);
    HYPRE_IJVectorDestroy(xij);

    // Reset back to host so subsequent CPU runs work.
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
    return r;
}

#endif

#endif

// ──────────────────── main ────────────────────
int main(int argc, char** argv) {
#ifdef HAVE_HYPRE
    HYPRE_Init();  // sequential build uses Hypre's MPI shim; no MPI_Init needed
#endif
    Args args = parse_args(argc, argv);

    // Thread setup must run for ALL solvers, not just apxchol_v1 (which had
    // its own omp_set_num_threads inside its `if(args.solvers.count(...))`
    // block). When omp_set_num_threads wasn't called, libgomp defaulted to 1
    // thread for AMGCL's setup → 13x slowdown on IPM iter10 (1.87s setup vs
    // 0.14s). Hoisted here so AMGCL/Hypre/etc. all see the right thread count.
    int tc_global = args.threads;
#ifdef _OPENMP
    if (tc_global <= 0) tc_global = omp_get_max_threads();
    omp_set_num_threads(tc_global);
    if (const char* env = std::getenv("OMP_NUM_THREADS")) {
        if (std::atoi(env) != tc_global) {
            std::cerr << "WARNING: OMP_NUM_THREADS=" << env
                      << " in env, --threads=" << tc_global
                      << ". omp_set_num_threads overrides for our parallel "
                         "regions but nested libraries may pick up the env.\n";
        }
    }
#endif

    // Build graph
    std::vector<std::vector<Edge>> adj;
    std::string graph_name;

    if (args.graph == "grid") {
        int gc = (args.ny > 0) ? args.ny : args.n;
        adj = grid_graph(args.n, gc);
        graph_name = (args.ny > 0)
            ? "grid_" + std::to_string(args.n) + "x" + std::to_string(args.ny)
            : "grid_" + std::to_string(args.n);
    } else if (args.graph == "grid3d") {
        adj = grid_graph_3d(args.n);
        graph_name = "grid3d_" + std::to_string(args.n);
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

    // --reg-rel: unify singular grid Laplacians to strictly-SDDM by adding
    // eps*I (eps = reg_rel * mean|diag|), the SAME shift for EVERY solver. This
    // removes the nullspace so no solver needs to pin a vertex, and every solver
    // is then measured against the identical operator b - (L+eps*I)x -- making
    // multigrid (BoomerAMG/CMG) directly comparable instead of flooring at
    // ~1e-4 on a pinned system. No-op (eps=0) for already-SDDM IPM/SuiteSparse.
    if (args.reg_rel > 0.0) {
        const double eps = args.reg_rel * L.diagonal().cwiseAbs().mean();
        for (int k = 0; k < L.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
                if (it.row() == it.col()) it.valueRef() += eps;
        std::cerr << "[reg] L += " << eps << " * I (reg_rel=" << args.reg_rel << ")\n";
    }

    int nnz = static_cast<int>(L.nonZeros());
    std::cerr << ", nnz=" << nnz << "\n";

    // --dump-mtx: write the built Laplacian (lower triangle, symmetric) to a
    // Matrix Market file and exit. Lets external solvers (e.g. ParAC, which
    // requires an AMD-reordered .mtx) consume the EXACT same matrix we benchmark.
    if (!args.dump_mtx.empty()) {
        std::ofstream out(args.dump_mtx);
        if (!out) { std::cerr << "cannot open " << args.dump_mtx << "\n"; return 1; }
        out << "%%MatrixMarket matrix coordinate real symmetric\n";
        // --pin-dump: emit the per-component Dirichlet-pinned matrix (the fair
        // grounded operator) rather than the raw singular L, so external solvers
        // solve the SAME system the in-process solvers do (no eps*I shift).
        Eigen::SparseMatrix<double> Lc;
        if (args.pin_dump) {
            std::vector<int> pinned;
            Lc = dirichlet_pin(L, pinned);
            std::cerr << "[pin-dump] grounded " << pinned.size()
                      << " node(s) (one per connected component)\n";
        } else if (args.giant_dump) {
            // The comp_rank-th LARGEST connected component's PURE Laplacian, relabeled
            // 0..cn-1. rank 0 = the giant (= the full pure L for a connected matrix). The
            // runner loops rank 0,1,2,... over components above a size threshold and runs
            // ParAC physics on each (single-node trim grounds a CONNECTED component) — the
            // per-component split. The "K components" count + per-rank size are printed so
            // the runner knows when to stop.
            auto comps = connected_components(L);
            std::vector<size_t> order(comps.size());
            std::iota(order.begin(), order.end(), static_cast<size_t>(0));
            std::sort(order.begin(), order.end(),
                      [&](size_t a, size_t b) { return comps[a].size() > comps[b].size(); });
            const int rank = args.comp_rank;
            if (rank < 0 || rank >= static_cast<int>(comps.size())) {
                std::cerr << "[component-dump] rank " << rank << " >= " << comps.size()
                          << " components; nothing to dump\n";
                return 0;
            }
            const auto& comp = comps[order[rank]];
            const int cn = static_cast<int>(comp.size());
            std::vector<int> remap(L.rows(), -1);
            for (int i = 0; i < cn; ++i) remap[comp[i]] = i;
            std::vector<Eigen::Triplet<double>> trips;
            trips.reserve(L.nonZeros());
            for (int k = 0; k < L.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                    const int ri = remap[it.row()], ci = remap[it.col()];
                    if (ri >= 0 && ci >= 0) trips.emplace_back(ri, ci, it.value());
                }
            Eigen::SparseMatrix<double> G(cn, cn);
            G.setFromTriplets(trips.begin(), trips.end());
            Lc = G;
            std::cerr << "[component-dump] rank " << rank << " / " << comps.size()
                      << " components: " << cn << " / " << L.rows() << " nodes\n";
        } else {
            Lc = L;
        }
        Lc.makeCompressed();
        long entries = 0;
        for (int k = 0; k < Lc.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(Lc, k); it; ++it)
                if (it.row() >= it.col()) ++entries;
        out << Lc.rows() << ' ' << Lc.cols() << ' ' << entries << '\n';
        out.precision(17);
        for (int k = 0; k < Lc.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(Lc, k); it; ++it)
                if (it.row() >= it.col())
                    out << (it.row() + 1) << ' ' << (it.col() + 1) << ' ' << it.value() << '\n';
        std::cerr << "dumped " << entries << " lower-triangle entries to " << args.dump_mtx << "\n";
        return 0;
    }

    Eigen::VectorXd b = make_rhs(L, args.seed);

    if (args.csv) print_csv_header();
    else print_header_pretty();

    auto print = [&](const BenchResult& r) {
        if (args.csv) print_result_csv(r);
        else print_result_pretty(r);
    };

    const int R = args.repeat;

    // Whether L is disconnected -- computed once (benchmark-only classification, not
    // timed); drives decompose=auto + the whole+coarse validity gate.
    const bool disconnected = connected_components(L).size() > 1;

    // De-singularization dispatch (2 axes; see resolve_desing). `single(L,b,nm,tol,mi,
    // ground)` solves one operator (whole or a single component) under `ground`. For
    // decompose=split, run_split decomposes and solves each component under the chosen
    // ground. Unsupported cells return an n/a sentinel (iterations & rel_residual = -1)
    // -- charted distinct from blank/X/T.
    auto make_na = [&](const std::string& base) {
        BenchResult r; r.solver_name = base + " [n/a]"; r.graph_name = graph_name;
        r.n = (int)L.rows(); r.nnz = (int)L.nonZeros();
        r.iterations = -1; r.rel_residual = -1.0;
        return r;
    };
    auto run_desing = [&](const std::string& key, const std::string& base,
                          auto single) -> BenchResult {
        auto plan = resolve_desing(key, args.decompose, args.ground, disconnected);
        if (!plan.supported) return make_na(base);
        if (plan.decompose == decompose_mode::split)
            return run_split([&, g = plan.ground](const Eigen::SparseMatrix<double>& sL,
                                                  const Eigen::VectorXd& sb, const std::string& nm,
                                                  double tl, int mi) {
                       return single(sL, sb, nm, tl, mi, g);
                   }, L, b, graph_name, args.tol, args.maxiter);
        return single(L, b, graph_name, args.tol, args.maxiter, plan.ground);
    };

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
    if (args.solvers.count("apxchol_v1") ||
        args.solvers.count("apxchol_gpu")) {
        struct V1Combo {
            const char* name;
            std::string is;
            apxchol::graph_storage storage = apxchol::graph_storage::forward_star;
            size_t exact_clique_max_degree = 0; // 0 = off; emit exact clique when deg <= this
            double degree_mult = 0.0;           // 0 = use fopts default (2.0); else override the IS cap
        };
        using gs = apxchol::graph_storage;
        static const V1Combo v1_combos[] = {
            // IS-selector x storage on the random-tree sampler (the only elim left).
            // forward_star (base)
            {"bg+tree",   "block_greedy"},
            {"bk+tree",   "baumann_kyng"},
            {"root+tree", "rootset"},
            {"luby+tree", "luby"},
            // vec
            {"bg+tree",   "block_greedy", gs::vec},
            {"bk+tree",   "baumann_kyng", gs::vec},
            {"root+tree", "rootset", gs::vec},
            {"luby+tree", "luby", gs::vec},
            // bstr (bit-string): remaining storage backend across
            // ALL four selectors -- completes the full selector x storage grid
            // (fwd_star / vec / bstr / vec_pool  x  bg / bk / root / luby) for
            // the ablation heatmap. fwd_star + vec are the bare-named combos above;
            // vec_pool below. Name carries the tag so --v1-configs selects it directly.
            {.name="bg+tree[bstr]",   .is="block_greedy", .storage=gs::bstr},
            {.name="bk+tree[bstr]",   .is="baumann_kyng", .storage=gs::bstr},
            {.name="root+tree[bstr]", .is="rootset",      .storage=gs::bstr},
            {.name="luby+tree[bstr]", .is="luby",         .storage=gs::bstr},
            // vec_pool (charted backend)
            {.name="bg+tree[vec_pool]",   .is="block_greedy", .storage=gs::vec_pool},
            {.name="bk+tree[vec_pool]",   .is="baumann_kyng", .storage=gs::vec_pool},
            {.name="root+tree[vec_pool]", .is="rootset",      .storage=gs::vec_pool},
            {.name="luby+tree[vec_pool]", .is="luby",         .storage=gs::vec_pool},
            // /hos: legacy heavy-oversample variant, kept so old --v1-configs strings still
            // resolve. It carries no extra knobs today (the oversampling levers were removed
            // from the library), so it behaves like bg+tree[vec_pool] -- which is the charted
            // headline config; /hos itself is not charted.
            {.name="bg+tree/hos[vec_pool]", .is="block_greedy", .storage=gs::vec_pool},
            // Quality levers: exact full clique at low degree (xcN), relaxed IS degree cap (capN).
            {.name="bg+tree/xc4[vec_pool]",  .is="block_greedy", .storage=gs::vec_pool, .exact_clique_max_degree=4},
            {.name="bg+tree/xc8[vec_pool]",  .is="block_greedy", .storage=gs::vec_pool, .exact_clique_max_degree=8},
            {.name="bg+tree/xc16[vec_pool]", .is="block_greedy", .storage=gs::vec_pool, .exact_clique_max_degree=16},
            {.name="bg+tree/cap4[vec_pool]", .is="block_greedy", .storage=gs::vec_pool, .degree_mult=4.0},
        };
        // Thread count: if --threads is set, use that; otherwise current OMP setting.
        int tc = args.threads;
        if (tc <= 0) {
#ifdef _OPENMP
            tc = omp_get_max_threads();
#else
            tc = 1;
#endif
        }
#ifdef _OPENMP
        omp_set_num_threads(tc);  // local re-pin in case of nested re-entry
#endif
        auto storage_tag = [](apxchol::graph_storage s) {
            switch (s) {
                case apxchol::graph_storage::vec:          return "[vec]";
                case apxchol::graph_storage::forward_star: return "[fwd_star]";
                case apxchol::graph_storage::bstr:         return "[bstr]";
                case apxchol::graph_storage::vec_pool:     return "[vec_pool]";
            }
            return "[?]";
        };
        for (const auto& combo : v1_combos) {
            // Subset filter: skip combo unless its bare name or its
            // qualified name (e.g. "bk+tree[vec]") is listed.
            if (!args.v1_configs.empty()) {
                std::string qual = std::string(combo.name) + storage_tag(combo.storage);
                if (!args.v1_configs.count(combo.name) &&
                    !args.v1_configs.count(qual)) continue;
            }
            std::string label = std::string("v1 ") + combo.name + " "
                + storage_tag(combo.storage)
                + " [" + std::to_string(tc) + "t]";
            // apxchol via run_desing. ground is fixed at native (rank-aware mean-
            // centering -- handles disconnection intrinsically); --decompose split
            // factorizes each component independently. The single-component callable
            // ignores `ground` (apxchol has only its native grounding).
            auto apx_single = [&combo, label](const Eigen::SparseMatrix<double>& L_,
                                              const Eigen::VectorXd& b_, const std::string& nm,
                                              double tl, int mi, ground_mode) {
                return run_apxchol_v1(L_, b_, nm, label, combo.is, combo.storage, tl, mi,
                                      false, combo.exact_clique_max_degree, combo.degree_mult);
            };
            print(median_run([&]() {
                return run_desing("apxchol", label, apx_single);
            }, R));
        }

#ifdef APXCHOL_USE_CUDA
        // apxchol with GPU-resident PCG (cuBLAS axpy/dot/nrm2 + cuSPARSE SpMV).
        // Preconditioner stays as-is (cuSPARSE sptrsv with H2D/D2H bounce).
        if (args.solvers.count("apxchol_gpu")) {
            for (const auto& combo : v1_combos) {
                if (!args.v1_configs.empty()) {
                    std::string qual = std::string(combo.name) + storage_tag(combo.storage);
                    if (!args.v1_configs.count(combo.name) &&
                        !args.v1_configs.count(qual)) continue;
                }
                apxchol::factor_options fopts{
                    .seed = 42, .is_select = combo.is,
                };
                if (combo.exact_clique_max_degree > 0) fopts.exact_clique_max_degree = combo.exact_clique_max_degree;
                if (combo.degree_mult > 0.0) fopts.partition.degree_multiplier = combo.degree_mult;
                std::string label = std::string(combo.name) + storage_tag(combo.storage)
                    + " /gpu_pcg";
                print(median_run([&]() {
                    return run_apxchol_gpu_pcg(L, b, graph_name, label,
                                               fopts, combo.storage,
                                               args.tol, args.maxiter);
                }, R));
            }
        }
#endif
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

    // Black-box AMG solvers route through run_desing(--desing). auto = AMGCL split,
    // BoomerAMG split; each a no-op on connected matrices (1 component). The
    // connected-grounding for AMGCL's split is `native` (default constant near-null-space).
#ifdef HAVE_AMGCL
    if (args.solvers.count("amgcl"))
        print(median_run([&]() { return run_desing("amgcl", "AMG+CG [AMGCL]",
            [](const Eigen::SparseMatrix<double>& L_, const Eigen::VectorXd& b_,
               const std::string& nm, double tl, int mi, ground_mode g) {
                return run_amgcl(L_, b_, nm, tl, mi, g); }); }, R));
#ifdef APXCHOL_USE_CUDA
    if (args.solvers.count("amgcl_cuda"))
        print(median_run([&]() { return run_desing("amgcl", "AMG+CG [AMGCL;cuda]",
            [](const Eigen::SparseMatrix<double>& L_, const Eigen::VectorXd& b_,
               const std::string& nm, double tl, int mi, ground_mode g) {
                return run_amgcl_cuda(L_, b_, nm, tl, mi, g); }); }, R));
#endif
#endif

#ifdef HAVE_HYPRE
    // BoomerAMG routes through run_desing. auto = split (A/B: ~3-5% faster than
    // whole-matrix multi-pin, both reach tol once the PCG stop is relative). The
    // connected-grounding under split is `pin` (one Dirichlet pin per component);
    // native/nullspace are n/a (Hypre has no scalar singular-nullspace API).
    if (args.solvers.count("hypre_boomeramg"))
        print(median_run([&]() { return run_desing("boomeramg", "BoomerAMG+PCG [Hypre]",
            [](const Eigen::SparseMatrix<double>& L_, const Eigen::VectorXd& b_,
               const std::string& nm, double tl, int mi, ground_mode) {
                return run_hypre_boomeramg(L_, b_, nm, tl, mi); }); }, R));

#ifdef APXCHOL_USE_CUDA
    if (args.solvers.count("hypre_boomeramg_gpu"))
        print(median_run([&]() { return run_desing("boomeramg", "BoomerAMG+PCG [Hypre;cuda]",
            [](const Eigen::SparseMatrix<double>& L_, const Eigen::VectorXd& b_,
               const std::string& nm, double tl, int mi, ground_mode) {
                return run_hypre_boomeramg_gpu(L_, b_, nm, tl, mi); }); }, R));
#endif

#endif

#ifdef HAVE_HYPRE
    HYPRE_Finalize();
#endif
    return 0;
}

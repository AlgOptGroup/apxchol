#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Sparse>
#include <fast_matrix_market/app/Eigen.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "apxchol/checkpoint.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/factorization_impl.h"
#include "apxchol/solver/solve.h"
#include "apxchol/solver/sptrsv/omp.h"
#include "apxchol/graph/incidence_list.h"
#include "apxchol/graph/conversions.h"
#include "mtx_input.h"

#if defined(APXCHOL_USE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(APXCHOL_USE_METAL)
#include "apxchol/solver/metal_solver.h"
#endif

namespace {

using apxchol::factor_options;
using apxchol::factorization;
using apxchol::graph_storage;
using apxchol::node_index;

struct cli_options {
    std::string input_path;
    graph_storage storage = graph_storage::vec_pool_aos;
    std::string is_select = "block_greedy";
    unsigned seed = 42;
    bool sweep_threads = false;
    bool profile = false;
    bool solve = false;
    bool bench_trsv = false;
    std::string backend = "cpu";   // --solve: cpu | metal
    long long columns = 1;         // --solve: compatible right-hand sides
    double tol = 1e-8;             // --solve: relative residual target
    int factor_threads = 0;        // --solve K-column path: factorization team (0 = inherit)
    double min_is_frac = 0.05;
    long long parallel_residual_threshold = -1;  // <0 = leave default (disabled)
    apxchol::residual_peel_strategy residual_peel = apxchol::residual_peel_strategy::natural;
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s <matrix.mtx> "
                 "[--graph-storage vec|bstr|vec_pool_aos]"
                 " [--is block_greedy|priority_greedy|baumann_kyng]"
                 " [--seed N]"
                 " [--min-is-frac FRACTION] [--parallel-residual-threshold N]"
                 " [--profile|--solve|--bench-trsv|--sweep-threads]"
                 " [--backend cpu|metal] [--columns K] [--tol T] [--factor-threads N]\n",
                 argv0);
    std::exit(1);
}

graph_storage parse_storage(const std::string& s) {
    if (s == "vec") return graph_storage::vec;
    if (s == "bstr") return graph_storage::bstr;
    if (s == "vec_pool_aos") return graph_storage::vec_pool_aos;
    throw std::invalid_argument("unknown graph storage: " + s);
}

std::string parse_is(const std::string& s) {
    if (s == "block_greedy" || s == "priority_greedy"
        || s == "baumann_kyng") return s;
    throw std::invalid_argument("unknown IS strategy: " + s);
}


cli_options parse_args(int argc, char* argv[]) {
    if (argc < 2) usage(argv[0]);
    if (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")
        usage(argv[0]);

    cli_options opts;
    opts.input_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after %s\n", flag);
                usage(argv[0]);
            }
            return argv[++i];
        };

        if (arg == "--graph-storage") {
            opts.storage = parse_storage(require_value("--graph-storage"));
        } else if (arg == "--is") {
            opts.is_select = parse_is(require_value("--is"));
        } else if (arg == "--seed") {
            const std::string value = require_value("--seed");
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
            if (value.empty() || *end != '\0'
                || parsed > std::numeric_limits<unsigned>::max()) {
                throw std::invalid_argument("invalid --seed: " + value);
            }
            opts.seed = static_cast<unsigned>(parsed);
        } else if (arg == "--sweep-threads") {
            opts.sweep_threads = true;
        } else if (arg == "--profile") {
            opts.profile = true;
        } else if (arg == "--solve") {
            opts.solve = true;
        } else if (arg == "--bench-trsv") {
            opts.bench_trsv = true;
        } else if (arg == "--backend") {
            opts.backend = require_value("--backend");
            if (opts.backend != "cpu" && opts.backend != "metal")
                throw std::invalid_argument("unknown --backend: " + opts.backend);
        } else if (arg == "--columns") {
            opts.columns = std::atoll(require_value("--columns").c_str());
            if (opts.columns < 1) throw std::invalid_argument("--columns must be positive");
        } else if (arg == "--factor-threads") {
            opts.factor_threads = std::atoi(require_value("--factor-threads").c_str());
            if (opts.factor_threads < 1) throw std::invalid_argument("--factor-threads must be positive");
        } else if (arg == "--tol") {
            opts.tol = std::atof(require_value("--tol").c_str());
            if (!(opts.tol > 0.0)) throw std::invalid_argument("--tol must be positive");
        } else if (arg == "--min-is-frac") {
            opts.min_is_frac = std::atof(require_value("--min-is-frac").c_str());
        } else if (arg == "--parallel-residual-threshold") {
            opts.parallel_residual_threshold =
                std::atoll(require_value("--parallel-residual-threshold").c_str());
        } else if (arg == "--residual-peel" && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "natural")    opts.residual_peel = apxchol::residual_peel_strategy::natural;
            else if (s == "min_degree") opts.residual_peel = apxchol::residual_peel_strategy::min_degree;
            else if (s == "bk_serial")  opts.residual_peel = apxchol::residual_peel_strategy::bk_serial;
            else { std::fprintf(stderr, "unknown --residual-peel: %s\n", s.c_str()); usage(argv[0]); }
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
        }
    }

    return opts;
}

struct level_stats {
    std::vector<int> depth;
    std::vector<int> level_size;
    std::vector<long long> level_nnz;
};

// A graph Laplacian has one null vector per connected component.  The usual
// globally centred random RHS is compatible only when the graph is connected;
// project component-wise for adjacency inputs, but leave connected inputs
// byte-for-byte unchanged.
void make_component_compatible(const Eigen::SparseMatrix<double>& A,
                               Eigen::VectorXd& b, bool report = true) {
    const Eigen::Index n = A.rows();
    std::vector<Eigen::Index> component(static_cast<std::size_t>(n), -1);
    std::vector<Eigen::Index> stack;
    std::vector<double> sums;
    std::vector<Eigen::Index> counts;
    for (Eigen::Index root = 0; root < n; ++root) {
        if (component[static_cast<std::size_t>(root)] >= 0) continue;
        const Eigen::Index id = static_cast<Eigen::Index>(sums.size());
        sums.push_back(0.0);
        counts.push_back(0);
        component[static_cast<std::size_t>(root)] = id;
        stack.push_back(root);
        while (!stack.empty()) {
            const Eigen::Index v = stack.back();
            stack.pop_back();
            sums[static_cast<std::size_t>(id)] += b[v];
            ++counts[static_cast<std::size_t>(id)];
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, v); it; ++it) {
                const Eigen::Index u = it.row();
                if (u == v || it.value() == 0.0
                    || component[static_cast<std::size_t>(u)] >= 0)
                    continue;
                component[static_cast<std::size_t>(u)] = id;
                stack.push_back(u);
            }
        }
    }
    if (sums.size() == 1) return;
    for (Eigen::Index v = 0; v < n; ++v) {
        const std::size_t id = static_cast<std::size_t>(
            component[static_cast<std::size_t>(v)]);
        b[v] -= sums[id] / static_cast<double>(counts[id]);
    }
    if (report) std::printf("rhs_components=%zu (projected component-wise)\n", sums.size());
}

// FNV-1a over raw bytes: digests of the factor (structure and values) and of
// the solution block, so repeated runs can be compared for bit identity.
std::uint64_t fnv1a(const void* data, std::size_t bytes, std::uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

std::uint64_t factor_digest(const factorization& F) {
    std::uint64_t h = fnv1a(F.L.outer_.data(), F.L.outer_.size() * sizeof(F.L.outer_[0]));
    h = fnv1a(F.L.inner_.data(), F.L.inner_.size() * sizeof(F.L.inner_[0]), h);
    h = fnv1a(F.L.vals_.data(), F.L.vals_.size() * sizeof(F.L.vals_[0]), h);
    return fnv1a(F.perm.data(), F.perm.size() * sizeof(F.perm[0]), h);
}

level_stats analyze_forward_levels(const Eigen::SparseMatrix<double>& L11) {
    const node_index m = static_cast<node_index>(L11.rows());
    level_stats stats;
    stats.depth.assign(m, 0);

    int max_depth = 0;
    for (node_index j = 0; j < m; ++j) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it) {
            const node_index i = static_cast<node_index>(it.row());
            if (i > j)
                stats.depth[i] = std::max(stats.depth[i], stats.depth[j] + 1);
        }
        max_depth = std::max(max_depth, stats.depth[j]);
    }

    stats.level_size.assign(max_depth + 1, 0);
    stats.level_nnz.assign(max_depth + 1, 0);
    for (node_index i = 0; i < m; ++i)
        stats.level_size[stats.depth[i]]++;
    for (node_index j = 0; j < m; ++j) {
        int col_nnz = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it)
            ++col_nnz;
        stats.level_nnz[stats.depth[j]] += col_nnz;
    }
    return stats;
}

// Build the top-left m×m block of an apxchol::sparse_csc factor as an
// Eigen::SparseMatrix<double> (analysis-only; replaces the old Eigen
// L.topLeftCorner(m,m) now that the factor owns its own CSC storage).
static Eigen::SparseMatrix<double> eigen_topleft(const apxchol::sparse_csc& L,
                                                 long long m) {
    const auto* outer = L.outerIndexPtr();
    const auto* inner = L.innerIndexPtr();
    const auto* vals  = L.valuePtr();
    std::vector<Eigen::Triplet<double>> trips;
    for (long long c = 0; c < m; ++c)
        for (apxchol::edge_index p = outer[c]; p < outer[c + 1]; ++p)
            if (static_cast<long long>(inner[p]) < m)
                trips.emplace_back(static_cast<int>(inner[p]),
                                   static_cast<int>(c), apxchol::widen(vals[p]));
    Eigen::SparseMatrix<double> M(static_cast<int>(m), static_cast<int>(m));
    M.setFromTriplets(trips.begin(), trips.end());
    M.makeCompressed();
    return M;
}

level_stats analyze_backward_levels(const Eigen::SparseMatrix<double>& L11) {
    const node_index m = static_cast<node_index>(L11.rows());
    level_stats stats;
    stats.depth.assign(m, 0);

    int max_depth = 0;
    for (node_index j = m; j-- > 0; ) {   // reverse: m-1 .. 0 (unsigned-safe)
        int d = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it) {
            const node_index row = static_cast<node_index>(it.row());
            if (row > j)
                d = std::max(d, stats.depth[row] + 1);
        }
        stats.depth[j] = d;
        max_depth = std::max(max_depth, d);
    }

    stats.level_size.assign(max_depth + 1, 0);
    stats.level_nnz.assign(max_depth + 1, 0);
    for (node_index j = 0; j < m; ++j) {
        stats.level_size[stats.depth[j]]++;
        int dep_nnz = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it) {
            if (it.row() > j)
                ++dep_nnz;
        }
        stats.level_nnz[stats.depth[j]] += dep_nnz;
    }
    return stats;
}

void print_level_summary(const char* label, const level_stats& stats, node_index m) {
    const int levels = static_cast<int>(stats.level_size.size());
    const int max_size = *std::max_element(stats.level_size.begin(), stats.level_size.end());
    const long long max_work = *std::max_element(stats.level_nnz.begin(), stats.level_nnz.end());
    const long long total_work =
        std::accumulate(stats.level_nnz.begin(), stats.level_nnz.end(), 0LL);
    const int singleton_levels =
        static_cast<int>(std::count(stats.level_size.begin(), stats.level_size.end(), 1));

    std::printf("%s levels: %d\n", label, levels);
    std::printf("%s max level size: %d (%.2f%% of rows)\n",
                label, max_size, 100.0 * max_size / std::max<node_index>(m, 1));
    std::printf("%s avg level size: %.2f\n",
                label, levels ? static_cast<double>(m) / levels : 0.0);
    std::printf("%s singleton levels: %d (%.2f%%)\n",
                label, singleton_levels, levels ? 100.0 * singleton_levels / levels : 0.0);
    std::printf("%s max per-level work: %lld (%.2f%% of total dependency nnz)\n",
                label, max_work, total_work ? 100.0 * max_work / total_work : 0.0);

    std::printf("%s first levels:\n", label);
    for (int l = 0; l < std::min(levels, 8); ++l) {
        std::printf("  L%-3d rows=%-8d work=%lld\n",
                    l, stats.level_size[l], stats.level_nnz[l]);
    }
    if (levels > 8) {
        std::printf("%s last levels:\n", label);
        for (int l = std::max(8, levels - 5); l < levels; ++l) {
            std::printf("  L%-3d rows=%-8d work=%lld\n",
                        l, stats.level_size[l], stats.level_nnz[l]);
        }
    }
}

const char* storage_name(graph_storage s) {
    switch (s) {
    case graph_storage::vec: return "vec";
    case graph_storage::bstr: return "bstr";
    case graph_storage::vec_pool_aos: return "vec_pool_aos";
    }
    return "unknown";
}

const char* is_name(const std::string& s) {
    return s.c_str();
}


} // namespace

int main(int argc, char* argv[]) {
    try {
        const cli_options cli = parse_args(argc, argv);

        Eigen::SparseMatrix<double> A;
        fast_matrix_market::matrix_market_header hdr;
        {
            std::ifstream f(cli.input_path);
            fast_matrix_market::read_matrix_market_eigen(f, hdr, A);
        }
        if (A.rows() != A.cols())
            throw std::runtime_error(
                "matrix must be square, got " + std::to_string(A.rows()) +
                "x" + std::to_string(A.cols()));
        // Same trap as the CLI: an adjacency/pattern .mtx handed straight to
        // factorize() gives negative edge weights and a fill-free factor, so
        // every number this tool prints would be about a graph that isn't
        // there. See src/mtx_input.h.
        apxchol::input_kind resolved_kind = apxchol::input_kind::automatic;
        apxchol::input_scan input_facts;
        {
            input_facts = apxchol::scan_input(A);
            std::string reason;
            resolved_kind = apxchol::resolve_input_kind(
                apxchol::input_kind::automatic, input_facts,
                hdr.field == fast_matrix_market::pattern, reason);
            std::printf("%s\n",
                        apxchol::describe_input(
                            resolved_kind, input_facts, reason).c_str());
            if (resolved_kind == apxchol::input_kind::adjacency)
                apxchol::adjacency_to_laplacian(A);
        }

        factor_options opts;
        opts.seed = cli.seed;
        opts.is_select = cli.is_select;
        opts.min_is_fraction = cli.min_is_frac;
        if (cli.parallel_residual_threshold >= 0)
            opts.parallel_residual_threshold = static_cast<size_t>(cli.parallel_residual_threshold);
        opts.residual_peel = cli.residual_peel;

#if defined(APXCHOL_USE_CUDA)
        // Analysis-only fairness: this CUDA build installs and benchmarks the
        // GPU solve backend regardless of which setup selector is chosen. The
        // benchmark driver prewarms the one-per-process CUDA context before
        // timing every solver, so do the same for every arm here.
        if (const cudaError_t err = cudaFree(nullptr); err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
#endif

        // ── Warm-context one-RHS benchmark mode ─────────────
        // The CUDA context was initialized above, outside all checkpointed
        // solver work, exactly as in the standalone benchmark driver. This
        // gives setup/solve measurements without pulling the competitor stack
        // into this small diagnostic executable.
        if (cli.solve) {
            std::srand(opts.seed);
            Eigen::VectorXd b = apxchol::generate_test_rhs(A.rows());
            // An assembled pure Laplacian can be disconnected too.  Project
            // it just like an adjacency input; leave SDDM/SPD operators alone.
            const bool project = resolved_kind == apxchol::input_kind::adjacency ||
                (input_facts.excess_rows == 0 && input_facts.deficient_rows == 0);
            if (project) make_component_compatible(A, b);
            apxchol::solve_options solve_opts;
            solve_opts.tol = cli.tol;
            solve_opts.max_iter = 500;
            solve_opts.storage = cli.storage;
            solve_opts.factor_opts = opts;
            if (cli.backend != "cpu" || cli.columns != 1) {
                // K right-hand sides: column 0 is the one-RHS b above, the
                // others continue the same random stream, each projected the
                // same way. Residuals are recomputed here in fp64 with Eigen.
                const Eigen::Index n = A.rows();
                const Eigen::Index K = static_cast<Eigen::Index>(cli.columns);
                Eigen::MatrixXd B(n, K), X(n, K);
                B.col(0) = b;
                for (Eigen::Index c = 1; c < K; ++c) {
                    Eigen::VectorXd bc = apxchol::generate_test_rhs(n);
                    if (project) make_component_compatible(A, bc, false);
                    B.col(c) = bc;
                }
                solve_opts.keep_factor_values = true;  // for the factor digest
                std::vector<long long> iterations(static_cast<std::size_t>(K));
                apxchol::checkpoint cp;
                double solve_ms = 0.0;
                std::uint64_t fdigest = 0;
                using clock = std::chrono::steady_clock;
                // The factor is built first, optionally on its own team, so that
                // runs at different OMP_NUM_THREADS can share one factor (factor
                // identity is checked through factor_digest).
                factorization F;
                {
#ifdef _OPENMP
                    const int team = omp_get_max_threads();
                    if (cli.factor_threads > 0) omp_set_num_threads(cli.factor_threads);
#endif
                    F = apxchol::factorize(A, cli.storage, opts, &cp);
#ifdef _OPENMP
                    omp_set_num_threads(team);
#endif
                }
                if (cli.backend == "cpu") {
                    const apxchol::cpu_solver slv(A, std::move(F), solve_opts, &cp);
                    fdigest = factor_digest(slv.preconditioner().factor());
                    const auto t0 = clock::now();
                    for (Eigen::Index c = 0; c < K; ++c) {
                        const Eigen::VectorXd bc = B.col(c);
                        const apxchol::solve_result r = slv.solve(bc);
                        X.col(c) = r.x;
                        iterations[static_cast<std::size_t>(c)] = static_cast<long long>(r.iterations);
                    }
                    solve_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
                } else {
#if defined(APXCHOL_USE_METAL)
                    if (!apxchol::metal_solver::available())
                        throw std::runtime_error("--backend metal: no usable Metal device");
                    const apxchol::metal_solver slv(A, std::move(F), solve_opts, &cp);
                    fdigest = factor_digest(slv.factor());
                    const auto st = slv.stats();
                    std::printf("metal device=\"%s\" block_columns=%d levels_fwd=%zu levels_bwd=%zu "
                                "steps_fwd=%zu steps_bwd=%zu operator=%s\n",
                                st.device.c_str(), st.block_columns, st.levels_forward,
                                st.levels_backward, st.steps_forward, st.steps_backward,
                                st.operator_double_float ? "double-float" : "fp32-exact");
                    const auto t0 = clock::now();
                    const apxchol::metal_block_result r =
                        slv.solve(apxchol::metal_solver::block_cref(B), X);
                    solve_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
                    for (Eigen::Index c = 0; c < K; ++c)
                        iterations[static_cast<std::size_t>(c)] =
                            static_cast<long long>(r.columns[static_cast<std::size_t>(c)].iterations);
#else
                    throw std::runtime_error("--backend metal: this build has no Metal backend "
                                             "(configure with -DAPXCHOL_USE_METAL=ON)");
#endif
                }
                long long converged = 0;
                double max_residual = 0.0;
                for (Eigen::Index c = 0; c < K; ++c) {
                    const double res = (B.col(c) - A * X.col(c)).norm() / B.col(c).norm();
                    converged += res < solve_opts.tol;
                    max_residual = std::max(max_residual, res);
                }
                const auto [it_min, it_max] = std::minmax_element(iterations.begin(), iterations.end());
                std::printf(
                    "block_result backend=%s columns=%lld tol=%g setup_ms=%.6f solve_ms=%.6f "
                    "iterations_min=%lld iterations_max=%lld converged=%lld/%lld "
                    "max_residual=%.17g factor_digest=%016llx solution_hash=%016llx\n",
                    cli.backend.c_str(), static_cast<long long>(K), solve_opts.tol,
                    cp.total("setup") * 1e3, solve_ms, *it_min, *it_max, converged,
                    static_cast<long long>(K), max_residual,
                    static_cast<unsigned long long>(fdigest),
                    static_cast<unsigned long long>(
                        fnv1a(X.data(), static_cast<std::size_t>(X.size()) * sizeof(double))));
                return converged == K ? 0 : 1;
            }
            const auto result = apxchol::solve(A, b, solve_opts);
            const double setup_ms = result.timings.total("setup") * 1e3;
            const double pcg_ms = result.timings.total("pcg") * 1e3;
            std::printf(
                "solve_result setup_ms=%.6f pcg_ms=%.6f total_ms=%.6f "
                "iterations=%lld residual=%.17g vram_mb=%.3f\n",
                setup_ms, pcg_ms, setup_ms + pcg_ms,
                static_cast<long long>(result.iterations), result.residual,
                result.solve_vram_mb);
            return result.residual < solve_opts.tol ? 0 : 1;
        }

        // ── Thread scaling sweep mode ────────────────────────
        if (cli.sweep_threads) {
            std::printf("matrix: %s\n", cli.input_path.c_str());
            std::printf("storage=%s is=%s elimination=%s\n",
                        storage_name(cli.storage), is_name(cli.is_select),
                        "tree");
            std::printf("A: n=%d nnz=%lld\n",
                        static_cast<int>(A.rows()),
                        static_cast<long long>(A.nonZeros()));
            std::printf("\n%-7s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
                        "thr", "find_is", "merge_is", "compute", "apply",
                        "elim", "elim_rem", "assembly",
                        "factor", "iters");
            std::printf("%s\n", std::string(110, '-').c_str());

            for (int t : {1, 2, 4, 8, 16, 32}) {
#ifdef _OPENMP
                omp_set_num_threads(t);
#endif
                apxchol::checkpoint cp;
                auto F = apxchol::factorize(A, cli.storage, opts, &cp);
                double find_is_ms  = cp.total("setup.find_partition")    * 1000;
                double merge_is_ms = cp.total("setup.eliminate.merge_is") * 1000;
                double compute_ms  = cp.total("setup.eliminate.compute") * 1000;
                double apply_ms    = cp.total("setup.eliminate.apply")   * 1000;
                double elim_ms     = cp.total("setup.eliminate")         * 1000;
                double elim_rem_ms = cp.total("setup.elim_remaining")    * 1000;
                double asm_ms      = cp.total("setup.assembly")          * 1000;
                double factor_ms   = cp.total("setup")                   * 1000;
                std::printf("%-7d %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10zu\n",
                            t, find_is_ms, merge_is_ms, compute_ms, apply_ms,
                            elim_ms, elim_rem_ms, asm_ms, factor_ms,
                            F.rounds.size());
            }
            return 0;
        }

        // ── Profile-only mode ────────────────────────────────
        if (cli.profile) {
            std::printf("matrix: %s\n", cli.input_path.c_str());
            std::printf("storage=%s is=%s elimination=%s\n",
                        storage_name(cli.storage), is_name(cli.is_select),
                        "tree");
            apxchol::checkpoint cp;
            auto F = apxchol::factorize(A, cli.storage, opts, &cp);
            std::cout << cp.report() << '\n';
            double avg_is = 0, avg_deg = 0;
            if (!F.rounds.empty()) {
                for (auto& r : F.rounds) { avg_is += r.is_size; avg_deg += r.avg_deg; }
                avg_is /= F.rounds.size();
                avg_deg /= F.rounds.size();
            }
            std::printf("rounds=%zu avg_is=%.1f avg_deg=%.1f nnz(L)=%lld\n",
                        F.rounds.size(), avg_is, avg_deg,
                        static_cast<long long>(F.L.nonZeros()));
            return 0;
        }

        // ── Triangular-solver microbenchmark ─────────────────
        if (cli.bench_trsv) {
            using Clock = std::chrono::high_resolution_clock;
            auto F = apxchol::factorize(A, cli.storage, opts);
            const Eigen::Index m = F.sddm ? A.rows() : A.rows() - 1;
            apxchol::omp_sptrsv trsv;
            trsv.setup(F.L, m);
            const int reps = 20;
            std::vector<double> rhs(F.L.rows(), 1.0), tmp(F.L.rows()), out(F.L.rows());

            std::printf("matrix: %s\n", cli.input_path.c_str());
            std::printf("is=%s  fwd_levels=%d  bck_levels=%d\n",
                        is_name(cli.is_select),
                        trsv.num_fwd_levels(), trsv.num_bck_levels());

            // L11 size and off-diagonal nnz, plus the column-length distribution.
            {
                Eigen::SparseMatrix<double> L11 = eigen_topleft(F.L, m);
                long long nnz = L11.nonZeros();
                long long off_diag = nnz - m;
                std::printf("L11: m=%lld nnz=%lld off_diag=%lld\n",
                            (long long)m, nnz, off_diag);
                // Column-length histogram.
                std::vector<long long> col_len(m);
                long long max_col = 0, sum_col = 0;
                for (Eigen::Index j = 0; j < m; ++j) {
                    long long c = L11.outerIndexPtr()[j+1] - L11.outerIndexPtr()[j] - 1;
                    col_len[j] = c; sum_col += c; if (c > max_col) max_col = c;
                }
                std::sort(col_len.begin(), col_len.end());
                std::printf("col_len: max=%lld p50=%lld p90=%lld p99=%lld mean=%.1f\n",
                            max_col,
                            col_len[m*50/100], col_len[m*90/100], col_len[m*99/100],
                            (double)sum_col / m);
            }

            // Per-direction level work distribution.
            for (bool fwd : {true, false}) {
                std::vector<int> sizes;
                std::vector<long long> work;
                trsv.level_stats(fwd, sizes, work);
                long long total_w = std::accumulate(work.begin(), work.end(), 0LL);
                int max_sz = sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end());
                long long max_w = work.empty() ? 0 : *std::max_element(work.begin(), work.end());
                int big = 0;            // levels with size > kSpTRSVOMPThreshold
                long long big_w = 0;
                for (size_t l = 0; l < sizes.size(); ++l)
                    if (sizes[l] > 1024) { ++big; big_w += work[l]; }
                std::printf("%s_levels: count=%zu total_work=%lld max_sz=%d max_w=%lld "
                            "big_levels=%d big_work_frac=%.3f\n",
                            fwd ? "fwd" : "bck", sizes.size(), total_w, max_sz, max_w,
                            big, total_w ? (double)big_w / total_w : 0.0);
            }
            std::printf("\n%-7s %14s %14s\n",
                        "thr", "fwd_levelset", "bck_levelset");
            std::printf("%s\n", std::string(40, '-').c_str());

            for (int t : {1, 2, 4, 8, 16, 32}) {
#ifdef _OPENMP
                omp_set_num_threads(t);
#endif
                // Warm up.
                trsv.forward_solve(rhs.data(), tmp.data());
                trsv.transpose_solve(tmp.data(), out.data());

                auto bench = [&](auto solve) {
                    auto t0 = Clock::now();
                    for (int r = 0; r < reps; ++r)
                        solve(rhs.data(), tmp.data());
                    auto t1 = Clock::now();
                    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
                };
                double fl = bench([&](const double* a, double* b) {
                    trsv.forward_solve(a, b);
                });
                double bl = bench([&](const double* a, double* b) {
                    trsv.transpose_solve(a, b);
                });
                std::printf("%-7d %14.3f %14.3f\n", t, fl, bl);
            }
            return 0;
        }

        factorization F = apxchol::factorize(A, cli.storage, opts);
        const node_index m = static_cast<node_index>(F.sddm ? A.rows() : A.rows() - 1);
        Eigen::SparseMatrix<double> L11 = eigen_topleft(F.L, m);

        std::printf("matrix: %s\n", cli.input_path.c_str());
        std::printf("storage=%s is=%s elimination=%s\n",
                    storage_name(cli.storage), is_name(cli.is_select),
                    "tree");
        std::printf("A: n=%d nnz=%lld sddm=%d\n",
                    static_cast<int>(A.rows()), static_cast<long long>(A.nonZeros()),
                    F.sddm ? 1 : 0);
        std::printf("factor: dim=%d nnz(L11)=%lld peak_graph_bytes=%zu\n",
                    static_cast<int>(m), static_cast<long long>(L11.nonZeros()),
                    F.peak_graph_bytes);
        if (!F.rounds.empty()) {
            const double avg_is = std::accumulate(
                F.rounds.begin(), F.rounds.end(), 0.0,
                [](double acc, const factorization::round_stats& r) {
                    return acc + static_cast<double>(r.is_size);
                }) / F.rounds.size();
            const double avg_deg = std::accumulate(
                F.rounds.begin(), F.rounds.end(), 0.0,
                [](double acc, const factorization::round_stats& r) {
                    return acc + r.avg_deg;
                }) / F.rounds.size();
            std::printf("rounds=%zu avg_is=%.2f avg_deg=%.2f\n",
                        F.rounds.size(), avg_is, avg_deg);
        }

        print_level_summary("forward", analyze_forward_levels(L11), m);
        print_level_summary("backward", analyze_backward_levels(L11), m);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}

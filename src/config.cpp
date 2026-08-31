#include "config.h"
#include "apxchol/version.h"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstdlib>
#include <map>

namespace apxchol {

static const std::map<std::string, input_kind> input_kind_map = {
    {"auto",      input_kind::automatic},
    {"laplacian", input_kind::laplacian},
    {"adjacency", input_kind::adjacency},
};

static const std::map<std::string, graph_storage> graph_storage_map = {
    {"vec",          graph_storage::vec},
    {"forward_star", graph_storage::forward_star},
    {"bstr",         graph_storage::bstr},
    {"vec_pool",     graph_storage::vec_pool},
    {"vec_pool_aos", graph_storage::vec_pool_aos},
};

static void setup_logging(bool quiet, bool verbose) {
    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
        spdlog::set_pattern("[%l] %v");
    } else if (quiet) {
        spdlog::set_level(spdlog::level::err);
        spdlog::set_pattern("%v");
    } else {
        spdlog::set_level(spdlog::level::info);
        spdlog::set_pattern("%v");
    }
}

run_config parse_args(int argc, char* argv[]) {
    run_config cfg;
    bool quiet   = false;
    bool verbose = false;
    std::string rhs_str, output_str;

    CLI::App app{"Solve a Laplacian or SDDM system Ax = b via approximate Cholesky + PCG. "
                 "The input .mtx may be an assembled Laplacian/SDDM operator or a graph "
                 "adjacency/pattern matrix; which one is auto-detected and reported (see "
                 "--input-kind), and Laplacian vs SDDM is auto-detected within the solver"};

    // ── version ──
    app.add_flag_callback("-V,--version", []() {
        std::printf("apxchol %s (commit %s)\n"
                    "  index widths: %d-bit vertex / %d-bit edge (unsigned)\n"
                    "  SpTRSV backend: %s\n",
                    APXCHOL_VERSION, APXCHOL_GIT_SHA,
                    APXCHOL_NODE_INDEX_BITS, APXCHOL_EDGE_INDEX_BITS,
                    APXCHOL_SPTRSV_BACKEND);
        std::exit(0);
    }, "Print version and build info");

    // ── positional ──
    app.add_option("input", cfg.input_path, "Input graph in MatrixMarket format")
        ->required()
        ->check(CLI::ExistingFile);

    // ── RHS source (exactly one of --rhs / --random-rhs) ──
    // Grouped with require_option(1) so --help states that one is required;
    // the excludes keep the "not both" error message specific.
    auto* rhs_group = app.add_option_group("Right-hand side (exactly one required)");
    auto* rhs_opt = rhs_group->add_option("--rhs", rhs_str,
                   "RHS vector in MatrixMarket format")
        ->check(CLI::ExistingFile);
    auto* rand_flag = rhs_group->add_flag("--random-rhs", cfg.random_rhs,
                 "Solve against a random zero-mean unit RHS (uses --seed)");
    rhs_opt->excludes(rand_flag);
    rand_flag->excludes(rhs_opt);
    rhs_group->require_option(1);

    // ── how to read the input matrix ──
    std::string input_kind_str = "auto";
    app.add_option("--input-kind", input_kind_str,
                   "How to read the input file: 'laplacian' = an already-assembled "
                   "Laplacian/SDDM operator, solved as given; 'adjacency' = a graph "
                   "adjacency/pattern matrix, from which L = D - A is assembled "
                   "(edge weight |value|, self-loops dropped); 'auto' = decide from "
                   "the off-diagonal signs and the diagonal, and refuse if ambiguous")
        ->capture_default_str()
        ->check(CLI::IsMember({"auto", "laplacian", "adjacency"}));

    // ── output ──
    app.add_option("-o,--output", output_str,
                   "Write solution vector to MatrixMarket file");

    // ── solver parameters ──
    app.add_option("--tol", cfg.solve_opts.tol, "Relative residual tolerance")
        ->capture_default_str();
    app.add_option("--maxiter", cfg.solve_opts.max_iter, "Maximum PCG iterations")
        ->capture_default_str();
    app.add_option("--seed", cfg.solve_opts.factor_opts.seed,
                   "Random seed for factorization and test RHS")
        ->capture_default_str();
    app.add_option("--clique-sampler",
                   cfg.solve_opts.factor_opts.clique_sampler,
                   "Clique sampler: gks (default) or bkz26 (BKZ26 Algorithm 1 "
                   "weighted-Pruefer clique sampler embedded in apxchol)")
        ->capture_default_str()
        ->check(CLI::IsMember({"gks", "bkz26"}));
    app.add_option("--degree-quantile", cfg.solve_opts.factor_opts.partition.degree_quantile,
                   "IS degree cap as a quantile of current degrees in (0,1); "
                   "0 = use --degree-multiplier")
        ->capture_default_str();
    app.add_option("--degree-multiplier", cfg.solve_opts.factor_opts.partition.degree_multiplier,
                   "IS degree threshold = multiplier × avg_degree "
                   "(fallback, used only when --degree-quantile is 0)")
        ->capture_default_str();
    app.add_option("--omp-threshold", cfg.solve_opts.factor_opts.omp_threshold,
                   "Min active vertices before engaging OpenMP")
        ->capture_default_str();
    app.add_option("--stagnation-window", cfg.solve_opts.stagnation_window,
                   "Check convergence every N iters, stop if <50% improvement (0=disable)")
        ->capture_default_str();

    std::string graph_storage_str = "vec_pool_aos";
    app.add_option("--graph-storage", graph_storage_str,
                   "Graph storage backend (vec_pool, vec_pool_aos, forward_star, vec, bstr)")
        ->capture_default_str()
        ->check(CLI::IsMember({"vec_pool", "vec_pool_aos", "forward_star", "vec", "bstr"}));

    app.add_option("--is", cfg.solve_opts.factor_opts.is_select,
                   "Independent set strategy (block_greedy, priority_greedy, baumann_kyng)")
        ->capture_default_str()
        ->check(CLI::IsMember(
            {"block_greedy", "priority_greedy", "baumann_kyng"}));

    app.add_option("--fs-compact", cfg.solve_opts.factor_opts.fs_compact_threshold,
                   "forward_star auto-compact threshold (live_fraction; 0=off)")
        ->capture_default_str();

    app.add_flag("--fs-filter-append,!--no-fs-filter-append",
                 cfg.solve_opts.factor_opts.fs_filter_append,
                 "forward_star: append survivors at pool end on filter (off by default)");

    // ── verbosity (mutually exclusive) ──
    auto* q_flag = app.add_flag("-q,--quiet", quiet, "Suppress non-error output");
    auto* v_flag = app.add_flag("-v,--verbose", verbose, "Verbose output");
    q_flag->excludes(v_flag);
    v_flag->excludes(q_flag);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    if (!rhs_str.empty())    cfg.rhs_path     = rhs_str;
    if (!output_str.empty()) cfg.output_path  = output_str;
    cfg.input = input_kind_map.at(input_kind_str);
    cfg.solve_opts.storage = graph_storage_map.at(graph_storage_str);
    setup_logging(quiet, verbose);
    return cfg;
}

} // namespace apxchol

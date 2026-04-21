#include "config.h"
#include "apxchol/version.h"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstdlib>
#include <map>

namespace apxchol {

static const std::map<std::string, graph_storage> graph_storage_map = {
    {"vec",          graph_storage::vec},
    {"forward_star", graph_storage::forward_star},
    {"small_vec",    graph_storage::small_vec},
    {"bstr",         graph_storage::bstr},
};

static const std::map<std::string, is_strategy> is_strategy_map = {
    {"block_greedy",  is_strategy::block_greedy},
    {"luby",          is_strategy::luby},
    {"baumann_kyng",  is_strategy::baumann_kyng},
    {"rootset",       is_strategy::rootset},
    {"hybrid",        is_strategy::hybrid},
};

static const std::map<std::string, elimination_strategy> elimination_strategy_map = {
    {"tree",    elimination_strategy::tree},
    {"star",    elimination_strategy::star},
    {"clique",  elimination_strategy::clique},
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

    CLI::App app{"Solve a Laplacian system Lx = b via approximate Cholesky + PCG"};

    // ── version ──
    app.add_flag_callback("-V,--version", []() {
        std::printf("apxchol %s (commit %s)\n"
                    "  index type: %d-bit signed\n"
                    "  SpTRSV backend: %s\n",
                    APXCHOL_VERSION, APXCHOL_GIT_SHA,
                    APXCHOL_INDEX_BITS, APXCHOL_SPTRSV_BACKEND);
        std::exit(0);
    }, "Print version and build info");

    // ── positional ──
    app.add_option("input", cfg.input_path, "Input graph in MatrixMarket format")
        ->required()
        ->check(CLI::ExistingFile);

    // ── RHS source (optional; test RHS generated when omitted) ──
    app.add_option("--rhs", rhs_str,
                   "RHS vector in MatrixMarket format (default: generate random test RHS)")
        ->check(CLI::ExistingFile);

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
    app.add_option("--degree-multiplier", cfg.solve_opts.factor_opts.degree_multiplier,
                   "IS degree threshold = multiplier × avg_degree")
        ->capture_default_str();
    app.add_option("--omp-threshold", cfg.solve_opts.factor_opts.omp_threshold,
                   "Min active vertices before engaging OpenMP")
        ->capture_default_str();
    app.add_option("--stagnation-window", cfg.solve_opts.stagnation_window,
                   "Check convergence every N iters, stop if <50% improvement (0=disable)")
        ->capture_default_str();

    std::string graph_storage_str = "vec";
    app.add_option("--graph-storage", graph_storage_str,
                   "Graph storage backend (vec, forward_star, small_vec, bstr)")
        ->capture_default_str()
        ->check(CLI::IsMember({"vec", "forward_star", "small_vec", "bstr"}));

    std::string is_strategy_str = "block_greedy";
    app.add_option("--is", is_strategy_str,
                   "Independent set strategy (block_greedy, luby, baumann_kyng, rootset)")
        ->capture_default_str()
        ->check(CLI::IsMember({"block_greedy", "luby", "baumann_kyng", "rootset"}));

    std::string elim_strategy_str = "tree";
    app.add_option("--elimination", elim_strategy_str,
                   "Elimination strategy (tree, star, clique)")
        ->capture_default_str()
        ->check(CLI::IsMember({"tree", "star", "clique"}));

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
    cfg.solve_opts.storage = graph_storage_map.at(graph_storage_str);
    cfg.solve_opts.factor_opts.is_select = is_strategy_map.at(is_strategy_str);
    cfg.solve_opts.factor_opts.elim = elimination_strategy_map.at(elim_strategy_str);

    setup_logging(quiet, verbose);
    return cfg;
}

} // namespace apxchol

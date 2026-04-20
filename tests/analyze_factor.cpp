#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Sparse>
#include <fast_matrix_market/app/Eigen.hpp>

#include "apxchol/solver/factorization.h"

namespace {

using apxchol::elimination_strategy;
using apxchol::factor_options;
using apxchol::factorization;
using apxchol::graph_storage;
using apxchol::index_t;
using apxchol::is_strategy;

struct cli_options {
    std::string input_path;
    graph_storage storage = graph_storage::forward_star;
    is_strategy is_select = is_strategy::block_greedy;
    elimination_strategy elimination = elimination_strategy::tree;
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s <matrix.mtx> [--graph-storage vec|forward_star|small_vec]"
                 " [--is block_greedy|luby|baumann_kyng|rootset]"
                 " [--elimination tree|star|clique]\n",
                 argv0);
    std::exit(1);
}

graph_storage parse_storage(const std::string& s) {
    if (s == "vec") return graph_storage::vec;
    if (s == "forward_star") return graph_storage::forward_star;
    if (s == "small_vec") return graph_storage::small_vec;
    throw std::invalid_argument("unknown graph storage: " + s);
}

is_strategy parse_is(const std::string& s) {
    if (s == "block_greedy") return is_strategy::block_greedy;
    if (s == "luby") return is_strategy::luby;
    if (s == "baumann_kyng") return is_strategy::baumann_kyng;
    if (s == "rootset") return is_strategy::rootset;
    throw std::invalid_argument("unknown IS strategy: " + s);
}

elimination_strategy parse_elimination(const std::string& s) {
    if (s == "tree") return elimination_strategy::tree;
    if (s == "star") return elimination_strategy::star;
    if (s == "clique") return elimination_strategy::clique;
    throw std::invalid_argument("unknown elimination strategy: " + s);
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
        } else if (arg == "--elimination") {
            opts.elimination = parse_elimination(require_value("--elimination"));
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

level_stats analyze_forward_levels(const Eigen::SparseMatrix<double>& L11) {
    const index_t m = static_cast<index_t>(L11.rows());
    level_stats stats;
    stats.depth.assign(m, 0);

    int max_depth = 0;
    for (index_t j = 0; j < m; ++j) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it) {
            const index_t i = static_cast<index_t>(it.row());
            if (i > j)
                stats.depth[i] = std::max(stats.depth[i], stats.depth[j] + 1);
        }
        max_depth = std::max(max_depth, stats.depth[j]);
    }

    stats.level_size.assign(max_depth + 1, 0);
    stats.level_nnz.assign(max_depth + 1, 0);
    for (index_t i = 0; i < m; ++i)
        stats.level_size[stats.depth[i]]++;
    for (index_t j = 0; j < m; ++j) {
        int col_nnz = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it)
            ++col_nnz;
        stats.level_nnz[stats.depth[j]] += col_nnz;
    }
    return stats;
}

level_stats analyze_backward_levels(const Eigen::SparseMatrix<double>& L11) {
    const index_t m = static_cast<index_t>(L11.rows());
    level_stats stats;
    stats.depth.assign(m, 0);

    int max_depth = 0;
    for (index_t j = m - 1; j >= 0; --j) {
        int d = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it) {
            const index_t row = static_cast<index_t>(it.row());
            if (row > j)
                d = std::max(d, stats.depth[row] + 1);
        }
        stats.depth[j] = d;
        max_depth = std::max(max_depth, d);
    }

    stats.level_size.assign(max_depth + 1, 0);
    stats.level_nnz.assign(max_depth + 1, 0);
    for (index_t j = 0; j < m; ++j) {
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

void print_level_summary(const char* label, const level_stats& stats, index_t m) {
    const int levels = static_cast<int>(stats.level_size.size());
    const int max_size = *std::max_element(stats.level_size.begin(), stats.level_size.end());
    const long long max_work = *std::max_element(stats.level_nnz.begin(), stats.level_nnz.end());
    const long long total_work =
        std::accumulate(stats.level_nnz.begin(), stats.level_nnz.end(), 0LL);
    const int singleton_levels =
        static_cast<int>(std::count(stats.level_size.begin(), stats.level_size.end(), 1));

    std::printf("%s levels: %d\n", label, levels);
    std::printf("%s max level size: %d (%.2f%% of rows)\n",
                label, max_size, 100.0 * max_size / std::max<index_t>(m, 1));
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
    case graph_storage::forward_star: return "forward_star";
    case graph_storage::small_vec: return "small_vec";
    default: return "unknown";
    }
}

const char* is_name(is_strategy s) {
    switch (s) {
    case is_strategy::block_greedy: return "block_greedy";
    case is_strategy::luby: return "luby";
    case is_strategy::baumann_kyng: return "baumann_kyng";
    case is_strategy::rootset: return "rootset";
    default: return "unknown";
    }
}

const char* elimination_name(elimination_strategy s) {
    switch (s) {
    case elimination_strategy::tree: return "tree";
    case elimination_strategy::star: return "star";
    case elimination_strategy::clique: return "clique";
    default: return "unknown";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const cli_options cli = parse_args(argc, argv);

        Eigen::SparseMatrix<double> A;
        {
            std::ifstream f(cli.input_path);
            fast_matrix_market::read_matrix_market_eigen(f, A);
        }

        factor_options opts;
        opts.is_select = cli.is_select;
        opts.elim = cli.elimination;

        factorization F = apxchol::factorize(A, cli.storage, opts);
        const index_t m = static_cast<index_t>(F.sddm ? A.rows() : A.rows() - 1);
        Eigen::SparseMatrix<double> L11 = F.L.topLeftCorner(m, m);
        L11.makeCompressed();

        std::printf("matrix: %s\n", cli.input_path.c_str());
        std::printf("storage=%s is=%s elimination=%s\n",
                    storage_name(cli.storage), is_name(cli.is_select),
                    elimination_name(cli.elimination));
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

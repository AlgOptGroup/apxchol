/// bench_setup.cpp – Benchmark factorization (setup stage) across graph storage backends.
///
/// Tests on multiple graph families:
///   - Grid: regular 2D lattice (degree ~4)
///   - Path: chain graph (degree 2, worst for fill)
///   - Star: hub-and-spoke (degree 1 except hub, tests high-degree vertex)
///   - Random geometric: random unit-square points, edges within radius r
///
/// Usage:  bench_setup [--csv]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <Eigen/Sparse>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "apxchol/graph/graph.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/checkpoint.h"

// ── Graph generators ─────────────────────────────────

template<typename Incidence>
static apxchol::graph<Incidence> make_grid(int rows, int cols) {
    apxchol::graph<Incidence> G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
        }
    return G;
}

template<typename Incidence>
static apxchol::graph<Incidence> make_path(int n) {
    apxchol::graph<Incidence> G(n);
    for (int i = 0; i + 1 < n; ++i)
        G.add_edge(i, i + 1, 1.0);
    return G;
}

template<typename Incidence>
static apxchol::graph<Incidence> make_star(int n) {
    apxchol::graph<Incidence> G(n);
    for (int i = 1; i < n; ++i)
        G.add_edge(0, i, 1.0);
    return G;
}

template<typename Incidence>
static apxchol::graph<Incidence> make_rgg(int n, double radius, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) { x[i] = dist(rng); y[i] = dist(rng); }

    apxchol::graph<Incidence> G(n);
    double r2 = radius * radius;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double dx = x[i] - x[j], dy = y[i] - y[j];
            if (dx * dx + dy * dy <= r2)
                G.add_edge(i, j, 1.0);
        }
    return G;
}

// ── Benchmark runner ─────────────────────────────────

struct bench_result {
    const char* storage;
    const char* graph_type;
    int n;
    int m;
    long long nnz_L;
    double build_ms;
    double factor_ms;
    double total_ms;
    double peak_MB; // peak graph memory in MB
    // Profile breakdown (filled when --profile)
    double find_is_ms  = 0;
    double eliminate_ms = 0;
    double perm_ms     = 0;
    double assembly_ms = 0;
};

template<typename Incidence, typename Builder>
static bench_result run_one(const char* sname, const char* graph_name,
                            Builder&& build_fn,
                            const apxchol::factor_options& opts = {}) {
    using Clock = std::chrono::high_resolution_clock;

    auto t0 = Clock::now();
    auto G = build_fn.template operator()<Incidence>();
    auto t1 = Clock::now();

    apxchol::checkpoint cp;
    auto F = apxchol::factorize(G, opts, &cp);
    auto t2 = Clock::now();

    double build_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double factor_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    bench_result r{sname, graph_name, G.n(), G.m(),
                   static_cast<long long>(F.L.nonZeros()),
                   build_ms, factor_ms, build_ms + factor_ms,
                   F.peak_graph_bytes / (1024.0 * 1024.0)};

    // Extract profile breakdown from checkpoint
    r.find_is_ms   = cp.total("setup.find_is")   * 1000;
    r.eliminate_ms  = cp.total("setup.eliminate")  * 1000;
    r.perm_ms      = cp.total("setup.permutation") * 1000;
    r.assembly_ms  = cp.total("setup.assembly")   * 1000;

    return r;
}

// ── Output ───────────────────────────────────────────

static void print_header() {
    std::printf("%-12s %-10s %8s %10s %10s %10s %10s %10s %8s\n",
                "storage", "graph", "n", "m",
                "nnz(L)", "build(ms)", "fact(ms)", "total(ms)", "peak MB");
    std::printf("%s\n", std::string(92, '-').c_str());
}

static void print_result(const bench_result& r) {
    std::printf("%-12s %-10s %8d %10d %10lld %10.2f %10.2f %10.2f %8.1f\n",
                r.storage, r.graph_type, r.n, r.m, r.nnz_L,
                r.build_ms, r.factor_ms, r.total_ms, r.peak_MB);
}

static void print_profile_header() {
    std::printf("%-12s %-10s %8s %10s %10s %10s %10s %10s %10s %8s\n",
                "storage", "graph", "n", "nnz(L)",
                "find_is", "elim", "perm", "assembly", "total(ms)", "peak MB");
    std::printf("%s\n", std::string(104, '-').c_str());
}

static void print_profile_result(const bench_result& r) {
    std::printf("%-12s %-10s %8d %10lld %10.2f %10.2f %10.2f %10.2f %10.2f %8.1f\n",
                r.storage, r.graph_type, r.n, r.nnz_L,
                r.find_is_ms, r.eliminate_ms, r.perm_ms, r.assembly_ms,
                r.factor_ms, r.peak_MB);
}

static void print_csv_header() {
    std::printf("storage,graph,n,m,nnz_L,build_ms,factor_ms,total_ms,"
                "find_is_ms,eliminate_ms,perm_ms,assembly_ms,peak_MB\n");
}

static void print_csv(const bench_result& r) {
    std::printf("%s,%s,%d,%d,%lld,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n",
                r.storage, r.graph_type, r.n, r.m, r.nnz_L,
                r.build_ms, r.factor_ms, r.total_ms,
                r.find_is_ms, r.eliminate_ms, r.perm_ms, r.assembly_ms,
                r.peak_MB);
}

template<typename Incidence>
static constexpr const char* storage_name();
template<> constexpr const char* storage_name<apxchol::vec_incidence>()            { return "vec"; }
template<> constexpr const char* storage_name<apxchol::forward_star_incidence>()   { return "fwd_star"; }
template<> constexpr const char* storage_name<apxchol::small_vec_incidence>()      { return "svec12"; }
template<> constexpr const char* storage_name<apxchol::bstr_incidence>()             { return "bstr"; }

enum class output_mode { table, csv, profile };

template<typename Builder>
static void run_all_storages(const char* graph_name, Builder&& build_fn,
                             output_mode mode,
                             const apxchol::factor_options& opts = {}) {
    auto r1 = run_one<apxchol::vec_incidence>(storage_name<apxchol::vec_incidence>(), graph_name, build_fn, opts);
    auto r2 = run_one<apxchol::forward_star_incidence>(storage_name<apxchol::forward_star_incidence>(), graph_name, build_fn, opts);
    auto r3 = run_one<apxchol::small_vec_incidence>(storage_name<apxchol::small_vec_incidence>(), graph_name, build_fn, opts);
    auto r4 = run_one<apxchol::bstr_incidence>(storage_name<apxchol::bstr_incidence>(), graph_name, build_fn, opts);

    auto out = mode == output_mode::csv     ? print_csv
             : mode == output_mode::profile ? print_profile_result
             :                                print_result;
    out(r1); out(r2); out(r3); out(r4);
    if (mode == output_mode::table || mode == output_mode::profile) std::printf("\n");
}

int main(int argc, char* argv[]) {
    output_mode mode = output_mode::table;
    bool sweep_omp = false;
    bool sweep_deg = false;
    bool sweep_threads = false;
    const char* graph_filter = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0)      mode = output_mode::csv;
        if (std::strcmp(argv[i], "--profile") == 0)   mode = output_mode::profile;
        if (std::strcmp(argv[i], "--sweep-omp") == 0)   sweep_omp = true;
        if (std::strcmp(argv[i], "--sweep-deg") == 0)   sweep_deg = true;
        if (std::strcmp(argv[i], "--sweep-threads") == 0) sweep_threads = true;
        if (std::strcmp(argv[i], "--graph") == 0 && i + 1 < argc)
            graph_filter = argv[++i];
    }

    // Helper: should we run this graph name?
    auto should_run = [&](const char* name) {
        return !graph_filter || std::strstr(name, graph_filter);
    };

    // ── OMP threshold sweep: grid2000 × fwd_star only ──
    if (sweep_omp) {
        std::printf("%-12s %8s %10s %10s %10s %10s\n",
                    "omp_thresh", "n", "find_is", "elim", "fact(ms)", "nnz(L)");
        std::printf("%s\n", std::string(64, '-').c_str());
        for (size_t thresh : {500UL, 1000UL, 2000UL, 5000UL, 10000UL, 50000UL}) {
            apxchol::factor_options opts;
            opts.omp_threshold = thresh;
            auto r = run_one<apxchol::forward_star_incidence>(
                "fwd_star", "grid2000",
                []<typename Incidence>() { return make_grid<Incidence>(2000, 2000); },
                opts);
            std::printf("%-12zu %8d %10.2f %10.2f %10.2f %10lld\n",
                        thresh, r.n, r.find_is_ms, r.eliminate_ms,
                        r.factor_ms, r.nnz_L);
        }
        return 0;
    }

    // ── degree_multiplier sweep: multiple graphs × fwd_star, with IS profiles ──
    if (sweep_deg) {
        struct graph_spec {
            const char* name;
            std::function<apxchol::graph<apxchol::forward_star_incidence>()> builder;
        };
        std::vector<graph_spec> graphs = {
            {"grid1000", [] { return make_grid<apxchol::forward_star_incidence>(1000, 1000); }},
            {"path2M",   [] { return make_path<apxchol::forward_star_incidence>(2000000); }},
            {"rgg10k",   [] { return make_rgg<apxchol::forward_star_incidence>(10000, 0.012); }},
            {"star1M",   [] { return make_star<apxchol::forward_star_incidence>(1000000); }},
        };

        for (auto& [gname, builder] : graphs) {
            std::printf("\n=== %s ===\n", gname);
            std::printf("%-8s %8s %8s %8s %10s %10s %10s %10s %10s\n",
                        "deg_mul", "rounds", "avg_IS", "avg_deg",
                        "find_is", "elim", "fact(ms)", "nnz(L)", "n");
            std::printf("%s\n", std::string(90, '-').c_str());

            for (double mult : {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0}) {
                apxchol::factor_options opts;
                opts.degree_multiplier = mult;
                apxchol::checkpoint cp;
                auto G = builder();
                auto F = apxchol::factorize(G, opts, &cp);

                // IS profile stats
                int rounds = static_cast<int>(F.rounds.size());
                double avg_is = 0, avg_deg = 0;
                for (auto& rs : F.rounds) {
                    avg_is  += rs.is_size;
                    avg_deg += rs.avg_deg;
                }
                if (rounds > 0) { avg_is /= rounds; avg_deg /= rounds; }

                double find_is_ms  = cp.total("setup.find_is")    * 1000;
                double elim_ms     = cp.total("setup.eliminate")   * 1000;
                double fact_ms     = cp.total("setup")             * 1000;

                std::printf("%-8.1f %8d %8.0f %8.1f %10.2f %10.2f %10.2f %10lld %10d\n",
                            mult, rounds, avg_is, avg_deg,
                            find_is_ms, elim_ms, fact_ms,
                            static_cast<long long>(F.L.nonZeros()), G.n());
            }
        }
        return 0;
    }

    // ── Thread scaling sweep: grid2000 × fwd_star, sub-stage breakdown ──
    if (sweep_threads) {
        std::printf("%-8s %10s %10s %10s %10s %10s %10s\n",
                    "threads", "find_is", "merge_is", "compute",
                    "apply", "elim", "total(ms)");
        std::printf("%s\n", std::string(72, '-').c_str());
        for (int t : {1, 2, 4, 8, 16, 32}) {
#ifdef _OPENMP
            omp_set_num_threads(t);
#endif
            apxchol::checkpoint cp;
            auto G = make_grid<apxchol::forward_star_incidence>(2000, 2000);
            auto F = apxchol::factorize(G, {}, &cp);

            double find_is_ms  = cp.total("setup.find_is")            * 1000;
            double merge_is_ms = cp.total("setup.eliminate.merge_is") * 1000;
            double compute_ms  = cp.total("setup.eliminate.compute")  * 1000;
            double apply_ms    = cp.total("setup.eliminate.apply")    * 1000;
            double elim_ms     = cp.total("setup.eliminate")          * 1000;
            double total_ms    = cp.total("setup")                    * 1000;

            std::printf("%-8d %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n",
                        t, find_is_ms, merge_is_ms, compute_ms,
                        apply_ms, elim_ms, total_ms);
        }
        return 0;
    }

    if (mode == output_mode::csv)      print_csv_header();
    else if (mode == output_mode::profile) print_profile_header();
    else                                   print_header();

    // Grid graphs: varying size (focus on larger sizes)
    for (int side : {100, 500, 1000, 1500, 2000}) {
        char name[32];
        std::snprintf(name, sizeof(name), "grid%d", side);
        if (!should_run(name)) continue;
        run_all_storages(name, [side]<typename Incidence>() {
            return make_grid<Incidence>(side, side);
        }, mode);
    }

    // Path graphs: linear chain
    for (int n : {10000, 100000, 500000, 1000000, 2000000}) {
        char name[32];
        if (n >= 1000000)
            std::snprintf(name, sizeof(name), "path%dM", n / 1000000);
        else
            std::snprintf(name, sizeof(name), "path%dk", n / 1000);
        if (!should_run(name)) continue;
        run_all_storages(name, [n]<typename Incidence>() {
            return make_path<Incidence>(n);
        }, mode);
    }

    // Star graphs: one hub
    for (int n : {10000, 100000, 500000, 1000000}) {
        char name[32];
        if (n >= 1000000)
            std::snprintf(name, sizeof(name), "star%dM", n / 1000000);
        else
            std::snprintf(name, sizeof(name), "star%dk", n / 1000);
        if (!should_run(name)) continue;
        run_all_storages(name, [n]<typename Incidence>() {
            return make_star<Incidence>(n);
        }, mode);
    }

    // Random geometric graphs
    for (auto [n, r] : std::initializer_list<std::pair<int, double>>{
            {1000, 0.05}, {5000, 0.02}, {10000, 0.012}, {20000, 0.008}}) {
        char name[32];
        std::snprintf(name, sizeof(name), "rgg%d", n);
        if (!should_run(name)) continue;
        run_all_storages(name, [n, r]<typename Incidence>() {
            return make_rgg<Incidence>(n, r);
        }, mode);
    }

    return 0;
}

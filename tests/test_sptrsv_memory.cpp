// Memory-lifetime guards for omp_sptrsv::setup (include/apxchol/solver/sptrsv/omp.h).
//
// Setup allocates several nnz-sized transients -- the Laplacian-path L11 copy,
// the compacted (drop) copy, the transpose bucket -- that must be released at
// their last use, not at return, and nothing but the SpTRSV's own arrays may
// survive setup. A `v = {}` / `.clear()` (which keep the capacity) or a
// transient promoted to a member would silently regress the setup peak; these
// tests catch that class of change through the kernel's own accounting:
//   * VmHWM is reset right before setup (/proc/self/clear_refs "5"), so
//     VmHWM - VmRSS_before is the setup-only peak, bounded by the intended
//     overlap (drop copy + CSR + transpose bucket, or L11 copy + drop copy);
//   * VmRSS after setup - before is bounded by omp_sptrsv::memory_bytes().
// glibc's malloc mmap threshold is pinned (mallopt) so every block above it is
// an mmap that free() returns to the OS immediately -- otherwise the dynamic
// threshold lets freed nnz-sized blocks linger in the arena and the RSS
// accounting would not reflect the release. Linux + glibc only (skipped
// elsewhere).
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "apxchol/graph/conversions.h"
#include "apxchol/graph/graph.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/sptrsv/omp.h"

using apxchol::edge_index;
using apxchol::node_index;
using apxchol::sparse_csc;
using apxchol::factor_value_t;
using apxchol::sptrsv_value_t;

namespace {

struct scoped_env {
    std::string name, saved; bool had = false;
    scoped_env(const char* var, const char* value) : name(var) {
        if (const char* e = std::getenv(var)) { had = true; saved = e; }
        if (value) setenv(var, value, 1); else unsetenv(var);
    }
    ~scoped_env() { if (had) setenv(name.c_str(), saved.c_str(), 1); else unsetenv(name.c_str()); }
};

long proc_status_kb(const char* key) {
    std::ifstream f("/proc/self/status"); std::string ln;
    const std::size_t klen = std::strlen(key);
    while (std::getline(f, ln))
        if (ln.rfind(key, 0) == 0) return std::stol(ln.substr(klen));
    return -1;
}

// Reset the process peak-RSS counter (VmHWM) to the current RSS.
bool reset_peak_rss() {
    std::FILE* f = std::fopen("/proc/self/clear_refs", "w");
    if (!f) return false;
    const bool ok = std::fputs("5", f) >= 0;
    return (std::fclose(f) == 0) && ok;
}

Eigen::SparseMatrix<double> grid_laplacian(int rows, int cols) {
    apxchol::graph<> G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
        }
    return apxchol::laplacian(G);
}

constexpr std::size_t MB = 1024 * 1024;

} // namespace

// The Laplacian path (m = n-1) with the compacting drop active exercises every
// setup transient: L11 copy -> drop copy (L11 freed) -> CSR + transpose bucket
// -> CSC copy (drop copy freed) -> level sets.
TEST(SpTRSVSetupMemory, TransientsAreReleasedAtLastUse) {
#if !defined(__linux__) || !defined(__GLIBC__)
    GTEST_SKIP() << "needs /proc/self/{status,clear_refs} and glibc mallopt";
#else
    // Deterministic RSS accounting: every block >= 256 KB is an mmap, returned
    // on free (disables glibc's dynamic mmap threshold for this process).
    ASSERT_EQ(mallopt(M_MMAP_THRESHOLD, 256 * 1024), 1);

    // ~640k-column factor, ~3.5M nnz: the transients are tens of MB, far above
    // the accounting noise (page rounding, per-thread RSS batching, the
    // per-level vectors of the level sets), while a leaked L11 copy would add
    // nnz * 8 B (~28 MB) to the peak.
    const auto L = grid_laplacian(800, 800);
    apxchol::factor_options fopts; fopts.seed = 7;
    const auto F = apxchol::factorize(L, apxchol::graph_storage::vec_pool, fopts);
    const node_index m = F.L.rows() - 1;                       // Laplacian path
    scoped_env drop("APXCHOL_FACTOR_DROP", "0.5");             // force a real drop

    { apxchol::omp_sptrsv warm; warm.setup(F.L, m); }          // OMP team + arenas exist
    if (!reset_peak_rss()) GTEST_SKIP() << "/proc/self/clear_refs not writable";
    const long rss0_kb = proc_status_kb("VmRSS:");
    ASSERT_GT(rss0_kb, 0);

    apxchol::omp_sptrsv trsv;
    trsv.setup(F.L, m);

    const long hwm_kb  = proc_status_kb("VmHWM:");
    const long rss1_kb = proc_status_kb("VmRSS:");
    ASSERT_GT(hwm_kb, 0); ASSERT_GT(rss1_kb, 0);

    const auto& st = trsv.drop_stats();
    const std::size_t nnz  = st.nnz_factor;                    // L11 as factorized
    const std::size_t kept = st.nnz_stored;                    // after the drop
    ASSERT_GT(st.dropped, nnz / 5) << "the test needs the drop-copy path";
    ASSERT_GT(nnz, 1000000u);

    // FB: the factor's value width (the L11 copy and the compacted copy are at
    // factor precision); VB: the SpTRSV's storage width (CSR / bucket) -- equal
    // to it at the default fp32 storage, 2 under APXCHOL_SPTRSV_FP16=1.
    constexpr std::size_t FB = sizeof(factor_value_t), VB = sizeof(sptrsv_value_t);
    constexpr std::size_t NB = sizeof(node_index), EB = sizeof(edge_index);
    const std::size_t slack = 6 * MB;   // page rounding + per-thread RSS batching + level-vector heap
    // Per-column member arrays the fp16 storage keeps alive through setup
    // (fp32 diag_, scale_ / inv_scale_ -- allocated before the drop): part of
    // memory_bytes(), and of every peak moment. Zero at the default fp32
    // storage, which is what these tests run.
    const std::size_t per_col_members = 0;

    // (a) Steady state: nothing survives setup but the SpTRSV's own arrays.
    const std::size_t live = trsv.memory_bytes();
    const std::size_t rss_delta = static_cast<std::size_t>(rss1_kb - rss0_kb) * 1024;
    EXPECT_LE(rss_delta, live + slack)
        << "setup left " << (rss_delta - live) / double(MB) << " MB beyond memory_bytes()="
        << live / double(MB) << " MB resident";

    // (b) Peak: only the intended overlap. Two candidates, whichever is larger:
    //   drop moment      L11 copy (nnz) + col_scale (m x 8) + drop copy (kept)
    //   transpose moment drop copy (kept) + CSR (kept) + bucket (kept x 12 B)
    // A leaked L11 copy would sit under the transpose moment and add nnz*8 B.
    const std::size_t drop_moment = nnz * (NB + FB) + (m + 1) * EB      // L11 copy
                                  + m * sizeof(double)                  // col_scale (fp32 or fp64)
                                  + kept * (NB + FB) + (m + 1) * EB;    // drop copy
    const std::size_t transpose_moment = kept * (NB + FB) + (m + 1) * EB   // drop copy
                                       + kept * (NB + VB) + (m + 1) * EB   // CSR
                                       + kept * (2 * NB + VB);             // bucket
    const std::size_t peak_bound = std::max(drop_moment, transpose_moment) + per_col_members + slack;
    const std::size_t peak = static_cast<std::size_t>(hwm_kb - rss0_kb) * 1024;
    EXPECT_LE(peak, peak_bound)
        << "setup peak " << peak / double(MB) << " MB exceeds the intended transient overlap "
        << peak_bound / double(MB) << " MB (nnz=" << nnz << " kept=" << kept << ")";
#endif
}

// setup_consuming releases the input factor's row/value arrays (capacity 0,
// nonZeros() intact) and yields the same solves as setup(const&).
TEST(SpTRSVSetupMemory, SetupConsumingReleasesTheFactorAndSolvesIdentically) {
    const auto L = grid_laplacian(120, 120);
    apxchol::factor_options fopts; fopts.seed = 3;
    auto F1 = apxchol::factorize(L, apxchol::graph_storage::vec_pool, fopts);
    auto F2 = apxchol::factorize(L, apxchol::graph_storage::vec_pool, fopts);
    ASSERT_EQ(F1.L.nonZeros(), F2.L.nonZeros());
    const node_index m = F1.L.rows() - 1;
    const auto nnz = F2.L.nonZeros();

    apxchol::omp_sptrsv a, b;
    a.setup(F1.L, m);
    b.setup_consuming(F2.L, m);
    EXPECT_EQ(F2.L.inner_.capacity(), 0u);
    EXPECT_EQ(F2.L.vals_.capacity(), 0u);
    EXPECT_EQ(F2.L.nonZeros(), nnz);            // column pointers stay
    EXPECT_EQ(F1.L.inner_.size(), static_cast<std::size_t>(nnz));  // const& setup keeps it

    std::vector<double> x(m), ya(m), yb(m), za(m), zb(m);
    for (node_index i = 0; i < m; ++i) x[i] = std::sin(0.37 * i) + 1.5;
    a.forward_solve(x.data(), ya.data());  a.transpose_solve(ya.data(), za.data());
    b.forward_solve(x.data(), yb.data());  b.transpose_solve(yb.data(), zb.data());
    EXPECT_EQ(std::memcmp(ya.data(), yb.data(), m * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(za.data(), zb.data(), m * sizeof(double)), 0);
    EXPECT_EQ(a.memory_bytes(), b.memory_bytes());
}

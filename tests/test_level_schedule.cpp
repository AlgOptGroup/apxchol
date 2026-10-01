// Portable parts of the Metal block PCG: the level schedules of the dropped
// factor (apxchol/solver/sptrsv/level_schedule.h) with their fp32 kernel
// emulation, and the host arithmetic (apxchol/solver/detail/metal_host.h).
// No device needed: these build and run on every platform.
#include <gtest/gtest.h>

#include "apxchol/solver/detail/metal_host.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include "apxchol/solver/sptrsv/level_schedule.h"
#include "apxchol/solver/sptrsv/omp.h"
#include "apxchol/sparse_csc.h"

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using apxchol::edge_index;
using apxchol::node_index;
using apxchol::sparse_csc;
namespace ls = apxchol::level_schedule;
namespace mh = apxchol::detail::metal_host;

struct scoped_env {
    std::string name, saved;
    bool had = false;
    scoped_env(const char* var, const char* value) : name(var) {
        if (const char* e = std::getenv(var)) { had = true; saved = e; }
        if (value) setenv(var, value, 1);
        else unsetenv(var);
    }
    ~scoped_env() {
        if (had) setenv(name.c_str(), saved.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

// Lower-triangular factor with positive diagonals, small negative
// off-diagonals over six decades (so the default drop removes some), hub
// columns (heavy backward rows) and hub rows (heavy forward rows).
sparse_csc random_factor(node_index n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> mag(-6.0, 0.0), diag(1.0, 3.0);
    std::vector<std::vector<std::pair<node_index, float>>> cols(n);
    auto add = [&](node_index i, node_index j, double scale) {
        cols[j].emplace_back(i, static_cast<float>(-scale * std::pow(10.0, mag(rng))));
    };
    for (node_index j = 0; j < n; ++j) {
        cols[j].emplace_back(j, static_cast<float>(diag(rng)));
        if (j + 1 >= n) continue;
        std::uniform_int_distribution<node_index> below(j + 1, n - 1);
        const int count = j % 211 == 0 ? 70 : 3;
        for (int t = 0; t < count; ++t) add(below(rng), j, j % 211 == 0 ? 0.01 : 0.1);
    }
    for (node_index i = 97; i < n; i += 263) {
        std::uniform_int_distribution<node_index> above(0, i - 1);
        for (int t = 0; t < 60; ++t) add(i, above(rng), 0.01);
    }
    sparse_csc L;
    L.n_ = n;
    L.outer_.assign(static_cast<std::size_t>(n) + 1, 0);
    for (node_index j = 0; j < n; ++j) {
        auto& c = cols[j];
        std::sort(c.begin() + 1, c.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        c.erase(std::unique(c.begin() + 1, c.end(), [](const auto& a, const auto& b) { return a.first == b.first; }),
                c.end());
        L.outer_[j + 1] = L.outer_[j] + static_cast<edge_index>(c.size());
        for (const auto& [i, v] : c) {
            L.inner_.push_back(i);
            L.vals_.push_back(v);
        }
    }
    return L;
}

// Grid (+ optional hubs joined to many vertices); Laplacian or SDDM.
Eigen::SparseMatrix<double> graph(int rows, int cols, double shift, int hubs, bool inexact) {
    const int g = rows * cols, n = g + hubs;
    std::vector<Eigen::Triplet<double>> t;
    std::vector<double> deg(n, shift);
    auto edge = [&](int a, int b, double w) {
        t.emplace_back(a, b, -w);
        t.emplace_back(b, a, -w);
        deg[a] += w;
        deg[b] += w;
    };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const int k = (r * 7 + c * 3) % 5;
            const double w = inexact ? 1.0 + k / 3.0 : 1.0 + 0.25 * k;
            if (r + 1 < rows) edge(r * cols + c, (r + 1) * cols + c, w);
            if (c + 1 < cols) edge(r * cols + c, r * cols + c + 1, w);
        }
    for (int h = 0; h < hubs; ++h)
        for (int v = h; v < g; v += 37 + h) edge(g + h, v, 0.5);
    for (int i = 0; i < n; ++i) t.emplace_back(i, i, deg[i]);
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    return A;
}

struct factor_case {
    std::string name;
    sparse_csc L;
    node_index m;
};

std::vector<factor_case> factor_cases() {
    std::vector<factor_case> out;
    const sparse_csc R = random_factor(6000, 11);
    out.push_back({"random SDDM path", R, 6000});
    out.push_back({"random Laplacian path", R, 5999});
    struct g { const char* name; int rows, cols; double shift; int hubs; bool inexact; };
    for (const g& c : {g{"grid Laplacian", 60, 50, 0.0, 0, false}, g{"weighted SDDM", 45, 40, 0.5, 0, true},
                       g{"hub Laplacian", 50, 50, 0.0, 3, true}}) {
        const apxchol::factorization F = apxchol::factorize(graph(c.rows, c.cols, c.shift, c.hubs, c.inexact));
        out.push_back({c.name, F.L, F.sddm ? F.L.rows() : F.L.rows() - 1});
    }
    return out;
}

std::vector<std::uint32_t> slot_levels(const ls::level_solve& s) {
    std::vector<std::uint32_t> lv(s.slots());
    for (std::uint32_t l = 0; l < s.levels(); ++l)
        for (std::uint32_t t = s.level_ptr[l]; t < s.level_ptr[l + 1]; ++t) lv[t] = l;
    return lv;
}

}  // namespace

TEST(LevelSchedule, EveryRowOnceDependenciesEarlierLightBeforeHeavy) {
    std::size_t heavy_rows = 0;
    for (const factor_case& fc : factor_cases()) {
        SCOPED_TRACE(fc.name);
        const ls::factor_schedules s = ls::build_factor_schedules(fc.L, fc.m, apxchol::factor_drop_rel_from_env());
        for (const ls::level_solve* sv : {&s.forward, &s.backward}) {
            const bool forward = sv == &s.forward;
            ASSERT_EQ(sv->slots(), fc.m);
            ASSERT_EQ(sv->level_ptr.back(), fc.m);
            std::vector<int> seen(fc.m, 0);
            std::vector<std::uint32_t> row_level(fc.m);
            const auto lv = slot_levels(*sv);
            for (std::uint32_t t = 0; t < sv->slots(); ++t) {
                ++seen[sv->rows[t]];
                row_level[sv->rows[t]] = lv[t];
            }
            EXPECT_EQ(std::count(seen.begin(), seen.end(), 1), static_cast<long>(fc.m));
            for (std::uint32_t l = 0; l < sv->levels(); ++l) {
                ASSERT_LT(sv->level_ptr[l], sv->level_ptr[l + 1]) << "empty level " << l;
                for (std::uint32_t t = sv->level_ptr[l]; t < sv->level_ptr[l + 1]; ++t) {
                    const bool heavy = sv->ptr[t + 1] - sv->ptr[t] > ls::kHeavyDeps;
                    EXPECT_EQ(heavy, t >= sv->heavy_ptr[l]) << "light rows precede heavy rows";
                    heavy_rows += heavy;
                    if (t + 1 < sv->heavy_ptr[l] || (t >= sv->heavy_ptr[l] && t + 1 < sv->level_ptr[l + 1]))
                        EXPECT_LT(sv->rows[t], sv->rows[t + 1]) << "ascending within each part";
                    for (std::uint32_t e = sv->ptr[t]; e < sv->ptr[t + 1]; ++e) {
                        EXPECT_LT(row_level[sv->col[e]], l) << "dependency solved earlier";
                        EXPECT_TRUE(forward ? sv->col[e] < sv->rows[t] : sv->col[e] > sv->rows[t]);
                    }
                }
            }
        }
    }
    EXPECT_GT(heavy_rows, 0u) << "the cases must exercise heavy rows";
}

// What the schedules apply is what omp_sptrsv stores on its fp32 storage (the
// same L11, drop and transpose), and the levels are its topological levels.
TEST(LevelSchedule, ArraysAndLevelsAreTheCpuStoredFactors) {
    const scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    for (const char* drop : {static_cast<const char*>(nullptr), "0", "1e-3"}) {
        const scoped_env env("APXCHOL_FACTOR_DROP", drop);
        for (const factor_case& fc : factor_cases()) {
            SCOPED_TRACE(fc.name + " APXCHOL_FACTOR_DROP=" + (drop ? drop : "<unset>"));
            apxchol::omp_sptrsv cpu;
            cpu.setup(fc.L, fc.m);  // no round metadata: topological levels
            const ls::factor_schedules s = ls::build_factor_schedules(fc.L, fc.m, apxchol::factor_drop_rel_from_env());
            EXPECT_EQ(s.drop.nnz_stored, cpu.stored_nnz());
            EXPECT_EQ(s.drop.dropped, cpu.drop_stats().dropped);
            if (drop && std::string(drop) == "1e-3" && fc.name.rfind("random", 0) == 0) EXPECT_GT(s.drop.dropped, 0u);
            std::size_t mismatches = 0;
            for (const bool forward : {true, false}) {
                const ls::level_solve& sv = forward ? s.forward : s.backward;
                const auto& ptr = forward ? cpu.csr_row_ptr() : cpu.csc_col_ptr();
                const auto& idx = forward ? cpu.csr_col_idx() : cpu.csc_row_idx();
                const auto& val = forward ? cpu.csr_vals() : cpu.csc_vals();
                for (std::uint32_t t = 0; t < sv.slots(); ++t) {
                    const node_index i = sv.rows[t];
                    const edge_index b = ptr[i] + (forward ? 0 : 1), e1 = ptr[i + 1] - (forward ? 1 : 0);
                    const float d = val[forward ? ptr[i + 1] - 1 : ptr[i]];
                    mismatches += sv.ptr[t + 1] - sv.ptr[t] != e1 - b;
                    for (edge_index k = b; k < e1 && k - b < sv.ptr[t + 1] - sv.ptr[t]; ++k) {
                        const std::uint32_t e = sv.ptr[t] + static_cast<std::uint32_t>(k - b);
                        mismatches += sv.col[e] != idx[k];
                        mismatches += std::memcmp(&sv.val[e], &val[k], sizeof(float)) != 0;
                    }
                    const float inv = static_cast<float>(1.0 / static_cast<double>(d));
                    mismatches += std::memcmp(&sv.dinv[t], &inv, sizeof(float)) != 0;
                }
                std::vector<int> sizes;
                std::vector<long long> work;
                cpu.level_stats(forward, sizes, work);
                ASSERT_EQ(sizes.size(), sv.levels());
                for (std::size_t l = 0; l < sizes.size(); ++l)
                    mismatches += static_cast<std::uint32_t>(sizes[l]) != sv.level_ptr[l + 1] - sv.level_ptr[l];
            }
            EXPECT_EQ(mismatches, 0u);
        }
    }
}

TEST(LevelSchedule, StepPlanCoversEachLevelOnce) {
    for (const factor_case& fc : factor_cases()) {
        const ls::factor_schedules s = ls::build_factor_schedules(fc.L, fc.m, apxchol::factor_drop_rel_from_env());
        for (const ls::level_solve* sv : {&s.forward, &s.backward})
            for (const std::uint32_t kc : {1u, 7u, 64u}) {
                SCOPED_TRACE(fc.name + " kc=" + std::to_string(kc));
                std::uint32_t next_level = 0, next_slot = 0, narrow = 0;
                for (const ls::level_step& st : ls::plan_steps(*sv, kc)) {
                    if (st.kind == ls::step_kind::narrow) {
                        ASSERT_EQ(st.first, next_level);
                        ASSERT_EQ(next_slot, sv->level_ptr[st.first]);
                        ASSERT_LT(st.first, st.last);
                        for (std::uint32_t l = st.first; l < st.last; ++l) {
                            EXPECT_EQ(sv->heavy_ptr[l], sv->level_ptr[l + 1]) << "light rows only";
                            EXPECT_LE(std::uint64_t{sv->level_ptr[l + 1] - sv->level_ptr[l]} * kc, ls::kNarrowItems);
                        }
                        next_level = st.last;
                        next_slot = sv->level_ptr[st.last];
                        ++narrow;
                        continue;
                    }
                    ASSERT_EQ(st.first, next_slot);
                    ASSERT_LT(st.first, st.last);
                    const std::uint32_t l = next_level;
                    if (st.kind == ls::step_kind::light) {
                        ASSERT_EQ(st.first, sv->level_ptr[l]);
                        ASSERT_EQ(st.last, sv->heavy_ptr[l]);
                    } else {
                        ASSERT_EQ(st.first, sv->heavy_ptr[l]);
                        ASSERT_EQ(st.last, sv->level_ptr[l + 1]);
                    }
                    next_slot = st.last;
                    if (next_slot == sv->level_ptr[l + 1]) ++next_level;
                }
                EXPECT_EQ(next_level, sv->levels());
                EXPECT_EQ(next_slot, sv->slots());
                if (kc == 1) EXPECT_GT(narrow, 0u);
            }
    }
}

// The fp32 emulation of the device kernels agrees with omp_sptrsv's fp64
// sweeps on the same stored factor to fp32 accuracy.
TEST(LevelSchedule, Fp32EmulationAgreesWithCpuSweeps) {
    const scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    for (const factor_case& fc : factor_cases()) {
        SCOPED_TRACE(fc.name);
        apxchol::omp_sptrsv cpu;
        cpu.setup(fc.L, fc.m);
        const ls::factor_schedules s = ls::build_factor_schedules(fc.L, fc.m, apxchol::factor_drop_rel_from_env());
        std::vector<float> bf(fc.m), yf(fc.m, 0.0f);
        std::vector<double> b(fc.m), y(fc.m), z(fc.m);
        for (node_index i = 0; i < fc.m; ++i) {
            bf[i] = static_cast<float>(std::sin(0.71 * (i + 1)) + 0.25);
            b[i] = bf[i];
        }
        cpu.forward_solve(b.data(), y.data());
        ls::emulate_sweep(s.forward, bf.data(), yf.data());
        double num = 0, den = 0;
        for (node_index i = 0; i < fc.m; ++i) { num += (y[i] - yf[i]) * (y[i] - yf[i]); den += y[i] * y[i]; }
        const double fwd = std::sqrt(num / den);
        cpu.transpose_solve(y.data(), z.data());
        ls::emulate_sweep(s.backward, yf.data(), yf.data());
        num = den = 0;
        for (node_index i = 0; i < fc.m; ++i) { num += (z[i] - yf[i]) * (z[i] - yf[i]); den += z[i] * z[i]; }
        const double both = std::sqrt(num / den);
        std::printf("[ schedule ] %-22s forward %.2e, forward+backward %.2e (relative to omp_sptrsv)\n",
                    fc.name.c_str(), fwd, both);
        EXPECT_LE(fwd, 1e-5);
        EXPECT_LE(both, 1e-5);
    }
}

TEST(LevelSchedule, RejectsMisplacedOrInvalidDiagonalsAndRanges) {
    auto csr = [](int m, std::vector<int> ptr, std::vector<int> idx, std::vector<float> vals) {
        apxchol::cuda_host::csr_int<float> A;
        A.m = m;
        A.nnz = static_cast<std::int64_t>(idx.size());
        A.ptr = std::move(ptr);
        A.idx = std::make_unique<int[]>(idx.size());
        A.vals = std::make_unique<float[]>(vals.size());
        std::copy(idx.begin(), idx.end(), A.idx.get());
        std::copy(vals.begin(), vals.end(), A.vals.get());
        return A;
    };
    // Forward CSR: row 1 = {L10, L11} with the diagonal last.
    EXPECT_NO_THROW(ls::build_forward(csr(2, {0, 1, 3}, {0, 0, 1}, {2.0f, -0.5f, 3.0f})));
    EXPECT_THROW(ls::build_forward(csr(2, {0, 1, 3}, {0, 1, 0}, {2.0f, 3.0f, -0.5f})), std::invalid_argument);
    EXPECT_THROW(ls::build_forward(csr(2, {0, 1, 3}, {0, 0, 1}, {2.0f, -0.5f, 0.0f})), std::domain_error);
    EXPECT_THROW(ls::build_forward(csr(2, {0, 1, 3}, {0, 0, 1}, {2.0f, -1e35f, 3.0f})), std::domain_error);
    EXPECT_THROW(ls::build_forward(csr(2, {0, 1, 3}, {0, 0, 1}, {2.0f, -0.5f, 1e-35f})), std::domain_error);
    // Backward CSC: column 0 = {L00, L10} with the diagonal first.
    EXPECT_NO_THROW(ls::build_backward(csr(2, {0, 2, 3}, {0, 1, 1}, {2.0f, -0.5f, 3.0f})));
    EXPECT_THROW(ls::build_backward(csr(2, {0, 2, 3}, {1, 0, 1}, {-0.5f, 2.0f, 3.0f})), std::invalid_argument);
}

TEST(MetalHost, DoubleFloatSplitIsExactOrBounded) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int i = 0; i < 20000; ++i) {
        const double v = std::ldexp(u(rng), static_cast<int>(rng() % 120) - 60);
        const mh::df d = mh::split(v);
        EXPECT_LE(std::fabs(mh::join(d) - v), std::ldexp(std::fabs(v), -48));
        EXPECT_LE(std::fabs(d.lo), std::ldexp(std::fabs(d.hi), -24));
        // A value with at most 48 significant bits splits exactly.
        const double w = static_cast<double>(static_cast<float>(v)) +
                         static_cast<double>(static_cast<float>(v * 0x1p-30));
        EXPECT_EQ(mh::join(mh::split(w)), w);
    }
}

TEST(MetalHost, PowerOfTwoScalingIsExact) {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(0.5, 2.0);
    for (int i = 0; i < 2000; ++i) {
        const double max_abs = std::ldexp(u(rng), static_cast<int>(rng() % 400) - 200);
        const double s = mh::pow2_scale(max_abs);
        int e = 0;
        EXPECT_EQ(std::frexp(s, &e), 0.5) << "a power of two";
        EXPECT_GE(max_abs * s, 1.0);
        EXPECT_LT(max_abs * s, 2.0);
        const double v = max_abs * u(rng) * 0.4;
        EXPECT_EQ(v * s / s, v);
        EXPECT_EQ(mh::split(v * s).hi, static_cast<float>(v * s));
    }
}

TEST(MetalHost, ChooseBlockColumnsRespectsEveryLimit) {
    mh::block_limits lim;
    lim.max_buffer_bytes = std::uint64_t{1} << 40;
    lim.working_set_bytes = 0;
    lim.tree_threads = 1024;
    lim.row_threads = 1024;
    EXPECT_EQ(mh::choose_block_columns(1000, lim), 64u);
    EXPECT_EQ(mh::choose_block_columns(std::uint64_t{1} << 26, lim), 63u) << "n kc < 2^32";
    EXPECT_EQ(mh::choose_block_columns(std::uint64_t{1} << 32, lim), 0u);
    mh::block_limits t = lim;
    t.tree_threads = 512;
    EXPECT_EQ(mh::choose_block_columns(1000, t), 32u) << "16 kc tree threadgroups";
    t = lim;
    t.row_threads = 20;
    EXPECT_EQ(mh::choose_block_columns(1000, t), 20u);
    t = lim;
    t.max_buffer_bytes = 1000 * 8 * 10;
    EXPECT_EQ(mh::choose_block_columns(1000, t), 10u) << "x as double-float within one buffer";
    t = lim;
    t.static_bytes = 1000;
    t.working_set_bytes = 1000 + 1000 * mh::kBlockBytesPerEntry * 5 + 2 * 4 * 5 * 8;
    EXPECT_EQ(mh::choose_block_columns(1000, t), 5u) << "working set";
    t.working_set_bytes = 10;
    EXPECT_EQ(mh::choose_block_columns(1000, t), 0u);
}

// tree_reduce is the definition the device kernels implement; restate it
// here per row (group g = i / 256, lane l = i % 16, steps in row order) and
// show that a column's result does not depend on the block it sits in.
TEST(MetalHost, ReductionTreeMatchesItsDefinitionAtEveryBlockWidth) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (const std::uint32_t n : {1u, 15u, 255u, 256u, 257u, 4097u, 70001u}) {
        SCOPED_TRACE(n);
        std::vector<mh::df> v(n);
        for (auto& x : v) x = mh::quick_two_sum(u(rng) * 1000.0f, u(rng) * 1e-5f);
        const std::uint32_t groups = static_cast<std::uint32_t>(mh::reduction_groups(n));
        std::vector<mh::df> lane(static_cast<std::size_t>(groups) * 16);
        for (std::uint32_t i = 0; i < n; ++i) {
            mh::df& acc = lane[(i / 256) * 16 + i % 16];
            acc = mh::df_add(acc, v[i]);
        }
        std::vector<mh::df> partial(groups), fin(16);
        for (std::uint32_t g = 0; g < groups; ++g)
            for (int l = 0; l < 16; ++l) partial[g] = mh::df_add(partial[g], lane[g * 16 + l]);
        for (std::uint32_t g = 0; g < groups; ++g) fin[g % 16] = mh::df_add(fin[g % 16], partial[g]);
        mh::df ref{};
        for (int l = 0; l < 16; ++l) ref = mh::df_add(ref, fin[l]);
        for (const std::uint32_t kc : {1u, 13u, 64u}) {
            std::vector<mh::df> block(static_cast<std::size_t>(n) * kc, mh::df{7.0f, 0.0f});
            const std::uint32_t c = kc - 1;
            for (std::uint32_t i = 0; i < n; ++i) block[static_cast<std::size_t>(i) * kc + c] = v[i];
            const mh::df t = mh::tree_reduce(n, [&](std::uint32_t i) { return block[static_cast<std::size_t>(i) * kc + c]; });
            EXPECT_EQ(std::memcmp(&t, &ref, sizeof t), 0) << "kc=" << kc;
        }
    }
}

TEST(MetalHost, FoldsAndPackingDoNotDependOnTheThreadCount) {
#ifndef _OPENMP
    GTEST_SKIP() << "serial build";
#else
    const std::size_t n = 50001;
    std::vector<double> v(n);
    std::vector<std::uint32_t> perm(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = std::sin(0.3 * static_cast<double>(i)) * std::ldexp(1.0, static_cast<int>(i % 40) - 20);
        perm[i] = static_cast<std::uint32_t>((i * 7919) % n);
    }
    const int saved = omp_get_max_threads();
    std::vector<double> sums;
    std::vector<std::vector<mh::df>> blocks;
    for (const int threads : {1, 2, 3, 6}) {
        omp_set_num_threads(threads);
        sums.push_back(mh::fold_sum_squares(v.data(), n));
        std::vector<double> vp(n);
        mh::scatter(v.data(), perm.data(), n, vp.data());
        blocks.emplace_back(n * 3);
        mh::pack_column(vp.data(), n, 0.25, 3, 1, blocks.back().data());
    }
    omp_set_num_threads(saved);
    for (std::size_t i = 1; i < sums.size(); ++i) {
        EXPECT_EQ(sums[i], sums[0]);
        EXPECT_EQ(std::memcmp(blocks[i].data(), blocks[0].data(), n * 3 * sizeof(mh::df)), 0);
    }
    std::vector<double> back(n), out(n);
    mh::unpack_column(blocks[0].data(), n, 0.25, 3, 1, back.data());
    mh::gather(back.data(), perm.data(), n, out.data());
    for (std::size_t i = 0; i < n; ++i) EXPECT_LE(std::fabs(out[i] - v[i]), std::ldexp(std::fabs(v[i]), -47));
#endif
}

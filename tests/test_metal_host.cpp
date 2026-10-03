// The Metal block PCG's portable host arithmetic
// (apxchol/solver/detail/metal_host.h): double-float splits, power-of-two
// scaling, the block-width choice, the device's reduction tree and the
// thread-count-independent folds and packing. No device needed: these build
// and run on every platform.
#include <gtest/gtest.h>

#include "apxchol/solver/detail/metal_host.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mh = apxchol::detail::metal_host;

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

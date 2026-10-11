#include <gtest/gtest.h>

#include "apxchol/solver/detail/metal_host.h"
#include "../src/metal_device.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

TEST(MetalHost, ScalingRemainsFiniteAtFloatingPointLimits) {
    for (const double value : {std::numeric_limits<double>::denorm_min(),
                               std::numeric_limits<double>::min(),
                               std::numeric_limits<double>::max()}) {
        SCOPED_TRACE(value);
        const double scale = mh::pow2_scale(value);
        EXPECT_TRUE(std::isfinite(scale));
        EXPECT_GT(scale, 0.0);
        EXPECT_GT(value * scale, 0.0);
        EXPECT_LT(value * scale, 2.0);
        EXPECT_EQ(value * scale / scale, value);
    }
}

TEST(MetalHost, VectorPackingRoundTrip) {
    const std::size_t n = 50001;
    std::vector<double> v(n), permuted(n), back(n), out(n);
    std::vector<std::uint32_t> perm(n);
    std::vector<apxchol::detail::metal::df32> packed(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = std::sin(0.3 * static_cast<double>(i)) * std::ldexp(1.0, static_cast<int>(i % 40) - 20);
        perm[i] = static_cast<std::uint32_t>((i * 7919) % n);
    }
    mh::scatter(v.data(), perm.data(), n, permuted.data());
    mh::pack(permuted.data(), n, 0.25, packed.data());
    mh::add_unpacked(packed.data(), n, 0.25, back.data());
    mh::gather(back.data(), perm.data(), n, out.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_LE(std::fabs(out[i] - v[i]), std::ldexp(std::fabs(v[i]), -47));
    mh::add_unpacked(packed.data(), n, 0.25, back.data());
    mh::gather(back.data(), perm.data(), n, out.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_LE(std::fabs(out[i] - 2 * v[i]), std::ldexp(std::fabs(v[i]), -46));
}

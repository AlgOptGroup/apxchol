#include <gtest/gtest.h>

#include "apxchol/lowprec.h"
#include "apxchol/solver/detail/metal_host.h"

#include <array>
#include <cmath>
#include <span>
#include <type_traits>

namespace {
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#define APXCHOL_GPU_FUNCTION inline
#include "../src/metal_setup/finalize.inc"
#undef APXCHOL_GPU_FUNCTION
#undef APXCHOL_GPU_CODE
namespace rf = resident_finalize;
namespace mh = apxchol::detail::metal_host;

template<class Value> struct Math {
    using wide = mh::df;
    wide from_float(float x) const { return {x, 0}; }
    float to_float(wide x) const { return x.hi + x.lo; }
    bool less(wide x, wide y) const { return mh::df_lt(x, y); }
    wide add(wide x, wide y) const { return mh::df_add(x, y); }
    wide subtract(wide x, wide y) const { return mh::df_add(x, mh::df_neg(y)); }
    wide multiply(wide x, wide y) const { return mh::df_mul(x, y); }
    wide divide(wide x, wide y) const { return mh::df_div(x, y); }
    float maximum(float x, float y) const { return std::max(x, y); }
    float absolute(float x) const { return std::abs(x); }
    bool finite_float(float x) const { return std::isfinite(x); }
    bool flushes(float x) const { return apxchol::detail::fp16_flushes(x); }
    wide drop(rf::Shape shape) const { return {shape.drop_hi, shape.drop_lo}; }
    float fraction(float x, int& exponent) const { return std::frexp(x, &exponent); }
    float scale_float(float x, int exponent) const { return std::ldexp(x, exponent); }
    Value store(float x, float scale, bool fp16) const {
        return fp16 ? Value(apxchol::detail::narrow_scaled_fp16(x, scale)) : Value(x);
    }
};

struct Edge { unsigned node; float value; };
struct Column { unsigned node, begin, count; float diagonal; };
template<class Value> struct Memory {
    rf::Shape shape;
    bool fp16;
    std::span<const Column> columns;
    Edge* entries;
    std::span<unsigned> perm, ptr, idx;
    std::span<rf::Row> rows;
    std::span<Value> val;
    std::span<rf::Metadata> metadata;
    unsigned* status;
    void fail(unsigned value) const { *status |= value; }
};

template<class Value> struct Fixture {
    // Column 4 is grounded; raw rows are contiguous and both ends have guards.
    std::array<Column, 5> columns{{{3, 0, 4, 8}, {0, 4, 3, 4}, {4, 7, 2, 2},
                                  {1, 9, 1, 1}, {2, 10, 0, 1}}};
    std::array<Edge, 12> raw{{{99, 99}, {2, 64}, {4, 3}, {0, 1}, {1, .5f},
                            {1, 2}, {2, 64}, {4, .25f}, {2, 64}, {1, 1}, {2, 64}, {99, 99}}};
    std::array<unsigned, 5> perm{}, ptr{};
    std::array<unsigned, 9> idx{};
    std::array<rf::Row, 4> rows{};
    std::array<Value, 9> val{};
    std::array<rf::Metadata, 4> metadata{};
    unsigned status = 0;
    rf::Shape shape{5, 4, .25f, 0, 0};

    Memory<Value> memory() {
        Memory<Value> s{shape, std::is_same_v<Value, _Float16>, columns, raw.data() + 1,
                        perm, ptr, idx, rows, val, metadata, &status};
        if (!s.fp16) s.metadata = {};
        return s;
    }
    void run(std::array<unsigned, 4> order) {
        auto s = memory();
        for (unsigned row = 0; row < columns.size(); ++row) rf::permutation(s, row);
        for (unsigned row : order) rf::prepare(s, Math<Value>{}, row);
        unsigned end = 0;
        for (auto& offset : ptr) {
            const unsigned count = offset;
            offset = end;
            end += count;
        }
        for (unsigned row : order) rf::emit(s, Math<Value>{}, row);
    }
    void check_guards() const {
        EXPECT_EQ(status, 0u);
        for (const auto& edge : {raw.front(), raw.back()}) {
            EXPECT_EQ(edge.node, 99u);
            EXPECT_EQ(edge.value, 99);
        }
        EXPECT_EQ(idx.back(), 0u);
        EXPECT_EQ(float(val.back()), 0);
    }
};

TEST(MetalFinalize, InPlaceCompactionGroundingAndDropPreserveOtherRows) {
    for (const auto order : {std::array{0u, 1u, 2u, 3u}, std::array{3u, 2u, 1u, 0u}}) {
        Fixture<float> f;
        f.run(order);
        f.check_guards();
        EXPECT_EQ(f.perm, (std::array{1u, 3u, 4u, 0u, 2u}));
        EXPECT_EQ(f.ptr, (std::array{0u, 3u, 5u, 7u, 8u}));
        const std::array expected_idx{0u, 2u, 1u, 1u, 3u, 2u, 3u, 3u};
        const std::array expected_val{8.f, -3.375f, -1.125f, 4.f, -2.25f, 2.f, -1.f, 1.f};
        for (unsigned p = 0; p < expected_idx.size(); ++p) {
            EXPECT_EQ(f.idx[p], expected_idx[p]);
            EXPECT_FLOAT_EQ(f.val[p], expected_val[p]);
        }
        const std::array scales{3.f, 2.f, 1.f, 1.f};
        for (unsigned row = 0; row < 4; ++row) {
            EXPECT_EQ(f.rows[row].count, f.ptr[row + 1] - f.ptr[row]);
            EXPECT_EQ(f.rows[row].scale, scales[row]);
            EXPECT_EQ(f.rows[row].diagonal, f.columns[row].diagonal);
        }
        // Each of the first two columns loses one off-diagonal to the threshold.
        const std::array factored{4u, 3u, 2u, 1u}, below{1u, 1u, 0u, 0u};
        for (unsigned row = 0; row < 4; ++row) {
            EXPECT_EQ(f.rows[row].factored, factored[row]);
            EXPECT_EQ(f.rows[row].dropped_threshold, below[row]);
        }
    }
}

TEST(MetalFinalize, InPlaceFloat16EmissionPreservesRoundingCompensation) {
    Fixture<_Float16> f;
    f.shape.drop_hi = 0x1p-30f;
    f.raw[4].value = 0x1p-20f;
    f.raw[5].value = f.raw[7].value = 1e-40f;
    f.run({2, 0, 3, 1});
    f.check_guards();
    EXPECT_EQ(f.ptr, (std::array{0u, 3u, 4u, 6u, 7u}));
    EXPECT_EQ(f.metadata[0].scale, 3.f);
    EXPECT_EQ(f.metadata[1].scale, 1.f); // Reciprocal of the raw scale would overflow.
    EXPECT_EQ(f.metadata[1].diagonal, 4.f);
    const float a = -3.f - .75f * 0x1p-20f, b = -1.f - .25f * 0x1p-20f;
    EXPECT_EQ(f.raw[1].value, a);
    EXPECT_EQ(f.raw[2].value, b);
    EXPECT_EQ(f.idx[1], 2u);
    EXPECT_EQ(f.idx[2], 1u);
    EXPECT_EQ(float(f.val[1]), float(_Float16(a / 3.f)));
    EXPECT_EQ(float(f.val[2]), float(_Float16(b / 3.f)));
    const double residual = double(a) / 3 - float(f.val[1]) + double(b) / 3 - float(f.val[2]);
    const float diagonal = float(double(8.f / 3.f) + residual) * 3.f;
    EXPECT_FLOAT_EQ(f.metadata[0].diagonal, diagonal);
    EXPECT_NE(f.metadata[0].diagonal, 8.f);
}
} // namespace

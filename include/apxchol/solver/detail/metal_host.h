#pragma once
// Host arithmetic for Metal PCG. Error-free transforms require explicit FMA
// and no implicit floating-point contraction or reassociation.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <omp.h>

namespace apxchol::detail::metal_host {

/// A double-float: the unevaluated sum hi + lo of two fp32 values, normalized
/// so that |lo| <= ulp(hi) / 2 (about 48 significant bits).
struct df {
    float hi = 0.0f;
    float lo = 0.0f;
};

inline df two_sum(float a, float b) {
    const float s = a + b;
    const float bb = s - a;
    const float e = (a - (s - bb)) + (b - bb);
    return {s, e};
}

inline df quick_two_sum(float a, float b) {
    const float s = a + b;
    const float e = b - (s - a);
    return {s, e};
}

inline df df_add(df a, df b) {
    df s = two_sum(a.hi, b.hi);
    const df t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}

inline df df_neg(df a) { return {-a.hi, -a.lo}; }

inline df two_prod(float a, float b) {
    const float p = a * b;
    const float e = std::fma(a, b, -p);
    return {p, e};
}

inline df df_mul(df a, df b) {
    const float p = a.hi * b.hi;
    float e = std::fma(a.hi, b.hi, -p);
    e = std::fma(a.hi, b.lo, e);
    e = std::fma(a.lo, b.hi, e);
    return quick_two_sum(p, e);
}

inline df df_mul_f(df a, float b) {
    const float p = a.hi * b;
    float e = std::fma(a.hi, b, -p);
    e = std::fma(a.lo, b, e);
    return quick_two_sum(p, e);
}

inline bool df_lt(df a, df b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }

/// fp64 -> double-float: hi = fp32(v), lo = fp32(v - hi). Exact for any v
/// with at most 48 significant bits whose parts stay in the fp32 normal range.
inline df split(double v) {
    const float hi = static_cast<float>(v);
    const float lo = static_cast<float>(v - static_cast<double>(hi));
    return {hi, lo};
}

/// hi + lo in fp64: exact for a normalized double-float.
inline double join(df v) { return static_cast<double>(v.hi) + static_cast<double>(v.lo); }

/// The exact power-of-two scale s = 2^-e with e = ilogb(max_abs), so that the
/// scaled vector's largest magnitude lies in [1, 2). Multiplying by s (and
/// dividing by it afterwards) is exact for every fp64 value that neither
/// underflows nor overflows, so the device computation is the same for
/// b and 2^k b. max_abs must be finite and positive.
inline double pow2_scale(double max_abs) { return std::ldexp(1.0, -std::ilogb(max_abs)); }

/// fp64 value as a double-float, saturated to the largest finite fp32 (used
/// for the stop threshold, whose overflow would mean "stop").
inline df split_saturated(double v) {
    constexpr double kMax = static_cast<double>(std::numeric_limits<float>::max());
    return split(std::min(v, kMax));
}

inline constexpr std::uint32_t kTreeRows = 256;
inline constexpr std::uint32_t kTreeLanes = 16;
inline constexpr std::uint32_t kTreeSteps = 16;
inline std::uint64_t reduction_groups(std::uint64_t n) { return (n + kTreeRows - 1) / kTreeRows; }

// ── Thread-count-independent fp64 folds ─────────────────────────────────────

/// Block size of the host folds: blocks are summed serially, block partials
/// in block order, so the result is the same at every OpenMP team size.
inline constexpr std::size_t kFoldBlock = 4096;

template <class Term>
double fold_sum(std::size_t n, Term&& term) {
    const std::size_t blocks = (n + kFoldBlock - 1) / kFoldBlock;
    std::vector<double> part(blocks, 0.0);
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(blocks); ++b) {
        const std::size_t lo = static_cast<std::size_t>(b) * kFoldBlock;
        const std::size_t hi = std::min(n, lo + kFoldBlock);
        double s = 0.0;
        for (std::size_t i = lo; i < hi; ++i) s += term(i);
        part[static_cast<std::size_t>(b)] = s;
    }
    double s = 0.0;
    for (const double v : part) s += v;
    return s;
}

inline double fold_sum_squares(const double* v, std::size_t n) {
    return fold_sum(n, [v](std::size_t i) { return v[i] * v[i]; });
}

inline double fold_max_abs(const double* v, std::size_t n) {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(v[i]));
    return m;
}

// Permutation and vector packing.

/// To the factor's permuted space: out[perm[v]] = v_in[v].
template <class Index>
void scatter(const double* v_in, const Index* perm, std::size_t n, double* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(n); ++v) out[perm[v]] = v_in[v];
}

/// Back from the permuted space: out[v] = v_perm[perm[v]].
template <class Index>
void gather(const double* v_perm, const Index* perm, std::size_t n, double* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(n); ++v) out[v] = v_perm[perm[v]];
}

inline void pack(const double* w, std::size_t n, double scale, df* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        out[i] = split(scale * w[i]);
}

inline void unpack(const df* v, std::size_t n, double scale, double* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        out[i] = join(v[i]) / scale;
}

// ── The device's reduction tree ─────────────────────────────────────────────

/// The fixed reduction tree of the device kernels. Group g
/// covers rows [256g, 256g + 256); lane l < 16 folds rows 256g + 16s + l for
/// s = 0..15 in order; lanes fold 0..15 in order into partial g; the final
/// lane l folds partials l, l + 16, ... in order, then lanes 0..15. Depends
/// only on n.
template <class Term>
df tree_reduce(std::uint32_t n, Term&& term) {
    const std::uint32_t groups = static_cast<std::uint32_t>(reduction_groups(n));
    std::vector<df> partial(groups);
    for (std::uint32_t g = 0; g < groups; ++g) {
        df lanes[kTreeLanes];
        for (std::uint32_t l = 0; l < kTreeLanes; ++l) {
            df acc{};
            for (std::uint32_t s = 0; s < kTreeSteps; ++s) {
                const std::uint64_t i = std::uint64_t{g} * kTreeRows + s * kTreeLanes + l;
                if (i >= n) break;
                acc = df_add(acc, term(static_cast<std::uint32_t>(i)));
            }
            lanes[l] = acc;
        }
        df t{};
        for (const df& v : lanes) t = df_add(t, v);
        partial[g] = t;
    }
    df lanes[kTreeLanes];
    for (std::uint32_t l = 0; l < kTreeLanes; ++l) {
        df acc{};
        for (std::uint32_t g = l; g < groups; g += kTreeLanes) acc = df_add(acc, partial[g]);
        lanes[l] = acc;
    }
    df t{};
    for (const df& v : lanes) t = df_add(t, v);
    return t;
}

/// 1 / n as the double-float the device multiplies sums by for the
/// Laplacian centring (mean = hi part of sum * inv_n).
inline df inverse_count(std::size_t n) { return split(1.0 / static_cast<double>(n)); }

}  // namespace apxchol::detail::metal_host

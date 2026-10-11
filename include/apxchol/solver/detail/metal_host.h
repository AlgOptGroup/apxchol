#pragma once
// Host arithmetic for Metal PCG. Error-free transforms require explicit FMA
// and no implicit floating-point contraction or reassociation.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <omp.h>

namespace apxchol::detail::metal_host { using std::fma; }
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#include "double_float.inc"
#undef APXCHOL_GPU_CODE

namespace apxchol::detail::metal_host {

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

template <class Pair>
void pack(const double* w, std::size_t n, double scale, Pair* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const df value = split(scale * w[i]);
        out[i] = {value.hi, value.lo};
    }
}

template <class Pair>
void add_unpacked(const Pair* v, std::size_t n, double scale, double* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        out[i] += join({v[i].hi, v[i].lo}) / scale;
}

}  // namespace apxchol::detail::metal_host

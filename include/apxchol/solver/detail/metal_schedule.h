#pragma once
// Host preparation for the Metal triangular-solve kernels. No device is needed
// to build or test these layouts; the CPU and CUDA solvers do not use them.
#include "apxchol/solver/sptrsv/cuda_host.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/types.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace apxchol::detail::metal_schedule {

/// Rows with more dependencies are heavy: one threadgroup per row, the
/// dependencies dealt over kHeavyDeps fixed virtual lanes.
inline constexpr std::uint32_t kHeavyDeps = 32;
/// A level with only light rows and width at most this many items runs
/// inside one merged single-threadgroup step with its narrow neighbours.
inline constexpr std::uint32_t kNarrowItems = 2048;
/// Nonzero factor and operator magnitudes the double-float PCG accepts:
/// [2^-kRangeExponent, 2^kRangeExponent]. Outside it fp32 products and the
/// low parts of double-floats leave the normal range.
inline constexpr int kRangeExponent = 100;

inline bool in_range(double v) {
    const double a = std::fabs(v);
    return a == 0.0 || (a >= std::ldexp(1.0, -kRangeExponent) && a <= std::ldexp(1.0, kRangeExponent));
}

/// One triangular solve ordered by levels. Level l owns slots
/// [level_ptr[l], level_ptr[l + 1]); slots [level_ptr[l], heavy_ptr[l]) are
/// light. Slot t solves row rows[t] from the dependencies
/// [ptr[t], ptr[t + 1]) of (col, val); dinv[t] = fp32(1 / fp64(L_ii)).
struct level_solve {
    std::vector<std::uint32_t> level_ptr;
    std::vector<std::uint32_t> heavy_ptr;
    std::vector<std::uint32_t> rows;
    std::vector<std::uint32_t> ptr;
    std::vector<std::uint32_t> col;
    std::vector<float> val;
    std::vector<float> dinv;

    std::size_t levels() const { return level_ptr.empty() ? 0 : level_ptr.size() - 1; }
    std::size_t slots() const { return rows.size(); }
};

namespace detail {

// deps of unknown i: [dep_begin(i), dep_end(i)) of (idx, vals); its diagonal
// at diag_slot(i). Forward (CSR of L) visits 0..m-1 with dependencies below
// the row, backward (CSC of L) m-1..0 with dependencies above the column.
inline level_solve build(const cuda_host::csr_int<float>& A, bool forward) {
    const std::uint32_t m = static_cast<std::uint32_t>(A.m);
    const int* ptr = A.ptr.data();
    const int* idx = A.idx.get();
    const float* vals = A.vals.get();
    const char* what = forward ? "forward" : "backward";
    auto diag_slot = [&](std::uint32_t i) { return forward ? ptr[i + 1] - 1 : ptr[i]; };
    auto dep_begin = [&](std::uint32_t i) { return forward ? ptr[i] : ptr[i] + 1; };
    auto dep_end = [&](std::uint32_t i) { return forward ? ptr[i + 1] - 1 : ptr[i + 1]; };

    std::vector<std::uint32_t> depth(m, 0);
    std::uint32_t levels = 0;
    for (std::uint32_t k = 0; k < m; ++k) {
        const std::uint32_t i = forward ? k : m - 1 - k;
        if (ptr[i + 1] <= ptr[i] || static_cast<std::uint32_t>(idx[diag_slot(i)]) != i)
            throw std::invalid_argument(std::string("apxchol level_schedule: ") + what +
                                        " row " + std::to_string(i) +
                                        " does not hold its diagonal where expected");
        std::uint32_t d = 0;
        for (int e = dep_begin(i); e < dep_end(i); ++e) {
            const int j = idx[e];
            if (forward ? !(j >= 0 && static_cast<std::uint32_t>(j) < i)
                        : !(static_cast<std::uint32_t>(j) > i && static_cast<std::uint32_t>(j) < m))
                throw std::invalid_argument(std::string("apxchol level_schedule: ") + what +
                                            " row " + std::to_string(i) +
                                            " has an entry outside its triangle");
            d = std::max(d, depth[static_cast<std::uint32_t>(j)] + 1);
        }
        depth[i] = d;
        levels = std::max(levels, d + 1);
    }

    level_solve s;
    s.level_ptr.assign(static_cast<std::size_t>(levels) + 1, 0);
    for (std::uint32_t i = 0; i < m; ++i) ++s.level_ptr[depth[i] + 1];
    for (std::uint32_t l = 0; l < levels; ++l) s.level_ptr[l + 1] += s.level_ptr[l];
    std::vector<std::uint32_t> fill(s.level_ptr.begin(), s.level_ptr.begin() + levels);
    s.rows.assign(m, 0);
    auto heavy = [&](std::uint32_t i) {
        return static_cast<std::uint32_t>(dep_end(i) - dep_begin(i)) > kHeavyDeps;
    };
    for (std::uint32_t i = 0; i < m; ++i)
        if (!heavy(i)) s.rows[fill[depth[i]]++] = i;
    s.heavy_ptr.assign(fill.begin(), fill.end());
    for (std::uint32_t i = 0; i < m; ++i)
        if (heavy(i)) s.rows[fill[depth[i]]++] = i;

    // At most ptr[m] <= INT_MAX dependencies: the 32-bit offsets cannot overflow.
    std::size_t deps = 0;
    for (std::uint32_t i = 0; i < m; ++i) deps += static_cast<std::size_t>(dep_end(i) - dep_begin(i));
    s.ptr.reserve(static_cast<std::size_t>(m) + 1);
    s.ptr.push_back(0);
    s.col.reserve(deps);
    s.val.reserve(deps);
    s.dinv.reserve(m);
    for (const std::uint32_t i : s.rows) {
        for (int e = dep_begin(i); e < dep_end(i); ++e) {
            if (!in_range(vals[e]))
                throw std::domain_error("apxchol level_schedule: factor value " + std::to_string(vals[e]) +
                                        " outside [2^-100, 2^100]");
            s.col.push_back(static_cast<std::uint32_t>(idx[e]));
            s.val.push_back(vals[e]);
        }
        s.ptr.push_back(static_cast<std::uint32_t>(s.col.size()));
        const double d = static_cast<double>(vals[diag_slot(i)]);
        const float inv = static_cast<float>(1.0 / d);
        if (!(std::isfinite(d) && d != 0.0 && in_range(d) && std::isfinite(inv)))
            throw std::domain_error("apxchol level_schedule: factor diagonal " + std::to_string(d) +
                                    " of row " + std::to_string(i) + " is zero, non-finite or outside [2^-100, 2^100]");
        s.dinv.push_back(inv);
    }
    return s;
}

}  // namespace detail

/// Forward solve L y = b from the CSR of L (diagonal last in each row).
inline level_solve build_forward(const cuda_host::csr_int<float>& L) { return detail::build(L, true); }
/// Backward solve L^T x = y from the CSC of L (diagonal first in each column).
inline level_solve build_backward(const cuda_host::csr_int<float>& LT) { return detail::build(LT, false); }

struct factor_schedules {
    level_solve forward;
    level_solve backward;
    factor_drop_stats drop;
};

/// Both schedules of L11 = L.topLeftCorner(m, m) after the compacting drop
/// at `drop_rel` (factor_drop_rel_from_env() for the solver): the drop and
/// transpose the GPU host preparation shares with omp_sptrsv's fp32 storage.
inline factor_schedules build_factor_schedules(const sparse_csc& L, std::int64_t m, double drop_rel) {
    auto LT = cuda_host::build_L11_csc_int<float>(L, m);
    const std::vector<float> scales = cuda_host::column_scales(LT);
    factor_schedules out;
    out.drop = cuda_host::apply_factor_drop(LT, scales, drop_rel, /*fp16_storage=*/false);
    const auto Lr = cuda_host::transpose_csr(LT);
    out.forward = build_forward(Lr);
    out.backward = build_backward(LT);
    return out;
}

enum class step_kind : std::uint32_t { light = 0, heavy = 1, narrow = 2 };

/// light / heavy: slots [first, last) of one level; narrow: levels [first, last).
struct level_step {
    step_kind kind;
    std::uint32_t first;
    std::uint32_t last;
};

/// Runs of narrow levels (light rows only, width <= kNarrowItems) merge
/// into one step; every other level becomes a light and/or a heavy step. The
/// plan changes dispatches only: a row's arithmetic depends on whether it is
/// heavy, never on the step that runs it.
inline std::vector<level_step> plan_steps(const level_solve& s) {
    std::vector<level_step> out;
    const std::size_t L = s.levels();
    auto narrow = [&](std::size_t l) {
        const std::uint64_t width = s.level_ptr[l + 1] - s.level_ptr[l];
        return s.heavy_ptr[l] == s.level_ptr[l + 1] && width <= kNarrowItems;
    };
    for (std::size_t l = 0; l < L;) {
        if (narrow(l)) {
            const std::size_t l0 = l;
            while (l < L && narrow(l)) ++l;
            out.push_back({step_kind::narrow, static_cast<std::uint32_t>(l0), static_cast<std::uint32_t>(l)});
            continue;
        }
        if (s.heavy_ptr[l] > s.level_ptr[l])
            out.push_back({step_kind::light, s.level_ptr[l], s.heavy_ptr[l]});
        if (s.level_ptr[l + 1] > s.heavy_ptr[l])
            out.push_back({step_kind::heavy, s.heavy_ptr[l], s.level_ptr[l + 1]});
        ++l;
    }
    return out;
}

}  // namespace apxchol::detail::metal_schedule

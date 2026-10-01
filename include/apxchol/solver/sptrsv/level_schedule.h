#pragma once
// Level schedules of the two triangular solves of a dropped apxchol factor in
// the layout the Metal block kernels read (src/metal_kernels.inc), plus an
// fp32 host emulation of exactly the arithmetic those kernels perform.
// Portable: no Metal, so the schedule and its emulation are tested on every
// platform (tests/test_level_schedule.cpp).
//
// The factor is prepared by the CUDA-free host code the GPU backends share
// (cuda_host.h: L11 extraction, the compacting drop of factor_drop.h with the
// fp32 keep predicate, the shared CSC -> CSR transpose), so the applied
// operator is the one omp_sptrsv stores on its fp32 storage. Rows are grouped
// by dependency depth (the topological levels omp_sptrsv builds when it has no
// elimination-round metadata). Within a level light rows (at most kHeavyDeps
// dependencies) come first, then heavy rows, each ascending. Dependencies keep
// the CSR / CSC order, which fixes every row's accumulation order.
#include "apxchol/solver/sptrsv/cuda_host.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/types.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace apxchol::level_schedule {

/// Rows with more dependencies are heavy: one threadgroup per row, the
/// dependencies dealt over kHeavyDeps fixed virtual lanes.
inline constexpr std::uint32_t kHeavyDeps = 32;
/// A level with only light rows and width * kc at most this many items runs
/// inside one merged single-threadgroup step with its narrow neighbours.
inline constexpr std::uint32_t kNarrowItems = 2048;
/// Nonzero factor and operator magnitudes the double-float block PCG accepts:
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

    std::size_t deps = 0;
    for (std::uint32_t i = 0; i < m; ++i) deps += static_cast<std::size_t>(dep_end(i) - dep_begin(i));
    if (deps > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("apxchol level_schedule: factor exceeds 32-bit offsets");
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

// ── Dispatch plan ───────────────────────────────────────────────────────────

enum class step_kind : std::uint32_t { light = 0, heavy = 1, narrow = 2 };

/// light / heavy: slots [first, last) of one level; narrow: levels [first, last).
struct level_step {
    step_kind kind;
    std::uint32_t first;
    std::uint32_t last;
};

/// Runs of narrow levels (light rows only, width * kc <= kNarrowItems) merge
/// into one step; every other level becomes a light and/or a heavy step. The
/// plan changes dispatches only: a row's arithmetic depends on whether it is
/// heavy, never on the step that runs it.
inline std::vector<level_step> plan_steps(const level_solve& s, std::uint32_t kc) {
    std::vector<level_step> out;
    const std::size_t L = s.levels();
    auto narrow = [&](std::size_t l) {
        const std::uint64_t width = s.level_ptr[l + 1] - s.level_ptr[l];
        return s.heavy_ptr[l] == s.level_ptr[l + 1] && width * kc <= kNarrowItems;
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

// ── fp32 emulation of the kernels ───────────────────────────────────────────

/// Slot t of one column: light rows run s = rhs; s = fma(-v, z_j, s) in
/// dependency order; z_i = s * dinv. Heavy rows deal dependency k to virtual
/// lane k mod 32 (each lane s_v = fma(v, z_j, s_v) from 0), fold lanes 0..31
/// in order and compute z_i = (rhs - total) * dinv.
inline float solve_slot(const level_solve& s, std::uint32_t t, float rhs, const float* z) {
    const std::uint32_t b = s.ptr[t], e1 = s.ptr[t + 1];
    if (e1 - b <= kHeavyDeps) {
        float acc = rhs;
        for (std::uint32_t e = b; e < e1; ++e) acc = std::fma(-s.val[e], z[s.col[e]], acc);
        return acc * s.dinv[t];
    }
    float lane[kHeavyDeps] = {};
    for (std::uint32_t e = b; e < e1; ++e) {
        float& v = lane[(e - b) % kHeavyDeps];
        v = std::fma(s.val[e], z[s.col[e]], v);
    }
    float total = lane[0];
    for (std::uint32_t v = 1; v < kHeavyDeps; ++v) total = total + lane[v];
    return (rhs - total) * s.dinv[t];
}

/// One sweep in slot order: z[rows[t]] = solve_slot(t, rhs[rows[t]], z).
/// rhs may alias z (the backward sweep runs in place on the forward result).
inline void emulate_sweep(const level_solve& s, const float* rhs, float* z) {
    for (std::uint32_t t = 0; t < s.slots(); ++t) {
        const std::uint32_t i = s.rows[t];
        z[i] = solve_slot(s, t, rhs[i], z);
    }
}

}  // namespace apxchol::level_schedule

#pragma once
#include "apxchol/solver/detail/metal_schedule.h"
#include <cmath>

// Host reference for the Metal kernels, shared by portable and device tests.
namespace apxchol::test {

using detail::metal_schedule::level_solve;
using detail::metal_schedule::kHeavyDeps;

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

}  // namespace apxchol::test

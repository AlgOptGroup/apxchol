#pragma once
// Fixed graph and factor digest for GpuOwnedIdentity (test_gpu_owned_identity.cpp).
// Public API only, so a driver linked against another revision can recompute
// the recorded digests.
#include "apxchol.h"

#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace apxchol_identity {

struct csc_data {
    int n = 0;
    std::vector<int> ptr, idx;
    std::vector<double> val;
    apxchol::csc_view view() const { return {n, ptr, idx, val}; }
};

// A ring with pseudo-random chords and a hub joined to every third vertex:
// many rounds, a small tail, and oversized rows in the first rounds.
inline csc_data identity_matrix(double shift) {
    constexpr int n = 3000;
    std::map<std::pair<int, int>, double> w;
    std::uint64_t state = 0x243F6A8885A308D3ULL;
    const auto weight = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return 0.5 + double(state >> 11) * 0x1.0p-53 * 1.5;
    };
    const auto add = [&](int u, int v) {
        if (u == v) return;
        if (u > v) std::swap(u, v);
        w[{u, v}] += weight();
    };
    for (int u = 0; u < n; ++u) {
        add(u, (u + 1) % n);
        add(u, int((std::int64_t(u) * 7 + 13) % n));
    }
    for (int u = 3; u < n; u += 3) add(0, u);
    std::vector<std::map<int, double>> cols(n);
    for (const auto& [e, x] : w) {
        cols[e.first][e.second] -= x; cols[e.second][e.first] -= x;
        cols[e.first][e.first] += x; cols[e.second][e.second] += x;
    }
    csc_data a;
    a.n = n;
    a.ptr.assign(1, 0);
    for (int c = 0; c < n; ++c) {
        cols[c][c] += shift;
        for (const auto& [r, x] : cols[c]) { a.idx.push_back(r); a.val.push_back(x); }
        a.ptr.push_back(int(a.idx.size()));
    }
    return a;
}

struct factor_digest {
    std::uint64_t structure = 0, values = 0;
    std::size_t rounds = 0;
};

inline void fnv_bytes(std::uint64_t& h, const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
}

template<class T> void fnv(std::uint64_t& h, std::span<const T> s) { fnv_bytes(h, s.data(), s.size_bytes()); }

// Structure: column pointers, row indices, permutation, per-round active and
// pivot counts. Values: fp32/fp16-widened bit patterns and column scales.
inline factor_digest digest(const apxchol::factorization& f) {
    factor_digest d;
    d.structure = d.values = 1469598103934665603ULL;
    fnv(d.structure, f.L.col_ptrs());
    fnv(d.structure, f.L.row_indices());
    fnv(d.structure, std::span<const apxchol::node_index>(f.perm));
    for (const auto& r : f.rounds) {
        const std::uint64_t pair[2] = {r.active, r.is_size};
        fnv(d.structure, std::span<const std::uint64_t>(pair));
    }
    fnv(d.values, f.L.values());
    fnv(d.values, std::span<const float>(f.scale));
    d.rounds = f.rounds.size();
    return d;
}

inline factor_digest owned_factor_digest(double shift, bool trace, unsigned seed) {
    const auto a = identity_matrix(shift);
    apxchol::solve_options so;
    so.backend = apxchol::solve_backend::gpu;
    so.factor_opts.seed = seed;
    if (trace) so.factor_opts.experimental.sampler = apxchol::clique_sampler::trace_cycle;
    apxchol::solver s(a.view(), so);
    return digest(s.factor());
}

} // namespace apxchol_identity

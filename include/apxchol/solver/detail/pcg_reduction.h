#pragma once
#include "apxchol/solver/detail/cpu_parallel.h"
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <omp.h>

namespace apxchol::detail {

// ── Deterministic parallel-reduction scaffolding ─────────────────────────────
// Shared by the fused vector passes of the CPU preconditioner and of the PCG
// loop (src/solve.cpp). Every reduction those passes perform follows one
// scheme so its fp summation order NEVER depends on scheduling:
//   * the index range is cut into the contiguous per-thread chunks that
//     `schedule(static)` (no chunk size) would produce -- a fixed partition
//     for a given (n, thread count);
//   * each thread accumulates its chunk into private registers (a 4-way
//     accumulator split where the pass has no other work to hide the FMA
//     dependency chain), then writes ONE partial into `part[tid * kStride]`;
//   * the partials are summed serially in thread order 0..nt-1.
// No `reduction(+:...)` clause anywhere: libgomp combines those in thread
// COMPLETION order, which varies run to run. Results are therefore
// bit-identical run-to-run for a fixed thread count (they do differ across
// thread counts -- the chunk boundaries move -- exactly like every other
// parallel pass in the solver).

/// Per-thread stride (in doubles) of the partial-sum buffer: one cache line
/// per thread, so the single end-of-chunk write never false-shares.
inline constexpr int kPartStride = 8;

/// Contiguous chunk of [0, n) owned by thread `tid` of `nt`: the first
/// (n mod nt) threads get one extra element, exactly like schedule(static).
inline std::pair<Eigen::Index, Eigen::Index>
static_chunk(Eigen::Index n, int tid, int nt) noexcept {
    const Eigen::Index base = n / nt, rem = n % nt;
    const Eigen::Index lo = static_cast<Eigen::Index>(tid) * base
                          + std::min<Eigen::Index>(tid, rem);
    return {lo, lo + base + (tid < rem ? 1 : 0)};
}

/// Number of doubles the partial buffer must hold: omp_get_max_threads() is,
/// per the OpenMP spec, an upper bound on the team size of any parallel
/// region encountered next using cpu_solve_threads -- so `kPartStride`
/// doubles per that many threads always suffices. Callers keep the buffer
/// as reusable workspace so repeated solves stay allocation-free.
inline std::size_t part_capacity() noexcept {
    return static_cast<std::size_t>(omp_get_max_threads()) * kPartStride;
}

/// Thread id / team size inside a parallel region (0 / 1 for a serial team).
inline void omp_ids(int& tid, int& nt) noexcept {
    tid = omp_get_thread_num();
    nt = omp_get_num_threads();
}

/// Serial thread-order sum of `nt` partials at slot `slot` (< stride).
inline double reduce_parts(const double* part, int nt, int slot = 0, int stride = kPartStride) noexcept {
    double s = 0.0;
    for (int t = 0; t < nt; ++t) s += part[static_cast<std::size_t>(t) * stride + slot];
    return s;
}

/// Deterministic parallel Σ v[i], i in [0, n). `part` must hold
/// part_capacity() doubles. Fixed 4-way accumulator split per chunk.
template<int Columns>
std::array<double, Columns> det_sum_columns(const double* v, Eigen::Index n, double* part) {
    static_assert(Columns > 0 && Columns <= kPartStride);
    int nt_used = 1;
    const int workers = cpu_solve_threads(n);
    #pragma omp parallel if(workers > 1) num_threads(workers)
    {
        int tid, nt; omp_ids(tid, nt);
        if (tid == 0) nt_used = nt;
        const auto [lo, hi] = static_chunk(n, tid, nt);
        std::array<double, Columns> s0{}, s1{}, s2{}, s3{};
        Eigen::Index i = lo;
        for (; i + 4 <= hi; i += 4)
            for (int c = 0; c < Columns; ++c) {
                s0[c] += v[(i + 0) * Columns + c];
                s1[c] += v[(i + 1) * Columns + c];
                s2[c] += v[(i + 2) * Columns + c];
                s3[c] += v[(i + 3) * Columns + c];
            }
        std::array<double, Columns> sum{};
        for (int c = 0; c < Columns; ++c)
            sum[c] = (s0[c] + s1[c]) + (s2[c] + s3[c]);
        for (; i < hi; ++i)
            for (int c = 0; c < Columns; ++c) sum[c] += v[i * Columns + c];
        for (int c = 0; c < Columns; ++c)
            part[static_cast<std::size_t>(tid) * kPartStride + c] = sum[c];
    }
    std::array<double, Columns> sum{};
    for (int c = 0; c < Columns; ++c) sum[c] = reduce_parts(part, nt_used, c);
    return sum;
}

} // namespace apxchol::detail

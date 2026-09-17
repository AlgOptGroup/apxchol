#pragma once

/// Internal helpers shared by the consumers of a caller's compressed operator:
/// the contract scan (src/operator_class.cpp), the graph builder
/// (graph/conversions.h), the owned SpMV operator copy and the PCG SpMV
/// (src/solve.cpp). Not part of the public API.

#include <cstdint>
#include <cstring>
#include <utility>

namespace apxchol::detail {

/// Contiguous range [lo, hi) of the n rows/columns of a compressed matrix that
/// thread `tid` of `nt` owns when the WORK, not the index count, is split
/// evenly. The work of index i is its stored entries plus one,
/// `ptr[i + 1] - ptr[i] + 1`, so empty rows still cost something and the
/// bounds stay strictly monotone in the limit of an all-empty matrix.
///
/// Why not `schedule(static)` by index count: every loop over the rows or
/// columns of the operator does work proportional to the row length, and on
/// matrices with hub rows the equal-count chunks are far from equal work. With
/// 16 threads and the natural labelling, the heaviest chunk carries 2.0x the
/// mean stored entries on the IPM normal equations (iter0040) and 4.0x on
/// as-Skitter; every other thread then waits for it. On uniform meshes the two
/// splits coincide to within one row.
///
/// The bounds are a pure function of (ptr, n, nt): a caller that reduces
/// per-thread partials in thread order stays bit-identical run to run for a
/// fixed thread count, exactly as with detail::static_chunk.
template<class Offset, class Index>
inline std::pair<Index, Index>
work_balanced_range(const Offset* ptr, Index n, int tid, int nt) noexcept {
    if (nt <= 1 || n <= 0) return {Index{0}, n};
    const std::int64_t base = static_cast<std::int64_t>(ptr[0]);
    const std::int64_t work = static_cast<std::int64_t>(ptr[n]) - base
                            + static_cast<std::int64_t>(n);
    auto bound = [&](int t) -> Index {
        if (t <= 0) return Index{0};
        if (t >= nt) return n;
        const std::int64_t target = work * t / nt;   // work < 2^33, t is a thread index
        Index a = 0, b = n;   // first i with (ptr[i] - base) + i >= target
        while (a < b) {
            const Index m = a + (b - a) / 2;
            if (static_cast<std::int64_t>(ptr[m]) - base
                    + static_cast<std::int64_t>(m) < target)
                a = m + 1;
            else
                b = m;
        }
        return a;
    };
    return {bound(tid), bound(tid + 1)};
}

/// Order-independent fingerprint of one stored off-diagonal entry, keyed by its
/// UNORDERED coordinate pair, its position inside a run of duplicate
/// coordinates, and the exact bit pattern of its value. Summing it (mod 2^64)
/// over the strictly lower and over the strictly upper triangle gives two
/// numbers that are equal whenever the triangles hold the same entries bit for
/// bit, in any traversal order and under any thread schedule. Unequal triangles
/// collide with probability about 2^-64; every consumer treats "sums differ"
/// as "fall back to the full per-entry check", never the reverse.
inline std::uint64_t symmetric_entry_hash(std::uint64_t smaller,
                                          std::uint64_t larger,
                                          std::uint64_t run_position,
                                          double value) noexcept {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    std::uint64_t z = (smaller << 32) ^ larger;
    z = (z ^ (bits * 0x9E3779B97F4A7C15ull))
      + 0x9E3779B97F4A7C15ull * (run_position + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

} // namespace apxchol::detail

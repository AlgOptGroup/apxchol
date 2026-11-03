#pragma once
/// Conversions between Eigen sparse matrices and graphs.
///
///   laplacian(g)     — graph → Eigen Laplacian
///   make_graph(L) — Eigen Laplacian → graph

#include "apxchol/csc_work.h"
#include "apxchol/detail/csc_matrix.h"
#include "apxchol/detail/eigen_boundary.h"
#include "apxchol/detail/operator_options.h"
#include "apxchol/graph/graph.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <ranges>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace apxchol::detail {

/// Build the graph Laplacian L = D - A as a sparse matrix.
template<typename G>
Eigen::SparseMatrix<double> laplacian(const G& g) {
    using T = Eigen::Triplet<double>;
    const node_index N = g.n();
    std::vector<T> trips;

    for (node_index v = 0; v < N; ++v) {
        double deg = 0.0;
        for (const auto& e : g.neighbors(v)) {
            trips.emplace_back(v, e.to, -e.w);
            deg += e.w;
        }
        trips.emplace_back(v, v, deg);
    }

    Eigen::SparseMatrix<double> L(N, N);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

/// make_graph with the caller's proof about the operator. `triangles_bit_identical`
/// must come from a fresh `operator_scan` of this very matrix
/// (`operator_scan::triangles_bit_identical`): the strictly lower and upper
/// triangles then hold the same entries bit for bit, so the full-column builder
/// below may take an upper entry's own value as the canonical LOWER value
/// instead of binary-searching its transpose partner -- about log2(column
/// length) cache misses per entry, landing in the hub columns of power-law and
/// IPM operators. Pass false whenever that proof is not at hand; the result is
/// identical either way, only the work differs.
inline graph make_graph_from_operator(csc_ref<double> L,
                           bool triangles_bit_identical, double regularization = 0) {
    validate_regularization(regularization);
    const node_index n = L.n;
    const node_index outer = L.n;
    graph g(n);

    std::vector<double> diag(n, 0.0);

    // This builder owns a fresh graph. A caller may already be in an unrelated
    // OpenMP team; do not partition this private graph across that team.
    const auto reserve_initial = [&](auto begin, auto end, const auto& incoming) {
        if (omp_in_parallel()) {
            for (auto it = begin; it != end; ++it)
                g.adj_reserve_for(*it, incoming[*it]);
        } else {
            g.adj_bulk_reserve_parallel(begin, end, incoming);
        }
    };
    // A full symmetric CSC already lists every incidence of a vertex in its
    // column. Give that column one writer instead of atomically counting and
    // appending both endpoints of each lower-triangle entry, then sorting.
    // The general path below remains authoritative for duplicate indices
    // and one-triangle/asymmetric stored patterns.
    bool built_from_columns = false;
    {
        {
            const edge_index* ptr = L.ptr;
            const node_index* idx = L.idx;
            const double* val = L.val;
            std::vector<node_index> incoming(static_cast<size_t>(n), 0);
            bool unique_sorted = true;
            // Columns are split by stored entries, not by count: with hub
            // columns the equal-count chunks are far from equal work (see
            // detail::work_balanced_range). Both loops reduce integers and
            // booleans only, so the split cannot change the result.
            #pragma omp parallel reduction(&& : unique_sorted)
            {
            const auto [col_lo, col_hi] = detail::work_balanced_range(
                ptr, n, omp_get_thread_num(), omp_get_num_threads());
            for (node_index col = col_lo; col < col_hi; ++col) {
                node_index count = 0;
                for (edge_index p = ptr[col]; p < ptr[col + 1]; ++p) {
                    if (p > ptr[col] && idx[p - 1] >= idx[p])
                        unique_sorted = false;
                    if (idx[p] == col) diag[col] = val[p];
                    else ++count;
                }
                incoming[col] = count;
            }
            }
            if (unique_sorted) {
                // Reserve in the same vertex order as the general builder.
                // iota avoids another materialized O(n) touched-vertex list.
                const auto vertices = std::views::iota(node_index{0}, n);
                reserve_initial(
                    vertices.begin(), vertices.end(), incoming);
                edge_index lower = 0, upper = 0;
                bool paired = true;
                #pragma omp parallel reduction(+ : lower, upper) reduction(&& : paired)
                {
                const auto [col_lo, col_hi] = detail::work_balanced_range(
                    ptr, n, omp_get_thread_num(), omp_get_num_threads());
                for (node_index col = col_lo; col < col_hi; ++col) {
                    node_index offset = 0;
                    for (edge_index p = ptr[col]; p < ptr[col + 1]; ++p) {
                        const node_index row = idx[p];
                        if (row == col) continue;
                        double weight = -val[p];
                        if (row < col && triangles_bit_identical) {
                            // Proven bit-identical triangles: val[p] IS the
                            // canonical lower value, and its partner exists.
                            ++upper;
                        } else if (row < col) {
                            ++upper;
                            const node_index* mate = std::lower_bound(
                                idx + ptr[row], idx + ptr[row + 1], col);
                            if (mate == idx + ptr[row + 1] || *mate != col) {
                                paired = false;
                            } else {
                                // Like the general builder, both incidences
                                // use the LOWER value, including tolerated
                                // asymmetry, signed zero, and fp32 narrowing.
                                weight = -val[mate - idx];
                            }
                        } else {
                            ++lower;
                        }
                        g.adj_write_reserved_directed_at(col, offset++, row, weight);
                    }
                    g.adj_commit_reserved_directed(col, offset);
                }
                }
                if (paired && upper == lower) {
                    g.record_edges_added(lower);
                    built_from_columns = true;
                } else {
                    // The speculative slabs have never escaped this call.
                    // Rebuild from the lower triangle under the old contract.
                    g = graph(n);
                }
            }
        }
    }
    if (!built_from_columns) {
        // Split both column passes by stored entries across the actual team.
        // Atomic counts and endpoint writes are independent of that split;
        // sorting each touched slab restores deterministic neighbor order.
        const int nt = std::max(1, omp_get_max_threads());
        const edge_index* outer_ptr = L.ptr;
        std::vector<node_index> incoming(static_cast<size_t>(n), 0);

        // PASS 1: atomic count of incoming + collect diag.
        #pragma omp parallel
        {
            const auto [col_lo, col_hi] = detail::work_balanced_range(outer_ptr, outer);
            for (node_index k = col_lo; k < col_hi; ++k) {
                for (edge_index p = L.ptr[k]; p < L.ptr[k + 1]; ++p) {
                    const node_index row = L.idx[p];
                    if (row > k) {
                        std::atomic_ref(incoming[row]).fetch_add(1, std::memory_order_relaxed);
                        std::atomic_ref(incoming[k]).fetch_add(1, std::memory_order_relaxed);
                    } else if (row == k) {
                        diag[k] = L.val[p];
                    }
                }
            }
        }

        // Touched-vertex list + total off-diag nnz from incoming[].
        // Parallel: per-thread touched lists over a static v-range (order within
        // touched is irrelevant -- it just drives bulk-reserve + per-slab sort),
        // total_inc via reduction. Replaces the serial O(n) scan.
        std::vector<node_index> touched;
        edge_index total_inc = 0;   // sum of incoming = 2 * #edges (edge-scale)
        {
            std::vector<std::vector<node_index>> tt(nt);
            parallel_exception error;
            // These are logical ranges, not runtime worker IDs. OpenMP may
            // provide fewer workers (thread limits or a nested region).
            #pragma omp parallel for num_threads(nt) schedule(static) reduction(+:total_inc)
            for (int t = 0; t < nt; ++t) error.run([&] {
                const node_index vb = static_cast<node_index>(int64_t(n) * t / nt);
                const node_index ve = static_cast<node_index>(int64_t(n) * (t + 1) / nt);
                auto& mine = tt[t];
                mine.reserve(static_cast<size_t>((ve - vb) / 2));
                for (node_index v = vb; v < ve; ++v) {
                    total_inc += incoming[v];
                    if (incoming[v] > 0) mine.push_back(v);
                }
            });
            error.rethrow();
            size_t total = 0;
            for (auto& l : tt) total += l.size();
            touched.reserve(total);
            for (auto& l : tt) touched.insert(touched.end(), l.begin(), l.end());
        }
        const edge_index off_diag_nnz = total_inc / 2;  // each edge counted twice

        if (off_diag_nnz > 0) {
            reserve_initial(touched.begin(), touched.end(), incoming);
            // incoming[] (n counts) is consumed by the reserve; the edge pass
            // below reads only the matrix. Free it here, not at return.
            std::vector<node_index>().swap(incoming);
            g.record_edges_added(off_diag_nnz);

            // PASS 2: write both incidences of each lower-triangle entry.
            #pragma omp parallel
            {
                const auto [col_lo, col_hi] = detail::work_balanced_range(outer_ptr, outer);
                for (node_index k = col_lo; k < col_hi; ++k)
                    for (edge_index p = L.ptr[k]; p < L.ptr[k + 1]; ++p) {
                        const node_index u = L.idx[p];
                        if (u > k) {
                            g.adj_atomic_push_directed(u, k, -L.val[p]);
                            g.adj_atomic_push_directed(k, u, -L.val[p]);
                        }
                    }
            }
            // Sort each touched vertex's adjacency slab for deterministic order
            // (atomic-push arrival order is thread-interleaved).
            #pragma omp parallel for schedule(static) if(touched.size() > 1024)
            for (std::size_t i = 0; i < touched.size(); ++i)
                g.adj_sort_slab(touched[i]);
        }
    }

    // Phase 2: excess = diag − weighted degree.  Zero for pure Laplacians.
    //
    // The weighted degree MUST come from the exact fp64 input values, not
    // from a re-walk of the stored edges: the
    // vec_pool slabs hold fp32-quantized weights, so exact diag[v] minus a
    // quantized wdeg leaves ≈1e-8·diag of phantom excess — far above the
    // 1e-12 gate — and a pure Laplacian gets misclassified as SDDM.
    //
    // L is symmetric (Laplacian/SDDM contract), so column v contains every
    // neighbor of v: summing −L(u,v) over the full column yields each
    // vertex's exact weighted degree with a fixed accumulation order
    // (the stored column order) — deterministic across thread counts
    // and identical for all storage backends. Per-column work is
    // independent → safe in parallel.
    //
    // The relative 1e-12 tolerance absorbs fp accumulation-order ulps
    // (column-sum order vs. the ingestion order that built diag[]).
    //
    // Optional diagonal regularization makes the preconditioner full-rank;
    // the solve still applies the caller's original operator.
    // Each column's sum stays serial, so the work-balanced split cannot
    // change the result.
    bool invalid_excess = false;
    #pragma omp parallel if(n > 16384) reduction(||:invalid_excess)
    {
        const auto [col_lo, col_hi] = detail::work_balanced_range(L.ptr, outer);
        for (node_index k = col_lo; k < col_hi; ++k) {
            double wdeg = 0.0;
            for (edge_index p = L.ptr[k]; p < L.ptr[k + 1]; ++p)
                if (L.idx[p] != k)
                    wdeg -= L.val[p];
            double excess = diag[k] - wdeg;
            excess = excess > diag[k] * 1e-12 ? excess : 0.0;
            if (regularization > 0.0)
                excess = std::max(excess, regularization * diag[k]);
            invalid_excess |= !std::isfinite(excess) ||
                (regularization > 0.0 && diag[k] > 0.0 && excess == 0.0);
            g.excess(k) = excess;
        }
    }
    if (invalid_excess)
        throw std::range_error("regularization produces an unrepresentable diagonal excess");
    return g;
}

/// Build a graph from a Laplacian or SDDM matrix.
///
/// Off-diagonal entries become weighted edges (negated).
/// Any excess diagonal (row sum > 0, i.e. SDDM) is stored
/// per-vertex in graph::excess() for use during factorization.
inline graph make_graph(const Eigen::SparseMatrix<double>& L) {
    const detail::eigen_csc input(L);
    if (!input.square()) throw std::invalid_argument("make_graph: matrix must be square");
    return detail::make_graph_from_operator(input, /*triangles_bit_identical=*/false);
}

} // namespace apxchol::detail

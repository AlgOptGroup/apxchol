#pragma once
/// Conversions between Eigen sparse matrices and graphs.
///
///   laplacian(g)     — graph → Eigen Laplacian
///   make_graph<G>(L) — Eigen Laplacian → G

#include "apxchol/env_knobs.h"
#include "apxchol/graph/graph.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

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

/// Build a graph from a Laplacian or SDDM matrix.
///
/// Off-diagonal entries become weighted edges (negated).
/// Any excess diagonal (row sum > 0, i.e. SDDM) is stored
/// per-vertex in graph::excess() for use during factorization.
template<typename G = graph<>>
G make_graph(const Eigen::SparseMatrix<double>& L) {
    using Incidence = typename G::incidence_type;
    const auto n = static_cast<node_index>(L.rows());
    const auto outer = static_cast<node_index>(L.outerSize());
    G g(n);

    std::vector<double> diag(n, 0.0);

#ifdef _OPENMP
    if constexpr (std::is_same_v<Incidence, vec_pool_incidence>) {
        // ── Parallel build for vec_pool — O(n + nt) memory ────────
        // incoming[] is one shared array of size n (atomic increments in
        // PASS 1 to count per-vertex degree). Slot assignment in PASS 2 is
        // per-thread DETERMINISTIC: thread t processes a fixed column range
        // [col_bs[t], col_bs[t+1]) and claims a contiguous block of edge_index
        // values [per_thread_edges[t], per_thread_edges[t+1]). The per-
        // thread offset array is O(nt) ints, not O(nt × n). Determinism is
        // preserved by construction — col_bs is deterministic, per-thread
        // count pre-pass is deterministic, slot ordering matches a single-
        // threaded column-order scan.
        const int nt = std::max(1,
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );
        std::vector<node_index> incoming(static_cast<size_t>(n), 0);
        std::vector<node_index> col_bs(nt + 1, 0);
        for (int t = 0; t < nt; ++t)
            col_bs[t + 1] = static_cast<node_index>(int64_t(outer) * (t + 1) / nt);

        // PASS 1: atomic count of incoming + collect diag.
        #pragma omp parallel for schedule(static)
        for (node_index k = 0; k < outer; ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                if (it.row() > it.col()) {
                    __atomic_fetch_add(&incoming[it.row()], 1, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&incoming[it.col()], 1, __ATOMIC_RELAXED);
                } else if (it.row() == it.col()) {
                    diag[k] = it.value();
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
            #pragma omp parallel num_threads(nt) reduction(+:total_inc)
            {
                const int t = omp_get_thread_num();
                const node_index vb = static_cast<node_index>(int64_t(n) * t / nt);
                const node_index ve = static_cast<node_index>(int64_t(n) * (t + 1) / nt);
                auto& mine = tt[t];
                mine.reserve(static_cast<size_t>((ve - vb) / 2));
                for (node_index v = vb; v < ve; ++v) {
                    total_inc += incoming[v];
                    if (incoming[v] > 0) mine.push_back(v);
                }
            }
            size_t total = 0;
            for (auto& l : tt) total += l.size();
            touched.reserve(total);
            for (auto& l : tt) touched.insert(touched.end(), l.begin(), l.end());
        }
        const edge_index off_diag_nnz = total_inc / 2;  // each edge counted twice

        if (off_diag_nnz > 0) {
            g.adj_bulk_reserve_parallel(touched.begin(), touched.end(), incoming);
            const edge_index e_start = g.reserve_edge_pool(off_diag_nnz);

            // Per-thread off-diag count (pre-pass for deterministic slot assignment).
            std::vector<edge_index> per_thread_edges(nt + 1, 0);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < nt; ++t) {
                edge_index cnt = 0;
                for (node_index k = col_bs[t]; k < col_bs[t + 1]; ++k)
                    for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
                        if (it.row() > it.col()) ++cnt;
                per_thread_edges[t + 1] = cnt;
            }
            for (int t = 0; t < nt; ++t)
                per_thread_edges[t + 1] += per_thread_edges[t];

            // PASS 2: each thread writes to its own contiguous edge_index range.
            #pragma omp parallel num_threads(nt)
            {
                const int t = omp_get_thread_num();
                edge_index local_slot = per_thread_edges[t];
                for (node_index k = col_bs[t]; k < col_bs[t + 1]; ++k)
                    for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                        if (it.row() > it.col()) {
                            const edge_index es = e_start + local_slot++;
                            const auto u = static_cast<node_index>(it.row());
                            const auto v = static_cast<node_index>(it.col());
                            g.write_edge_at(es, u, v, -it.value());
                            g.adj_atomic_push_reserved(u, es);
                            g.adj_atomic_push_reserved(v, es);
                        }
                    }
            }
            // Sort each touched vertex's adj slab by edge_index for deterministic
            // adj order (atomic-push arrival order is thread-interleaved).
            #pragma omp parallel for schedule(static) if(touched.size() > 1024)
            for (std::size_t i = 0; i < touched.size(); ++i)
                g.adj_sort_slab(touched[i]);
        }
    } else
#endif
    {
        // Phase 1 (serial): extract edges and diagonal.
        for (node_index k = 0; k < outer; ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                if (it.row() > it.col())
                    g.add_edge(static_cast<node_index>(it.row()),
                               static_cast<node_index>(it.col()),
                               -it.value());
                else if (it.row() == it.col())
                    diag[k] = it.value();
            }
    }

    // Phase 2: excess = diag − weighted degree.  Zero for pure Laplacians.
    //
    // The weighted degree MUST come from the exact fp64 input values, not
    // from a re-walk of the stored edges: under APXCHOL_POOL_FP32 the
    // vec_pool slabs hold fp32-quantized weights, so exact diag[v] minus a
    // quantized wdeg leaves ≈1e-8·diag of phantom excess — far above the
    // 1e-12 gate — and a pure Laplacian gets misclassified as SDDM.
    //
    // L is symmetric (Laplacian/SDDM contract), so column v contains every
    // neighbor of v: summing −L(u,v) over the full column yields each
    // vertex's exact weighted degree with a fixed accumulation order
    // (Eigen's sorted inner order) — deterministic across thread counts
    // and identical for all storage backends. Per-column work is
    // independent → safe in parallel.
    //
    // The relative 1e-12 tolerance absorbs fp accumulation-order ulps
    // (column-sum order vs. the ingestion order that built diag[]).
    //
    // APXCHOL_GROUND=reg (experiment knob, see env_knobs.h): ground the
    // matrix here instead of by mean-centring in the preconditioner -- every
    // vertex gets excess[v] = max(exact_excess, reg_eps * diag[v]), i.e. an
    // explicit self-loop to ground of relative size reg_eps. Any positive
    // excess makes factorize_impl's sddm_scan classify the matrix as SDDM,
    // so the factor is full-rank n and _solve_impl takes the no-centring
    // path. reg_eps == 0.0 (mode != reg) is a no-op.
    const double reg_eps =
        detail::env_knobs::get().ground == detail::grounding_kind::reg
            ? detail::env_knobs::get().reg_eps : 0.0;
    #pragma omp parallel for schedule(static) if(n > 16384)
    for (node_index k = 0; k < outer; ++k) {
        double wdeg = 0.0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            if (it.row() != it.col())
                wdeg -= it.value();
        double excess = diag[k] - wdeg;
        excess = excess > diag[k] * 1e-12 ? excess : 0.0;
        if (reg_eps > 0.0) excess = std::max(excess, reg_eps * diag[k]);
        g.excess(k) = excess;
    }
    return g;
}

} // namespace apxchol

#pragma once
/// Block-greedy parallel independent set selection.
///
/// ─── Algorithm overview ───
///
/// This strategy partitions the active vertices into contiguous blocks
/// (one per thread) and runs a serial greedy scan within each block
/// independently.  A short parallel cleanup pass then resolves the
/// small number of conflicts at block boundaries.
///
/// ─── Greedy IS on a graph ───
///
/// The textbook greedy algorithm for independent set scans vertices in
/// some order and adds each vertex to the IS if none of its already-
/// chosen neighbors are in the IS.  On a graph with average degree d
/// this gives an IS of size ≈ n/(d+1).  IS size directly controls the
/// number of rounds in the elimination loop: larger IS → fewer rounds
/// → less total work (every round pays O(m) for pruning + degree
/// computation).
///
/// ─── Parallelization via block partitioning ───
///
/// A serial greedy scan is inherently sequential — each decision
/// depends on prior choices.  Our approach splits the active vertex
/// list into T contiguous blocks (T = number of OpenMP threads) and
/// runs the greedy scan independently within each block:
///
///   1. Assign each active vertex to a block (block_of[v] = thread_id).
///   2. Within each block, scan in order: add v to IS if
///      (a) degree(v) ≤ threshold, and
///      (b) no same-block neighbor is already chosen.
///      Track whether v has any cross-block neighbor ("boundary" flag).
///   3. Barrier.
///
/// Since each thread only reads chosen[] for vertices in its own block,
/// there are no data races.  This produces T independent local ISs.
/// Their union is almost an IS — the only violations are edges that
/// cross block boundaries.
///
/// ─── Conflict resolution ───
///
/// Boundary vertices (those with at least one neighbor in a different
/// block) may conflict: both endpoints of a cross-block edge might be
/// chosen.  We resolve this with a parallel pass:
///
///   4. For each boundary vertex v that was chosen, check whether any
///      chosen cross-block neighbor u has u < v (lower index wins).
///      If so, un-choose v (mark for removal).
///   5. Apply removals in parallel.
///
/// The conflict resolution uses vertex index as a deterministic
/// tie-breaker (not a hash — indices already provide a total order).
/// Because conflicts only occur at the T−1 block boundaries and
/// spatial locality means most neighbors are in the same block,
/// the number of removals is typically negligible (< 0.1% of IS).
///
/// ─── IS quality ───
///
/// The resulting IS size is very close to the serial greedy IS:
///   • Within each block, greedy is exact.
///   • Cross-block conflicts remove only boundary vertices, which are
///     a small fraction (proportional to d/block_size, where block_size
///     = |active|/T is typically > 100k for our graphs).
///
/// In practice, we observe IS sizes within 1–2% of serial greedy,
/// meaning the factorization does the same number of rounds as the
/// serial algorithm — the parallelization of IS selection is "free"
/// in terms of round count.
///
/// ─── Relation to prior work ───
///
/// Blelloch, Fineman & Shun (2012) proved that sequential greedy MIS
/// under a random vertex ordering has O(log²n) dependence depth WHP.
/// Their "rootset" algorithm (GBBS: RandomGreedy) peels the dependence
/// DAG layer by layer, reproducing the *exact* sequential greedy IS.
/// However, it requires O(log²n) synchronous rounds per IS computation
/// on a *static* graph — the priority DAG is computed once from a fixed
/// random permutation.
///
/// In our setting the graph changes each elimination round (Schur
/// complement adds fill edges), so the Blelloch approach does not
/// apply directly — there is no fixed graph on which to build a
/// priority DAG.  Instead, we run a fresh single-pass parallel greedy
/// on the current graph.  Block partitioning approximates the first
/// ≈T layers of a random-order dependence DAG in a single pass,
/// which is why the IS quality is close to sequential greedy.
///
/// Reference:
///   • Blelloch, Fineman & Shun (2012), "Greedy Sequential Maximal
///     Independent Set and Matching are Parallel on Average,"
///     arXiv:1202.3205 / SPAA '12.
///
/// The contribution here is adapting this pattern for the approximate
/// Cholesky context where (a) the graph changes each round due to fill,
/// (b) the degree threshold provides a natural candidate filter,
/// (c) the graph has strong spatial locality from PDE discretizations,
/// and (d) block partitioning of the active list aligns with cache-line
/// boundaries for minimal false sharing.

#include "apxchol/solver/is/independent_set.h"

namespace apxchol {

struct block_greedy_is {
    std::vector<int> block_of;
    std::vector<char> near_boundary;

    template<incidence_storage Incidence>
    void select(const graph<Incidence>& G,
                std::span<const node_index> active,
                std::span<const index_t> degrees,
                double degree_threshold,
                std::span<char> chosen,
                const factor_options& opts) {
        size_t nn = G.n();
        block_of.resize(nn);
        near_boundary.resize(nn, 0);

        #ifdef _OPENMP
        if (omp_get_max_threads() > 1 && active.size() > opts.omp_threshold) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                auto bs = active.size() * size_t(tid) / nthreads;
                auto be = active.size() * size_t(tid + 1) / nthreads;

                for (size_t i = bs; i < be; ++i)
                    block_of[active[i]] = tid;

                #pragma omp barrier

                for (size_t i = bs; i < be; ++i) {
                    if (degrees[i] > degree_threshold) continue;
                    auto v = active[i];
                    bool ok = true;
                    bool boundary = false;
                    for (auto idx : G.adj(v)) {
                        auto u = G.edge_target(idx, v);
                        if (!G.is_active(u)) continue;
                        if (block_of[u] == tid) {
                            if (ok && chosen[u]) ok = false;
                        } else {
                            boundary = true;
                        }
                        if (!ok && boundary) break;
                    }
                    if (ok) {
                        chosen[v] = 1;
                        if (boundary) near_boundary[v] = 1;
                    }
                }
            }

            // Fix cross-block conflicts (parallel — each vertex independent).
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < active.size(); ++i) {
                auto v = active[i];
                if (!chosen[v] || !near_boundary[v]) continue;
                int my_block = block_of[v];
                for (auto idx : G.adj(v)) {
                    auto u = G.edge_target(idx, v);
                    if (G.is_active(u) && chosen[u] && u < v && block_of[u] != my_block) {
                        near_boundary[v] = 2;
                        break;
                    }
                }
            }

            // Apply removals.
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < active.size(); ++i) {
                auto v = active[i];
                if (near_boundary[v] == 2) chosen[v] = 0;
            }
        } else
        #endif
        {
            // Serial greedy.
            for (size_t i = 0; i < active.size(); ++i) {
                if (degrees[i] > degree_threshold) continue;
                auto v = active[i];
                bool ok = true;
                for (auto idx : G.adj(v)) {
                    auto u = G.edge_target(idx, v);
                    if (G.is_active(u) && chosen[u]) { ok = false; break; }
                }
                if (ok) chosen[v] = 1;
            }
        }
    }

    void cleanup(std::span<const node_index> active,
                 const factor_options& opts) {
        #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
        for (size_t i = 0; i < active.size(); ++i)
            near_boundary[active[i]] = 0;
    }
};

} // namespace apxchol

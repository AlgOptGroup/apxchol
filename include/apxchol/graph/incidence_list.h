#pragma once
/// Incidence list storage backends for graph.
///
/// Each backend manages per-vertex lists of one value type. Most backends store
/// edge_index values into graph's flat undirected-edge pool. The experimental
/// directed vec_pool stores {neighbor, weight} records inline instead.
///
/// Required interface:
///   init(n)                   — set up n empty vertex lists
///   operator[](v) const       — range of backend values for vertex v
///   push(v, value)            — append one backend value to vertex v
///   clear(v)                  — clear vertex v's list
///   filter(v, pred)           — erase entries where pred is false
///   filter(v, pred, on_keep)  — same, calling on_keep for survivors
///   memory_bytes() const      — approximate heap usage in bytes
///   tag                       — static constexpr graph_storage value

#include "apxchol/types.h"
#include "apxchol/trivial_char_traits.h"
#include "apxchol/graph/forward_star.h"
#include "apxchol/big_alloc.h"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace apxchol {

namespace detail {
    struct edge_pred  {
        template<typename T> bool operator()(const T&) const;
    };
    struct edge_visit {
        template<typename T> void operator()(const T&) const;
    };
}

/// Concept for incidence list storage backends.
template<typename T>
concept incidence_storage = requires(T t, const T ct, node_index n, node_index v,
                                      detail::edge_pred pred, detail::edge_visit visit) {
    t.init(n);
    { ct[v] } -> std::ranges::input_range;
    t.push(v, *std::ranges::begin(ct[v]));
    t.clear(v);
    t.filter(v, pred);
    t.filter(v, pred, visit);
    { ct.memory_bytes() } -> std::convertible_to<std::size_t>;
    { T::tag } -> std::convertible_to<const graph_storage&>;
};

// ── Contiguous storage (vector / basic_string / …) ──

template<typename Container, graph_storage Tag>
struct contiguous_incidence {
    static constexpr graph_storage tag = Tag;

    void init(node_index n) { adj_.assign(n, {}); }

    std::span<const edge_index> operator[](node_index v) const {
        return {adj_[v].data(), adj_[v].size()};
    }

    void push(node_index v, edge_index idx) { adj_[v].push_back(idx); }
    void clear(node_index v) { adj_[v].clear(); }

    /// Remove elements from v's list where pred(idx) is false.
    /// Calls on_keep(idx) for each surviving element.
    void filter(node_index v, auto&& pred, auto&& on_keep) {
        auto& list = adj_[v];
        list.erase(std::ranges::remove_if(list, std::not_fn(pred)).begin(),
                   list.end());
        for (auto idx : list) on_keep(idx);
    }

    void filter(node_index v, auto&& pred) {
        filter(v, pred, [](edge_index) {});
    }

    /// Approximate memory usage in bytes (heap only).
    std::size_t memory_bytes() const {
        std::size_t total = adj_.capacity() * sizeof(Container);
        for (const auto& c : adj_)
            total += c.capacity() * sizeof(edge_index);
        return total;
    }

private:
    std::vector<Container> adj_;
};

using vec_incidence = contiguous_incidence<std::vector<edge_index>, graph_storage::vec>;

// ── basic_string SSO storage ──

using bstr_incidence = contiguous_incidence<
    std::basic_string<edge_index>, graph_storage::bstr>;

// ── forward_star<edge_index> satisfies incidence_storage directly ──

using forward_star_incidence = forward_star<edge_index>;

// ── vec_pool storage ──
//
// One contiguous arena of Value slots, with per-vertex
// {base, count, capacity} triples.  Push appends to a vertex's
// reserved slab; if the slab fills, the vertex's slab is moved to
// the end of the pool with doubled capacity (in-place doubling
// would shuffle later vertices).
//
// Compared to vec_incidence: same per-vertex contiguity for
// iteration, but no per-vertex std::vector wrapper (saves 3 words
// of header per vertex) and a single pool reallocation grows ALL
// vertices' headroom amortized via the global arena.
//
// Compared to forward_star_incidence: no linked-list pointer chase
// during iteration — adjacency reads are sequential.  Cost: doubling
// reallocates by copy (linked-list just appends a node).
//
// Parallel push API:
//   reserve_for(v, need)         — grow v's slab so cap_[v] >= need (serial)
//   atomic_push_reserved(v, idx) — lock-free push assuming caller has called
//                                  reserve_for to size the slab; multiple
//                                  threads may invoke concurrently on the same
//                                  or different v's. count_[v] is the atomic
//                                  counter.
//
// Tunable knobs (compile-time):
//   APXCHOL_VEC_POOL_SLAB_ALIGN  -- align each newly-allocated slab to this
//       number of edge_index slots (1 = no alignment; 16 = 64B = one cache
//       line for int32, also covers AVX2/AVX-512 vector boundaries).
//   APXCHOL_VEC_POOL_ALIGN_THRESHOLD -- only align when slab cap >= this
//       threshold (default = slab_align, so tiny slabs avoid padding waste).
//       Set to 1 to align every slab; set huge to disable alignment.
//
// On grids (avg final degree ~5), every grow() of a tiny vertex would waste
// ~slab_align/2 slots per grow on average → at 5 grows × 4M vertices ×
// 32 bytes ≈ 640 MB pure padding on grid_2000.  Conditional alignment
// (only large slabs) keeps padding small while still cache-aligning the
// slabs that actually span multiple cache lines.
#ifndef APXCHOL_VEC_POOL_SLAB_ALIGN
#define APXCHOL_VEC_POOL_SLAB_ALIGN 16
#endif
#ifndef APXCHOL_VEC_POOL_ALIGN_THRESHOLD
#define APXCHOL_VEC_POOL_ALIGN_THRESHOLD APXCHOL_VEC_POOL_SLAB_ALIGN
#endif

// Runtime tunables (env-var driven, evaluated once at first init).
// These let us A/B alignment + initial-cap policies without rebuilding.
//
//   APXCHOL_VEC_POOL_SLAB_ALIGN_ENV  (int)  -- override slab_align in slots.
//                                              0 = use compile-time default.
//   APXCHOL_VEC_POOL_ALIGN_MODE_ENV  (0|1)  -- 0 (default) = align only when
//                                              slab cap >= slab_align;
//                                              1 = align always (subject to mode A).
//   APXCHOL_VEC_POOL_MIN_POW2_K      (0|1)  -- 1 = align per-slab to
//                                              min(next_pow2(cap), slab_align).
//                                              0 (default) = always slab_align.
//   APXCHOL_VEC_POOL_INITIAL_CAP     (int)  -- first slab cap (slots).
//                                              Default 4. Try 6, 8, 12 to capture
//                                              typical post-fill degree on grids.
//   APXCHOL_VEC_POOL_GROWTH_NUM/DENOM       -- grow cap by num/denom each grow.
//                                              Default 2/1 (doubling). Try 3/2 for
//                                              1.5× to reduce overshoot.
struct vec_pool_config {
    node_index slab_align;
    node_index align_threshold;
    bool    min_pow2_k;
    node_index initial_cap;
    int     growth_num;
    int     growth_denom;

    static const vec_pool_config& get() {
        static const vec_pool_config c = []() {
            vec_pool_config x;
            auto env_int = [](const char* k, int def) {
                const char* v = std::getenv(k);
                return v ? std::atoi(v) : def;
            };
            int sa = env_int("APXCHOL_VEC_POOL_SLAB_ALIGN_ENV", 0);
            x.slab_align       = sa > 0 ? sa : APXCHOL_VEC_POOL_SLAB_ALIGN;
            x.align_threshold  = env_int("APXCHOL_VEC_POOL_ALIGN_MODE_ENV", 0) ? 1
                                     : x.slab_align;
            x.min_pow2_k       = env_int("APXCHOL_VEC_POOL_MIN_POW2_K", 0) != 0;
            x.initial_cap      = env_int("APXCHOL_VEC_POOL_INITIAL_CAP", 4);
            x.growth_num       = env_int("APXCHOL_VEC_POOL_GROWTH_NUM",   2);
            x.growth_denom     = env_int("APXCHOL_VEC_POOL_GROWTH_DENOM", 1);
            return x;
        }();
        return c;
    }
};

inline node_index next_pow2_ge(node_index x) {
    if (x <= 1) return 1;
    node_index p = 1;
    while (p < x) p <<= 1;
    return p;
}

template<typename Value, graph_storage Tag>
struct basic_vec_pool_incidence {
    static constexpr graph_storage tag = Tag;
    static constexpr node_index slab_align = APXCHOL_VEC_POOL_SLAB_ALIGN;
    static constexpr node_index align_threshold = APXCHOL_VEC_POOL_ALIGN_THRESHOLD;
    using value_type = Value;

    void init(node_index n) {
        n_ = n;
        base_.assign(static_cast<size_t>(n), 0);
        count_.assign(static_cast<size_t>(n), 0);
        cap_.assign(static_cast<size_t>(n), 0);
        pool_.clear();
        abandoned_ = 0;
    }

    std::span<const value_type> operator[](node_index v) const {
        return std::span<const value_type>(
            pool_.data() + base_[v], static_cast<size_t>(count_[v]));
    }

    void push(node_index v, value_type idx) {
        if (count_[v] == cap_[v])
            grow(v);
        pool_[base_[v] + count_[v]++] = idx;
    }

    /// Ensure cap_[v] >= need. Caller must hold a serial pre-pass before
    /// invoking atomic_push_reserved in parallel — grow() reallocates the
    /// vertex's slab and is NOT thread-safe.
    ///
    /// Grows geometrically (configurable growth ratio, default 2x doubling)
    /// just like std::vector::reserve. On dense matrices (IPM), this
    /// dramatically reduces the number of grows over successive rounds.
    /// Earlier "tight" reserve_for (cap = need exactly) caused every round
    /// to trigger a grow + std::copy on touched vertices, making vec_pool
    /// 80% slower than vec on IPM bg+tree/x2/ps1.
    void reserve_for(node_index v, node_index need) {
        if (cap_[v] >= need) return;
        abandoned_ += cap_[v];
        const auto& cfg = vec_pool_config::get();
        // Geometric over-allocation: at least `need`, but try cap * growth.
        node_index new_cap = need;
        if (cap_[v] > 0) {
            const node_index geo = static_cast<node_index>(
                int64_t(cap_[v]) * cfg.growth_num / cfg.growth_denom);
            if (geo > new_cap) new_cap = geo;
        }
        // Pad to slab_align only when the new slab is large enough to
        // benefit from cache-line alignment. Tiny slabs (< align_threshold)
        // skip padding to avoid wasting more memory than the slab itself.
        node_index align_to = cfg.min_pow2_k
            ? std::min(next_pow2_ge(new_cap), cfg.slab_align)
            : cfg.slab_align;
        if (align_to > 1 && new_cap >= cfg.align_threshold) {
            size_t pad = (align_to - (pool_.size() % align_to)) % align_to;
            if (pad) pool_.resize(pool_.size() + pad);
            new_cap = (new_cap + align_to - 1) & ~(align_to - 1);
        }
        const std::size_t want = pool_.size() + static_cast<std::size_t>(new_cap);
        if (want > static_cast<std::size_t>(kPoolMax))
            edge_index_overflow("vec_pool slab growth");
        const edge_index new_base = static_cast<edge_index>(pool_.size());
        pool_.resize(want);
        if (count_[v] > 0) {
            std::copy_n(pool_.begin() + base_[v],
                        static_cast<size_t>(count_[v]),
                        pool_.begin() + new_base);
        }
        base_[v] = new_base;
        cap_[v] = new_cap;
    }

    /// Lock-free push.  Pre: caller has reserved a slot via reserve_for so
    /// count_[v] < cap_[v] when this call enters.  Multiple threads may
    /// invoke concurrently for any (v, slot) since the slot is atomically
    /// claimed.
    void atomic_push_reserved(node_index v, value_type idx) {
        const node_index slot = __atomic_fetch_add(
            &count_[v], 1, __ATOMIC_RELAXED);
        pool_[base_[v] + static_cast<size_t>(slot)] = idx;
    }

    void clear(node_index v) { count_[v] = 0; }

    /// Sort vertex v's slab in value order. Used after a parallel
    /// atomic_push_reserved phase to restore deterministic ordering (the
    /// atomic-push arrival order depends on thread interleaving; the values
    /// themselves are deterministic).
    void sort_slab(node_index v) {
        value_type* p = pool_.data() + base_[v];
        std::sort(p, p + count_[v]);
    }

    void filter(node_index v, auto&& pred, auto&& on_keep) {
        value_type* p = pool_.data() + base_[v];
        const node_index count = count_[v];

        // Do not copy the clean prefix onto itself. In the common no-removal
        // case this makes filter a read-only pass; after the first hole, the
        // usual stable compaction resumes. This is the same first-hole shape
        // used by remove_if, written out so on_keep still runs exactly once
        // for every survivor.
        node_index out = 0;
        while (out < count && pred(p[out])) {
            on_keep(p[out]);
            ++out;
        }
        if (out == count) return;

        for (node_index i = out + 1; i < count; ++i)
            if (pred(p[i])) {
                p[out] = p[i];
                on_keep(p[out]);
                ++out;
            }
        count_[v] = out;
    }

    void filter(node_index v, auto&& pred) {
        filter(v, pred, [](const value_type&) {});
    }

    std::size_t memory_bytes() const {
        return pool_.capacity() * sizeof(value_type)
             + base_.capacity() * sizeof(edge_index)
             + count_.capacity() * sizeof(node_index)
             + cap_.capacity() * sizeof(node_index);
    }

    /// Fraction of pool slots that are part of an active slab.
    /// Used as auto-compact trigger to limit abandoned-slab buildup.
    double live_fraction() const {
        return pool_.empty() ? 1.0
             : 1.0 - double(abandoned_) / double(pool_.size());
    }

    // Diagnostics: how many per-vertex slab grows happened, how many compactions
    // fired, and the worst (lowest) live fraction seen across the build. Lets us
    // verify whether compaction actually triggers and how fragmented the pool got.
    size_t grow_count() const    { return grow_count_; }
    size_t compact_count() const { return compact_count_; }
    double min_live_fraction() const { return min_live_frac_; }
    void note_live_fraction() { min_live_frac_ = std::min(min_live_frac_, live_fraction()); }

    /// Compact: rebuild the pool with each vertex's live slab laid out
    /// contiguously, reclaiming the abandoned dead slabs.  O(n + sum(count)).
    ///   with_headroom == false : tighten cap=count (explicit/manual use).
    ///   with_headroom == true  : leave cap = count*growth headroom per slab,
    ///       so the next pushes don't immediately re-grow. Used by the
    ///       defrag-on-extension path so reclaiming dead slabs doesn't just
    ///       trade abandoned space for a storm of re-grows.
    /// `round_incoming == nullptr` : tighten every slab to cap=count (manual use).
    /// non-null : a vertex receiving fill THIS round (incoming[v] > 0) is sized to
    ///   its KNOWN need `count+incoming` plus the same geometric future-headroom a
    ///   normal grow would give (`max(need, cap*growth)`), so the round's grows
    ///   become a no-op; untouched vertices are tightened to count. Reclaims all
    ///   abandoned slabs in the same pass.
    /// Rebuild the pool, reclaiming abandoned dead slabs + pruning bloat. Each
    /// live slab is sized to `max(count(+incoming), count*growth)` -- downsizing
    /// over-grown slabs to 2x their CURRENT count while keeping room for this
    /// round's known incoming.
    ///   round_incoming : per-vertex fill counts for the imminent round (or null).
    void compact(const std::vector<node_index>* round_incoming = nullptr) {
        const auto& cfg = vec_pool_config::get();
        auto target_cap = [&](node_index v) -> node_index {
            const node_index cnt = count_[v];
            if (cnt == 0) return 0;                          // empty/eliminated -> reclaim
            const node_index inc  = round_incoming ? (*round_incoming)[v] : 0;
            const node_index need = cnt + inc;
            const node_index hr   = static_cast<node_index>(
                int64_t(cnt) * cfg.growth_num / cfg.growth_denom);
            return hr > need ? hr : need;                    // 2x count, fit round need
        };
        size_t total = 0;
        for (node_index v = 0; v < n_; ++v) total += target_cap(v);
        decltype(pool_) new_pool;
        new_pool.resize(total);            // pre-sized (headroom slots are gaps)
        size_t cursor = 0;
        for (node_index v = 0; v < n_; ++v) {
            const node_index cnt = count_[v];
            const node_index nc  = target_cap(v);
            if (cnt) std::copy_n(pool_.begin() + base_[v],
                                 static_cast<size_t>(cnt), new_pool.begin() + cursor);
            base_[v] = static_cast<edge_index>(cursor);
            cap_[v]  = nc;
            cursor  += nc;
        }
        pool_.swap(new_pool);
        abandoned_ = 0;
        ++compact_count_;
    }

    /// Bulk parallel reserve_for. Caller is inside an OpenMP parallel region.
    /// Each thread computes the grow metadata for a stable slice of touched
    /// vertices; one thread prefix-sums those sizes and grows the pool once;
    /// then every thread copies its own old slabs. The trailing barrier makes
    /// the new bases and contents visible before the caller's apply phase.
    template<typename TouchedIter>
    void bulk_reserve_parallel(TouchedIter touched_begin, TouchedIter touched_end,
                               const std::vector<node_index>& incoming) {
        static_assert(std::random_access_iterator<TouchedIter>);
#ifdef _OPENMP
        #pragma omp single
        {
            // Downsize-defrag: when >25% of the pool is dead slabs, rebuild it
            // keeping each live slab sized to max(this-round-need, 2*count). This
            // reclaims BOTH the abandoned slabs left by past doublings AND the
            // "pruning bloat" -- slabs whose count shrank far below their cap as
            // neighbours were eliminated (up to 20x on dense IPM). Sizing to the
            // round's incoming makes the grow loop below a no-op when it fires.
            // Numerically transparent (factor byte-identical), perf-neutral, frees
            // 3-12% RSS on fill-heavy graphs; a no-op on low-fill grids.
            // (The 25% trigger / 2x headroom are unswept heuristic guesses;
            // a dedicated sweep could tune them if pool overhead shows up.)
            if (abandoned_ > pool_.size() / 4)
                compact(&incoming);
            bulk_touched_n_ = static_cast<size_t>(touched_end - touched_begin);
            bulk_team_ = omp_get_num_threads();
            bulk_old_pool_size_ = pool_.size();
            bulk_cap_offsets_.resize(static_cast<size_t>(bulk_team_) + 1);
            bulk_abandoned_by_thread_.resize(static_cast<size_t>(bulk_team_));
            bulk_thread_grows_.resize(static_cast<size_t>(bulk_team_));
        }

        const int tid = omp_get_thread_num();
        const size_t begin = bulk_touched_n_ * static_cast<size_t>(tid)
                           / static_cast<size_t>(bulk_team_);
        const size_t end = bulk_touched_n_ * static_cast<size_t>(tid + 1)
                         / static_cast<size_t>(bulk_team_);
        const auto& cfg = vec_pool_config::get();
        auto& my_grows = bulk_thread_grows_[static_cast<size_t>(tid)];
        my_grows.clear();
        size_t new_slots = 0;
        size_t abandoned = 0;
        for (size_t i = begin; i < end; ++i) {
            const node_index v = touched_begin[static_cast<std::ptrdiff_t>(i)];
            const node_index need = count_[v] + incoming[v];
            if (incoming[v] == 0 || cap_[v] >= need) continue;
            node_index new_cap = need;
            if (cap_[v] > 0) {
                const node_index geo = static_cast<node_index>(
                    int64_t(cap_[v]) * cfg.growth_num / cfg.growth_denom);
                if (geo > new_cap) new_cap = geo;
            }
            my_grows.push_back({v, base_[v], cap_[v], 0, new_cap});
            new_slots += new_cap;
            abandoned += cap_[v];
        }
        bulk_cap_offsets_[static_cast<size_t>(tid) + 1] = new_slots;
        bulk_abandoned_by_thread_[static_cast<size_t>(tid)] = abandoned;

        #pragma omp barrier
        #pragma omp single
        {
            bulk_cap_offsets_[0] = bulk_old_pool_size_;
            size_t total_abandoned = 0;
            size_t total_grows = 0;
            for (int t = 0; t < bulk_team_; ++t) {
                const size_t add =
                    bulk_cap_offsets_[static_cast<size_t>(t) + 1];
                if (add > static_cast<size_t>(kPoolMax)
                              - bulk_cap_offsets_[static_cast<size_t>(t)])
                    edge_index_overflow("vec_pool parallel bulk reserve");
                bulk_cap_offsets_[static_cast<size_t>(t) + 1] =
                    bulk_cap_offsets_[static_cast<size_t>(t)] + add;
                total_abandoned +=
                    bulk_abandoned_by_thread_[static_cast<size_t>(t)];
                total_grows +=
                    bulk_thread_grows_[static_cast<size_t>(t)].size();
            }
            const size_t cursor =
                bulk_cap_offsets_[static_cast<size_t>(bulk_team_)];
            if (cursor != pool_.size()) pool_.resize(cursor);
            abandoned_ += total_abandoned;
            grow_count_ += total_grows;
        }

        size_t cursor = bulk_cap_offsets_[static_cast<size_t>(tid)];
        for (auto& g : my_grows) {
            g.new_base = static_cast<edge_index>(cursor);
            base_[g.v] = g.new_base;
            cap_[g.v] = g.new_cap;
            if (g.old_cap > 0 && count_[g.v] > 0)
                std::copy_n(pool_.begin() + g.old_base,
                            static_cast<size_t>(count_[g.v]),
                            pool_.begin() + g.new_base);
            cursor += g.new_cap;
        }
        #pragma omp barrier
#else
        if (abandoned_ > pool_.size() / 4) compact(&incoming);
        const auto& cfg = vec_pool_config::get();
        std::vector<Grow> grows;
        size_t cursor = pool_.size();
        for (auto it = touched_begin; it != touched_end; ++it) {
            const node_index v = *it;
            const node_index need = count_[v] + incoming[v];
            if (incoming[v] == 0 || cap_[v] >= need) continue;
            node_index new_cap = need;
            if (cap_[v] > 0) {
                const node_index geo = static_cast<node_index>(
                    int64_t(cap_[v]) * cfg.growth_num / cfg.growth_denom);
                if (geo > new_cap) new_cap = geo;
            }
            if (static_cast<size_t>(new_cap) >
                static_cast<size_t>(kPoolMax) - cursor)
                edge_index_overflow("vec_pool bulk reserve");
            grows.push_back({v, base_[v], cap_[v],
                             static_cast<edge_index>(cursor), new_cap});
            cursor += new_cap;
            abandoned_ += cap_[v];
        }
        if (cursor != pool_.size()) pool_.resize(cursor);
        for (const auto& g : grows) {
            base_[g.v] = g.new_base;
            cap_[g.v] = g.new_cap;
            if (g.old_cap > 0 && count_[g.v] > 0)
                std::copy_n(pool_.begin() + g.old_base,
                            static_cast<size_t>(count_[g.v]),
                            pool_.begin() + g.new_base);
        }
        grow_count_ += grows.size();
#endif
    }

private:
    void grow(node_index v) {
        // Account for the slab we are about to abandon.
        abandoned_ += cap_[v];
        ++grow_count_;
        const auto& cfg = vec_pool_config::get();
        node_index new_cap;
        if (cap_[v] == 0) {
            new_cap = cfg.initial_cap;
        } else {
            // Geometric grow: cap * (growth_num / growth_denom).
            new_cap = static_cast<node_index>(
                int64_t(cap_[v]) * cfg.growth_num / cfg.growth_denom);
            if (new_cap <= cap_[v]) new_cap = cap_[v] + 1;
        }
        // Align new_cap and pool base if requested.
        node_index align_to = cfg.min_pow2_k
            ? std::min(next_pow2_ge(new_cap), cfg.slab_align)
            : cfg.slab_align;
        if (align_to > 1 && new_cap >= cfg.align_threshold) {
            size_t pad = (align_to - (pool_.size() % align_to)) % align_to;
            if (pad) pool_.resize(pool_.size() + pad);
            new_cap = (new_cap + align_to - 1) & ~(align_to - 1);
        }
        const std::size_t want = pool_.size() + static_cast<std::size_t>(new_cap);
        if (want > static_cast<std::size_t>(kPoolMax))
            edge_index_overflow("vec_pool slab growth");
        const edge_index new_base = static_cast<edge_index>(pool_.size());
        pool_.resize(want);
        if (count_[v] > 0) {
            std::copy_n(pool_.begin() + base_[v],
                        static_cast<size_t>(count_[v]),
                        pool_.begin() + new_base);
        }
        base_[v] = new_base;
        cap_[v] = new_cap;
    }

    node_index n_ = 0;
    // The pool can grow to hundreds of MB on grid_3000+. Keep big_alloc's THP
    // advice, but leave pages lazy: std::vector's unused geometric capacity can
    // be hundreds of MB and must not become resident merely because it was
    // reserved. resize() still initializes and commits every live slot.
    std::vector<value_type, util::big_alloc<value_type, 32, false>> pool_;
    // base_[v] is an OFFSET into pool_ (can exceed 2^31 on dense factors) ->
    // edge_index. count_/cap_ are per-vertex slab sizes (<= degree < n) ->
    // node_index. base_[v] + count_[v] promotes to edge_index (pool index).
    std::vector<edge_index> base_;
    std::vector<node_index>    count_;
    std::vector<node_index>    cap_;
    size_t abandoned_ = 0;  // total slots in dead slabs (pre-grow)
    size_t grow_count_ = 0;     // per-vertex slab grows over the build
    size_t compact_count_ = 0;  // compactions that fired
    double min_live_frac_ = 1.0; // worst fragmentation seen (via note_live_fraction)

    // Pool offsets are bounded by this; exceeding it would silently wrap base_.
    static constexpr edge_index kPoolMax = std::numeric_limits<edge_index>::max();

    struct Grow { node_index v; edge_index old_base; node_index old_cap; edge_index new_base; node_index new_cap; };
    size_t bulk_touched_n_ = 0;
    size_t bulk_old_pool_size_ = 0;
    int bulk_team_ = 1;
    std::vector<size_t> bulk_cap_offsets_;
    std::vector<size_t> bulk_abandoned_by_thread_;
    std::vector<std::vector<Grow>> bulk_thread_grows_;
};

using vec_pool_incidence = basic_vec_pool_incidence<
    edge_index, graph_storage::vec_pool>;
using directed_vec_pool_incidence = basic_vec_pool_incidence<
    directed_pool_edge, graph_storage::vec_pool_aos>;

template<typename T>
inline constexpr bool is_vec_pool_incidence_v =
    std::same_as<T, vec_pool_incidence> ||
    std::same_as<T, directed_vec_pool_incidence>;

} // namespace apxchol

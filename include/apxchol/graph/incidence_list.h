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
#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
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
// One logically contiguous, physically segmented arena of Value slots, with
// per-vertex {base, count, capacity} triples. Push appends to a vertex's
// reserved slab; if the slab fills, the slab is copied to a newly claimed
// doubled-capacity range. Stable mmap segments keep every existing offset
// valid while workers claim growth ranges concurrently.
//
// Compared to vec_incidence: same per-vertex contiguity for
// iteration, but no per-vertex std::vector wrapper (saves 3 words
// of header per vertex). Growth maps only the segment it first touches; unused
// claimed pages remain lazy.
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
// Runtime tunables (env-var driven, evaluated once at first init).
//   APXCHOL_VEC_POOL_INITIAL_CAP     (int)  -- first slab cap (slots).
//                                              Default 4. Try 6, 8, 12 to
//                                              capture typical post-fill degree
//                                              on grids.
//   APXCHOL_VEC_POOL_GROWTH_NUM/DENOM       -- grow cap by num/denom each grow.
//                                              Default 2/1 (doubling). Try 3/2
//                                              for 1.5× to reduce overshoot.
struct vec_pool_config {
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
            x.initial_cap      = env_int("APXCHOL_VEC_POOL_INITIAL_CAP", 4);
            x.growth_num       = env_int("APXCHOL_VEC_POOL_GROWTH_NUM",   2);
            x.growth_denom     = env_int("APXCHOL_VEC_POOL_GROWTH_DENOM", 1);
            return x;
        }();
        return c;
    }
};

// Stable-address, segmented pool. Slabs never straddle a segment, so their
// compact integer offsets remain valid while workers claim disjoint ranges
// independently. Only used segments are mmap'ed; unlike one giant lazy
// reservation, this remains usable under an ordinary RLIMIT_AS. This removes
// the whole-pool resize and prefix-sum rendezvous from parallel elimination.
template<typename T, size_t SegmentSlots = (size_t{1} << 27),
         size_t MaxVirtualBytes = (size_t{1} << 40)>
class segmented_pool {
public:
    segmented_pool() = default;
    ~segmented_pool() { release(); }

    segmented_pool(const segmented_pool& other) {
        const size_t n = other.size();
        ensure_table();
        for (size_t segment = 0; segment < segment_count(n); ++segment) {
            if (!other.segment(segment))
                continue;
            T* dst = ensure_segment(segment);
            const size_t count = std::min(segment_slots, n - segment * segment_slots);
            if constexpr (std::is_trivially_copyable_v<T>)
                std::memcpy(dst, other.segment(segment), count * sizeof(T));
            else
                std::uninitialized_copy_n(other.segment(segment), count, dst);
        }
        size_.store(n, std::memory_order_relaxed);
    }
    segmented_pool& operator=(const segmented_pool& other) {
        if (this == &other)
            return *this;
        segmented_pool copy(other);
        swap(copy);
        return *this;
    }
    segmented_pool(segmented_pool&& other) noexcept { take(other); }
    segmented_pool& operator=(segmented_pool&& other) noexcept {
        if (this != &other) {
            release();
            take(other);
        }
        return *this;
    }

    T* ptr(size_t i) {
        if (i < segment_slots) {
            if (T* first = first_segment_.load(std::memory_order_relaxed))
                return first + i;
            else
                return nullptr;
        }
        return segment(i / segment_slots) + i % segment_slots;
    }
    const T* ptr(size_t i) const {
        if (i < segment_slots) {
            if (T* first = first_segment_.load(std::memory_order_relaxed))
                return first + i;
            else
                return nullptr;
        }
        return segment(i / segment_slots) + i % segment_slots;
    }
    T& operator[](size_t i) { return *ptr(i); }
    const T& operator[](size_t i) const { return *ptr(i); }
    size_t size() const { return size_.load(std::memory_order_relaxed); }
    size_t capacity() const { return size(); } // logical bytes, not VA reservation
    bool empty() const { return size() == 0; }
    static constexpr size_t max_claim_slots() { return segment_slots; }
    void prepare() { ensure_table(); }
    void clear() {
        const size_t n = size();
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t segment = 0; segment < segment_count(n); ++segment) {
                T* data = this->segment(segment);
                if (!data)
                    continue;
                const size_t count = std::min(segment_slots, n - segment * segment_slots);
                std::destroy_n(data, count);
            }
        }
        size_.store(0, std::memory_order_relaxed);
    }

    size_t claim(size_t n) {
        if (n > segment_slots)
            throw std::bad_alloc{};
        ensure_table();
        size_t observed = size_.load(std::memory_order_relaxed);
        if (n == 0)
            return observed;
        size_t begin = 0;
        while (true) {
            const size_t offset = observed % segment_slots;
            begin = offset + n <= segment_slots ? observed : observed + (segment_slots - offset);
            if (begin > slot_limit() || n > slot_limit() - begin)
                throw std::bad_alloc{};
            if (size_.compare_exchange_weak(observed, begin + n, std::memory_order_relaxed))
                break;
        }
        if constexpr (!std::is_trivially_copyable_v<T>) {
            // A crossing leaves a logical gap at the end of the old segment.
            // Keep every slot below size() alive so clear/copy can traverse
            // the logical range without touching unconstructed objects.
            if (begin > observed) {
                T* gap = ensure_segment(observed / segment_slots) + observed % segment_slots;
                std::uninitialized_default_construct_n(gap, begin - observed);
            }
        }
        T* destination = ensure_segment(begin / segment_slots) + begin % segment_slots;
        if constexpr (!std::is_trivially_copyable_v<T>)
            std::uninitialized_default_construct_n(destination, n);
        return begin;
    }

    void swap(segmented_pool& other) noexcept {
        segments_.swap(other.segments_);
        std::swap(max_segments_, other.max_segments_);
        T* mine_first = first_segment_.load(std::memory_order_relaxed);
        T* other_first = other.first_segment_.load(std::memory_order_relaxed);
        first_segment_.store(other_first, std::memory_order_relaxed);
        other.first_segment_.store(mine_first, std::memory_order_relaxed);
        const size_t mine = size();
        const size_t theirs = other.size();
        size_.store(theirs, std::memory_order_relaxed);
        other.size_.store(mine, std::memory_order_relaxed);
    }

private:
    // 2^27 slots = 1 GiB for the 8-byte directed AoS record and 512 MiB for a
    // 32-bit indexed incidence. This keeps segment crossings rare while mapping
    // only memory proportional to the graph that actually exists.
    static constexpr size_t segment_slots = SegmentSlots;
    static_assert(std::has_single_bit(segment_slots));
    static constexpr size_t max_virtual_bytes = MaxVirtualBytes;
    static constexpr size_t slot_limit() {
        const auto edge_limit = std::numeric_limits<edge_index>::max();
        const size_t by_address = max_virtual_bytes / sizeof(T);
        if constexpr (sizeof(edge_index) < sizeof(size_t))
            return std::min(by_address, static_cast<size_t>(edge_limit));
        else
            return by_address;
    }
    static constexpr size_t segment_count(size_t slots) {
        return slots == 0 ? 0 : (slots - 1) / segment_slots + 1;
    }
    void ensure_table() {
        if (segments_)
            return;
        max_segments_ = segment_count(slot_limit());
        segments_ = std::make_unique<std::atomic<T*>[]>(max_segments_);
        for (size_t i = 0; i < max_segments_; ++i)
            segments_[i].store(nullptr, std::memory_order_relaxed);
    }
    T* segment(size_t i) const {
        return segments_ ? segments_[i].load(std::memory_order_acquire) : nullptr;
    }
    T* ensure_segment(size_t i) {
        if (T* existing = segment(i)) {
            if (i == 0)
                first_segment_.store(existing, std::memory_order_relaxed);
            return existing;
        }
        const size_t bytes = segment_slots * sizeof(T);
        void* raw = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (raw == MAP_FAILED)
            throw std::bad_alloc{};
        madvise(raw, bytes, MADV_HUGEPAGE);
        T* candidate = static_cast<T*>(raw);
        T* expected = nullptr;
        if (!segments_[i].compare_exchange_strong(expected, candidate, std::memory_order_release,
                                                  std::memory_order_acquire)) {
            munmap(candidate, bytes);
            if (i == 0)
                first_segment_.store(expected, std::memory_order_relaxed);
            return expected;
        }
        if (i == 0)
            first_segment_.store(candidate, std::memory_order_relaxed);
        return candidate;
    }
    void release() noexcept {
        clear();
        if (segments_) {
            const size_t bytes = segment_slots * sizeof(T);
            for (size_t i = 0; i < max_segments_; ++i)
                if (T* data = segment(i))
                    munmap(data, bytes);
        }
        segments_.reset();
        max_segments_ = 0;
        first_segment_.store(nullptr, std::memory_order_relaxed);
    }
    void take(segmented_pool& other) noexcept {
        segments_ = std::move(other.segments_);
        max_segments_ = other.max_segments_;
        first_segment_.store(other.first_segment_.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        size_.store(other.size(), std::memory_order_relaxed);
        other.max_segments_ = 0;
        other.first_segment_.store(nullptr, std::memory_order_relaxed);
        other.size_.store(0, std::memory_order_relaxed);
    }

    std::unique_ptr<std::atomic<T*>[]> segments_;
    size_t max_segments_ = 0;
    std::atomic<T*> first_segment_{nullptr};
    std::atomic<size_t> size_{0};
};

template<typename Value, graph_storage Tag> struct basic_vec_pool_incidence {
    static constexpr graph_storage tag = Tag;
    using value_type = Value;

    void init(node_index n) {
        n_ = n;
        base_.assign(static_cast<size_t>(n), 0);
        count_.assign(static_cast<size_t>(n), 0);
        cap_.assign(static_cast<size_t>(n), 0);
        pool_.clear();
        // claim() is parallel during elimination; materialize its segment table
        // here, while graph initialization is still serial.
        pool_.prepare();
        int team = 1;
#ifdef _OPENMP
        team = std::max(1, omp_get_max_threads());
#endif
        bulk_thread_grows_.resize(static_cast<size_t>(team));
        abandoned_ = 0;
    }

    std::span<const value_type> operator[](node_index v) const {
        return std::span<const value_type>(pool_.ptr(base_[v]), static_cast<size_t>(count_[v]));
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
        if (cap_[v] >= need)
            return;
        abandoned_ += cap_[v];
        const auto& cfg = vec_pool_config::get();
        // Geometric over-allocation: at least `need`, but try cap * growth.
        node_index new_cap = need;
        if (cap_[v] > 0) {
            const node_index geo =
                static_cast<node_index>(int64_t(cap_[v]) * cfg.growth_num / cfg.growth_denom);
            if (geo > new_cap)
                new_cap = geo;
        }
        const edge_index new_base =
            static_cast<edge_index>(pool_.claim(static_cast<size_t>(new_cap)));
        if (count_[v] > 0) {
            std::copy_n(pool_.ptr(base_[v]), static_cast<size_t>(count_[v]), pool_.ptr(new_base));
        }
        base_[v] = new_base;
        cap_[v] = new_cap;
    }

    /// Lock-free push.  Pre: caller has reserved a slot via reserve_for so
    /// count_[v] < cap_[v] when this call enters.  Multiple threads may
    /// invoke concurrently for any (v, slot) since the slot is atomically
    /// claimed.
    void atomic_push_reserved(node_index v, value_type idx) {
        const node_index slot = __atomic_fetch_add(&count_[v], 1, __ATOMIC_RELAXED);
        pool_[base_[v] + static_cast<size_t>(slot)] = idx;
    }

    /// Write a caller-assigned slot relative to the current end of v's slab.
    /// The caller must have reserved the whole batch and keep count_[v]
    /// unchanged until every disjoint offset has been materialized.
    void write_reserved_at(node_index v, node_index offset, value_type value) {
        pool_[base_[v] + static_cast<size_t>(count_[v]) + offset] = value;
    }

    /// Publish a batch written through write_reserved_at(). One owner calls
    /// this once for v after every writer has completed.
    void commit_reserved(node_index v, node_index added) { count_[v] += added; }

    void clear(node_index v) { count_[v] = 0; }

    /// Sort vertex v's slab in value order. Used after a parallel
    /// atomic_push_reserved phase to restore deterministic ordering (the
    /// atomic-push arrival order depends on thread interleaving; the values
    /// themselves are deterministic).
    void sort_slab(node_index v) {
        value_type* p = pool_.ptr(base_[v]);
        std::sort(p, p + count_[v]);
    }

    /// Directed-inline diagnostic: discard inactive endpoints and combine all
    /// parallel records with the same target into one fp32 record. Sorting by
    /// (target,weight) gives both endpoint slabs the same deterministic fp64
    /// accumulation order.
    template <class IsLive>
        requires std::same_as<value_type, directed_pool_edge>
    std::pair<node_index, node_index> coalesce_slab(
            node_index v, IsLive&& is_live) {
        // claim() guarantees that a vertex slab never crosses a segment, so
        // ptr(base) exposes the same contiguous range without assuming the
        // whole pool has one backing allocation.
        value_type* p = pool_.ptr(base_[v]);
        const node_index before = count_[v];
        std::sort(p, p + before);
        node_index out = 0;
        for (node_index i = 0; i < before;) {
            const node_index target = p[i].to;
            double weight = 0.0;
            do {
                weight += static_cast<double>(p[i].w);
                ++i;
            } while (i < before && p[i].to == target);
            if (is_live(target))
                p[out++] = {target, static_cast<pool_value_t>(weight)};
        }
        count_[v] = out;
        return {before, out};
    }

    void filter(node_index v, auto&& pred, auto&& on_keep) {
        value_type* p = pool_.ptr(base_[v]);
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
        if (out == count)
            return;

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
        return pool_.capacity() * sizeof(value_type) + base_.capacity() * sizeof(edge_index) +
               count_.capacity() * sizeof(node_index) + cap_.capacity() * sizeof(node_index);
    }

    /// Fraction of pool slots that are part of an active slab.
    /// Used as auto-compact trigger to limit abandoned-slab buildup.
    double live_fraction() const {
        return pool_.empty() ? 1.0 : 1.0 - double(abandoned_) / double(pool_.size());
    }

    // Diagnostics: how many per-vertex slab grows happened, how many compactions
    // fired, and the worst (lowest) live fraction seen across the build. Lets us
    // verify whether compaction actually triggers and how fragmented the pool
    // got.
    size_t grow_count() const { return grow_count_; }
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
    /// `round_incoming == nullptr` : tighten every slab to cap=count (manual
    /// use). non-null : a vertex receiving fill THIS round (incoming[v] > 0) is
    /// sized to
    ///   its KNOWN need `count+incoming` plus the same geometric future-headroom
    ///   a normal grow would give (`max(need, cap*growth)`), so the round's grows
    ///   become a no-op; untouched vertices are tightened to count. Reclaims all
    ///   abandoned slabs in the same pass.
    /// Rebuild the pool, reclaiming abandoned dead slabs + pruning bloat. Each
    /// live slab is sized to `max(count(+incoming), count*growth)` -- downsizing
    /// over-grown slabs to 2x their CURRENT count while keeping room for this
    /// round's known incoming.
    ///   round_incoming : per-vertex fill counts for the imminent round (or
    ///   null).
    void compact(const std::vector<node_index>* round_incoming = nullptr) {
        const auto& cfg = vec_pool_config::get();
        auto target_cap = [&](node_index v) -> node_index {
            const node_index cnt = count_[v];
            if (cnt == 0)
                return 0; // empty/eliminated -> reclaim
            const node_index inc = round_incoming ? (*round_incoming)[v] : 0;
            const node_index need = cnt + inc;
            const node_index hr =
                static_cast<node_index>(int64_t(cnt) * cfg.growth_num / cfg.growth_denom);
            return hr > need ? hr : need; // 2x count, fit round need
        };
        decltype(pool_) new_pool;
        new_pool.prepare();
        for (node_index v = 0; v < n_; ++v) {
            const node_index cnt = count_[v];
            const node_index nc = target_cap(v);
            edge_index new_base = 0;
            if (nc != 0)
                new_base = static_cast<edge_index>(new_pool.claim(nc));
            if (cnt)
                std::copy_n(pool_.ptr(base_[v]), static_cast<size_t>(cnt), new_pool.ptr(new_base));
            base_[v] = new_base;
            cap_[v] = nc;
        }
        pool_.swap(new_pool);
        abandoned_ = 0;
        ++compact_count_;
    }

    /// Serial decision/rebuild for the imminent parallel reserve. The caller
    /// owns the surrounding OpenMP single region; its existing implicit
    /// barrier publishes the decision and rebuilt pool to every worker before
    /// bulk_reserve_parallel starts.
    void compact_for_round_if_needed(
            const std::vector<node_index>& incoming) {
        if (abandoned_ > pool_.size() / 2)
            compact(&incoming);
    }

    /// Bulk parallel reserve_for. Caller is inside an OpenMP parallel region.
    /// Each thread computes and copies the grows for a stable slice of touched
    /// vertices. Atomic segment claims replace the old team prefix sum and
    /// whole-pool resize; the trailing barrier publishes every new slab before
    /// the caller's apply phase.
    template<typename TouchedIter>
    void bulk_reserve_parallel(TouchedIter touched_begin, TouchedIter touched_end,
                               const std::vector<node_index>& incoming) {
        static_assert(std::random_access_iterator<TouchedIter>);
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int team = omp_get_num_threads();
#else
        const int tid = 0;
        const int team = 1;
#endif
        const size_t touched_n = static_cast<size_t>(touched_end - touched_begin);
        const size_t begin = touched_n * static_cast<size_t>(tid) / static_cast<size_t>(team);
        const size_t end = touched_n * static_cast<size_t>(tid + 1) / static_cast<size_t>(team);
        const auto& cfg = vec_pool_config::get();
        auto& grows = bulk_thread_grows_[static_cast<size_t>(tid)];
        grows.clear();
        size_t local_abandoned = 0;
        for (size_t i = begin; i < end; ++i) {
            const node_index v = touched_begin[static_cast<std::ptrdiff_t>(i)];
            const node_index need = count_[v] + incoming[v];
            if (incoming[v] == 0 || cap_[v] >= need)
                continue;
            node_index new_cap = need;
            if (cap_[v] > 0) {
                const node_index geo =
                    static_cast<node_index>(int64_t(cap_[v]) * cfg.growth_num / cfg.growth_denom);
                if (geo > new_cap)
                    new_cap = geo;
            }
            grows.push_back({v, base_[v], cap_[v], 0, new_cap});
            local_abandoned += cap_[v];
        }
        // A giant graph can give one worker more than one segment worth of
        // aggregate growth even though every individual vertex slab fits.
        // Claim maximal consecutive batches that fit in one segment. This
        // keeps the common path at O(segments claimed per worker), rather than
        // performing one atomic claim per grown vertex.
        size_t grow_begin = 0;
        while (grow_begin < grows.size()) {
            size_t grow_end = grow_begin;
            size_t batch_slots = 0;
            while (grow_end < grows.size() &&
                   grows[grow_end].new_cap <=
                       segmented_pool<value_type>::max_claim_slots() - batch_slots) {
                batch_slots += grows[grow_end].new_cap;
                ++grow_end;
            }
            if (grow_end == grow_begin)
                throw std::bad_alloc{};
            size_t cursor = pool_.claim(batch_slots);
            for (size_t i = grow_begin; i < grow_end; ++i) {
                auto& grow = grows[i];
                grow.new_base = static_cast<edge_index>(cursor);
                if (grow.old_cap > 0 && count_[grow.v] > 0)
                    std::copy_n(pool_.ptr(grow.old_base), static_cast<size_t>(count_[grow.v]),
                                pool_.ptr(grow.new_base));
                base_[grow.v] = grow.new_base;
                cap_[grow.v] = grow.new_cap;
                cursor += grow.new_cap;
            }
            grow_begin = grow_end;
        }
#ifdef _OPENMP
#pragma omp atomic update
#endif
        abandoned_ += local_abandoned;
#ifdef _OPENMP
#pragma omp atomic update
#endif
        grow_count_ += grows.size();
#ifdef _OPENMP
#pragma omp barrier
#endif
        return;
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
            new_cap = static_cast<node_index>(int64_t(cap_[v]) * cfg.growth_num / cfg.growth_denom);
            if (new_cap <= cap_[v])
                new_cap = cap_[v] + 1;
        }
        const edge_index new_base = static_cast<edge_index>(pool_.claim(new_cap));
        if (count_[v] > 0) {
            std::copy_n(pool_.ptr(base_[v]), static_cast<size_t>(count_[v]), pool_.ptr(new_base));
        }
        base_[v] = new_base;
        cap_[v] = new_cap;
    }

    node_index n_ = 0;
    // Stable mmap segments are lazy: a claim advances the logical offset, but
    // pages become resident only when a slab is written.
    segmented_pool<value_type> pool_;
    // base_[v] is an OFFSET into pool_ (can exceed 2^31 on dense factors) ->
    // edge_index. count_/cap_ are per-vertex slab sizes (<= degree < n) ->
    // node_index. base_[v] + count_[v] promotes to edge_index (pool index).
    std::vector<edge_index> base_;
    std::vector<node_index> count_;
    std::vector<node_index> cap_;
    size_t abandoned_ = 0;       // total slots in dead slabs (pre-grow)
    size_t grow_count_ = 0;      // per-vertex slab grows over the build
    size_t compact_count_ = 0;   // compactions that fired
    double min_live_frac_ = 1.0; // worst fragmentation seen (via note_live_fraction)

    struct Grow {
        node_index v;
        edge_index old_base;
        node_index old_cap;
        edge_index new_base;
        node_index new_cap;
    };
    std::vector<std::vector<Grow>> bulk_thread_grows_;
};

using vec_pool_incidence =
    basic_vec_pool_incidence<edge_index, graph_storage::vec_pool>;
using directed_vec_pool_incidence =
    basic_vec_pool_incidence<directed_pool_edge, graph_storage::vec_pool_aos>;

template <typename T>
inline constexpr bool is_vec_pool_incidence_v =
    std::same_as<T, vec_pool_incidence> ||
    std::same_as<T, directed_vec_pool_incidence>;

} // namespace apxchol

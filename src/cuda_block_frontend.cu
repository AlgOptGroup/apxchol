#include "apxchol/solver/gpu_block_frontend.h"
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace apxchol::detail {
namespace {

using clock_type = gpu_setup_diagnostic_clock;

static_assert(std::is_standard_layout_v<deferred_edge>);
static_assert(offsetof(deferred_edge, u) == 0);
static_assert(offsetof(deferred_edge, v) == sizeof(node_index));
static_assert(offsetof(deferred_edge, w) == sizeof(gpu_topology_edge));

[[noreturn]] void cuda_failure(cudaError_t err, const char *what) {
    throw std::runtime_error(std::string("GPU block front-end: ") + what + ": " +
                             cudaGetErrorString(err));
}

void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess)
        cuda_failure(err, what);
}

template <class T> class device_buffer {
public:
    device_buffer() = default;
    ~device_buffer() {
        if (ptr_)
            cudaFree(ptr_);
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;

    void reserve(std::size_t count) {
        if (count <= capacity_)
            return;
        T *next = nullptr;
        cuda_check(
            cudaMalloc(reinterpret_cast<void **>(&next), count * sizeof(T)),
            "cudaMalloc");
        if (ptr_)
            cuda_check(cudaFree(ptr_), "cudaFree during growth");
        ptr_ = next;
        capacity_ = count;
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }
    std::size_t capacity() const { return capacity_; }

private:
    T *ptr_ = nullptr;
    std::size_t capacity_ = 0;
};

struct residual_endpoints {
    __host__ __device__ gpu_topology_edge operator()(
            const gpu_round_shadow_incidence& incidence) const {
        return {incidence.owner, incidence.neighbor};
    }
};
struct canonical_endpoint_pair {
    __host__ __device__ bool operator()(const gpu_topology_edge& edge) const {
        return edge.u < edge.v;
    }
};

constexpr int kBlock = 256;
constexpr std::size_t kTopologyStageEdges = std::size_t{1} << 20;

int blocks_for(std::size_t n) {
    const std::size_t blocks = (n + kBlock - 1) / kBlock;
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("GPU block front-end: CUDA grid is too large");
    return static_cast<int>(blocks);
}

template <class T> __device__ T atomic_increment(T *p) {
    if constexpr (sizeof(T) == sizeof(unsigned int)) {
        return static_cast<T>(
            atomicAdd(reinterpret_cast<unsigned int *>(p), 1U));
    } else {
        return static_cast<T>(
            atomicAdd(reinterpret_cast<unsigned long long *>(p), 1ULL));
    }
}

struct device_selection_digest {
    unsigned long long xor_hash;
    unsigned long long sum_hash;
};

struct device_prepare_status {
    unsigned long long degree_sum;
    device_selection_digest topology;
    device_selection_digest active;
};
static_assert(sizeof(device_prepare_status) == 40);

struct device_selected_status {
    unsigned long long degree_sum;
    device_selection_digest selection;
};
static_assert(sizeof(device_selected_status) == 24);

__device__ void block_selection_digest_add(
        unsigned long long item, device_selection_digest* digest,
        unsigned long long* warp_xor,
        unsigned long long* warp_sum) {
    constexpr unsigned kWarp = 32;
    constexpr unsigned kWarpsPerBlock = kBlock / kWarp;
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned warp = threadIdx.x / kWarp;
    unsigned long long xor_value = item;
    unsigned long long sum_value = item;
    for (unsigned offset = kWarp / 2; offset; offset >>= 1) {
        xor_value ^= __shfl_down_sync(0xffffffffU, xor_value, offset);
        sum_value += __shfl_down_sync(0xffffffffU, sum_value, offset);
    }
    if (lane == 0) {
        warp_xor[warp] = xor_value;
        warp_sum[warp] = sum_value;
    }
    __syncthreads();
    if (warp == 0) {
        xor_value = lane < kWarpsPerBlock ? warp_xor[lane] : 0;
        sum_value = lane < kWarpsPerBlock ? warp_sum[lane] : 0;
        for (unsigned offset = kWarp / 2; offset; offset >>= 1) {
            xor_value ^= __shfl_down_sync(0xffffffffU, xor_value, offset);
            sum_value += __shfl_down_sync(0xffffffffU, sum_value, offset);
        }
        if (lane == 0) {
            atomicXor(&digest->xor_hash, xor_value);
            atomicAdd(&digest->sum_hash, sum_value);
        }
    }
}

__global__ void count_degrees(const gpu_topology_edge *edges, std::size_t m,
                              edge_index *degrees,
                              device_selection_digest* topology_digest) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    unsigned long long item_hash = 0;
    if (i < m) {
        const auto e = edges[i];
        atomic_increment(&degrees[e.u]);
        atomic_increment(&degrees[e.v]);
        item_hash = gpu_device_selection_topology_hash(e.u, e.v);
    }
    __shared__ unsigned long long warp_xor[kBlock / 32];
    __shared__ unsigned long long warp_sum[kBlock / 32];
    block_selection_digest_add(
        item_hash, topology_digest, warp_xor, warp_sum);
}

__global__ void scatter_csr(const gpu_topology_edge *edges, std::size_t m,
                            edge_index *cursor, node_index *neighbors) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= m)
        return;
    const auto e = edges[i];
    const edge_index pu = atomic_increment(&cursor[e.u]);
    const edge_index pv = atomic_increment(&cursor[e.v]);
    neighbors[pu] = e.v;
    neighbors[pv] = e.u;
}

struct selector_csr_view {
    const edge_index* offsets;
    const node_index* neighbors;
    const edge_index* degrees;
    __host__ __device__ edge_index begin(node_index v) const { return offsets[v]; }
    __host__ __device__ edge_index end(node_index v) const { return offsets[v + 1]; }
    __host__ __device__ node_index neighbor(edge_index p) const { return neighbors[p]; }
    __host__ __device__ node_index degree(node_index v) const {
        return static_cast<node_index>(degrees[v]);
    }
};

struct selector_owned_view {
    const std::uint32_t* offsets;
    const gpu_round_shadow_incidence* incidences;
    __host__ __device__ edge_index begin(node_index v) const { return offsets[v]; }
    __host__ __device__ edge_index end(node_index v) const { return offsets[v + 1]; }
    __host__ __device__ node_index neighbor(edge_index p) const {
        return incidences[p].neighbor;
    }
    __host__ __device__ node_index degree(node_index v) const {
        return static_cast<node_index>(offsets[v + 1] - offsets[v]);
    }
};

template<class View, bool Audit = true>
__global__ void gather_active_degrees(const node_index *active_ids,
                                      std::size_t count,
                                      View view,
                                      node_index *output,
                                      device_selection_digest* active_digest) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    unsigned long long item_hash = 0;
    if (i < count) {
        const node_index vertex = active_ids[i];
        output[i] = view.degree(vertex);
        if constexpr (Audit) item_hash = gpu_device_selection_active_hash(vertex);
    }
    __shared__ unsigned long long warp_xor[kBlock / 32];
    __shared__ unsigned long long warp_sum[kBlock / 32];
    if constexpr (Audit)
        block_selection_digest_add(item_hash, active_digest, warp_xor, warp_sum);
}

__global__ void initialize_vertex_ids(std::size_t count, node_index *ids) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        ids[i] = static_cast<node_index>(i);
}

__global__ void sum_degree_values(const node_index *values, std::size_t count,
                                  unsigned long long *sum) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        atomicAdd(sum, static_cast<unsigned long long>(values[i]));
}

template<class View>
__global__ void sum_vertex_degrees(const node_index *ids, std::size_t count,
                                   View view,
                                   device_selected_status *status) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    unsigned long long item_hash = 0;
    unsigned long long degree_sum = 0;
    if (i < count) {
        const node_index vertex = ids[i];
        degree_sum = static_cast<unsigned long long>(view.degree(vertex));
        item_hash = gpu_device_selection_selected_hash(i, vertex);
    }
    // Every lane participates, including the zero-filled last block. Publish
    // warp sums before the digest helper's existing unconditional block barrier.
    for (unsigned offset = 16; offset; offset >>= 1)
        degree_sum += __shfl_down_sync(0xffffffffU, degree_sum, offset);
    __shared__ unsigned long long warp_degrees[kBlock / 32];
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    if (lane == 0)
        warp_degrees[warp] = degree_sum;
    __shared__ unsigned long long warp_xor[kBlock / 32];
    __shared__ unsigned long long warp_sum[kBlock / 32];
    block_selection_digest_add(
        item_hash, &status->selection, warp_xor, warp_sum);
    if (warp == 0) {
        degree_sum = lane < kBlock / 32 ? warp_degrees[lane] : 0;
        for (unsigned offset = 16; offset; offset >>= 1)
            degree_sum += __shfl_down_sync(0xffffffffU, degree_sum, offset);
        if (lane == 0)
            atomicAdd(&status->degree_sum, degree_sum);
    }
}

__global__ void set_status(const node_index *ids, std::size_t count,
                           int *status, int value) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        status[ids[i]] = value;
}

__global__ void initialize_status(std::size_t count, int *status) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        status[i] = 2;
}

__global__ void deactivate_vertices(const node_index *ids, std::size_t count,
                                    unsigned char *active) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        active[ids[i]] = 0;
}

__global__ void extract_topology_edges(const deferred_edge *input,
                                       std::size_t count,
                                       gpu_topology_edge *output) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        output[i] = {input[i].u, input[i].v};
}

template<class View>
__device__ bool block_priority_precedes(node_index a, node_index b,
                                        View view,
                                        bool degree_tiebreak) {
    if (!degree_tiebreak)
        return a < b;
    const node_index da = view.degree(a);
    const node_index db = view.degree(b);
    return da != db ? da < db : a < b;
}

__global__ void initialize_block_candidates(const node_index *candidates,
                                             std::size_t count,
                                             std::size_t region_count,
                                             int *status,
                                             node_index *region_of,
                                             int *frontier,
                                             unsigned char *scratch) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count)
        return;
    const node_index v = candidates[i];
    status[v] = 0;
    region_of[v] = static_cast<node_index>(
        (i * region_count) / count);
    frontier[v] = 0;
    scratch[v] = 0;
}

template<class View>
__global__ void block_greedy_regions(
    const node_index *candidates, std::size_t count,
    std::size_t region_count, View view,
    const node_index *region_of, int *status) {
    constexpr unsigned kFullWarp = 0xffffffffU;
    const unsigned lane = threadIdx.x & 31U;
    const std::size_t warp =
        (blockIdx.x * std::size_t(blockDim.x) + threadIdx.x) >> 5;
    if (warp >= region_count)
        return;
    // initialize_block_candidates assigns i to floor(i * R / N).  The exact
    // inverse interval is [ceil(N*r/R), ceil(N*(r+1)/R)); floor bounds would
    // hand boundary candidates to the next warp while retaining the previous
    // region label, defeating the within-region independence guarantee.
    const std::size_t begin =
        (count * warp + region_count - 1) / region_count;
    const std::size_t end =
        (count * (warp + 1) + region_count - 1) / region_count;
    for (std::size_t i = begin; i < end; ++i) {
        const node_index v = candidates[i];
        bool blocked = false;
        for (edge_index p = view.begin(v) + lane;
             p < view.end(v); p += 32) {
            const node_index u = view.neighbor(p);
            blocked |= status[u] == 1 && region_of[u] == warp;
        }
        const bool any = __any_sync(kFullWarp, blocked);
        if (lane == 0 && !any)
            status[v] = 1;
        __syncwarp(kFullWarp);
    }
}

template<class View>
__global__ void decide_block_conflicts(
    const node_index *candidates, std::size_t count,
    View view, const node_index *region_of, int *status,
    unsigned char *drop, bool degree_tiebreak) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count)
        return;
    const node_index v = candidates[i];
    bool loses = false;
    if (status[v] == 1) {
        const node_index region = region_of[v];
        for (edge_index p = view.begin(v); p < view.end(v); ++p) {
            const node_index u = view.neighbor(p);
            if (status[u] == 1 && region_of[u] != region &&
                block_priority_precedes(u, v, view, degree_tiebreak)) {
                loses = true;
                break;
            }
        }
    }
    drop[v] = static_cast<unsigned char>(loses);
}

__global__ void apply_block_drops(const node_index *candidates,
                                  std::size_t count, int *status,
                                  int *frontier, unsigned char *drop) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count)
        return;
    const node_index v = candidates[i];
    if (drop[v])
        status[v] = 0;
    // Regional greedy picks dominate every candidate before conflict drops.
    // Thus every now-free unselected candidate is a drop or its neighbor.
    // The repair's first free scan produces exactly the old explicit frontier's
    // pending set when started from all unselected candidates. This avoids the
    // dropped-row scatter and repeated atomic marks, at the cost of testing
    // additional blocked rows once. Later repair passes are unchanged.
    frontier[v] = status[v] == 1 ? 0 : 1;
    drop[v] = 0;
}

template<class View>
__global__ void block_greedy_repair(
    const node_index *candidates, std::size_t count,
    View view, int *status, int *frontier,
    unsigned char *winner, bool degree_tiebreak, int *has_pending) {
    // Keep the private launch ABI; decisions now commit status directly and
    // leave the caller's zeroed winner scratch untouched.
    (void)winner;
    namespace cg = cooperative_groups;
    const cg::grid_group grid = cg::this_grid();
    constexpr unsigned full_warp = 0xffffffffU;
    const unsigned lane = threadIdx.x & 31U;
    const std::size_t first =
        blockIdx.x * std::size_t(blockDim.x) + threadIdx.x - lane;
    const std::size_t stride = gridDim.x * std::size_t(blockDim.x);

    for (;;) {
        if (grid.thread_rank() == 0)
            *has_pending = 0;
        grid.sync();

        // A frontier vertex blocked by a committed pick can never become free:
        // the repair only adds picks. Free vertices become pending (2).
        bool thread_pending = false;
        // Loop bounds are warp-uniform even for the final partial group.
        for (std::size_t group = first; group < count; group += stride) {
            const std::size_t i = group + lane;
            const node_index v = i < count ? candidates[i] : 0;
            bool active = i < count && frontier[v] != 0;
            if (active && status[v] == 1) {
                frontier[v] = 0;
                active = false;
            }
            const bool long_row = active && view.degree(v) > 32;
            if (active && !long_row) {
                bool free = true;
                for (std::size_t p = view.begin(v); p < std::size_t(view.end(v)); ++p) {
                    if (status[view.neighbor(p)] == 1) {
                        free = false;
                        break;
                    }
                }
                frontier[v] = free ? 2 : 0;
                thread_pending |= free;
            }
            unsigned rows = __ballot_sync(full_warp, long_row);
            while (rows) {
                const unsigned owner = __ffs(rows) - 1;
                const node_index row = __shfl_sync(full_warp, v, owner);
                bool free = true;
                const std::size_t end = view.end(row);
                for (std::size_t base = view.begin(row); base < end; base += 32) {
                    const std::size_t p = base + lane;
                    const bool blocked = p < end && status[view.neighbor(p)] == 1;
                    if (__any_sync(full_warp, blocked)) {
                        free = false;
                        break;
                    }
                }
                if (lane == owner) {
                    frontier[row] = free ? 2 : 0;
                    thread_pending |= free;
                }
                rows &= rows - 1;
            }
        }
        // This reduction is outside the divergent scan: even threads with no
        // candidate reach it. The following grid barrier publishes each block's
        // OR before any thread tests the shared termination flag.
        const int block_pending = __syncthreads_or(thread_pending);
        if (threadIdx.x == 0 && block_pending)
            atomicExch(has_pending, 1);
        grid.sync();
        if (*has_pending == 0)
            break;

        // Pick all minima of the pending frontier under the same strict order
        // as the CPU repair. Decisions read a barrier-separated snapshot.
        for (std::size_t group = first; group < count; group += stride) {
            const std::size_t i = group + lane;
            const node_index v = i < count ? candidates[i] : 0;
            const bool active = i < count && frontier[v] == 2;
            const bool long_row = active && view.degree(v) > 32;
            if (active && !long_row) {
                bool wins = true;
                for (std::size_t p = view.begin(v); p < std::size_t(view.end(v)); ++p) {
                    const node_index u = view.neighbor(p);
                    if (frontier[u] == 2 &&
                        block_priority_precedes(u, v, view, degree_tiebreak)) {
                        wins = false;
                        break;
                    }
                }
                if (wins)
                    status[v] = 1;
            }
            unsigned rows = __ballot_sync(full_warp, long_row);
            while (rows) {
                const unsigned owner = __ffs(rows) - 1;
                const node_index row = __shfl_sync(full_warp, v, owner);
                bool wins = true;
                const std::size_t end = view.end(row);
                for (std::size_t base = view.begin(row); base < end; base += 32) {
                    const std::size_t p = base + lane;
                    bool loses = false;
                    if (p < end) {
                        const node_index u = view.neighbor(p);
                        loses = frontier[u] == 2 &&
                            block_priority_precedes(u, row, view, degree_tiebreak);
                    }
                    if (__any_sync(full_warp, loses)) {
                        wins = false;
                        break;
                    }
                }
                // Only the owning lane publishes; decision scans read the
                // unchanged frontier and priority, never another pick's status.
                if (lane == owner && wins)
                    status[row] = 1;
                rows &= rows - 1;
            }
        }
        // Besides publishing picks, this ensures every thread has consumed
        // has_pending before rank0 may clear it at the next loop entrance.
        grid.sync();
    }
}

// Research policy: Yves' strict (degree, phase/seed hash, vertex) order, fixed
// throughout at most four global snapshot passes. CSR rows remain authoritative;
// no segmented residual representation or maximality-completion pass is added.
__device__ bool bounded_priority_precedes(node_index a, node_index b,
        selector_owned_view view, bool degree_tiebreak,
        std::uint64_t seed, std::uint64_t phase) {
    if (degree_tiebreak && view.degree(a) != view.degree(b))
        return view.degree(a) < view.degree(b);
    const std::uint64_t key = seed ^ (phase * 0xD6E8FEB86659FD93ULL);
    const auto ha = gpu_device_selection_mix(key ^ a);
    const auto hb = gpu_device_selection_mix(key ^ b);
    return ha != hb ? ha < hb : a < b;
}

__global__ void bounded_decide(const node_index* candidates, std::size_t count,
        selector_owned_view view, const int* status, unsigned char* winners,
        bool degree_tiebreak, std::uint64_t seed, std::uint64_t phase) {
    const std::size_t index =
        (blockIdx.x * std::size_t(blockDim.x) + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x & 31U;
    if (index >= count) return; // whole warp, including the final partial block
    const node_index v = candidates[index];
    bool wins = true;
    const std::size_t end = view.end(v);
    for (std::size_t begin = view.begin(v); begin < end; begin += 32) {
        const std::size_t at = begin + lane;
        bool loses = false;
        if (at < end) {
            const auto u = view.neighbor(static_cast<edge_index>(at));
            loses = status[u] == 0 &&
                bounded_priority_precedes(u, v, view, degree_tiebreak, seed, phase);
        }
        if (__any_sync(0xffffffffU, loses)) { wins = false; break; }
    }
    if (lane == 0) winners[index] = wins;
}

__global__ void bounded_commit(const node_index* candidates, std::size_t count,
        selector_owned_view view, const unsigned char* winners, int* status) {
    const std::size_t index =
        (blockIdx.x * std::size_t(blockDim.x) + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x & 31U;
    if (index >= count || !winners[index]) return; // warp-uniform
    const node_index v = candidates[index];
    if (lane == 0) atomicCAS(status + v, 0, 1);
    // Winners are independent in the preceding immutable snapshot. Multiple
    // winners/parallel incidences may mark the same neighbor; the mark is idempotent.
    for (std::size_t at = std::size_t(view.begin(v)) + lane;
         at < std::size_t(view.end(v)); at += 32)
        atomicCAS(status + view.neighbor(static_cast<edge_index>(at)), 0, 2);
}

struct status_equals {
    const int *status;
    int value;
    __host__ __device__ bool operator()(node_index v) const {
        return status[v] == value;
    }
};

struct vertex_is_active {
    const unsigned char *active;
    __host__ __device__ bool operator()(node_index v) const {
        return active[v] != 0;
    }
};

template<class View> struct degree_is_eligible {
    View view;
    double threshold;
    __host__ __device__ bool operator()(node_index v) const {
        return static_cast<double>(view.degree(v)) <= threshold;
    }
};

struct edge_is_live {
    const unsigned char *active;
    __host__ __device__ bool operator()(gpu_topology_edge e) const {
        return active[e.u] && active[e.v];
    }
};

double elapsed_ms(clock_type::time_point start) {
    if (!gpu_setup_diagnostics())
        return std::numeric_limits<double>::quiet_NaN();
    return std::chrono::duration<double, std::milli>(clock_type::now() - start)
        .count();
}

bool trace_enabled() {
    if (!gpu_setup_diagnostics()) return false;
    const char *e = std::getenv("APXCHOL_GPU_BLOCK_TRACE");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool add_allocation(std::size_t &total, std::size_t count,
                    std::size_t item_bytes) {
    if (count && item_bytes > std::numeric_limits<std::size_t>::max() / count)
        return false;
    const std::size_t bytes = count * item_bytes;
    if (bytes > std::numeric_limits<std::size_t>::max() - total)
        return false;
    total += bytes;
    return true;
}

} // namespace

struct gpu_block_frontend::impl {
    explicit impl(node_index n_in,
                  std::span<const gpu_topology_edge> initial_edges,
                  std::shared_ptr<gpu_device_selection_producer> producer,
                  std::size_t owned_edges = 0, bool owning_initial = false)
        : n(n_in), live_edge_count(owning_initial ? owned_edges : initial_edges.size()), active_count(n_in),
          selection_producer(std::move(producer)) {
        if (static_cast<std::uint64_t>(n) + 1 >
            static_cast<std::uint64_t>(INT_MAX))
            throw std::overflow_error(
                "GPU block front-end currently requires n < INT_MAX");
        if (live_edge_count > static_cast<std::size_t>(INT_MAX))
            throw std::overflow_error(
                "GPU block front-end currently requires live edges < INT_MAX");

        int device = 0;
        int cooperative = 0;
        cuda_check(cudaGetDevice(&device), "query active CUDA device");
        selection_producer->bind_device(device);
        cuda_check(cudaDeviceGetAttribute(&cooperative,
                                          cudaDevAttrCooperativeLaunch, device),
                   "query cooperative-launch support");
        if (!cooperative)
            throw std::runtime_error("GPU block front-end requires "
                                     "cooperative-kernel launch support");
        int block_repair_blocks_per_sm = 0;
        int block_scan_blocks_per_sm = 0;
        int sms = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &block_repair_blocks_per_sm, block_greedy_repair<selector_csr_view>,
                       kBlock, 0),
                   "query cooperative block-greedy repair occupancy");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &block_scan_blocks_per_sm, block_greedy_regions<selector_csr_view>,
                       kBlock, 0),
                   "query block-greedy region-scan occupancy");
        cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                          device),
                   "query CUDA multiprocessor count");
        block_repair_grid_limit = block_repair_blocks_per_sm * sms;
        block_region_limit = block_scan_blocks_per_sm * sms * (kBlock / 32);
        int owned_repair_blocks_per_sm = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &owned_repair_blocks_per_sm,
                       block_greedy_repair<selector_owned_view>, kBlock, 0),
                   "query owned block-greedy repair occupancy");
        owned_block_repair_grid_limit = owned_repair_blocks_per_sm * sms;
        if (owned_block_repair_grid_limit <= 0)
            throw std::runtime_error("GPU owned selector has no repair residency");
        // Logical regions remain those of the existing selector policy. The
        // region scan is not cooperative, so its residency is not a launch
        // constraint; changing its region count would change the sampled factor.
        if (block_repair_grid_limit <= 0)
            throw std::runtime_error(
                "GPU block-greedy front-end found no cooperative-kernel residency");
        if (block_region_limit <= 0)
            throw std::runtime_error(
                "GPU block-greedy front-end found no region-scan residency");

        if (!owning_initial) {
            coo[0].reserve(initial_edges.size());
            coo[1].reserve(initial_edges.size());
            if (!initial_edges.empty())
                cuda_check(cudaMemcpy(coo[0].get(), initial_edges.data(),
                           initial_edges.size() * sizeof(gpu_topology_edge),
                           cudaMemcpyHostToDevice), "upload initial topology");
            active_mask.reserve(n);
            degrees.reserve(n + 1);
            row_offsets.reserve(n + 1);
            row_cursor.reserve(n);
            csr_neighbors.reserve(initial_edges.size() * 2);
        }
        status.reserve(n);
        pick.reserve(n);
        active_ids[0].reserve(n);
        active_ids[1].reserve(n);
        active_degrees.reserve(n);
        sorted_degrees.reserve(n);
        candidates_original.reserve(n);
        selected_ids.reserve(n);
        if (!owning_initial) {
            update_ids.reserve(n);
            topology_staging.reserve(
                std::min(initial_edges.size(), kTopologyStageEdges));
        }
        selected_count.reserve(1);
        selected_status.reserve(1);
        prepare_status.reserve(1);
        cooperative_flag.reserve(1);
        cuda_check(cudaMemset(prepare_status.get(), 0,
                              sizeof(device_prepare_status)),
                   "initialize producer content status");
        if (!owning_initial)
            cuda_check(cudaMemset(active_mask.get(), 1, n),
                       "initialize active mask");
        if (n) {
            initialize_status<<<blocks_for(n), kBlock>>>(n, status.get());
            cuda_check(cudaGetLastError(), "initialize block-greedy status");
            initialize_vertex_ids<<<blocks_for(n), kBlock>>>(
                n, active_ids[current_active].get());
            cuda_check(cudaGetLastError(), "initialize active vertex ids");
        }

        // Reserve the ordinary host-update path's largest CUB workspace;
        // its counts only shrink after round zero. Resident projection selects
        // directed incidences and may grow this scratch; advance_resident
        // accounts for that additional allocation.
        std::size_t cub_bytes = 0;
        std::size_t bytes = 0;
        if (!owning_initial) {
            cuda_check(cub::DeviceScan::ExclusiveSum(
                           nullptr, bytes, degrees.get(), row_offsets.get(),
                           static_cast<int>(std::size_t(n) + 1)),
                       "query initial CUB scan workspace");
            cub_bytes = std::max(cub_bytes, bytes);
        }
        if (!initial_edges.empty()) {
            bytes = 0;
            cuda_check(
                cub::DeviceSelect::If(nullptr, bytes, coo[0].get(),
                                      coo[1].get(), selected_count.get(),
                                      static_cast<int>(initial_edges.size()),
                                      edge_is_live{active_mask.get()}),
                "query initial edge-selection workspace");
            cub_bytes = std::max(cub_bytes, bytes);
        }
        if (n) {
            bytes = 0;
            cuda_check(cub::DeviceSelect::If(
                           nullptr, bytes, candidates_original.get(),
                           selected_ids.get(), selected_count.get(),
                           static_cast<int>(n), status_equals{status.get(), 1}),
                       "query initial vertex-selection workspace");
            cub_bytes = std::max(cub_bytes, bytes);
            bytes = 0;
            cuda_check(cub::DeviceRadixSort::SortKeys(
                           nullptr, bytes, active_degrees.get(),
                           sorted_degrees.get(), static_cast<int>(n)),
                       "query initial degree-sort workspace");
            cub_bytes = std::max(cub_bytes, bytes);
        }
        cub_temp.reserve(cub_bytes);
    }

    void require_current_device(bool read_residual = true) const {
        int current_device = -1;
        cuda_check(cudaGetDevice(&current_device),
                   "query active CUDA device for producer operation");
        if (current_device != selection_producer->cuda_device())
            throw std::invalid_argument(
                "GPU block front-end operation uses the wrong CUDA device");
        selection_producer->require_usable();
        if (read_residual && owns_residual_view && residual_epoch.expired())
            throw std::logic_error("GPU selector residual generation is retired");
    }

    selector_csr_view csr_view() const {
        return {row_offsets.get(), csr_neighbors.get(), degrees.get()};
    }

    /// Invalidate before, rather than after, every producer mutation. Thus an
    /// exception cannot leave an earlier view apparently current.
    void invalidate_selection() {
        selected_size = 0;
        selection_producer->invalidate_selection();
    }

    void publish_selection() {
        selection_producer->publish_selection(
            selected_ids.get(), selected_size, selection_content,
            /*independence_certified=*/true);
    }

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    void inject_device_selection_fault_for_test(
            std::span<const node_index> replacement) {
        if (selected_size == 0 || replacement.size() != selected_size)
            throw std::logic_error(
                "GPU block front-end test fault has the wrong selection size");
        cuda_check(cudaMemcpy(selected_ids.get(), replacement.data(),
                              replacement.size_bytes(),
                              cudaMemcpyHostToDevice),
                   "inject device-selection test fault");
    }
#endif

    void begin_topology_advance() {
        selection_producer->begin_topology_advance();
    }

    void poison_selection_producer() noexcept {
        selection_producer->poison();
        selected_size = 0;
    }

    void retire_selection_producer() noexcept {
        selection_producer->retire();
    }

    void require_cub_count(std::size_t count, const char *what) const {
        if (count > static_cast<std::size_t>(INT_MAX))
            throw std::overflow_error(std::string("GPU block front-end: ") +
                                      what + " exceeds CUB's item-count limit");
    }

    void exclusive_sum(const edge_index *input, edge_index *output,
                       std::size_t count) {
        require_cub_count(count, "scan");
        std::size_t bytes = 0;
        cuda_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, input, output,
                                                 static_cast<int>(count)),
                   "query CUB scan workspace");
        cub_temp.reserve(bytes);
        cuda_check(cub::DeviceScan::ExclusiveSum(cub_temp.get(), bytes, input,
                                                 output,
                                                 static_cast<int>(count)),
                   "CUB exclusive scan");
    }

    template <class Input, class T, class Predicate>
    int select_if(Input input, T *output, std::size_t count,
                  Predicate predicate) {
        require_cub_count(count, "selection");
        selected_count.reserve(1);
        std::size_t bytes = 0;
        cuda_check(cub::DeviceSelect::If(nullptr, bytes, input, output,
                                         selected_count.get(),
                                         static_cast<int>(count), predicate),
                   "query CUB selection workspace");
        cub_temp.reserve(bytes);
        cuda_check(cub::DeviceSelect::If(cub_temp.get(), bytes, input, output,
                                         selected_count.get(),
                                         static_cast<int>(count), predicate),
                   "CUB ordered selection");
        int host_count = 0;
        cuda_check(cudaMemcpy(&host_count, selected_count.get(),
                              sizeof(host_count), cudaMemcpyDeviceToHost),
                   "download selection count");
        return host_count;
    }

    void sort_degrees(std::size_t count) {
        require_cub_count(count, "degree sort");
        if (!count)
            return;
        std::size_t bytes = 0;
        cuda_check(cub::DeviceRadixSort::SortKeys(
                       nullptr, bytes, active_degrees.get(),
                       sorted_degrees.get(), static_cast<int>(count)),
                   "query degree-sort workspace");
        cub_temp.reserve(bytes);
        cuda_check(cub::DeviceRadixSort::SortKeys(
                       cub_temp.get(), bytes, active_degrees.get(),
                       sorted_degrees.get(), static_cast<int>(count)),
                   "sort active degrees");
    }

    unsigned long long sum_active_degree_values(std::size_t count) {
        cuda_check(cudaMemset(&prepare_status.get()->degree_sum, 0,
                              sizeof(unsigned long long)),
                   "clear active-degree sum");
        if (count) {
            sum_degree_values<<<blocks_for(count), kBlock>>>(
                active_degrees.get(), count,
                &prepare_status.get()->degree_sum);
            cuda_check(cudaGetLastError(), "sum active degrees");
        }
        device_prepare_status host_status{};
        // This replaces the old 8-byte degree-sum readback: the same bounded
        // synchronization also publishes the producer-owned content digest.
        cuda_check(cudaMemcpy(&host_status, prepare_status.get(),
                              sizeof(host_status), cudaMemcpyDeviceToHost),
                   "download active-degree sum and selection content");
        selection_content = {
            n,
            count,
            {host_status.topology.xor_hash,
             host_status.topology.sum_hash},
            {host_status.active.xor_hash,
             host_status.active.sum_hash},
            {}};
        return host_status.degree_sum;
    }

    unsigned long long sum_selected_degrees(std::size_t count) {
        selection_content.selection = {};
        if (!count)
            return 0;
        cuda_check(cudaMemset(selected_status.get(), 0,
                              sizeof(device_selected_status)),
                   "clear selected-degree/content status");
        if (owns_residual_view)
            sum_vertex_degrees<<<blocks_for(count), kBlock>>>(
                selected_ids.get(), count, owned_view, selected_status.get());
        else
            sum_vertex_degrees<<<blocks_for(count), kBlock>>>(
                selected_ids.get(), count, csr_view(), selected_status.get());
        cuda_check(cudaGetLastError(), "sum selected degrees");
        device_selected_status host_status{};
        cuda_check(cudaMemcpy(&host_status, selected_status.get(),
                              sizeof(host_status), cudaMemcpyDeviceToHost),
                   "download selected-degree sum and ordered selection digest");
        selection_content.selection = {
            host_status.selection.xor_hash,
            host_status.selection.sum_hash};
        return host_status.degree_sum;
    }

    void rebuild_topology() {
        if (!topology_dirty)
            return;

        if (live_edge_count > std::numeric_limits<std::size_t>::max() / 2)
            throw std::overflow_error("GPU block front-end CSR size overflow");
        const std::size_t directed = live_edge_count * 2;
        if (directed >
            static_cast<std::size_t>(std::numeric_limits<edge_index>::max()))
            throw std::overflow_error(
                "GPU block front-end directed CSR exceeds edge_index range");

        cuda_check(cudaMemset(degrees.get(), 0,
                              (std::size_t(n) + 1) * sizeof(edge_index)),
                   "clear degree counts");
        cuda_check(cudaMemset(&prepare_status.get()->topology, 0,
                              sizeof(device_selection_digest)),
                   "clear producer topology digest");
        if (live_edge_count) {
            count_degrees<<<blocks_for(live_edge_count), kBlock>>>(
                coo[current_coo].get(), live_edge_count, degrees.get(),
                &prepare_status.get()->topology);
            cuda_check(cudaGetLastError(), "launch degree count");
        }
        exclusive_sum(degrees.get(), row_offsets.get(), std::size_t(n) + 1);

        csr_neighbors.reserve(directed);
        if (n) {
            cuda_check(cudaMemcpy(row_cursor.get(), row_offsets.get(),
                                  std::size_t(n) * sizeof(edge_index),
                                  cudaMemcpyDeviceToDevice),
                       "initialize CSR cursors");
        }
        if (live_edge_count) {
            scatter_csr<<<blocks_for(live_edge_count), kBlock>>>(
                coo[current_coo].get(), live_edge_count, row_cursor.get(),
                csr_neighbors.get());
            cuda_check(cudaGetLastError(), "launch CSR scatter");
        }

        topology_dirty = false;
    }

    gpu_block_frontend::prepare_result
    prepare(std::span<const node_index> active,
            const partition_options &options) {
        const auto start = clock_type::now();
        rebuild_topology();

        active_degrees.reserve(active.size());
        sorted_degrees.reserve(active.size());
        candidates_original.reserve(active.size());
        selected_ids.reserve(active.size());
        if (active.size() != active_count)
            throw std::logic_error(
                "GPU block active-list size diverged from CPU state");
        host_active_degrees.clear();
        host_candidate_ids.clear();
        host_active_degrees_valid = false;
        host_candidate_ids_valid = false;

        unsigned long long total_degree = 0;
        if (owns_residual_view) {
            if (!active.empty()) {
                gather_active_degrees<selector_owned_view, false>
                    <<<blocks_for(active.size()), kBlock>>>(
                        active_ids[current_active].get(), active.size(), owned_view,
                        active_degrees.get(), nullptr);
                cuda_check(cudaGetLastError(), "gather owned active degrees");
            }
            // Paired owner incidences are exactly the active degree sum.
            total_degree = 2ULL * live_edge_count;
            selection_content.selection = {};
        } else {
            cuda_check(cudaMemset(&prepare_status.get()->active, 0,
                                  sizeof(device_selection_digest)),
                       "clear producer active-set digest");
            if (!active.empty()) {
                gather_active_degrees<<<blocks_for(active.size()), kBlock>>>(
                    active_ids[current_active].get(), active.size(), csr_view(),
                    active_degrees.get(), &prepare_status.get()->active);
                cuda_check(cudaGetLastError(), "launch active-degree gather");
            }
            total_degree = sum_active_degree_values(active.size());
        }
        const double average =
            active.empty() ? 0.0
                           : static_cast<double>(total_degree) /
                                 static_cast<double>(active.size());

        double threshold = options.degree_multiplier * average;
        const double q = options.degree_quantile;
        if (q > 0.0 && q < 1.0 && !active.empty()) {
            std::size_t quantile_count = active.size();
            if (owns_residual_view) --quantile_count; // Yves' floor(q*(active-1))
            std::size_t rank = static_cast<std::size_t>(q * quantile_count);
            if (rank >= active.size()) rank = active.size() - 1;
            sort_degrees(active.size());
            node_index quantile = 0;
            cuda_check(cudaMemcpy(&quantile, sorted_degrees.get() + rank,
                                  sizeof(quantile), cudaMemcpyDeviceToHost),
                       "download degree quantile");
            threshold = static_cast<double>(quantile);
        }

        if (owns_residual_view)
            candidate_count = static_cast<std::size_t>(select_if(
                active_ids[current_active].get(), candidates_original.get(),
                active.size(), degree_is_eligible{owned_view, threshold}));
        else
            candidate_count = static_cast<std::size_t>(select_if(
                active_ids[current_active].get(), candidates_original.get(),
                active.size(), degree_is_eligible{csr_view(), threshold}));

        last_prepare = elapsed_ms(start);
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[gpu-block] prepare active=%zu candidates=%zu "
                         "edges=%zu %.3f ms\n",
                         active.size(), candidate_count, live_edge_count,
                         last_prepare);
        }
        return {.candidate_count = candidate_count, .average_degree = average};
    }

    std::span<const node_index> download_host_candidates() {
        if (!host_candidate_ids_valid) {
            host_candidate_ids.resize(candidate_count);
            if (candidate_count) {
                cuda_check(cudaMemcpy(host_candidate_ids.data(),
                                      candidates_original.get(),
                                      candidate_count * sizeof(node_index),
                                      cudaMemcpyDeviceToHost),
                           "download debug candidate ids");
            }
            host_candidate_ids_valid = true;
        }
        return host_candidate_ids;
    }

    std::span<const node_index> download_host_active_degrees() {
        if (!host_active_degrees_valid) {
            host_active_degrees.resize(active_count);
            if (active_count) {
                cuda_check(cudaMemcpy(host_active_degrees.data(),
                                      active_degrees.get(),
                                      active_count * sizeof(node_index),
                                      cudaMemcpyDeviceToHost),
                           "download debug active degrees");
            }
            host_active_degrees_valid = true;
        }
        return host_active_degrees;
    }

    std::size_t block_region_count() const {
        const char *e = std::getenv("APXCHOL_GPU_BLOCKS");
        if (e && *e) {
            char *end = nullptr;
            const unsigned long long parsed = std::strtoull(e, &end, 10);
            if (end == e || *end != '\0' || parsed == 0)
                throw std::invalid_argument(
                    "APXCHOL_GPU_BLOCKS must be a positive integer");
            return std::min<std::size_t>(candidate_count,
                                         static_cast<std::size_t>(parsed));
        }
        // One contiguous candidate region per warp that can be resident in
        // the region-scan kernel. This is a hardware occupancy choice, not a
        // graph-size cutoff; APXCHOL_GPU_BLOCKS pins alternatives for A/B.
        return std::min<std::size_t>(candidate_count,
                                     static_cast<std::size_t>(block_region_limit));
    }

    template<class View>
    void launch_selection(View view, std::size_t regions, int repair_grid_limit) {
        constexpr std::size_t warps_per_block = kBlock / 32;
        const std::size_t scan_blocks =
            (regions + warps_per_block - 1) / warps_per_block;
        block_greedy_regions<<<static_cast<int>(scan_blocks), kBlock>>>(
            candidates_original.get(), candidate_count, regions,
            view, block_region.get(),
            status.get());
        cuda_check(cudaGetLastError(), "scan block-greedy regions");

        decide_block_conflicts<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, view, block_region.get(), status.get(),
            pick.get(), current_options.degree_tiebreak);
        cuda_check(cudaGetLastError(), "decide block-greedy conflicts");
        apply_block_drops<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, status.get(),
            block_frontier.get(), pick.get());
        cuda_check(cudaGetLastError(), "apply block-greedy conflicts");

        const int wanted = blocks_for(candidate_count);
        const int grid =
            std::max(1, std::min(wanted, repair_grid_limit));
        const node_index *candidate_ptr = candidates_original.get();
        int *status_ptr = status.get();
        int *frontier_ptr = block_frontier.get();
        unsigned char *winner_ptr = pick.get();
        bool degree_tiebreak = current_options.degree_tiebreak;
        int *pending_ptr = cooperative_flag.get();
        void *args[] = {
            &candidate_ptr, &candidate_count, &view,
            &status_ptr, &frontier_ptr, &winner_ptr,
            &degree_tiebreak, &pending_ptr};
        cuda_check(cudaLaunchCooperativeKernel(
                       reinterpret_cast<void *>(block_greedy_repair<View>), grid,
                       kBlock, args, 0, nullptr),
                   "launch cooperative block-greedy repair");
    }

    void launch_bounded_selection() {
        // selected_ids is already reserved. update_ids is otherwise unused on
        // the owned path; these buffers alternate without touching the immutable
        // original candidates. No region/frontier buffers are allocated here.
        update_ids.reserve(candidate_count);
        set_status<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, status.get(), 0);
        cuda_check(cudaGetLastError(), "initialize bounded candidates");
        const node_index* current = candidates_original.get();
        node_index* next = update_ids.get();
        std::size_t count = candidate_count;
        bounded_passes = 0;
        bounded_candidate_visits = 0;
        for (unsigned step = 0; step < 4 && count; ++step) {
            ++bounded_passes;
            bounded_candidate_visits += count;
            // n<INT_MAX and at most32 lanes/candidate: checked size_t arithmetic,
            // launch grid remains well within CUDA's signed-int X dimension.
            if (count > (std::numeric_limits<std::size_t>::max() - (kBlock - 1)) / 32)
                throw std::overflow_error("bounded selector launch size overflow");
            const int blocks = blocks_for(count * std::size_t{32});
            bounded_decide<<<blocks, kBlock>>>(current, count, owned_view,
                status.get(), pick.get(), current_options.degree_tiebreak,
                bounded_seed, bounded_phase);
            cuda_check(cudaGetLastError(), "decide bounded independent set");
            bounded_commit<<<blocks, kBlock>>>(current, count, owned_view,
                pick.get(), status.get());
            cuda_check(cudaGetLastError(), "commit bounded independent set");
            if (step == 3) break;
            // Stable compaction is also the completion boundary before the next
            // snapshot. Selected/blocked vertices never reenter this call.
            count = static_cast<std::size_t>(select_if(current, next, count,
                status_equals{status.get(), 0}));
            current = next;
            next = next == update_ids.get() ? selected_ids.get() : update_ids.get();
        }
    }

    const partition_result &select_block_greedy() {
        const auto start = clock_type::now();
        result.data.clear();
        if (!candidate_count) {
            selected_size = 0;
            selected_degree_work = 0;
            last_select = elapsed_ms(start);
            return result;
        }

        std::size_t regions = 0;
        if (owns_residual_view) {
            launch_bounded_selection();
        } else
        {
            block_region.reserve(n);
            block_frontier.reserve(n);
            regions = block_region_count();
            initialize_block_candidates<<<blocks_for(candidate_count), kBlock>>>(
                candidates_original.get(), candidate_count, regions, status.get(),
                block_region.get(), block_frontier.get(), pick.get());
            cuda_check(cudaGetLastError(), "initialize block-greedy candidates");

            if (owns_residual_view)
                launch_selection(owned_view, regions, owned_block_repair_grid_limit);
            else
                launch_selection(csr_view(), regions, block_repair_grid_limit);
        }
        const std::size_t chosen = static_cast<std::size_t>(
            select_if(candidates_original.get(), selected_ids.get(),
                      candidate_count, status_equals{status.get(), 1}));
        if (owns_residual_view && !chosen)
            throw std::logic_error("bounded selector made no progress on nonempty candidates");
        selected_size = chosen;
        selected_degree_work = sum_selected_degrees(chosen);
        result.data.resize(chosen);
        if (chosen) {
            cuda_check(cudaMemcpy(result.data.data(), selected_ids.get(),
                                  chosen * sizeof(node_index),
                                  cudaMemcpyDeviceToHost),
                       "download block-greedy selection");
        }
        set_status<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, status.get(), 2);
        cuda_check(cudaGetLastError(), "restore block-greedy candidate status");
        cuda_check(cudaDeviceSynchronize(), "finish block-greedy selection");

        last_select = elapsed_ms(start);
        if (owns_residual_view && trace_enabled())
            std::fprintf(stderr,
                "[gpu-bounded-selection] passes=%u candidate_visits=%llu "
                "seed=%llu phase=%llu scratch_capacity_bytes=%zu\n",
                bounded_passes, static_cast<unsigned long long>(bounded_candidate_visits),
                static_cast<unsigned long long>(bounded_seed),
                static_cast<unsigned long long>(bounded_phase),
                update_ids.capacity() * sizeof(node_index));
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[gpu-block] select candidates=%zu regions=%zu "
                         "chosen=%zu %.3f ms\n",
                         candidate_count, regions, chosen, last_select);
        }
        return result;
    }

    void bind_owned_residual(const gpu_round_shadow_incidence* incidences,
                             const std::uint32_t* offsets,
                             const std::uint8_t* active,
                             std::size_t directed_count,
                             const gpu_device_selection_content& expected,
                             std::weak_ptr<const void> epoch) {
        if (epoch.expired() || expected.vertex_count != n ||
            expected.active_count > active_count || directed_count % 2 ||
            directed_count / 2 > live_edge_count || !offsets || (n && !active) ||
            (directed_count && !incidences))
            throw std::logic_error("GPU selector owned residual has invalid dimensions or lifetime");
        if (active_count) {
            const int next = 1 - current_active;
            const std::size_t count = std::size_t(select_if(
                active_ids[current_active].get(), active_ids[next].get(),
                active_count, vertex_is_active{active}));
            if (count != expected.active_count)
                throw std::logic_error("GPU selector owned active count mismatch");
            active_count = count;
            current_active = next;
        } else if (expected.active_count != 0) {
            throw std::logic_error("GPU selector owned active count increased");
        }
        // The private caller binds producer/device/consumed-selection identity
        // before entry. The arrays and digests belong to this exact immutable
        // numerical generation; copying/re-hashing them adds no validation.
        owned_view = {offsets, incidences};
        residual_epoch = std::move(epoch);
        owns_residual_view = true;
        live_edge_count = directed_count / 2;
        selection_content = expected;
        topology_dirty = false;
        candidate_count = 0;
        host_active_degrees_valid = host_candidate_ids_valid = false;
        ++transfers.resident_advances;
        ++transfers.owned_residual_binds;
    }

    void advance_resident(const gpu_round_shadow_incidence* incidences,
                          std::size_t directed_count, const std::uint8_t* active,
                          const gpu_device_selection_content& expected) {
        if (owns_residual_view)
            throw std::logic_error("GPU owned selector cannot resume residual projection");
        if (expected.vertex_count != n || expected.active_count > active_count ||
            directed_count % 2 || directed_count / 2 > live_edge_count)
            throw std::logic_error("GPU selector resident projection has invalid dimensions");
        const std::size_t scratch_before = cub_temp.capacity();
        const int next = 1 - current_coo;
        // The accepted residual has paired directed incidences. Selecting one
        // orientation retains EVERY multigraph edge; neither weights nor owner
        // encounter order in the resident allocation are modified. Its edge
        // count cannot grow after independent tree elimination, so the existing
        // selector COO capacity suffices. No endpoint staging array is needed.
        const std::size_t count = directed_count ? std::size_t(select_if(
            thrust::make_transform_iterator(incidences, residual_endpoints{}), coo[next].get(),
            directed_count, canonical_endpoint_pair{})) : 0;
        if (count != directed_count / 2)
            throw std::logic_error("GPU selector resident projection is not paired");
        if (n)
            cuda_check(cudaMemcpy(active_mask.get(), active, std::size_t(n),
                                  cudaMemcpyDeviceToDevice),
                       "copy resident active mask");
        if (active_count) {
            const int next_active = 1 - current_active;
            active_count = std::size_t(select_if(
                active_ids[current_active].get(), active_ids[next_active].get(),
                active_count, vertex_is_active{active_mask.get()}));
            current_active = next_active;
        }
        current_coo = next;
        live_edge_count = count;
        topology_dirty = true;
        rebuild_topology();
        cuda_check(cudaMemset(&prepare_status.get()->active, 0,
                              sizeof(device_selection_digest)),
                   "clear projected active digest");
        if (active_count) {
            gather_active_degrees<<<blocks_for(active_count), kBlock>>>(
                active_ids[current_active].get(), active_count, csr_view(),
                active_degrees.get(), &prepare_status.get()->active);
            cuda_check(cudaGetLastError(), "gather projected active digest");
        }
        sum_active_degree_values(active_count);
        if (!gpu_device_selection_state_matches(selection_content, expected))
            throw std::logic_error("GPU selector resident projection content mismatch");
        candidate_count = 0;
        host_active_degrees_valid = host_candidate_ids_valid = false;
        cuda_check(cudaDeviceSynchronize(), "finish resident selector projection");
        ++transfers.resident_advances;
        const std::size_t scratch_after = cub_temp.capacity();
        if (scratch_after > scratch_before)
            transfers.projection_scratch_peak_extra_bytes = std::max(
                transfers.projection_scratch_peak_extra_bytes,
                2 * scratch_after - scratch_before);
    }

    void advance(std::span<const node_index> eliminated,
                 std::span<const gpu_topology_edge> new_edges,
                 std::span<const gpu_topology_batch> new_edge_batches) {
        const auto start = clock_type::now();
        if (owns_residual_view)
            throw std::logic_error("GPU owned selector cannot resume CPU topology updates");

        update_ids.reserve(eliminated.size());
        if (!eliminated.empty()) {
            cuda_check(cudaMemcpy(update_ids.get(), eliminated.data(),
                                  eliminated.size() * sizeof(node_index),
                                  cudaMemcpyHostToDevice),
                       "upload eliminated vertices");
            transfers.host_update_bytes += eliminated.size_bytes();
            deactivate_vertices<<<blocks_for(eliminated.size()), kBlock>>>(
                update_ids.get(), eliminated.size(), active_mask.get());
            cuda_check(cudaGetLastError(), "launch vertex deactivation");
        }

        if (active_count) {
            const int next = 1 - current_active;
            active_count = static_cast<std::size_t>(select_if(
                active_ids[current_active].get(), active_ids[next].get(),
                active_count, vertex_is_active{active_mask.get()}));
            current_active = next;
        }

        require_cub_count(live_edge_count, "edge compaction");
        const int next = 1 - current_coo;
        const std::size_t old_count = live_edge_count;
        const std::size_t kept =
            live_edge_count
                ? static_cast<std::size_t>(select_if(
                      coo[current_coo].get(), coo[next].get(), live_edge_count,
                      edge_is_live{active_mask.get()}))
                : 0;
        std::size_t added = new_edges.size();
        for (const auto batch : new_edge_batches) {
            if (batch.size > std::numeric_limits<std::size_t>::max() - added)
                throw std::overflow_error(
                    "GPU block front-end topology batch size overflow");
            added += batch.size;
        }
        if (added > std::numeric_limits<std::size_t>::max() - kept)
            throw std::overflow_error(
                "GPU block front-end live topology size overflow");
        const std::size_t total = kept + added;
        if (total > static_cast<std::size_t>(INT_MAX))
            throw std::overflow_error(
                "GPU block front-end live topology exceeds INT_MAX edges");
        // Eliminating an independent set removes sum(deg(v)) old edges and
        // emits at most sum(deg(v)-1) sampled edges. Isolated vertices emit
        // nothing, so multiplicity-aware topology never grows. The old COO
        // capacity is therefore sufficient for compacted + appended edges.
        if (total > old_count)
            throw std::logic_error("GPU block front-end topology "
                                   "unexpectedly grew after elimination");

        std::size_t offset = kept;
        if (!new_edges.empty()) {
            cuda_check(cudaMemcpy(coo[next].get() + offset, new_edges.data(),
                                  new_edges.size() * sizeof(gpu_topology_edge),
                                  cudaMemcpyHostToDevice),
                       "append contiguous sampled topology edges");
            transfers.host_update_bytes += new_edges.size_bytes();
            offset += new_edges.size();
        }
        for (const auto batch : new_edge_batches) {
            for (std::size_t begin = 0; begin < batch.size;) {
                const std::size_t count =
                    std::min(batch.size - begin, kTopologyStageEdges);
                topology_staging.reserve(count);
                cuda_check(cudaMemcpy(topology_staging.get(),
                                      batch.data + begin,
                                      count * sizeof(deferred_edge),
                                      cudaMemcpyHostToDevice),
                           "stage sampled topology edges");
                transfers.host_update_bytes += count * sizeof(deferred_edge);
                extract_topology_edges<<<blocks_for(count), kBlock>>>(
                    topology_staging.get(), count, coo[next].get() + offset);
                cuda_check(cudaGetLastError(),
                           "extract sampled topology endpoints");
                begin += count;
                offset += count;
            }
        }
        current_coo = next;
        live_edge_count = total;
        topology_dirty = true;
        cuda_check(cudaDeviceSynchronize(), "finish topology advance");
        last_advance = elapsed_ms(start);
        if (trace_enabled()) {
            std::fprintf(
                stderr, "[gpu-block] advance eliminated=%zu added=%zu %.3f ms\n",
                eliminated.size(), added, last_advance);
        }
    }

    gpu_block_frontend::transfer_stats transfers;
    node_index n;
    device_buffer<gpu_topology_edge> coo[2];
    int current_coo = 0;
    std::size_t live_edge_count = 0;

    device_buffer<unsigned char> active_mask;
    device_buffer<edge_index> degrees;
    device_buffer<edge_index> row_offsets;
    device_buffer<edge_index> row_cursor;
    device_buffer<node_index> csr_neighbors;
    selector_owned_view owned_view{};
    std::weak_ptr<const void> residual_epoch;
    bool owns_residual_view = false;

    device_buffer<node_index> active_ids[2];
    int current_active = 0;
    std::size_t active_count = 0;
    device_buffer<node_index> active_degrees;
    device_buffer<node_index> sorted_degrees;
    device_buffer<node_index> candidates_original;
    device_buffer<node_index> selected_ids;
    device_buffer<node_index> update_ids;
    device_buffer<node_index> block_region;
    device_buffer<int> block_frontier;
    device_buffer<deferred_edge> topology_staging;
    device_buffer<int> status;
    device_buffer<unsigned char> pick;
    device_buffer<int> selected_count;
    device_buffer<device_selected_status> selected_status;
    device_buffer<device_prepare_status> prepare_status;
    device_buffer<int> cooperative_flag;
    device_buffer<unsigned char> cub_temp;

    std::vector<node_index> host_active_degrees;
    std::vector<node_index> host_candidate_ids;
    mutable bool host_active_degrees_valid = false;
    mutable bool host_candidate_ids_valid = false;
    partition_result result;
    partition_options current_options;
    std::uint64_t bounded_seed = 42, bounded_phase = 0;
    unsigned bounded_passes = 0;
    std::uint64_t bounded_candidate_visits = 0;
    std::size_t candidate_count = 0;
    std::size_t selected_size = 0;
    std::size_t selected_degree_work = 0;
    int block_repair_grid_limit = 0;
    int owned_block_repair_grid_limit = 0;
    int block_region_limit = 0;
    bool topology_dirty = true;
    gpu_device_selection_content selection_content;
    std::shared_ptr<gpu_device_selection_producer> selection_producer;

    double last_prepare = 0.0;
    double last_select = 0.0;
    double last_advance = 0.0;
};

gpu_block_frontend::mode gpu_block_frontend::configured_block_mode() {
    const char *e = std::getenv("APXCHOL_GPU_BLOCK_FRONTEND");
    if (!e || !*e) return mode::disabled;
    if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0 ||
        std::strcmp(e, "false") == 0)
        return mode::disabled;
    if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0 ||
        std::strcmp(e, "force") == 0)
        return mode::forced;
    throw std::invalid_argument(
        std::string("unknown APXCHOL_GPU_BLOCK_FRONTEND='") + e +
        "'; expected 0|off|false|1|on|force");
}

gpu_block_frontend::runtime_probe
gpu_block_frontend::probe_runtime(node_index n, std::size_t initial_edges) {
    runtime_probe result;
    int device = 0;
    int cooperative = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                               device) != cudaSuccess) {
        (void)cudaGetLastError();
        return result;
    }
    result.cooperative_launch = cooperative != 0;
    if (cudaMemGetInfo(&result.free_bytes, &result.total_bytes) !=
        cudaSuccess) {
        (void)cudaGetLastError();
        return result;
    }

    std::size_t bytes = 0;
    bool valid = true;
    // Two ping-pong COOs plus the rebuilt directed CSR. The residual edge
    // count decreases by one per eliminated vertex, so round zero is the cap.
    if (initial_edges > std::numeric_limits<std::size_t>::max() / 2) {
        valid = false;
    } else {
        valid &=
            add_allocation(bytes, initial_edges * 2, sizeof(gpu_topology_edge));
        valid &= add_allocation(bytes, initial_edges * 2, sizeof(node_index));
    }
    const std::size_t nv = static_cast<std::size_t>(n);
    valid &= add_allocation(bytes, nv, sizeof(unsigned char)); // active
    valid &= add_allocation(bytes, nv, sizeof(int));           // status
    valid &= add_allocation(bytes, nv, sizeof(unsigned char)); // pick
    if (nv > (std::numeric_limits<std::size_t>::max() - 2) / 3) {
        valid = false;
    } else {
        valid &= add_allocation(bytes, nv * 3 + 2, sizeof(edge_index));
    }
    if (nv > std::numeric_limits<std::size_t>::max() / 7) {
        valid = false;
    } else {
        valid &= add_allocation(bytes, nv * 7, sizeof(node_index));
    }
    valid &= add_allocation(bytes,
                            std::min(initial_edges, kTopologyStageEdges),
                            sizeof(deferred_edge));
    valid &= add_allocation(bytes, nv, sizeof(node_index));
    valid &= add_allocation(bytes, nv, sizeof(int));
    valid &= add_allocation(bytes, 1, sizeof(int)); // selected count
    valid &= add_allocation(bytes, 1, sizeof(device_selected_status));
    valid &= add_allocation(bytes, 1, sizeof(device_prepare_status));
    valid &= add_allocation(bytes, 1, sizeof(int)); // cooperative flag
    // CUB scan/select scratch is implementation-dependent and much smaller
    // than the edge arrays; retain a conservative 64 MiB floor plus 5%.
    const std::size_t cub_floor = std::size_t{64} << 20;
    const std::size_t cub_margin = bytes / 20;
    valid &= add_allocation(bytes, 1, std::max(cub_floor, cub_margin));
    if (!valid) {
        result.estimated_bytes = std::numeric_limits<std::size_t>::max();
        return result;
    }
    result.estimated_bytes = bytes;
    const std::size_t reserve =
        std::max(result.total_bytes / 10, std::size_t{256} << 20);
    result.memory_fits =
        bytes <= result.free_bytes && reserve <= result.free_bytes - bytes;
    return result;
}

gpu_block_frontend::gpu_block_frontend(
    node_index n, std::span<const gpu_topology_edge> initial_edges)
    : p_(std::make_unique<impl>(
          n, initial_edges,
          std::shared_ptr<gpu_device_selection_producer>(
              new gpu_device_selection_producer()))) {}

gpu_block_frontend::gpu_block_frontend(
    node_index n, std::size_t undirected_edges, owned_initialization_tag)
    : p_(std::make_unique<impl>(n, std::span<const gpu_topology_edge>{},
          std::shared_ptr<gpu_device_selection_producer>(new gpu_device_selection_producer()),
          undirected_edges, true)) {}

void gpu_block_frontend::reset() noexcept {
    if (!p_) return;
    p_->retire_selection_producer();
    // Destruction is nonconcurrent by contract. If the caller left another
    // device current, free on the producer's device and restore it best-effort.
    int original_device = -1;
    bool switched_device = false;
    if (cudaGetDevice(&original_device) == cudaSuccess &&
        original_device != p_->selection_producer->cuda_device() &&
        cudaSetDevice(p_->selection_producer->cuda_device()) == cudaSuccess)
        switched_device = true;
    p_.reset();
    if (switched_device) (void)cudaSetDevice(original_device);
}

gpu_block_frontend::~gpu_block_frontend() { reset(); }
gpu_block_frontend::gpu_block_frontend(gpu_block_frontend &&) noexcept = default;
gpu_block_frontend &
gpu_block_frontend::operator=(gpu_block_frontend && other) noexcept {
    if (this != &other) {
        reset();
        p_ = std::move(other.p_);
    }
    return *this;
}

gpu_block_frontend::prepare_result
gpu_block_frontend::prepare(std::span<const node_index> active,
                           const partition_options &options
                           , std::uint64_t seed, std::uint64_t phase
                           ) {
    try {
        p_->require_current_device();
        p_->invalidate_selection();
        p_->current_options = options;
        p_->bounded_seed = seed;
        p_->bounded_phase = phase;
        return p_->prepare(active, options);
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

std::size_t gpu_block_frontend::resident_region_capacity() const {
    try {
        p_->require_current_device();
        return static_cast<std::size_t>(p_->block_region_limit);
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

std::span<const node_index> gpu_block_frontend::host_candidates() const {
    try {
        p_->require_current_device();
        return p_->download_host_candidates();
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

std::span<const node_index> gpu_block_frontend::host_active_degrees() const {
    try {
        p_->require_current_device();
        return p_->download_host_active_degrees();
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

const partition_result &gpu_block_frontend::select_block_greedy() {
    try {
        p_->require_current_device();
        p_->invalidate_selection();
        const auto& selected = p_->select_block_greedy();
        if (!selected.data.empty()) p_->publish_selection();
        return selected;
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

std::size_t gpu_block_frontend::selected_degree_work() const {
    try {
        p_->require_current_device();
        return p_->selected_degree_work;
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

gpu_device_selection gpu_block_frontend::device_selection() const {
    try {
        p_->require_current_device();
        return gpu_device_selection::issue(p_->selection_producer);
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
void gpu_block_frontend::inject_device_selection_fault_for_test(
        std::span<const node_index> replacement) {
    try {
        p_->require_current_device();
        p_->inject_device_selection_fault_for_test(replacement);
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}
#endif

void gpu_block_frontend::advance(std::span<const node_index> eliminated,
                                std::span<const gpu_topology_edge> new_edges,
                                std::span<const gpu_topology_batch>
                                    new_edge_batches) {
    try {
        p_->require_current_device();
        p_->invalidate_selection();
        p_->begin_topology_advance();
        p_->advance(eliminated, new_edges, new_edge_batches);
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

gpu_device_selection gpu_block_frontend::selection_for_residual_handoff() const {
    try {
        p_->require_current_device(/*read_residual=*/false);
        return gpu_device_selection::issue(p_->selection_producer);
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

void gpu_block_frontend::bind_owned_residual(
        const gpu_round_shadow_incidence* incidences, const std::uint32_t* offsets,
        const std::uint8_t* active, std::size_t directed_count,
        const gpu_device_selection_content& expected,
        std::weak_ptr<const void> epoch) {
    try {
        p_->require_current_device(/*read_residual=*/false);
        p_->invalidate_selection();
        p_->begin_topology_advance();
        p_->bind_owned_residual(incidences, offsets, active, directed_count,
                                expected, std::move(epoch));
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

void gpu_block_frontend::advance_resident(
        const gpu_round_shadow_incidence* incidences, std::size_t directed_count,
        const std::uint8_t* active,
        const gpu_device_selection_content& expected) {
    try {
        p_->require_current_device();
        p_->invalidate_selection();
        p_->begin_topology_advance();
        p_->advance_resident(incidences, directed_count, active, expected);
    } catch (...) {
        p_->poison_selection_producer();
        throw;
    }
}

gpu_block_frontend::transfer_stats gpu_block_frontend::transfers() const {
    p_->require_current_device(/*read_residual=*/false);
    return p_->transfers;
}

} // namespace apxchol::detail

#include "apxchol/solver/gpu_priority_frontend.h"

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace apxchol::detail {
namespace {

using clock_type = std::chrono::steady_clock;

static_assert(std::is_standard_layout_v<deferred_edge>);
static_assert(offsetof(deferred_edge, u) == 0);
static_assert(offsetof(deferred_edge, v) == sizeof(node_index));
static_assert(offsetof(deferred_edge, w) == sizeof(gpu_topology_edge));

[[noreturn]] void cuda_failure(cudaError_t err, const char *what) {
    throw std::runtime_error(std::string("GPU priority-greedy front-end: ") + what + ": " +
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

constexpr int kBlock = 256;
constexpr std::size_t kTopologyStageEdges = std::size_t{1} << 20;

int blocks_for(std::size_t n) {
    const std::size_t blocks = (n + kBlock - 1) / kBlock;
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("GPU priority-greedy front-end: CUDA grid is too large");
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

__global__ void count_degrees(const gpu_topology_edge *edges, std::size_t m,
                              edge_index *degrees) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= m)
        return;
    const auto e = edges[i];
    atomic_increment(&degrees[e.u]);
    atomic_increment(&degrees[e.v]);
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

__global__ void gather_active_degrees(const node_index *active_ids,
                                      std::size_t count,
                                      const edge_index *degrees,
                                      node_index *output) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        output[i] = static_cast<node_index>(degrees[active_ids[i]]);
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

__global__ void sum_vertex_degrees(const node_index *ids, std::size_t count,
                                   const edge_index *degrees,
                                   unsigned long long *sum) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        atomicAdd(sum, static_cast<unsigned long long>(
                           static_cast<node_index>(degrees[ids[i]])));
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

__device__ bool priority_precedes(node_index a, node_index degree_a,
                                  node_index b, node_index degree_b,
                                  std::uint64_t round_seed,
                                  bool degree_tiebreak) {
    if (degree_tiebreak && degree_a != degree_b)
        return degree_a < degree_b;
    const std::uint64_t ha =
        (std::uint64_t(a) ^ round_seed) * 11400714819323198485ULL;
    const std::uint64_t hb =
        (std::uint64_t(b) ^ round_seed) * 11400714819323198485ULL;
    return ha != hb ? ha < hb : a < b;
}

__global__ void
priority_greedy_cooperative(const node_index *candidates, std::size_t count,
                 const edge_index *row_offsets, const node_index *neighbors,
                 const edge_index *degrees, const unsigned char *active,
                 int *status, unsigned char *pick, std::uint64_t round_seed,
                 bool degree_tiebreak, int *has_undecided) {
    namespace cg = cooperative_groups;
    const cg::grid_group grid = cg::this_grid();
    const std::size_t first =
        blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    const std::size_t stride = gridDim.x * std::size_t(blockDim.x);

    for (;;) {
        for (std::size_t i = first; i < count; i += stride) {
            const node_index v = candidates[i];
            if (status[v] != 0) {
                pick[v] = 0;
                continue;
            }
            bool local_minimum = true;
            for (edge_index p = row_offsets[v]; p < row_offsets[v + 1]; ++p) {
                const node_index u = neighbors[p];
                if (active[u] && status[u] == 0 &&
                    priority_precedes(u, static_cast<node_index>(degrees[u]), v,
                                      static_cast<node_index>(degrees[v]), round_seed,
                                      degree_tiebreak)) {
                    local_minimum = false;
                    break;
                }
            }
            pick[v] = static_cast<unsigned char>(local_minimum);
        }
        grid.sync();

        for (std::size_t i = first; i < count; i += stride) {
            const node_index v = candidates[i];
            if (status[v] != 0 || !pick[v])
                continue;
            status[v] = 1;
            for (edge_index p = row_offsets[v]; p < row_offsets[v + 1]; ++p) {
                const node_index u = neighbors[p];
                if (active[u])
                    atomicCAS(&status[u], 0, 2);
            }
        }
        grid.sync();

        if (grid.thread_rank() == 0)
            *has_undecided = 0;
        grid.sync();
        for (std::size_t i = first; i < count; i += stride) {
            if (status[candidates[i]] == 0)
                atomicExch(has_undecided, 1);
        }
        grid.sync();
        if (*has_undecided == 0)
            break;
    }
}

__device__ bool block_priority_precedes(node_index a, node_index b,
                                        const edge_index *degrees,
                                        bool degree_tiebreak) {
    if (!degree_tiebreak)
        return a < b;
    const node_index da = static_cast<node_index>(degrees[a]);
    const node_index db = static_cast<node_index>(degrees[b]);
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

__global__ void block_greedy_regions(
    const node_index *candidates, std::size_t count,
    std::size_t region_count, const edge_index *row_offsets,
    const node_index *neighbors, const node_index *region_of, int *status) {
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
        for (edge_index p = row_offsets[v] + lane;
             p < row_offsets[v + 1]; p += 32) {
            const node_index u = neighbors[p];
            blocked |= status[u] == 1 && region_of[u] == warp;
        }
        const bool any = __any_sync(kFullWarp, blocked);
        if (lane == 0 && !any)
            status[v] = 1;
        __syncwarp(kFullWarp);
    }
}

__global__ void decide_block_conflicts(
    const node_index *candidates, std::size_t count,
    const edge_index *row_offsets, const node_index *neighbors,
    const edge_index *degrees, const node_index *region_of, int *status,
    unsigned char *drop, bool degree_tiebreak) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count)
        return;
    const node_index v = candidates[i];
    bool loses = false;
    if (status[v] == 1) {
        const node_index region = region_of[v];
        for (edge_index p = row_offsets[v]; p < row_offsets[v + 1]; ++p) {
            const node_index u = neighbors[p];
            if (status[u] == 1 && region_of[u] != region &&
                block_priority_precedes(u, v, degrees, degree_tiebreak)) {
                loses = true;
                break;
            }
        }
    }
    drop[v] = static_cast<unsigned char>(loses);
}

__global__ void apply_block_drops(const node_index *candidates,
                                  std::size_t count, int *status,
                                  const unsigned char *drop) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count && drop[candidates[i]])
        status[candidates[i]] = 0;
}

__global__ void mark_block_repair_frontier(
    const node_index *candidates, std::size_t count,
    const edge_index *row_offsets, const node_index *neighbors, int *status,
    int *frontier, unsigned char *drop) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count)
        return;
    const node_index v = candidates[i];
    if (!drop[v])
        return;
    frontier[v] = 1;
    for (edge_index p = row_offsets[v]; p < row_offsets[v + 1]; ++p) {
        const node_index u = neighbors[p];
        if (status[u] != 2)
            atomicExch(&frontier[u], 1);
    }
    drop[v] = 0;
}

__global__ void block_greedy_repair(
    const node_index *candidates, std::size_t count,
    const edge_index *row_offsets, const node_index *neighbors,
    const edge_index *degrees, int *status, int *frontier,
    unsigned char *winner, bool degree_tiebreak, int *has_pending) {
    namespace cg = cooperative_groups;
    const cg::grid_group grid = cg::this_grid();
    const std::size_t first =
        blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    const std::size_t stride = gridDim.x * std::size_t(blockDim.x);

    for (;;) {
        if (grid.thread_rank() == 0)
            *has_pending = 0;
        grid.sync();

        // A frontier vertex blocked by a committed pick can never become free:
        // the repair only adds picks. Free vertices become pending (2).
        for (std::size_t i = first; i < count; i += stride) {
            const node_index v = candidates[i];
            if (frontier[v] == 0)
                continue;
            if (status[v] == 1) {
                frontier[v] = 0;
                continue;
            }
            bool free = true;
            for (edge_index p = row_offsets[v]; p < row_offsets[v + 1]; ++p) {
                if (status[neighbors[p]] == 1) {
                    free = false;
                    break;
                }
            }
            if (free) {
                frontier[v] = 2;
                atomicExch(has_pending, 1);
            } else {
                frontier[v] = 0;
            }
        }
        grid.sync();
        if (*has_pending == 0)
            break;

        // Pick all minima of the pending frontier under the same strict order
        // as the CPU repair. Decisions read a barrier-separated snapshot.
        for (std::size_t i = first; i < count; i += stride) {
            const node_index v = candidates[i];
            if (frontier[v] != 2)
                continue;
            bool wins = true;
            for (edge_index p = row_offsets[v]; p < row_offsets[v + 1]; ++p) {
                const node_index u = neighbors[p];
                if (frontier[u] == 2 &&
                    block_priority_precedes(u, v, degrees,
                                            degree_tiebreak)) {
                    wins = false;
                    break;
                }
            }
            winner[v] = static_cast<unsigned char>(wins);
        }
        grid.sync();

        for (std::size_t i = first; i < count; i += stride) {
            const node_index v = candidates[i];
            if (frontier[v] != 2)
                continue;
            if (winner[v]) {
                status[v] = 1;
                frontier[v] = 0;
            } else {
                frontier[v] = 1;
            }
            winner[v] = 0;
        }
        grid.sync();
    }
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

struct degree_is_eligible {
    const edge_index *degrees;
    double threshold;
    __host__ __device__ bool operator()(node_index v) const {
        return static_cast<double>(static_cast<node_index>(degrees[v])) <=
               threshold;
    }
};

struct edge_is_live {
    const unsigned char *active;
    __host__ __device__ bool operator()(gpu_topology_edge e) const {
        return active[e.u] && active[e.v];
    }
};

double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start)
        .count();
}

bool trace_enabled() {
    const char *e = std::getenv("APXCHOL_GPU_PRIORITY_TRACE");
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

struct gpu_priority_frontend::impl {
    explicit impl(node_index n_in,
                  std::span<const gpu_topology_edge> initial_edges)
        : n(n_in), live_edge_count(initial_edges.size()), active_count(n_in) {
        if (static_cast<std::uint64_t>(n) + 1 >
            static_cast<std::uint64_t>(INT_MAX))
            throw std::overflow_error(
                "GPU priority-greedy front-end currently requires n < INT_MAX");
        if (initial_edges.size() > static_cast<std::size_t>(INT_MAX))
            throw std::overflow_error(
                "GPU priority-greedy front-end currently requires live edges < INT_MAX");

        int device = 0;
        int cooperative = 0;
        cuda_check(cudaGetDevice(&device), "query active CUDA device");
        cuda_check(cudaDeviceGetAttribute(&cooperative,
                                          cudaDevAttrCooperativeLaunch, device),
                   "query cooperative-launch support");
        if (!cooperative)
            throw std::runtime_error("GPU priority-greedy front-end requires "
                                     "cooperative-kernel launch support");
        int blocks_per_sm = 0;
        int block_repair_blocks_per_sm = 0;
        int block_scan_blocks_per_sm = 0;
        int sms = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &blocks_per_sm, priority_greedy_cooperative, kBlock, 0),
                   "query cooperative priority-greedy occupancy");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &block_repair_blocks_per_sm, block_greedy_repair,
                       kBlock, 0),
                   "query cooperative block-greedy repair occupancy");
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &block_scan_blocks_per_sm, block_greedy_regions,
                       kBlock, 0),
                   "query block-greedy region-scan occupancy");
        cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                          device),
                   "query CUDA multiprocessor count");
        cooperative_grid_limit = blocks_per_sm * sms;
        block_repair_grid_limit = block_repair_blocks_per_sm * sms;
        block_region_limit = block_scan_blocks_per_sm * sms * (kBlock / 32);
        if (cooperative_grid_limit <= 0)
            throw std::runtime_error(
                "GPU priority-greedy front-end found no cooperative-kernel residency");
        if (block_repair_grid_limit <= 0)
            throw std::runtime_error(
                "GPU block-greedy front-end found no cooperative-kernel residency");
        if (block_region_limit <= 0)
            throw std::runtime_error(
                "GPU block-greedy front-end found no region-scan residency");

        coo[0].reserve(initial_edges.size());
        coo[1].reserve(initial_edges.size());
        if (!initial_edges.empty()) {
            cuda_check(
                cudaMemcpy(coo[0].get(), initial_edges.data(),
                           initial_edges.size() * sizeof(gpu_topology_edge),
                           cudaMemcpyHostToDevice),
                "upload initial topology");
        }

        active_mask.reserve(n);
        status.reserve(n);
        pick.reserve(n);
        degrees.reserve(n + 1);
        row_offsets.reserve(n + 1);
        row_cursor.reserve(n);
        csr_neighbors.reserve(initial_edges.size() * 2);
        active_ids[0].reserve(n);
        active_ids[1].reserve(n);
        active_degrees.reserve(n);
        sorted_degrees.reserve(n);
        candidates_original.reserve(n);
        selected_ids.reserve(n);
        update_ids.reserve(n);
        topology_staging.reserve(
            std::min(initial_edges.size(), kTopologyStageEdges));
        selected_count.reserve(1);
        degree_sum.reserve(1);
        cooperative_flag.reserve(1);
        cuda_check(cudaMemset(active_mask.get(), 1, n),
                   "initialize active mask");
        if (n) {
            initialize_status<<<blocks_for(n), kBlock>>>(n, status.get());
            cuda_check(cudaGetLastError(), "initialize priority-greedy status");
            initialize_vertex_ids<<<blocks_for(n), kBlock>>>(
                n, active_ids[current_active].get());
            cuda_check(cudaGetLastError(), "initialize active vertex ids");
        }

        // Reserve the largest CUB workspace while construction can still
        // cleanly fall back to the CPU. Counts only shrink after round zero.
        std::size_t cub_bytes = 0;
        std::size_t bytes = 0;
        cuda_check(cub::DeviceScan::ExclusiveSum(
                       nullptr, bytes, degrees.get(), row_offsets.get(),
                       static_cast<int>(std::size_t(n) + 1)),
                   "query initial CUB scan workspace");
        cub_bytes = std::max(cub_bytes, bytes);
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

    void require_cub_count(std::size_t count, const char *what) const {
        if (count > static_cast<std::size_t>(INT_MAX))
            throw std::overflow_error(std::string("GPU priority-greedy front-end: ") +
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

    template <class T, class Predicate>
    int select_if(const T *input, T *output, std::size_t count,
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
        if (!count)
            return 0;
        cuda_check(cudaMemset(degree_sum.get(), 0,
                              sizeof(unsigned long long)),
                   "clear degree sum");
        sum_degree_values<<<blocks_for(count), kBlock>>>(
            active_degrees.get(), count, degree_sum.get());
        cuda_check(cudaGetLastError(), "sum active degrees");
        unsigned long long result = 0;
        cuda_check(cudaMemcpy(&result, degree_sum.get(), sizeof(result),
                              cudaMemcpyDeviceToHost),
                   "download active-degree sum");
        return result;
    }

    unsigned long long sum_selected_degrees(std::size_t count) {
        if (!count)
            return 0;
        cuda_check(cudaMemset(degree_sum.get(), 0,
                              sizeof(unsigned long long)),
                   "clear selected-degree sum");
        sum_vertex_degrees<<<blocks_for(count), kBlock>>>(
            selected_ids.get(), count, degrees.get(), degree_sum.get());
        cuda_check(cudaGetLastError(), "sum selected degrees");
        unsigned long long result = 0;
        cuda_check(cudaMemcpy(&result, degree_sum.get(), sizeof(result),
                              cudaMemcpyDeviceToHost),
                   "download selected-degree sum");
        return result;
    }

    void rebuild_topology() {
        if (!topology_dirty)
            return;

        if (live_edge_count > std::numeric_limits<std::size_t>::max() / 2)
            throw std::overflow_error("GPU priority-greedy front-end CSR size overflow");
        const std::size_t directed = live_edge_count * 2;
        if (directed >
            static_cast<std::size_t>(std::numeric_limits<edge_index>::max()))
            throw std::overflow_error(
                "GPU priority-greedy front-end directed CSR exceeds edge_index range");

        cuda_check(cudaMemset(degrees.get(), 0,
                              (std::size_t(n) + 1) * sizeof(edge_index)),
                   "clear degree counts");
        if (live_edge_count) {
            count_degrees<<<blocks_for(live_edge_count), kBlock>>>(
                coo[current_coo].get(), live_edge_count, degrees.get());
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

    gpu_priority_frontend::prepare_result
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
                "GPU priority-greedy active-list size diverged from CPU state");
        host_active_degrees.clear();
        host_candidate_ids.clear();
        host_active_degrees_valid = false;
        host_candidate_ids_valid = false;

        if (!active.empty()) {
            gather_active_degrees<<<blocks_for(active.size()), kBlock>>>(
                active_ids[current_active].get(), active.size(), degrees.get(),
                active_degrees.get());
            cuda_check(cudaGetLastError(), "launch active-degree gather");
        }

        const auto total_degree = sum_active_degree_values(active.size());
        const double average =
            active.empty() ? 0.0
                           : static_cast<double>(total_degree) /
                                 static_cast<double>(active.size());

        double threshold = options.degree_multiplier * average;
        const double q = options.degree_quantile;
        if (q > 0.0 && q < 1.0 && !active.empty()) {
            std::size_t rank = static_cast<std::size_t>(q * active.size());
            if (rank >= active.size()) rank = active.size() - 1;
            sort_degrees(active.size());
            node_index quantile = 0;
            cuda_check(cudaMemcpy(&quantile, sorted_degrees.get() + rank,
                                  sizeof(quantile), cudaMemcpyDeviceToHost),
                       "download degree quantile");
            threshold = static_cast<double>(quantile);
        }

        candidate_count = static_cast<std::size_t>(select_if(
            active_ids[current_active].get(), candidates_original.get(),
            active.size(), degree_is_eligible{degrees.get(), threshold}));

        last_prepare = elapsed_ms(start);
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[gpu-priority] prepare active=%zu candidates=%zu "
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

    const partition_result &select(unsigned seed, std::uint64_t round) {
        const auto start = clock_type::now();
        result.data.clear();
        if (!candidate_count) {
            selected_degree_work = 0;
            last_select = elapsed_ms(start);
            return result;
        }

        set_status<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, status.get(), 0);
        cuda_check(cudaGetLastError(), "initialize candidate status");

        const std::uint64_t round_seed =
            std::uint64_t(seed) ^
            (round * 6364136223846793005ULL + 1442695040888963407ULL);

        const int wanted = blocks_for(candidate_count);
        const int grid =
            std::max(1, std::min(wanted, cooperative_grid_limit));
        const node_index *candidate_ptr = candidates_original.get();
        const edge_index *row_ptr = row_offsets.get();
        const node_index *neighbor_ptr = csr_neighbors.get();
        const edge_index *degree_ptr = degrees.get();
        const unsigned char *active_ptr = active_mask.get();
        int *status_ptr = status.get();
        std::uint64_t kernel_round_seed = round_seed;
        bool degree_tiebreak = current_options.degree_tiebreak;
        cooperative_flag.reserve(1);
        int *flag_ptr = cooperative_flag.get();
        unsigned char *pick_ptr = pick.get();
        void *args[] = {
            &candidate_ptr,   &candidate_count, &row_ptr,
            &neighbor_ptr,    &degree_ptr,      &active_ptr,
            &status_ptr,      &pick_ptr,        &kernel_round_seed,
            &degree_tiebreak, &flag_ptr};
        cuda_check(cudaLaunchCooperativeKernel(
                       reinterpret_cast<void *>(priority_greedy_cooperative),
                       grid, kBlock, args, 0, nullptr),
                   "launch cooperative priority-greedy passes");

        const std::size_t chosen = static_cast<std::size_t>(
            select_if(candidates_original.get(), selected_ids.get(),
                      candidate_count, status_equals{status.get(), 1}));
        selected_degree_work = sum_selected_degrees(chosen);
        result.data.resize(chosen);
        if (chosen) {
            cuda_check(cudaMemcpy(result.data.data(), selected_ids.get(),
                                  chosen * sizeof(node_index),
                                  cudaMemcpyDeviceToHost),
                       "download selected vertices");
        }
        set_status<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, status.get(), 2);
        cuda_check(cudaGetLastError(), "restore candidate status");
        cuda_check(cudaDeviceSynchronize(), "finish priority-greedy selection");

        last_select = elapsed_ms(start);
        if (trace_enabled()) {
            std::fprintf(stderr, "[gpu-priority] select chosen=%zu %.3f ms\n",
                         chosen, last_select);
        }
        return result;
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

    const partition_result &select_block_greedy() {
        const auto start = clock_type::now();
        result.data.clear();
        if (!candidate_count) {
            selected_degree_work = 0;
            last_select = elapsed_ms(start);
            return result;
        }

        block_region.reserve(n);
        block_frontier.reserve(n);
        const std::size_t regions = block_region_count();
        initialize_block_candidates<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, regions, status.get(),
            block_region.get(), block_frontier.get(), pick.get());
        cuda_check(cudaGetLastError(), "initialize block-greedy candidates");

        constexpr std::size_t warps_per_block = kBlock / 32;
        const std::size_t scan_blocks =
            (regions + warps_per_block - 1) / warps_per_block;
        block_greedy_regions<<<static_cast<int>(scan_blocks), kBlock>>>(
            candidates_original.get(), candidate_count, regions,
            row_offsets.get(), csr_neighbors.get(), block_region.get(),
            status.get());
        cuda_check(cudaGetLastError(), "scan block-greedy regions");

        decide_block_conflicts<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, row_offsets.get(),
            csr_neighbors.get(), degrees.get(), block_region.get(), status.get(),
            pick.get(), current_options.degree_tiebreak);
        cuda_check(cudaGetLastError(), "decide block-greedy conflicts");
        apply_block_drops<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, status.get(), pick.get());
        cuda_check(cudaGetLastError(), "apply block-greedy conflicts");
        mark_block_repair_frontier<<<blocks_for(candidate_count), kBlock>>>(
            candidates_original.get(), candidate_count, row_offsets.get(),
            csr_neighbors.get(), status.get(), block_frontier.get(), pick.get());
        cuda_check(cudaGetLastError(), "mark block-greedy repair frontier");

        const int wanted = blocks_for(candidate_count);
        const int grid =
            std::max(1, std::min(wanted, block_repair_grid_limit));
        const node_index *candidate_ptr = candidates_original.get();
        const edge_index *row_ptr = row_offsets.get();
        const node_index *neighbor_ptr = csr_neighbors.get();
        const edge_index *degree_ptr = degrees.get();
        int *status_ptr = status.get();
        int *frontier_ptr = block_frontier.get();
        unsigned char *winner_ptr = pick.get();
        bool degree_tiebreak = current_options.degree_tiebreak;
        int *pending_ptr = cooperative_flag.get();
        void *args[] = {
            &candidate_ptr, &candidate_count, &row_ptr,      &neighbor_ptr,
            &degree_ptr,    &status_ptr,      &frontier_ptr, &winner_ptr,
            &degree_tiebreak, &pending_ptr};
        cuda_check(cudaLaunchCooperativeKernel(
                       reinterpret_cast<void *>(block_greedy_repair), grid,
                       kBlock, args, 0, nullptr),
                   "launch cooperative block-greedy repair");
        const std::size_t chosen = static_cast<std::size_t>(
            select_if(candidates_original.get(), selected_ids.get(),
                      candidate_count, status_equals{status.get(), 1}));
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
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[gpu-block] select candidates=%zu regions=%zu "
                         "chosen=%zu %.3f ms\n",
                         candidate_count, regions, chosen, last_select);
        }
        return result;
    }

    void advance(std::span<const node_index> eliminated,
                 std::span<const gpu_topology_edge> new_edges,
                 std::span<const gpu_topology_batch> new_edge_batches) {
        const auto start = clock_type::now();

        update_ids.reserve(eliminated.size());
        if (!eliminated.empty()) {
            cuda_check(cudaMemcpy(update_ids.get(), eliminated.data(),
                                  eliminated.size() * sizeof(node_index),
                                  cudaMemcpyHostToDevice),
                       "upload eliminated vertices");
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
                    "GPU priority-greedy front-end topology batch size overflow");
            added += batch.size;
        }
        if (added > std::numeric_limits<std::size_t>::max() - kept)
            throw std::overflow_error(
                "GPU priority-greedy front-end live topology size overflow");
        const std::size_t total = kept + added;
        if (total > static_cast<std::size_t>(INT_MAX))
            throw std::overflow_error(
                "GPU priority-greedy front-end live topology exceeds INT_MAX edges");
        // Eliminating an independent set removes sum(deg(v)) old edges and
        // emits at most sum(deg(v)-1) sampled edges. Isolated vertices emit
        // nothing, so multiplicity-aware topology never grows. The old COO
        // capacity is therefore sufficient for compacted + appended edges.
        if (total > old_count)
            throw std::logic_error("GPU priority-greedy front-end topology "
                                   "unexpectedly grew after elimination");

        std::size_t offset = kept;
        if (!new_edges.empty()) {
            cuda_check(cudaMemcpy(coo[next].get() + offset, new_edges.data(),
                                  new_edges.size() * sizeof(gpu_topology_edge),
                                  cudaMemcpyHostToDevice),
                       "append contiguous sampled topology edges");
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
                stderr, "[gpu-priority] advance eliminated=%zu added=%zu %.3f ms\n",
                eliminated.size(), added, last_advance);
        }
    }

    node_index n;
    device_buffer<gpu_topology_edge> coo[2];
    int current_coo = 0;
    std::size_t live_edge_count = 0;

    device_buffer<unsigned char> active_mask;
    device_buffer<edge_index> degrees;
    device_buffer<edge_index> row_offsets;
    device_buffer<edge_index> row_cursor;
    device_buffer<node_index> csr_neighbors;

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
    device_buffer<unsigned long long> degree_sum;
    device_buffer<int> cooperative_flag;
    device_buffer<unsigned char> cub_temp;

    std::vector<node_index> host_active_degrees;
    std::vector<node_index> host_candidate_ids;
    mutable bool host_active_degrees_valid = false;
    mutable bool host_candidate_ids_valid = false;
    partition_result result;
    partition_options current_options;
    std::size_t candidate_count = 0;
    std::size_t selected_degree_work = 0;
    int cooperative_grid_limit = 0;
    int block_repair_grid_limit = 0;
    int block_region_limit = 0;
    bool topology_dirty = true;

    double last_prepare = 0.0;
    double last_select = 0.0;
    double last_advance = 0.0;
};

gpu_priority_frontend::mode gpu_priority_frontend::configured_mode() {
    const char *e = std::getenv("APXCHOL_GPU_PRIORITY_FRONTEND");
    if (!e || !*e || std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0 ||
        std::strcmp(e, "false") == 0)
        return mode::disabled;
    if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0 ||
        std::strcmp(e, "force") == 0)
        return mode::forced;
    static std::atomic_flag warned = ATOMIC_FLAG_INIT;
    if (!warned.test_and_set())
        std::fprintf(
            stderr,
            "[apxchol] unknown APXCHOL_GPU_PRIORITY_FRONTEND='%s'; expected "
            "0|off|1|on|force, disabling the optional GPU setup front-end\n",
            e);
    return mode::disabled;
}

gpu_priority_frontend::mode gpu_priority_frontend::configured_block_mode() {
    const char *e = std::getenv("APXCHOL_GPU_BLOCK_FRONTEND");
    if (!e || !*e || std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0 ||
        std::strcmp(e, "false") == 0)
        return mode::disabled;
    if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0 ||
        std::strcmp(e, "force") == 0)
        return mode::forced;
    static std::atomic_flag warned = ATOMIC_FLAG_INIT;
    if (!warned.test_and_set())
        std::fprintf(
            stderr,
            "[apxchol] unknown APXCHOL_GPU_BLOCK_FRONTEND='%s'; expected "
            "0|off|1|on|force, disabling the GPU block-greedy prototype\n",
            e);
    return mode::disabled;
}

gpu_priority_frontend::runtime_probe
gpu_priority_frontend::probe_runtime(node_index n, std::size_t initial_edges,
                                     bool block_selector) {
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
    if (block_selector) {
        valid &= add_allocation(bytes, nv, sizeof(node_index));
        valid &= add_allocation(bytes, nv, sizeof(int));
    }
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

gpu_priority_frontend::gpu_priority_frontend(
    node_index n, std::span<const gpu_topology_edge> initial_edges)
    : p_(std::make_unique<impl>(n, initial_edges)) {}

gpu_priority_frontend::~gpu_priority_frontend() = default;
gpu_priority_frontend::gpu_priority_frontend(gpu_priority_frontend &&) noexcept = default;
gpu_priority_frontend &
gpu_priority_frontend::operator=(gpu_priority_frontend &&) noexcept = default;

gpu_priority_frontend::prepare_result
gpu_priority_frontend::prepare(std::span<const node_index> active,
                           const partition_options &options) {
    p_->current_options = options;
    return p_->prepare(active, options);
}

std::span<const node_index> gpu_priority_frontend::host_candidates() const {
    return p_->download_host_candidates();
}

std::span<const node_index> gpu_priority_frontend::host_active_degrees() const {
    return p_->download_host_active_degrees();
}

const partition_result &gpu_priority_frontend::select(unsigned seed,
                                                  std::uint64_t round) {
    return p_->select(seed, round);
}

const partition_result &gpu_priority_frontend::select_block_greedy() {
    return p_->select_block_greedy();
}

std::size_t gpu_priority_frontend::selected_degree_work() const {
    return p_->selected_degree_work;
}

void gpu_priority_frontend::advance(std::span<const node_index> eliminated,
                                std::span<const gpu_topology_edge> new_edges,
                                std::span<const gpu_topology_batch> new_edge_batches) {
    p_->advance(eliminated, new_edges, new_edge_batches);
}

} // namespace apxchol::detail

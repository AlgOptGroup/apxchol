#include "apxchol/solver/gpu_priority_frontend.h"

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace apxchol::detail {
namespace {

using clock_type = std::chrono::steady_clock;

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

__device__ bool priority_precedes(node_index a, edge_index degree_a,
                                  node_index b, edge_index degree_b,
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
                    priority_precedes(u, degrees[u], v, degrees[v], round_seed,
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

struct status_equals {
    const int *status;
    int value;
    __host__ __device__ bool operator()(node_index v) const {
        return status[v] == value;
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
        : n(n_in), live_edge_count(initial_edges.size()) {
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
        int sms = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &blocks_per_sm, priority_greedy_cooperative, kBlock, 0),
                   "query cooperative priority-greedy occupancy");
        cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                          device),
                   "query CUDA multiprocessor count");
        cooperative_grid_limit = blocks_per_sm * sms;
        if (cooperative_grid_limit <= 0)
            throw std::runtime_error(
                "GPU priority-greedy front-end found no cooperative-kernel residency");

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
        active_ids.reserve(n);
        active_degrees.reserve(n);
        candidates_original.reserve(n);
        selected_ids.reserve(n);
        update_ids.reserve(n);
        selected_count.reserve(1);
        cooperative_flag.reserve(1);
        cuda_check(cudaMemset(active_mask.get(), 1, n),
                   "initialize active mask");
        if (n) {
            initialize_status<<<blocks_for(n), kBlock>>>(n, status.get());
            cuda_check(cudaGetLastError(), "initialize priority-greedy status");
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

    void rebuild_topology() {
        if (!topology_dirty)
            return;

        if (prepared_once) {
            require_cub_count(live_edge_count, "edge compaction");
            const int next = 1 - current_coo;
            const std::size_t old_count = live_edge_count;
            const std::size_t kept =
                live_edge_count
                    ? static_cast<std::size_t>(select_if(
                          coo[current_coo].get(), coo[next].get(),
                          live_edge_count, edge_is_live{active_mask.get()}))
                    : 0;
            const std::size_t total = kept + pending_edges.size();
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
            if (!pending_edges.empty()) {
                cuda_check(
                    cudaMemcpy(coo[next].get() + kept, pending_edges.data(),
                               pending_edges.size() * sizeof(gpu_topology_edge),
                               cudaMemcpyHostToDevice),
                    "append sampled topology edges");
            }
            current_coo = next;
            live_edge_count = total;
            pending_edges.clear();
        }

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
        prepared_once = true;
    }

    gpu_priority_frontend::prepare_result
    prepare(std::span<const node_index> active,
            const partition_options &options) {
        const auto start = clock_type::now();
        rebuild_topology();

        active_ids.reserve(active.size());
        active_degrees.reserve(active.size());
        candidates_original.reserve(active.size());
        selected_ids.reserve(active.size());
        host_active_degrees.resize(active.size());
        host_candidate_ids.clear();

        if (!active.empty()) {
            cuda_check(cudaMemcpy(active_ids.get(), active.data(),
                                  active.size() * sizeof(node_index),
                                  cudaMemcpyHostToDevice),
                       "upload active vertex list");
            gather_active_degrees<<<blocks_for(active.size()), kBlock>>>(
                active_ids.get(), active.size(), degrees.get(),
                active_degrees.get());
            cuda_check(cudaGetLastError(), "launch active-degree gather");
            cuda_check(cudaMemcpy(host_active_degrees.data(),
                                  active_degrees.get(),
                                  active.size() * sizeof(node_index),
                                  cudaMemcpyDeviceToHost),
                       "download active degrees");
        }

        double total_degree = 0.0;
        for (node_index d : host_active_degrees)
            total_degree += static_cast<double>(d);
        const double average =
            active.empty() ? 0.0
                           : total_degree / static_cast<double>(active.size());

        double threshold = options.degree_multiplier * average;
        const double q = options.degree_quantile;
        if (q > 0.0 && q < 1.0 && !active.empty()) {
            quantile_scratch = host_active_degrees;
            std::size_t rank = static_cast<std::size_t>(q * active.size());
            if (rank >= quantile_scratch.size())
                rank = quantile_scratch.size() - 1;
            std::nth_element(quantile_scratch.begin(),
                             quantile_scratch.begin() + rank,
                             quantile_scratch.end());
            threshold = static_cast<double>(quantile_scratch[rank]);
        }

        host_candidate_ids.reserve(active.size());
        for (std::size_t i = 0; i < active.size(); ++i)
            if (static_cast<double>(host_active_degrees[i]) <= threshold)
                host_candidate_ids.push_back(active[i]);
        candidate_count = host_candidate_ids.size();

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

    const partition_result &select(unsigned seed, std::uint64_t round) {
        const auto start = clock_type::now();
        result.data.clear();
        if (!candidate_count) {
            last_select = elapsed_ms(start);
            return result;
        }

        cuda_check(cudaMemcpy(candidates_original.get(),
                              host_candidate_ids.data(),
                              candidate_count * sizeof(node_index),
                              cudaMemcpyHostToDevice),
                   "upload degree-eligible candidates");
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
        unsigned char *pick_ptr = pick.get();
        std::uint64_t kernel_round_seed = round_seed;
        bool degree_tiebreak = current_options.degree_tiebreak;
        cooperative_flag.reserve(1);
        int *flag_ptr = cooperative_flag.get();
        void *args[] = {
            &candidate_ptr,   &candidate_count, &row_ptr,
            &neighbor_ptr,    &degree_ptr,      &active_ptr,
            &status_ptr,      &pick_ptr,        &kernel_round_seed,
            &degree_tiebreak, &flag_ptr};
        cuda_check(cudaLaunchCooperativeKernel(
                       reinterpret_cast<void *>(priority_greedy_cooperative), grid,
                       kBlock, args, 0, nullptr),
                   "launch cooperative priority-greedy passes");

        const std::size_t chosen = static_cast<std::size_t>(
            select_if(candidates_original.get(), selected_ids.get(),
                      candidate_count, status_equals{status.get(), 1}));
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

    void advance(std::span<const node_index> eliminated,
                 std::span<const gpu_topology_edge> new_edges) {
        const auto start = clock_type::now();
        if (!pending_edges.empty())
            throw std::logic_error(
                "GPU priority-greedy front-end advance called before pending "
                "edges were consumed");

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

        pending_edges.assign(new_edges.begin(), new_edges.end());
        topology_dirty = true;
        cuda_check(cudaDeviceSynchronize(), "finish topology advance");
        last_advance = elapsed_ms(start);
        if (trace_enabled()) {
            std::fprintf(
                stderr, "[gpu-priority] advance eliminated=%zu added=%zu %.3f ms\n",
                eliminated.size(), new_edges.size(), last_advance);
        }
    }

    node_index n;
    device_buffer<gpu_topology_edge> coo[2];
    int current_coo = 0;
    std::size_t live_edge_count = 0;
    std::vector<gpu_topology_edge> pending_edges;

    device_buffer<unsigned char> active_mask;
    device_buffer<edge_index> degrees;
    device_buffer<edge_index> row_offsets;
    device_buffer<edge_index> row_cursor;
    device_buffer<node_index> csr_neighbors;

    device_buffer<node_index> active_ids;
    device_buffer<node_index> active_degrees;
    device_buffer<node_index> candidates_original;
    device_buffer<node_index> selected_ids;
    device_buffer<node_index> update_ids;
    device_buffer<int> status;
    device_buffer<unsigned char> pick;
    device_buffer<int> selected_count;
    device_buffer<int> cooperative_flag;
    device_buffer<unsigned char> cub_temp;

    std::vector<node_index> host_active_degrees;
    std::vector<node_index> host_candidate_ids;
    std::vector<node_index> quantile_scratch;
    partition_result result;
    partition_options current_options;
    std::size_t candidate_count = 0;
    int cooperative_grid_limit = 0;
    bool topology_dirty = true;
    bool prepared_once = false;

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

gpu_priority_frontend::runtime_probe
gpu_priority_frontend::probe_runtime(node_index n, std::size_t initial_edges) {
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
    if (nv > std::numeric_limits<std::size_t>::max() / 5) {
        valid = false;
    } else {
        valid &= add_allocation(bytes, nv * 5, sizeof(node_index));
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
    return p_->host_candidate_ids;
}

std::span<const node_index> gpu_priority_frontend::host_active_degrees() const {
    return p_->host_active_degrees;
}

const partition_result &gpu_priority_frontend::select(unsigned seed,
                                                  std::uint64_t round) {
    return p_->select(seed, round);
}

void gpu_priority_frontend::advance(std::span<const node_index> eliminated,
                                std::span<const gpu_topology_edge> new_edges) {
    p_->advance(eliminated, new_edges);
}

} // namespace apxchol::detail

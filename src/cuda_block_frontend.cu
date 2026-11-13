#include "cuda_buffer.cuh"
#include "cuda_digest.cuh"
#include "cuda_small_scan.cuh"
#include "apxchol/solver/gpu_block_frontend.h"
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/solver/detail/cuda_device_scope.h"
#include "apxchol/solver/detail/cuda_error.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
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
#include <utility>
#include <vector>

namespace apxchol::detail {
namespace {

using clock_type = gpu_setup_diagnostic_clock;

void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess)
        check_cuda(err, (std::string("GPU block front-end: ") + what).c_str());
}

template<class T> using device_buffer = cuda_buffer<T, cuda_check>;

constexpr int kBlock = 256;

int blocks_for(std::size_t n) {
    const std::size_t blocks = (n + kBlock - 1) / kBlock;
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("GPU block front-end: CUDA grid is too large");
    return static_cast<int>(blocks);
}

struct device_selection_digest {
    unsigned long long xor_hash;
    unsigned long long sum_hash;
};

struct device_selected_status {
    unsigned long long degree_sum;
    device_selection_digest selection;
};
static_assert(sizeof(device_selected_status) == 24);

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

__global__ void gather_active_degrees(const node_index* active_ids,
                                      std::size_t count,
                                      selector_owned_view view,
                                      node_index* output) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        output[i] = view.degree(active_ids[i]);
}

__global__ void initialize_vertex_ids(std::size_t count, node_index *ids) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        ids[i] = static_cast<node_index>(i);
}

// count_on_device, when given, bounds the ids below bound; zero-valued
// lanes past the count leave the xor/sum digest and the degree sum unchanged.
// With restore_ids, it also sets restore_status[restore_ids[i]] = 2 for
// i < bound (the candidates leave the selection; the selection compaction,
// the last status reader, ran before).
__global__ void sum_vertex_degrees(const node_index *ids, std::size_t bound,
                                   const int *count_on_device,
                                   selector_owned_view view,
                                   device_selected_status *status,
                                   const node_index* restore_ids = nullptr,
                                   int* restore_status = nullptr) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (restore_ids && i < bound) restore_status[restore_ids[i]] = 2;
    const std::size_t count = count_on_device
        ? static_cast<std::size_t>(*count_on_device) : bound;
    unsigned long long item_hash = 0;
    unsigned long long degree_sum = 0;
    if (i < count && i < bound) {
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
    __shared__ device_selection_digest warp_values[kBlock / 32];
    digest_add_block<kBlock>({item_hash, item_hash}, &status->selection, warp_values);
    if (warp == 0) {
        degree_sum = lane < kBlock / 32 ? warp_degrees[lane] : 0;
        for (unsigned offset = 16; offset; offset >>= 1)
            degree_sum += __shfl_down_sync(0xffffffffU, degree_sum, offset);
        if (lane == 0)
            atomicAdd(&status->degree_sum, degree_sum);
    }
}

__global__ void initialize_status(std::size_t count, int *status) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count)
        status[i] = 2;
}

// Warp callers must supply the same vertex on all 32 participating lanes.
template<class Predicate>
__device__ bool any_neighbor(node_index v, selector_owned_view view, Predicate predicate) {
    const std::size_t end = view.end(v);
    const unsigned lane = threadIdx.x & 31U;
    for (std::size_t base = view.begin(v); base < end; base += 32) {
        const std::size_t at = base + lane;
        const bool found = at < end && predicate(view.neighbor(static_cast<edge_index>(at)));
        if (__any_sync(0xffffffffU, found))
            return true;
    }
    return false;
}

// Keep one strict (degree, phase/seed hash, vertex) order through at most
// four snapshot passes. The selected set is independent, not necessarily maximal.
__device__ bool bounded_priority_precedes(node_index a, node_index b,
        selector_owned_view view,
        std::uint64_t seed, std::uint64_t phase) {
    if (view.degree(a) != view.degree(b))
        return view.degree(a) < view.degree(b);
    const std::uint64_t key = seed ^ (phase * 0xD6E8FEB86659FD93ULL);
    const auto ha = gpu_device_selection_mix(key ^ a);
    const auto hb = gpu_device_selection_mix(key ^ b);
    return ha != hb ? ha < hb : a < b;
}

__device__ void bounded_decide_one(const node_index* candidates, std::size_t index,
        selector_owned_view view, const int* status, unsigned char* winners,
        std::uint64_t seed, std::uint64_t phase) {
    const unsigned lane = threadIdx.x & 31U;
    const node_index v = candidates[index];
    // Decided candidates sit out the pass; the warp reads one status word.
    if (status[v] != 0) {
        if (lane == 0) winners[index] = 0;
        return;
    }
    const bool wins = !any_neighbor(v, view, [&](node_index u) {
        return status[u] == 0 &&
            bounded_priority_precedes(u, v, view, seed, phase);
    });
    if (lane == 0) winners[index] = wins;
}

__device__ void bounded_commit_one(const node_index* candidates, std::size_t index,
        selector_owned_view view, const unsigned char* winners, int* status) {
    const unsigned lane = threadIdx.x & 31U;
    if (!winners[index]) return; // warp-uniform
    const node_index v = candidates[index];
    if (lane == 0) atomicCAS(status + v, 0, 1);
    // Winners are independent in the preceding immutable snapshot. Multiple
    // winners/parallel incidences may mark the same neighbor; the mark is idempotent.
    for (std::size_t at = std::size_t(view.begin(v)) + lane;
         at < std::size_t(view.end(v)); at += 32)
        atomicCAS(status + view.neighbor(static_cast<edge_index>(at)), 0, 2);
}

__global__ void bounded_decide(const node_index* candidates, std::size_t count,
        selector_owned_view view, const int* status, unsigned char* winners,
        std::uint64_t seed, std::uint64_t phase) {
    const std::size_t index =
        (blockIdx.x * std::size_t(blockDim.x) + threadIdx.x) / 32;
    if (index >= count) return; // whole warp, including the final partial block
    bounded_decide_one(candidates, index, view, status, winners, seed, phase);
}

__global__ void bounded_commit(const node_index* candidates, std::size_t count,
        selector_owned_view view, const unsigned char* winners, int* status) {
    const std::size_t index =
        (blockIdx.x * std::size_t(blockDim.x) + threadIdx.x) / 32;
    if (index >= count) return; // warp-uniform
    bounded_commit_one(candidates, index, view, winners, status);
}

// Selection start: candidates become undecided (status 0); the selected ids
// (copied up to the candidate bound, so cleared for initcheck) and the
// degree/digest status are cleared.
__global__ void begin_bounded_selection(const node_index* candidates, std::size_t count,
        int* status, node_index* selected_ids, device_selected_status* selected_status) {
    const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i == 0) *selected_status = {};
    if (i < count) {
        status[candidates[i]] = 0;
        selected_ids[i] = 0;
    }
}

constexpr unsigned kBoundedBlock = 1024;
// Candidate lists up to this size run all passes in one block.
constexpr std::size_t kBoundedOneBlockLimit = 1024;

// begin_bounded_selection and the four decide/commit passes in one block.
// __syncthreads separates the phases as the kernel boundaries do: every
// decide reads the status left by the previous commit, and no commit starts
// before every decide of its pass has finished.
__global__ void __launch_bounds__(kBoundedBlock)
bounded_selection_one_block(const node_index* candidates, std::size_t count,
        selector_owned_view view, int* status, unsigned char* winners,
        node_index* selected_ids, device_selected_status* selected_status,
        std::uint64_t seed, std::uint64_t phase, unsigned passes) {
    if (threadIdx.x == 0) *selected_status = {};
    for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
        status[candidates[i]] = 0;
        selected_ids[i] = 0;
    }
    __syncthreads();
    const std::size_t warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (unsigned step = 0; step < passes; ++step) {
        for (std::size_t index = warp; index < count; index += warps)
            bounded_decide_one(candidates, index, view, status, winners, seed, phase);
        __syncthreads();
        for (std::size_t index = warp; index < count; index += warps)
            bounded_commit_one(candidates, index, view, winners, status);
        __syncthreads();
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

// With quantile set, the threshold is the device-resident sorted degree
// (the host path compared the same value widened to double).
struct degree_is_eligible {
    selector_owned_view view;
    double threshold;
    const node_index* quantile = nullptr;
    __host__ __device__ bool operator()(node_index v) const {
        const double limit = quantile ? static_cast<double>(*quantile) : threshold;
        return static_cast<double>(view.degree(v)) <= limit;
    }
};

// Page-locked host words for the per-round selection read.
struct pinned_selection_readback {
    int chosen;
    int compacted;
    device_selected_status status;
};

template<class T>
class pinned_host_array {
public:
    pinned_host_array() = default;
    ~pinned_host_array() { if (data_) cudaFreeHost(data_); }
    pinned_host_array(const pinned_host_array&) = delete;
    pinned_host_array& operator=(const pinned_host_array&) = delete;
    T* reserve(std::size_t count) {
        if (count > count_) {
            if (data_) { cudaFreeHost(data_); data_ = nullptr; count_ = 0; }
            cuda_check(cudaMallocHost(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                       "allocate pinned selection readback");
            count_ = count;
        }
        return data_;
    }
    T* get() noexcept { return data_; }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

double elapsed_ms(clock_type::time_point start, bool verbose) {
    if (!verbose)
        return std::numeric_limits<double>::quiet_NaN();
    return std::chrono::duration<double, std::milli>(clock_type::now(verbose) - start)
        .count();
}

template<class Operation>
std::size_t cub_bytes(Operation operation) {
    std::size_t bytes = 0;
    cuda_check(operation(nullptr, bytes), "query CUB workspace");
    return bytes;
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
    bool verbose;
    template<class Input, class T, class Predicate>
    auto select_operation(Input input, T* output, std::size_t count,
                          Predicate predicate, int slot = 0) {
        require_cub_count(count, "selection");
        int* const counter = selected_count.get() + slot;
        return [=] [[gnu::always_inline]] (void* storage, std::size_t& bytes) {
            return cub::DeviceSelect::If(storage, bytes, input, output,
                                        counter, static_cast<int>(count), predicate);
        };
    }

    auto degree_sort_operation(std::size_t count) {
        require_cub_count(count, "degree sort");
        return [=, this] [[gnu::always_inline]] (void* storage, std::size_t& bytes) {
            return cub::DeviceRadixSort::SortKeys(storage, bytes, active_degrees.get(),
                                                sorted_degrees.get(), static_cast<int>(count));
        };
    }

    explicit impl(node_index n_in, std::size_t initial_edges,
                  std::shared_ptr<gpu_device_selection_producer> producer, bool diagnostics)
        : verbose(diagnostics), n(n_in), live_edge_count(initial_edges), active_count(n_in),
          selection_producer(std::move(producer)) {
        if (n >= static_cast<node_index>(INT_MAX))
            throw std::overflow_error(
                "GPU block front-end currently requires n < INT_MAX");
        if (live_edge_count > static_cast<std::size_t>(INT_MAX))
            throw std::overflow_error(
                "GPU block front-end currently requires live edges < INT_MAX");

        int device = 0;
        cuda_check(cudaGetDevice(&device), "query active CUDA device");
        selection_producer->bind_device(device);
        status.reserve(n);
        pick.reserve(n);
        active_ids[0].reserve(n);
        active_ids[1].reserve(n);
        active_degrees.reserve(n);
        sorted_degrees.reserve(n);
        candidates_original.reserve(n);
        selected_ids.reserve(n);
        // [0] selection counts, [1] the last active compaction (checked later).
        selected_count.reserve(2);
        selected_status.reserve(1);
        pinned.reserve(1);
        pinned_ids.reserve(std::max<std::size_t>(n, 1));
        if (n) {
            initialize_status<<<blocks_for(n), kBlock>>>(n, status.get());
            cuda_check(cudaGetLastError(), "initialize bounded selector status");
            initialize_vertex_ids<<<blocks_for(n), kBlock>>>(
                n, active_ids[current_active].get());
            cuda_check(cudaGetLastError(), "initialize active vertex ids");
        }
    }

    void require_current_device(bool read_residual = true) const {
        int current_device = -1;
        cuda_check(cudaGetDevice(&current_device),
                   "query active CUDA device for producer operation");
        if (current_device != selection_producer->cuda_device())
            throw std::invalid_argument(
                "GPU block front-end operation uses the wrong CUDA device");
        selection_producer->require_usable();
        if (read_residual && residual_epoch.expired())
            throw std::logic_error("GPU selector residual generation is retired");
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
    const partition_result& select_for_test(std::span<const node_index> pivots) {
        if (pivots.empty() || pivots.size() > active_count)
            throw std::invalid_argument("test selection must contain active vertices");
        std::vector<node_index> active(active_count);
        std::vector<std::uint32_t> offsets(std::size_t(n) + 1);
        std::vector<gpu_round_shadow_incidence> incidences(2 * live_edge_count);
        cuda_check(cudaMemcpy(active.data(), active_ids[current_active].get(),
                              active.size() * sizeof(node_index), cudaMemcpyDeviceToHost),
                   "download active ids for test selection");
        cuda_check(cudaMemcpy(offsets.data(), owned_view.offsets,
                              offsets.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                   "download offsets for test selection");
        if (!incidences.empty())
            cuda_check(cudaMemcpy(incidences.data(), owned_view.incidences,
                                  incidences.size() * sizeof(gpu_round_shadow_incidence),
                                  cudaMemcpyDeviceToHost), "download residual for test selection");
        std::vector<unsigned char> marked(n);
        for (const auto v : active) marked[v] = 1;
        for (const auto v : pivots) {
            if (v >= n || marked[v] != 1)
                throw std::invalid_argument("test selection has invalid, inactive or repeated vertices");
            marked[v] = 2;
        }
        for (const auto v : pivots)
            for (auto at = offsets[v]; at < offsets[v + 1]; ++at)
                if (marked[incidences[at].neighbor] == 2)
                    throw std::invalid_argument("test selection is not independent");
        cuda_check(cudaMemcpy(selected_ids.get(), pivots.data(), pivots.size_bytes(),
                              cudaMemcpyHostToDevice), "upload test selection");
        selected_size = pivots.size();
        selected_degree_work = sum_selected_degrees(selected_size);
        result.data.assign(pivots.begin(), pivots.end());
        return result;
    }

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

    template<class Operation>
    decltype(auto) checked_operation(Operation&& operation, bool read_residual = true) {
        try {
            require_current_device(read_residual);
            return operation();
        } catch (...) {
            selection_producer->poison();
            selected_size = 0;
            throw;
        }
    }

    void require_cub_count(std::size_t count, const char *what) const {
        if (count > static_cast<std::size_t>(INT_MAX))
            throw std::overflow_error(std::string("GPU block front-end: ") +
                                      what + " exceeds CUB's item-count limit");
    }

    template<class Operation>
    void execute_cub(Operation operation, const char* what) {
        auto bytes = cub_bytes(operation);
        cub_temp.reserve(bytes);
        cuda_check(operation(cub_temp.get(), bytes), what);
    }

    // Short inputs take one launch (cuda_small_scan.cuh), same output.
    template<class Input, class T, class Predicate>
    void ordered_select(Input input, T* output, std::size_t count, Predicate predicate,
                        int slot) {
        if (count && count <= kSmallScanLimit) {
            small_stable_select<<<1, kSmallScanBlock>>>(
                input, output, static_cast<int>(count), selected_count.get() + slot,
                predicate_holds<Input, Predicate>{input, predicate});
            cuda_check(cudaGetLastError(), "ordered selection");
            return;
        }
        execute_cub(select_operation(input, output, count, predicate, slot),
                    "CUB ordered selection");
    }

    template<class Input, class T, class Predicate>
    int select_if(Input input, T* output, std::size_t count, Predicate predicate) {
        ordered_select(input, output, count, predicate, 0);
        int host_count = 0;
        cuda_check(cudaMemcpy(&host_count, selected_count.get(),
                              sizeof(host_count), cudaMemcpyDeviceToHost),
                   "download selection count");
        return host_count;
    }

    template<class Input, class T, class Predicate>
    void select_to_slot(Input input, T* output, std::size_t count, Predicate predicate, int slot) {
        ordered_select(input, output, count, predicate, slot);
    }

    // The compacted count is the expected one; its device value is checked
    // with the next blocking read (prepare's candidate count).
    void compact_active_ids(const std::uint8_t* mask, std::size_t expected) {
        if (!active_count) {
            if (expected) throw std::logic_error("GPU selector owned active count mismatch");
            return;
        }
        verify_pending_compaction();
        const int next = 1 - current_active;
        select_to_slot(active_ids[current_active].get(), active_ids[next].get(),
                       active_count, vertex_is_active{mask}, 1);
        current_active = next;
        active_count = expected;
        pending_compaction = true;
    }

    void check_compaction(int device_count) {
        pending_compaction = false;
        if (device_count < 0 || std::size_t(device_count) != active_count)
            throw std::logic_error("GPU selector owned active count mismatch");
    }

    void verify_pending_compaction() {
        if (!pending_compaction) return;
        int host = 0;
        cuda_check(cudaMemcpy(&host, selected_count.get() + 1, sizeof(host),
                              cudaMemcpyDeviceToHost), "download active compaction count");
        check_compaction(host);
    }

    unsigned long long sum_selected_degrees(std::size_t count) {
        selection_content.selection = {};
        if (!count)
            return 0;
        cuda_check(cudaMemset(selected_status.get(), 0,
                              sizeof(device_selected_status)),
                   "clear selected-degree/content status");
        sum_vertex_degrees<<<blocks_for(count), kBlock>>>(
            selected_ids.get(), count, nullptr, owned_view, selected_status.get());
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

    gpu_block_frontend::prepare_result
    prepare(std::span<const node_index> active,
            const partition_options &options) {
        const auto start = clock_type::now(verbose);

        if (active.size() != active_count)
            throw std::logic_error(
                "GPU block active-list size diverged from CPU state");
        host_active_degrees.clear();
        host_candidate_ids.clear();
        host_active_degrees_valid = false;
        host_candidate_ids_valid = false;

        if (!active.empty()) {
            gather_active_degrees<<<blocks_for(active.size()), kBlock>>>(
                active_ids[current_active].get(), active.size(), owned_view,
                active_degrees.get());
            cuda_check(cudaGetLastError(), "gather owned active degrees");
        }
        selection_content.selection = {};
        // Each retained edge contributes to two active vertices.
        const double average =
            active.empty() ? 0.0
                           : static_cast<double>(2ULL * live_edge_count) /
                                 static_cast<double>(active.size());

        degree_is_eligible eligible{owned_view, options.degree_multiplier * average};
        const double q = options.degree_quantile.value_or(degree_quantile_host_default);
        if (q > 0.0 && q < 1.0 && !active.empty()) {
            std::size_t rank = static_cast<std::size_t>(q * (active.size() - 1));
            if (rank >= active.size()) rank = active.size() - 1;
            execute_cub(degree_sort_operation(active.size()), "sort active degrees");
            eligible.quantile = sorted_degrees.get() + rank;
        }

        if (active.empty()) {
            verify_pending_compaction();
            candidate_count = 0;
        } else {
            select_to_slot(active_ids[current_active].get(), candidates_original.get(),
                           active.size(), eligible, 0);
            int counts[2]{};
            cuda_check(cudaMemcpy(counts, selected_count.get(),
                                  (pending_compaction ? 2 : 1) * sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       "download selection count");
            if (pending_compaction) check_compaction(counts[1]);
            candidate_count = static_cast<std::size_t>(counts[0]);
        }

        last_prepare = elapsed_ms(start, verbose);
        if (verbose) {
            std::fprintf(stderr,
                         "[gpu-block] prepare active=%zu candidates=%zu "
                         "edges=%zu %.3f ms\n",
                         active.size(), candidate_count, live_edge_count,
                         last_prepare);
        }
        return {.candidate_count = candidate_count, .average_degree = average};
    }

    std::span<const node_index> cached_download(std::vector<node_index>& host, bool& valid,
                                               const node_index* device, std::size_t count,
                                               const char* what) {
        if (!valid) {
            host.resize(count);
            if (count)
                cuda_check(cudaMemcpy(host.data(), device, count * sizeof(node_index),
                                      cudaMemcpyDeviceToHost), what);
            valid = true;
        }
        return host;
    }

    void launch_bounded_selection() {
        // Alternate scratch buffers without touching the original candidates.
        update_ids.reserve(candidate_count);
        // Every pass visits the whole candidate list; vertices decided in an
        // earlier pass skip out (bounded_decide), so each pass decides exactly
        // the still-undecided set a compaction would have kept, without
        // reading its size back. Kernel order is the snapshot boundary.
        const node_index* current = candidates_original.get();
        const std::size_t count = candidate_count;
        bounded_passes = 0;
        bounded_candidate_visits = 0;
        // n<INT_MAX and at most32 lanes/candidate: checked size_t arithmetic,
        // launch grid remains well within CUDA's signed-int X dimension.
        if (count > (std::numeric_limits<std::size_t>::max() - (kBlock - 1)) / 32)
            throw std::overflow_error("bounded selector launch size overflow");
        const int blocks = blocks_for(count * std::size_t{32});
        // Without diagnostics all four passes run; a short list runs them
        // in one block (same decisions, see bounded_selection_one_block).
        if (!verbose && count <= kBoundedOneBlockLimit) {
            bounded_passes = 4;
            bounded_candidate_visits = 4 * count;
            bounded_selection_one_block<<<1, kBoundedBlock>>>(current, count, owned_view,
                status.get(), pick.get(), selected_ids.get(), selected_status.get(),
                bounded_seed, bounded_phase, 4);
            cuda_check(cudaGetLastError(), "select bounded independent set in one block");
            return;
        }
        begin_bounded_selection<<<blocks_for(count), kBlock>>>(current, count, status.get(),
            selected_ids.get(), selected_status.get());
        cuda_check(cudaGetLastError(), "initialize bounded candidates");
        std::size_t undecided = count;
        for (unsigned step = 0; step < 4 && undecided; ++step) {
            ++bounded_passes;
            bounded_candidate_visits += undecided;
            bounded_decide<<<blocks, kBlock>>>(current, count, owned_view,
                status.get(), pick.get(),
                bounded_seed, bounded_phase);
            cuda_check(cudaGetLastError(), "decide bounded independent set");
            bounded_commit<<<blocks, kBlock>>>(current, count, owned_view,
                pick.get(), status.get());
            cuda_check(cudaGetLastError(), "commit bounded independent set");
            // Diagnostics only: count the undecided set as before.
            if (verbose && step < 3)
                undecided = static_cast<std::size_t>(select_if(current, update_ids.get(), count,
                    status_equals{status.get(), 0}));
        }
    }

    const partition_result &select_block_greedy() {
        const auto start = clock_type::now(verbose);
        result.data.clear();
        if (!candidate_count) {
            selected_size = 0;
            selected_degree_work = 0;
            last_select = elapsed_ms(start, verbose);
            return result;
        }

        // Also clears the selected ids and the selected-degree/content status.
        launch_bounded_selection();
        // Chosen count, degree sum, digest and ids leave the device in one
        // synchronization; ids are copied up to the candidate count bound
        // (cleared above, so the copied tail is defined for initcheck).
        select_to_slot(candidates_original.get(), selected_ids.get(),
                       candidate_count, status_equals{status.get(), 1}, 0);
        // Also restores the candidates' status to 2 after the compaction.
        sum_vertex_degrees<<<blocks_for(candidate_count), kBlock>>>(
            selected_ids.get(), candidate_count, selected_count.get(), owned_view,
            selected_status.get(), candidates_original.get(), status.get());
        cuda_check(cudaGetLastError(), "sum selected degrees");
        auto* staged = pinned.get();
        auto* staged_ids = pinned_ids.get();
        cuda_check(cudaMemcpyAsync(&staged->chosen, selected_count.get(), sizeof(int),
                                   cudaMemcpyDeviceToHost), "queue selection count download");
        cuda_check(cudaMemcpyAsync(&staged->status, selected_status.get(),
                                   sizeof(device_selected_status), cudaMemcpyDeviceToHost),
                   "queue selected-degree sum and ordered selection digest download");
        cuda_check(cudaMemcpyAsync(staged_ids, selected_ids.get(),
                                   candidate_count * sizeof(node_index), cudaMemcpyDeviceToHost),
                   "queue block-greedy selection download");
        cuda_check(cudaStreamSynchronize(nullptr), "finish block-greedy selection");
        if (staged->chosen < 0 || std::size_t(staged->chosen) > candidate_count)
            throw std::logic_error("bounded selector returned an invalid selection count");
        const std::size_t chosen = static_cast<std::size_t>(staged->chosen);
        if (!chosen)
            throw std::logic_error("bounded selector made no progress on nonempty candidates");
        selected_size = chosen;
        selection_content.selection = {staged->status.selection.xor_hash,
                                       staged->status.selection.sum_hash};
        selected_degree_work = staged->status.degree_sum;
        result.data.assign(staged_ids, staged_ids + chosen);

        last_select = elapsed_ms(start, verbose);
        if (verbose)
            std::fprintf(stderr,
                "[gpu-bounded-selection] passes=%u candidate_visits=%llu "
                "seed=%llu phase=%llu scratch_capacity_bytes=%zu\n",
                bounded_passes, static_cast<unsigned long long>(bounded_candidate_visits),
                static_cast<unsigned long long>(bounded_seed),
                static_cast<unsigned long long>(bounded_phase),
                update_ids.count() * sizeof(node_index));
        if (verbose) {
            std::fprintf(stderr,
                         "[gpu-block] select candidates=%zu chosen=%zu %.3f ms\n",
                         candidate_count, chosen, last_select);
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
        compact_active_ids(active, expected.active_count);
        // The private caller binds producer/device/consumed-selection identity
        // before entry. The arrays and digests belong to this exact immutable
        // numerical generation; copying/re-hashing them adds no validation.
        owned_view = {offsets, incidences};
        residual_epoch = std::move(epoch);
        live_edge_count = directed_count / 2;
        selection_content = expected;
        candidate_count = 0;
        host_active_degrees_valid = host_candidate_ids_valid = false;
        ++transfers.owned_residual_binds;
    }

    gpu_block_frontend::transfer_stats transfers;
    node_index n;
    std::size_t live_edge_count = 0;
    selector_owned_view owned_view{};
    std::weak_ptr<const void> residual_epoch;

    device_buffer<node_index> active_ids[2];
    int current_active = 0;
    std::size_t active_count = 0;
    device_buffer<node_index> active_degrees;
    device_buffer<node_index> sorted_degrees;
    device_buffer<node_index> candidates_original;
    device_buffer<node_index> selected_ids;
    device_buffer<node_index> update_ids;
    device_buffer<int> status;
    device_buffer<unsigned char> pick;
    device_buffer<int> selected_count;
    device_buffer<device_selected_status> selected_status;
    device_buffer<unsigned char> cub_temp;
    pinned_host_array<pinned_selection_readback> pinned;
    pinned_host_array<node_index> pinned_ids;
    bool pending_compaction = false;

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
    gpu_device_selection_content selection_content;
    std::shared_ptr<gpu_device_selection_producer> selection_producer;

    double last_prepare = 0.0;
    double last_select = 0.0;
};

gpu_block_frontend::runtime_probe
gpu_block_frontend::probe_runtime(node_index n) {
    runtime_probe result;
    if (cudaMemGetInfo(&result.free_bytes, &result.total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return result;
    }

    std::size_t bytes = 0;
    const std::size_t nv = static_cast<std::size_t>(n);
    bool valid = n < static_cast<node_index>(INT_MAX);
    // Seven vertex arrays, plus the old scratch allocation during growth.
    valid &= add_allocation(bytes, nv, 8 * sizeof(node_index));
    valid &= add_allocation(bytes, nv, sizeof(int) + sizeof(unsigned char));
    valid &= add_allocation(bytes, 1, 2 * sizeof(int) + sizeof(device_selected_status));
    if (!valid) {
        result.estimated_bytes = std::numeric_limits<std::size_t>::max();
        return result;
    }
    // Null-storage queries use only iterator types and counts. Match every
    // production selection instantiation, including the const bounded frontier.
    const int count = static_cast<int>(n);
    node_index* ids = nullptr;
    int* selected = nullptr;
    const auto selection_bytes = [&](auto input, auto predicate) {
        return cub_bytes([&](void* storage, std::size_t& size) {
            return cub::DeviceSelect::If(storage, size, input, ids, selected,
                                        count, predicate);
        });
    };
    const auto scratch = std::max({
        cub_bytes([&](void* storage, std::size_t& size) {
            return cub::DeviceRadixSort::SortKeys(storage, size, ids, ids, count);
        }),
        selection_bytes(ids, vertex_is_active{nullptr}),
        selection_bytes(ids, degree_is_eligible{selector_owned_view{}, 0.0}),
        selection_bytes(ids, status_equals{nullptr, 1}),
        selection_bytes(static_cast<const node_index*>(ids), status_equals{nullptr, 0})});
    // Reserve for simultaneous old/new CUB allocations during growth, keeping
    // the existing floor for implementation overhead outside these arrays.
    valid &= add_allocation(bytes, 2, scratch);
    valid &= add_allocation(bytes, 1, std::size_t{64} << 20);
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
    node_index n, std::size_t undirected_edges, owned_initialization_tag, bool verbose)
    : p_(std::make_unique<impl>(n, undirected_edges,
          std::shared_ptr<gpu_device_selection_producer>(new gpu_device_selection_producer()), verbose)) {}

void gpu_block_frontend::reset() noexcept {
    if (!p_) return;
    p_->selection_producer->retire();
    // Destruction is nonconcurrent by contract. If the caller left another
    // device current, free on the producer's device and restore it best-effort.
    const cuda_device_scope device(p_->selection_producer->cuda_device());
    p_.reset();
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
                           const partition_options &options,
                           std::uint64_t seed, std::uint64_t phase) {
    return p_->checked_operation([&] {
        p_->invalidate_selection();
        p_->current_options = options;
        p_->bounded_seed = seed;
        p_->bounded_phase = phase;
        return p_->prepare(active, options);
    });
}

std::span<const node_index> gpu_block_frontend::host_candidates() const {
    return p_->checked_operation([&] {
        return p_->cached_download(p_->host_candidate_ids, p_->host_candidate_ids_valid,
            p_->candidates_original.get(), p_->candidate_count, "download debug candidate ids");
    });
}

std::span<const node_index> gpu_block_frontend::host_active_degrees() const {
    return p_->checked_operation([&] {
        return p_->cached_download(p_->host_active_degrees, p_->host_active_degrees_valid,
            p_->active_degrees.get(), p_->active_count, "download debug active degrees");
    });
}

const partition_result &gpu_block_frontend::select_block_greedy() {
    return p_->checked_operation([&]() -> const partition_result& {
        p_->invalidate_selection();
        const auto& selected = p_->select_block_greedy();
        if (!selected.data.empty()) p_->publish_selection();
        return selected;
    });
}

std::size_t gpu_block_frontend::selected_degree_work() const {
    return p_->checked_operation([&] {
        return p_->selected_degree_work;
    });
}

gpu_device_selection gpu_block_frontend::device_selection() const {
    return p_->checked_operation([&] {
        return gpu_device_selection::issue(p_->selection_producer);
    });
}

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
const partition_result& gpu_block_frontend::select_for_test(std::span<const node_index> pivots) {
    return p_->checked_operation([&]() -> const partition_result& {
        p_->invalidate_selection();
        const auto& selected = p_->select_for_test(pivots);
        p_->publish_selection();
        return selected;
    });
}

void gpu_block_frontend::inject_device_selection_fault_for_test(
        std::span<const node_index> replacement) {
    return p_->checked_operation([&] {
        p_->inject_device_selection_fault_for_test(replacement);
    });
}
#endif

gpu_device_selection gpu_block_frontend::selection_for_residual_handoff() const {
    return p_->checked_operation([&] {
        return gpu_device_selection::issue(p_->selection_producer);
    }, /*read_residual=*/false);
}

void gpu_block_frontend::bind_owned_residual(
        const gpu_round_shadow_incidence* incidences, const std::uint32_t* offsets,
        const std::uint8_t* active, std::size_t directed_count,
        const gpu_device_selection_content& expected,
        std::weak_ptr<const void> epoch) {
    return p_->checked_operation([&] {
        p_->invalidate_selection();
        p_->selection_producer->begin_topology_advance();
        p_->bind_owned_residual(incidences, offsets, active, directed_count,
                                expected, std::move(epoch));
    }, /*read_residual=*/false);
}

void gpu_block_frontend::verify_bound_residual() {
    return p_->checked_operation([&] { p_->verify_pending_compaction(); },
                                 /*read_residual=*/false);
}

gpu_block_frontend::transfer_stats gpu_block_frontend::transfers() const {
    p_->require_current_device(/*read_residual=*/false);
    return p_->transfers;
}

} // namespace apxchol::detail

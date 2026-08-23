#pragma once
/// GPU-resident residual topology for the priority and block-region selectors.
///
/// This is deliberately a narrow setup front-end: numerical elimination and
/// factor construction stay on the CPU.  The GPU owns an unweighted COO copy
/// of the live residual topology, rebuilds CSR after each elimination round,
/// applies the same degree cap as the CPU partitioners, and returns selected
/// vertex ids in candidate order. The priority selector is an explicit
/// research path; the block-region selector is automatic through T=8.
///
/// The implementation lives in src/cuda_priority_frontend.cu. At higher host
/// thread counts, incompatible options, or a failed AUTO fitting probe, the
/// CPU path has no topology capture or device-allocation overhead.

#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/factorize_workspace.h"
#include "apxchol/solver/partition.h"
#include <cstdint>
#include <memory>
#include <span>

namespace apxchol::detail {

class gpu_priority_frontend {
public:
    enum class mode { disabled, automatic, forced };

    struct prepare_result {
        std::size_t candidate_count = 0;
        double average_degree = 0.0;
    };

    struct runtime_probe {
        bool cooperative_launch = false;
        bool memory_fits = false;
        std::size_t estimated_bytes = 0;
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
    };

    /// Parse APXCHOL_GPU_PRIORITY_FRONTEND. Accepted values are 0/off and
    /// 1/on/force. Unknown values disable the optional path with a note.
    static mode configured_mode();
    /// Parse the independent GPU block-greedy policy. Unset and `auto` select
    /// automatic mode; 0/off disable it; 1/on/force bypass the governor.
    static mode configured_block_mode();
    /// CPU block-greedy overtakes the GPU selector beyond this measured
    /// host-thread crossover. Kept pure so the policy is unit-testable.
    static bool block_auto_enabled(int max_threads) noexcept;
    static runtime_probe probe_runtime(node_index n, std::size_t initial_edges,
                                       bool block_selector = false);

    gpu_priority_frontend(node_index n,
                      std::span<const gpu_topology_edge> initial_edges);
    ~gpu_priority_frontend();

    gpu_priority_frontend(const gpu_priority_frontend &) = delete;
    gpu_priority_frontend &operator=(const gpu_priority_frontend &) = delete;
    gpu_priority_frontend(gpu_priority_frontend &&) noexcept;
    gpu_priority_frontend &operator=(gpu_priority_frontend &&) noexcept;

    prepare_result prepare(std::span<const node_index> active,
                           const partition_options &options);
    /// Debug/test views. Materializing either view downloads a full device
    /// array; the production factorization path does not call these methods.
    std::span<const node_index> host_candidates() const;
    std::span<const node_index> host_active_degrees() const;
    const partition_result &select(unsigned seed, std::uint64_t round);
    const partition_result &select_block_greedy();
    std::size_t selected_degree_work() const;

    /// Commit a selection after CPU elimination succeeded and enqueue the
    /// sampled clique endpoints that must be present in the next round.
    void advance(std::span<const node_index> eliminated,
                 std::span<const gpu_topology_edge> new_edges,
                 std::span<const gpu_topology_batch> new_edge_batches = {});

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace apxchol::detail

#pragma once
/// Optional GPU-resident topology and exact fixed-priority Luby selector.
///
/// This is deliberately a narrow setup front-end: numerical elimination and
/// factor construction stay on the CPU.  The GPU owns an unweighted COO copy
/// of the live residual topology, rebuilds CSR after each elimination round,
/// applies the same degree cap and priority rule as luby_partitioner, and
/// returns selected vertex ids in candidate order.
///
/// The implementation lives in src/cuda_luby_frontend.cu.  Nothing constructs
/// this class unless APXCHOL_GPU_LUBY_FRONTEND=auto|1, so the shipped/default
/// path has no topology capture or device-allocation overhead.

#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/factorize_workspace.h"
#include "apxchol/solver/partition.h"
#include <cstdint>
#include <memory>
#include <span>

namespace apxchol::detail {

class gpu_luby_frontend {
public:
    enum class mode { disabled, automatic, forced };
    static constexpr node_index automatic_min_vertices = node_index{500000};

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

    /// Parse APXCHOL_GPU_LUBY_FRONTEND. Accepted values are 0/off, auto,
    /// and 1/on/force. Unknown values disable the optional path with a note.
    static mode configured_mode();
    static bool automatic_worthwhile(node_index n, std::size_t incidence,
                                     std::size_t incidence_ge32);
    static runtime_probe probe_runtime(node_index n, std::size_t initial_edges);

    gpu_luby_frontend(node_index n,
                      std::span<const gpu_topology_edge> initial_edges);
    ~gpu_luby_frontend();

    gpu_luby_frontend(const gpu_luby_frontend &) = delete;
    gpu_luby_frontend &operator=(const gpu_luby_frontend &) = delete;
    gpu_luby_frontend(gpu_luby_frontend &&) noexcept;
    gpu_luby_frontend &operator=(gpu_luby_frontend &&) noexcept;

    prepare_result prepare(std::span<const node_index> active,
                           const partition_options &options);
    std::span<const node_index> host_candidates() const;
    std::span<const node_index> host_active_degrees() const;
    const partition_result &select(unsigned seed, std::uint64_t round);

    /// Commit a selection after CPU elimination succeeded and enqueue the
    /// sampled clique endpoints that must be present in the next round.
    void advance(std::span<const node_index> eliminated,
                 std::span<const gpu_topology_edge> new_edges);

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace apxchol::detail

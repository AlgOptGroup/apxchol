#pragma once
/// GPU-resident residual topology for the block-region selector.
///
/// For ordinary CPU elimination, the selector owns an unweighted COO copy and
/// rebuilds CSR after each round. The consuming GPU path instead borrows the
/// numerical owner's immutable current CSR through a private generation token.
/// Both routes use the same degree cap and block-region selection policy.
/// Enable explicitly with APXCHOL_GPU_BLOCK_FRONTEND=1|on|force (unset disables).
/// The implementation lives in src/cuda_block_frontend.cu; the default CPU setup
/// has no topology capture or device-allocation overhead.

#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/factorize_workspace.h"
#include "apxchol/solver/gpu_device_selection.h"
#include "apxchol/solver/partition.h"
#include <cstdint>
#include <memory>
#include <span>

namespace apxchol::detail {

struct gpu_round_shadow_incidence;
class gpu_round_shadow_device_state;

/// Single-threaded producer. No method call, destruction, or consumption of a
/// returned gpu_device_selection may overlap another operation on this object.
/// The borrowed host spans and partition_result remain valid only until the
/// next non-const producer operation or destruction. This explicit contract
/// avoids locks that could not protect a borrow after its returning call.
/// When paired with a numerical GPU owner, selector calls must not overlap the
/// owner's mutation or destruction either; a private epoch rejects serial use
/// after the borrowed numerical generation has been retired.
class gpu_block_frontend {
public:
    enum class mode { disabled, forced };

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

    /// Unset/empty and 0/off/false disable; 1/on/force enable. Other values
    /// (including the removed auto policy) throw invalid_argument.
    static mode configured_block_mode();
    static runtime_probe probe_runtime(node_index n, std::size_t initial_edges);

    gpu_block_frontend(node_index n,
                      std::span<const gpu_topology_edge> initial_edges);
    ~gpu_block_frontend();

    gpu_block_frontend(const gpu_block_frontend &) = delete;
    gpu_block_frontend &operator=(const gpu_block_frontend &) = delete;
    gpu_block_frontend(gpu_block_frontend &&) noexcept;
    gpu_block_frontend &operator=(gpu_block_frontend &&) noexcept;

    prepare_result prepare(std::span<const node_index> active,
                           const partition_options &options
                           , std::uint64_t seed = 42, std::uint64_t phase = 0
                           );
    /// Maximum number of candidate regions that the region-scan kernel can
    /// keep resident at once (one warp per region).
    std::size_t resident_region_capacity() const;
    /// Debug/test borrowed views. Materializing either view downloads a full
    /// device array; the production factorization path does not call these
    /// methods. See the class-level nonconcurrency/lifetime contract.
    std::span<const node_index> host_candidates() const;
    std::span<const node_index> host_active_degrees() const;
    const partition_result &select_block_greedy();
    std::size_t selected_degree_work() const;
    /// The exact selection returned by the most recent
    /// select_block_greedy(), before its CPU copy.  This lets a device-owned
    /// elimination round consume the same ordered ids without re-uploading
    /// them. The returned immutable capability carries checked producer,
    /// device, generation, active-set and topology identity. The next prepare,
    /// selection, or advance invalidates it; producer destruction is detected
    /// before dereferencing the retired allocation. Consumption may not race
    /// producer mutation or destruction (see the class contract).
    /// Empty host selections are not published as device capabilities;
    /// requesting one fails closed and poisons the producer.
    gpu_device_selection device_selection() const;

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    /// Compile-time-only corruption seam for the dedicated fault build. It
    /// overwrites the current selected ids without changing their producer
    /// digest, and never exposes the producer-owned CUDA allocation.
    void inject_device_selection_fault_for_test(
        std::span<const node_index> replacement);
#endif

    /// Commit a selection after CPU elimination succeeded and enqueue the
    /// sampled clique endpoints that must be present in the next round.
    void advance(std::span<const node_index> eliminated,
                 std::span<const gpu_topology_edge> new_edges,
                 std::span<const gpu_topology_batch> new_edge_batches = {});

    /// Internal audit counters exclude initial construction and scalar downloads.
    struct transfer_stats {
        std::size_t host_update_bytes = 0;
        std::size_t resident_advances = 0;
        std::size_t owned_residual_binds = 0;
        // Conservative extra peak from any CUB scratch growth during projection.
        std::size_t projection_scratch_peak_extra_bytes = 0;
    };
    transfer_stats transfers() const;

private:
    friend class gpu_round_shadow_device_state;
    struct owned_initialization_tag {};
    gpu_block_frontend(node_index n, std::size_t undirected_edges,
                       owned_initialization_tag);
    // Generic CPU-certified resident producers retain projection and auditing.
    // The caller binds its consumed selection identity/generations before entry;
    // synchronous completion keeps the borrowed weighted state immutable/alive.
    void advance_resident(const gpu_round_shadow_incidence* incidences,
                          std::size_t directed_count,
                          const std::uint8_t* active,
                          const gpu_device_selection_content& expected);
    // Owning generations borrow the numerical CSR instead of projecting it.
    // The private epoch is revoked before mutation, failure, or destruction.
    void bind_owned_residual(const gpu_round_shadow_incidence* incidences,
                             const std::uint32_t* offsets,
                             const std::uint8_t* active,
                             std::size_t directed_count,
                             const gpu_device_selection_content& expected,
                             std::weak_ptr<const void> epoch);
    gpu_device_selection selection_for_residual_handoff() const;
    void reset() noexcept;
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace apxchol::detail

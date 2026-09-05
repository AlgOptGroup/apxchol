#pragma once

#include "apxchol/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace apxchol::detail {

class gpu_block_frontend;
class gpu_round_shadow_device_state;

struct gpu_device_selection_digest {
    std::uint64_t xor_hash = 0;
    std::uint64_t sum_hash = 0;

    friend bool operator==(const gpu_device_selection_digest&,
                           const gpu_device_selection_digest&) = default;
};

/// Producer-owned semantic identity of the residual graph against which a
/// device selection was made. Topology hashes the undirected edge multiset;
/// active hashes the active-vertex set; selection hashes ordered selected ids.
/// Multiplicity is retained by sum_hash.
struct gpu_device_selection_content {
    node_index vertex_count = 0;
    std::size_t active_count = 0;
    gpu_device_selection_digest topology;
    gpu_device_selection_digest active;
    // Order-sensitive through the ordinal embedded in each item hash.
    gpu_device_selection_digest selection;

    friend bool operator==(const gpu_device_selection_content&,
                           const gpu_device_selection_content&) = default;
};

#if defined(__CUDACC__)
#define APXCHOL_GPU_SELECTION_HD __host__ __device__
#else
#define APXCHOL_GPU_SELECTION_HD
#endif

APXCHOL_GPU_SELECTION_HD inline constexpr std::uint64_t
gpu_device_selection_mix(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

APXCHOL_GPU_SELECTION_HD inline constexpr std::uint64_t
gpu_device_selection_item_hash(std::uint64_t tag, std::uint64_t a,
                               std::uint64_t b,
                               std::uint64_t value) noexcept {
    std::uint64_t h = gpu_device_selection_mix(tag);
    h = gpu_device_selection_mix(h ^ gpu_device_selection_mix(a));
    h = gpu_device_selection_mix(h ^ gpu_device_selection_mix(b));
    return gpu_device_selection_mix(h ^ gpu_device_selection_mix(value));
}

inline constexpr std::uint64_t gpu_device_selection_active_tag = 4;
inline constexpr std::uint64_t gpu_device_selection_topology_tag = 9;
inline constexpr std::uint64_t gpu_device_selection_selected_tag = 10;

APXCHOL_GPU_SELECTION_HD inline constexpr std::uint64_t
gpu_device_selection_active_hash(node_index vertex) noexcept {
    return gpu_device_selection_item_hash(
        gpu_device_selection_active_tag, vertex, 0, 1);
}

APXCHOL_GPU_SELECTION_HD inline constexpr std::uint64_t
gpu_device_selection_topology_hash(node_index u, node_index v) noexcept {
    if (v < u) {
        const node_index swap = u;
        u = v;
        v = swap;
    }
    return gpu_device_selection_item_hash(
        gpu_device_selection_topology_tag, u, v, 1);
}

APXCHOL_GPU_SELECTION_HD inline constexpr std::uint64_t
gpu_device_selection_selected_hash(std::size_t ordinal,
                                   node_index vertex) noexcept {
    return gpu_device_selection_item_hash(
        gpu_device_selection_selected_tag, ordinal, vertex, 1);
}

inline void gpu_device_selection_digest_add(
        gpu_device_selection_digest& digest, std::uint64_t item) noexcept {
    digest.xor_hash ^= item;
    digest.sum_hash += item;
}

inline bool gpu_device_selection_state_matches(
        const gpu_device_selection_content& selected,
        const gpu_device_selection_content& state) noexcept {
    return selected.vertex_count == state.vertex_count &&
        selected.active_count == state.active_count &&
        selected.topology == state.topology &&
        selected.active == state.active;
}

#undef APXCHOL_GPU_SELECTION_HD

/// Mutable producer state is deliberately not caller-constructible and is
/// never exposed by gpu_device_selection. gpu_block_frontend owns the only
/// instances; a published capability snapshots and later rechecks this state.
class gpu_device_selection_producer final {
public:
    gpu_device_selection_producer(const gpu_device_selection_producer&) =
        delete;
    gpu_device_selection_producer& operator=(
        const gpu_device_selection_producer&) = delete;

private:
    struct identity_token final {};

    gpu_device_selection_producer()
        : identity_(std::make_shared<const identity_token>()) {}

    void bind_device(int cuda_device) {
        if (cuda_device_ >= 0)
            throw std::logic_error(
                "GPU block front-end producer device was already bound");
        cuda_device_ = cuda_device;
    }

    void require_usable() const {
        if (!alive_)
            throw std::logic_error(
                "GPU block front-end selection producer is no longer alive");
        if (poisoned_)
            throw std::logic_error(
                "GPU block front-end selection producer is poisoned");
    }

    void invalidate_selection() {
        require_usable();
        data_ = nullptr;
        size_ = 0;
        ready_ = false;
        if (generation_ == UINT64_MAX) {
            poisoned_ = true;
            throw std::overflow_error(
                "GPU block front-end selection generation overflow");
        }
        ++generation_;
    }

    void publish_selection(const node_index* data, std::size_t size,
                           const gpu_device_selection_content& content,
                           bool independence_certified) {
        if (!data || size == 0) {
            poison();
            throw std::logic_error(
                "GPU block front-end cannot publish an empty device selection");
        }
        data_ = data;
        size_ = size;
        content_ = content;
        independence_certified_ = independence_certified;
        ready_ = true;
    }

    void begin_topology_advance() {
        require_usable();
        if (topology_generation_ == UINT64_MAX) {
            poisoned_ = true;
            throw std::overflow_error(
                "GPU block front-end topology generation overflow");
        }
        ++topology_generation_;
    }

    void poison() noexcept {
        data_ = nullptr;
        size_ = 0;
        ready_ = false;
        independence_certified_ = false;
        poisoned_ = true;
    }

    void retire() noexcept {
        poison();
        alive_ = false;
    }

    int cuda_device() const noexcept { return cuda_device_; }
    const node_index* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    std::uint64_t generation() const noexcept { return generation_; }
    std::uint64_t topology_generation() const noexcept {
        return topology_generation_;
    }
    const gpu_device_selection_content& content() const noexcept {
        return content_;
    }
    bool alive() const noexcept { return alive_; }
    bool ready() const noexcept { return ready_; }
    bool poisoned() const noexcept { return poisoned_; }
    bool independence_certified() const noexcept {
        return independence_certified_;
    }
    const std::shared_ptr<const void>& identity() const noexcept {
        return identity_;
    }

    friend class gpu_block_frontend;
    friend class gpu_device_selection;

    std::shared_ptr<const void> identity_;
    const node_index* data_ = nullptr;
    std::size_t size_ = 0;
    std::uint64_t generation_ = 0;
    std::uint64_t topology_generation_ = 0;
    gpu_device_selection_content content_;
    int cuda_device_ = -1;
    bool alive_ = true;
    bool ready_ = false;
    bool poisoned_ = false;
    bool independence_certified_ = false;
};

/// Immutable, producer-issued capability for selected ids in CUDA memory.
/// Callers can default-construct an unissued sentinel or copy an issued value,
/// but cannot create or mutate its identity, generations, device, or content.
class gpu_device_selection final {
public:
    gpu_device_selection() = default;

private:
    struct snapshot {
        const node_index* data = nullptr;
        std::size_t size = 0;
        int cuda_device = -1;
        std::uint64_t generation = 0;
        std::uint64_t topology_generation = 0;
        gpu_device_selection_content content;
        bool independence_certified = false;
        // Separate private token prevents an opaque identity from being cast
        // back into producer state, even by code including this internal header.
        std::shared_ptr<const void> producer_identity;
    };

    snapshot inspect() const {
        if (!capability_ || !capability_->producer)
            throw std::invalid_argument(
                "GPU round shadow: device selection was not issued by a producer");
        const auto& producer = *capability_->producer;
        if (!producer.alive())
            throw std::invalid_argument(
                "GPU round shadow: device selection producer is no longer alive");
        if (producer.poisoned())
            throw std::invalid_argument(
                "GPU round shadow: device selection producer is poisoned");
        if (!producer.ready())
            throw std::invalid_argument(
                "GPU round shadow: device selection generation is not published");
        if (capability_->generation != producer.generation())
            throw std::invalid_argument(
                "GPU round shadow: stale device selection generation");
        if (capability_->data != producer.data() ||
            capability_->size != producer.size() ||
            capability_->cuda_device != producer.cuda_device() ||
            capability_->topology_generation !=
                producer.topology_generation() ||
            capability_->content != producer.content() ||
            capability_->independence_certified !=
                producer.independence_certified())
            throw std::invalid_argument(
                "GPU round shadow: device selection capability no longer matches its producer");
        return {
            capability_->data,
            capability_->size,
            capability_->cuda_device,
            capability_->generation,
            capability_->topology_generation,
            capability_->content,
            capability_->independence_certified,
            producer.identity()};
    }

    struct capability {
        std::shared_ptr<const gpu_device_selection_producer> producer;
        const node_index* data = nullptr;
        std::size_t size = 0;
        int cuda_device = -1;
        std::uint64_t generation = 0;
        std::uint64_t topology_generation = 0;
        gpu_device_selection_content content;
        bool independence_certified = false;
    };

    static gpu_device_selection issue(
            const std::shared_ptr<gpu_device_selection_producer>& producer) {
        producer->require_usable();
        if (!producer->ready())
            throw std::logic_error(
                "GPU block front-end has no current nonempty device selection");
        if (!producer->data() || producer->size() == 0) {
            producer->poison();
            throw std::logic_error(
                "GPU block front-end refused an empty device selection capability");
        }
        auto issued = std::shared_ptr<capability>(new capability{
            producer,
            producer->data(),
            producer->size(),
            producer->cuda_device(),
            producer->generation(),
            producer->topology_generation(),
            producer->content(),
            producer->independence_certified()});
        gpu_device_selection result;
        result.capability_ = std::move(issued);
        return result;
    }

    std::shared_ptr<const capability> capability_;
    friend class gpu_block_frontend;
    friend class gpu_round_shadow_device_state;
};

} // namespace apxchol::detail

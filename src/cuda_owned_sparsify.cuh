// Included after the numerical owner's allocation/CUB helpers, inside its
// anonymous namespace. Input is paired active CSR; output is a distinct spare.
// The forest priority is the CPU's (high 12 complemented pool bits, ordinal),
// not exact-weight Kruskal. Normalization preserves its full-stream partition.
struct owned_sparsify_scale {
    double value = 0.0, target = 0.0;
    unsigned passes = 0, advance = 0;
};

__global__ void owned_sparsify_canonical_flags(
        const gpu_round_shadow_incidence* in, std::size_t count, node_index n,
        const std::uint8_t* active, std::uint8_t* flags, unsigned* error) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    const auto e = in[i];
    if (e.owner >= n || e.neighbor >= n || e.owner == e.neighbor ||
        !isfinite(e.weight) || e.weight < 0.0) {
        flags[i] = 0; atomicExch(error, 1U); return;
    }
    if (!active[e.owner] || !active[e.neighbor]) {
        flags[i] = 0; atomicExch(error, 1U); return;
    }
    flags[i] = e.owner < e.neighbor;
    if (flags[i]) atomicAdd(error + 1, 1U);
}

__global__ void owned_sparsify_keys(const gpu_round_shadow_incidence* in,
        std::size_t count, std::uint64_t* keys, bool endpoints) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    const auto e = in[i];
    // All dimensions are checked <= INT_MAX, including wide-index builds.
    // Canonical +0/-0 compare equal in the CPU ascending-weight ordering.
    keys[i] = endpoints ? (std::uint64_t(e.owner) << 32) | e.neighbor :
        (e.weight == 0.0 ? 0ULL : std::uint64_t(__double_as_longlong(e.weight)));
}

__global__ void owned_sparsify_coalesce(
        const gpu_round_shadow_incidence* sorted, std::size_t count,
        gpu_round_shadow_incidence* folded, std::uint8_t* flags, unsigned* error) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    const auto e = sorted[i];
    flags[i] = i == 0 || sorted[i - 1].owner != e.owner ||
        sorted[i - 1].neighbor != e.neighbor;
    folded[i] = e; // CUB may load every item before applying its flag.
    if (!flags[i]) return;
    double sum = 0.0;
    for (auto j = i; j < count && sorted[j].owner == e.owner &&
         sorted[j].neighbor == e.neighbor; ++j)
        sum = __dadd_rn(sum, sorted[j].weight);
    const pool_value_t stored = static_cast<pool_value_t>(sum);
    if (!isfinite(sum) || !isfinite(double(stored))) atomicExch(error, 1U);
    folded[i] = {e.owner, e.neighbor, double(stored)};
}

__device__ std::uint64_t owned_sparsify_priority(double weight, unsigned id) {
    std::uint64_t bucket;
    if constexpr (sizeof(pool_value_t) == 4)
        bucket = (std::uint64_t(~__float_as_uint(static_cast<float>(weight))) >> 20) & 4095ULL;
    else bucket = (~std::uint64_t(__double_as_longlong(weight)) >> 52) & 4095ULL;
    return (bucket << 32) | id;
}

__global__ void owned_sparsify_ids(std::size_t count, std::uint32_t* ids) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count) ids[i] = unsigned(i);
}

__global__ void owned_sparsify_initialize(node_index n,
        std::uint32_t* parents, std::uint32_t* labels) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) parents[i] = labels[i] = unsigned(i);
}

__global__ void owned_sparsify_best(const gpu_round_shadow_incidence* edges,
        std::size_t count, const std::uint32_t* labels,
        unsigned long long* best) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    const auto e = edges[i]; const auto u = labels[e.owner], v = labels[e.neighbor];
    if (u == v) return;
    const auto priority = static_cast<unsigned long long>(owned_sparsify_priority(e.weight, unsigned(i)));
    atomicMin(best + u, priority); atomicMin(best + v, priority);
}

__device__ unsigned owned_sparsify_root(unsigned v, unsigned* parent) {
    for (;;) { const auto p = atomicAdd(parent + v, 0U); if (p == v) return v; v = p; }
}

__global__ void owned_sparsify_contract(node_index n,
        const gpu_round_shadow_incidence* edges, const std::uint32_t* labels,
        const unsigned long long* best, std::uint32_t* parent,
        unsigned* backbone, unsigned* counts) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= n || labels[i] != i || best[i] == ~0ULL) return;
    const auto id = static_cast<unsigned>(best[i]); const auto e = edges[id];
    if (atomicExch(backbone + id, 1U) == 0) atomicAdd(counts + 1, 1U);
    // Unique strict priorities imply an acyclic Boruvka choice set and the
    // same unique forest as ordered Kruskal. Monotone parent IDs prevent races
    // from introducing parent cycles; every chosen edge is actually united.
    for (;;) {
        const auto u = owned_sparsify_root(unsigned(e.owner), parent);
        const auto v = owned_sparsify_root(unsigned(e.neighbor), parent);
        if (u == v) return;
        const auto high = u > v ? u : v, low = u > v ? v : u;
        if (atomicCAS(parent + high, high, low) == high) { atomicAdd(counts, 1U); return; }
    }
}

__global__ void owned_sparsify_labels(node_index n,
        const std::uint32_t* parent, std::uint32_t* labels) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    auto v = unsigned(i); while (parent[v] != v) v = parent[v]; labels[i] = v;
}

__global__ void owned_sparsify_importance(const gpu_round_shadow_incidence* edges,
        std::size_t count, double* importance) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count) importance[i] = __dsqrt_rn(edges[i].weight);
}

__global__ void owned_sparsify_partials(std::size_t count,
        const unsigned* backbone, const double* importance,
        const owned_sparsify_scale* scale, double* partials, bool initial) {
    const auto block = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    const auto begin = block * 16384;
    if (begin >= count || (!initial && !scale->advance)) return;
    const auto end = begin + ((count - begin < 16384) ? count - begin : 16384);
    double sum = 0.0;
    for (auto i = begin; i < end; ++i) {
        if (backbone[i]) continue; // Literal skip; never compact off-tree items.
        const double term = initial ? importance[i] : fmin(1.0, __dmul_rn(scale->value, importance[i]));
        sum = __dadd_rn(sum, term);
    }
    partials[block] = sum;
}

__global__ void owned_sparsify_fold(std::size_t blocks, const double* partials,
        owned_sparsify_scale* scale, bool initial) {
    if (blockIdx.x || threadIdx.x || (!initial && !scale->advance)) return;
    double sum = 0.0;
    for (std::size_t i = 0; i < blocks; ++i) sum = __dadd_rn(sum, partials[i]);
    if (initial) scale->value = scale->target > 0.0 && sum > 0.0 ? __ddiv_rn(scale->target, sum) : 0.0;
    else {
        ++scale->passes;
        if (sum <= 0.0) { scale->advance = 0; return; }
        scale->value = __dmul_rn(scale->value, __ddiv_rn(scale->target, sum));
    }
    scale->advance = scale->value > 0.0;
}

__device__ double owned_sparsify_draw(node_index u, node_index v, std::uint64_t seed) {
    auto z = seed ^ (std::uint64_t(u) * 0x9E3779B97F4A7C15ULL) ^
        (std::uint64_t(v) * 0xBF58476D1CE4E5B9ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; z ^= z >> 31;
    return __dmul_rn(double(z >> 11), 0x1.0p-53);
}

__global__ void owned_sparsify_sample(const gpu_round_shadow_incidence* edges,
        std::size_t count, const unsigned* backbone, double* importance_probability,
        const owned_sparsify_scale* scale, double override_scale, std::uint64_t seed,
        std::uint8_t* kept, unsigned* error) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    const auto e = edges[i]; const double lambda = override_scale >= 0.0 ? override_scale : scale->value;
    const double p = backbone[i] ? 1.0 : fmin(1.0, __dmul_rn(lambda, importance_probability[i]));
    importance_probability[i] = p;
    kept[i] = p >= 1.0 || owned_sparsify_draw(e.owner, e.neighbor, seed) < p;
    if (!isfinite(lambda) || !isfinite(p) || p < 0.0) atomicExch(error, 1U);
    if (!kept[i]) return; // p==0 never divides, including an exactly-zero draw.
    const double value = backbone[i] ? e.weight : __dmul_rn(e.weight, __ddiv_rn(1.0, p));
    if (!isfinite(value) || !isfinite(double(static_cast<pool_value_t>(value)))) atomicExch(error, 1U);
}

struct owned_sparsify_diagnostic_partial {
    double total, backbone, expected, minimum;
};

__global__ void owned_sparsify_statistics_partials(
        const gpu_round_shadow_incidence* edges, std::size_t count,
        const unsigned* backbone, const double* probabilities,
        owned_sparsify_diagnostic_partial* partials) {
    __shared__ double sums[4][kBlock];
    const auto begin = std::size_t(blockIdx.x) * 16384;
    const auto end = begin + ((count - begin < 16384) ? count - begin : 16384);
    const auto lane = threadIdx.x;
    double total = 0.0, tree = 0.0, expected = 0.0, minimum = 1.0;
    for (auto i = begin + lane; i < end; i += kBlock) {
        total = __dadd_rn(total, edges[i].weight);
        expected = __dadd_rn(expected, probabilities[i]);
        if (backbone[i]) tree = __dadd_rn(tree, edges[i].weight);
        else minimum = fmin(minimum, probabilities[i]);
    }
    sums[0][lane] = total; sums[1][lane] = tree;
    sums[2][lane] = expected; sums[3][lane] = minimum;
    __syncthreads();
    for (unsigned stride = kBlock / 2; stride; stride /= 2) {
        if (lane < stride) {
            for (unsigned k = 0; k < 3; ++k)
                sums[k][lane] = __dadd_rn(sums[k][lane], sums[k][lane + stride]);
            sums[3][lane] = fmin(sums[3][lane], sums[3][lane + stride]);
        }
        __syncthreads();
    }
    if (!lane) partials[blockIdx.x] = {sums[0][0], sums[1][0], sums[2][0], sums[3][0]};
}

__global__ void owned_sparsify_statistics_fold(std::size_t blocks,
        const owned_sparsify_diagnostic_partial* partials,
        gpu_owned_sparsify_stats* stats) {
    if (blockIdx.x || threadIdx.x) return;
    double total = 0.0, tree = 0.0, expected = 0.0, minimum = 1.0;
    for (std::size_t i = 0; i < blocks; ++i) {
        total = __dadd_rn(total, partials[i].total);
        tree = __dadd_rn(tree, partials[i].backbone);
        expected = __dadd_rn(expected, partials[i].expected);
        minimum = fmin(minimum, partials[i].minimum);
    }
    // These three aggregate sums are diagnostics: fixed parallel reduction
    // order, not bitwise CPU serial sums. They never feed the sampling law.
    stats->total_weight = total; stats->backbone_weight = tree;
    stats->expected_kept_edges = expected; stats->min_offtree_probability = minimum;
    stats->max_inverse_probability = __ddiv_rn(1.0, minimum);
}

__global__ void owned_sparsify_expand(const gpu_round_shadow_incidence* edges,
        const unsigned* backbone, const double* probabilities,
        const std::uint32_t* ids, std::size_t count, gpu_round_shadow_incidence* output) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    const auto j = ids[i]; auto e = edges[j];
    if (!backbone[j]) e.weight = double(static_cast<pool_value_t>(
        __dmul_rn(e.weight, __ddiv_rn(1.0, probabilities[j]))));
    output[2 * i] = e; output[2 * i + 1] = {e.neighbor, e.owner, e.weight};
}

__global__ void owned_sparsify_backbone_bytes(const unsigned* source,
        std::size_t count, std::uint8_t* target) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < count) target[i] = source[i] != 0;
}

gpu_owned_sparsify_stats sparsify_owned_residual_device(
        node_index n, std::size_t live, const gpu_round_shadow_incidence* input,
        const std::uint8_t* active, double keep_probability, std::uint64_t seed,
        gpu_round_shadow_incidence* output, std::uint32_t* output_offsets,
        std::uint32_t* output_degrees, allocation_tracker& tracker,
        gpu_owned_sparsify_test_output* test = nullptr, double scale_override = -1.0) {
    if (std::size_t(n) > INT_MAX || live > INT_MAX || live % 2 ||
        !std::isfinite(keep_probability) || !std::isfinite(scale_override) ||
        (live && (!input || !output || input == output || !active)) ||
        !output_offsets || (n && !output_degrees))
        throw std::invalid_argument("GPU owned sparsification dimensions or pointers are invalid");
    keep_probability = std::clamp(keep_probability, 1e-6, 1.0);
    gpu_owned_sparsify_stats result{}; result.physical_before = live / 2;
    result.importance_scale = scale_override >= 0.0 ? scale_override : 0.0;
    if (test) { test->canonical.clear(); test->backbone.clear(); test->kept.clear(); test->probabilities.clear(); }
    if (!live) {
        cuda_check(cudaMemset(output_offsets, 0, (std::size_t(n) + 1) * sizeof(std::uint32_t)), "clear empty sparsification offsets");
        if (n) cuda_check(cudaMemset(output_degrees, 0, std::size_t(n) * sizeof(std::uint32_t)), "clear empty sparsification degrees");
        cuda_check(cudaDeviceSynchronize(), "finish empty sparsification");
        result.peak_device_bytes = tracker.peak; if (test) test->stats = result; return result;
    }
    const auto edges = live / 2; const auto max_blocks = (edges + 16383) / 16384;
    device_buffer<gpu_round_shadow_incidence> canonical(tracker), scratch(tracker), expanded(tracker);
    device_buffer<std::uint8_t> flags(tracker);
    device_buffer<std::uint64_t> keys_a(tracker), keys_b(tracker);
    device_buffer<std::uint32_t> ids(tracker), ordered_ids(tracker), parent(tracker), labels(tracker);
    device_buffer<unsigned> backbone(tracker), counts(tracker);
    device_buffer<unsigned long long> best(tracker);
    device_buffer<double> importance(tracker), partials(tracker);
    device_buffer<owned_sparsify_scale> scale(tracker);
    device_buffer<gpu_owned_sparsify_stats> device_stats(tracker);
    device_buffer<owned_sparsify_diagnostic_partial> diagnostic_partials(tracker);
    device_buffer<std::byte> temporary(tracker); device_buffer<int> selected_count(tracker);
    std::size_t cub_bytes = 0, bytes = 0;
    const auto note = [&] { cub_bytes = std::max(cub_bytes, bytes); bytes = 0; };
    for (const auto count : {edges, live}) {
        cuda_check(cub::DeviceSelect::Flagged(nullptr, bytes,
            static_cast<const gpu_round_shadow_incidence*>(nullptr), static_cast<const std::uint8_t*>(nullptr),
            static_cast<gpu_round_shadow_incidence*>(nullptr), static_cast<int*>(nullptr), int(count)), "query sparsification incidence select"); note();
        cuda_check(cub::DeviceRadixSort::SortPairs(nullptr, bytes,
            static_cast<const std::uint64_t*>(nullptr), static_cast<std::uint64_t*>(nullptr),
            static_cast<const gpu_round_shadow_incidence*>(nullptr), static_cast<gpu_round_shadow_incidence*>(nullptr), int(count)), "query sparsification incidence sort"); note();
    }
    cuda_check(cub::DeviceSelect::Flagged(nullptr, bytes, static_cast<const std::uint32_t*>(nullptr),
        static_cast<const std::uint8_t*>(nullptr), static_cast<std::uint32_t*>(nullptr), static_cast<int*>(nullptr), int(edges)), "query sparsification ID select"); note();
    cuda_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, output_degrees, output_offsets, int(n)), "query sparsification degree scan"); note();
    // Conservative sum of all phase allocations (larger than the live maximum).
    // Existing input, spare, offsets, degrees, factor log and retained scratch
    // already consume free memory and remain in the caller's allocation tracker.
    std::size_t required = cub_bytes;
    required = add_bytes(required, edges, 2 * sizeof(gpu_round_shadow_incidence) + 3 * sizeof(unsigned) + sizeof(double), "sparsification canonical workspace");
    required = add_bytes(required, live, sizeof(gpu_round_shadow_incidence) + 2 * sizeof(std::uint64_t) + sizeof(std::uint8_t), "sparsification paired workspace");
    required = add_bytes(required, n, 2 * sizeof(unsigned) + sizeof(unsigned long long), "sparsification forest workspace");
    required = add_bytes(required, max_blocks, sizeof(double) + sizeof(owned_sparsify_diagnostic_partial), "sparsification reduction blocks");
    required = add_bytes(required, 1, sizeof(owned_sparsify_scale) + sizeof(gpu_owned_sparsify_stats) + 3 * sizeof(unsigned), "sparsification scalar workspace");
    std::size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "query sparsification workspace fit");
    if (required > free_bytes || total_bytes / 10 > free_bytes - required)
        throw std::runtime_error("GPU owned sparsification workspace does not fit with device margin");
    canonical.allocate(edges, "allocate sparsification canonical edges");
    scratch.allocate(edges, "allocate sparsification coalescing scratch");
    flags.allocate(live, "allocate sparsification flags");
    keys_a.allocate(live, "allocate sparsification keys"); keys_b.allocate(live, "allocate sparsification sorted keys");
    counts.allocate(2, "allocate sparsification status"); selected_count.allocate(1, "allocate sparsification CUB count");
    temporary.allocate(cub_bytes, "allocate sparsification CUB workspace"); cub_workspace cub(temporary, selected_count);
    const auto check_error = [&] {
        unsigned error = 0; copy_to_host(&error, counts.get(), 1, "download sparsification validation");
        result.scalar_download_bytes += sizeof(error);
        if (error) throw std::runtime_error("GPU owned sparsification has invalid or overflowing edge weights");
    };
    cuda_check(cudaMemset(counts.get(), 0, 2 * sizeof(unsigned)), "clear sparsification validation");
    owned_sparsify_canonical_flags<<<blocks_for(live), kBlock>>>(input, live, n, active, flags.get(), counts.get());
    cuda_check(cudaGetLastError(), "validate sparsification input");
    // Prove the half-sized canonical output capacity before CUB writes it;
    // malformed orientation counts must not turn a rejected input into an OOB.
    check_error();
    unsigned canonical_count = 0;
    copy_to_host(&canonical_count, counts.get() + 1, 1, "download canonical capacity check");
    result.scalar_download_bytes += sizeof(canonical_count);
    if (canonical_count != edges) throw std::logic_error("GPU owned sparsification requires paired input");
    const int count = cub.select(input, flags.get(), canonical.get(), live, "select canonical sparsification incidences");
    result.scalar_download_bytes += sizeof(int); check_error();
    if (count != int(edges)) throw std::logic_error("GPU owned sparsification requires paired input");
    owned_sparsify_keys<<<blocks_for(edges), kBlock>>>(canonical.get(), edges, keys_a.get(), false);
    cuda_check(cudaGetLastError(), "build sparsification weight keys");
    cub.sort_pairs(keys_a.get(), keys_b.get(), canonical.get(), scratch.get(), edges, "sort sparsification weights");
    owned_sparsify_keys<<<blocks_for(edges), kBlock>>>(scratch.get(), edges, keys_a.get(), true);
    cuda_check(cudaGetLastError(), "build sparsification endpoint keys");
    cub.sort_pairs(keys_a.get(), keys_b.get(), scratch.get(), canonical.get(), edges, "sort sparsification endpoint pairs");
    owned_sparsify_coalesce<<<blocks_for(edges), kBlock>>>(canonical.get(), edges, scratch.get(), flags.get(), counts.get());
    cuda_check(cudaGetLastError(), "coalesce sparsification duplicate pairs");
    const int distinct_count = cub.select(scratch.get(), flags.get(), canonical.get(), edges, "compact coalesced canonical edges");
    if (distinct_count <= 0 || std::size_t(distinct_count) > edges)
        throw std::logic_error("GPU sparsification returned invalid coalesced count");
    const auto distinct = std::size_t(distinct_count);
    result.scalar_download_bytes += sizeof(int); check_error(); result.distinct_before = distinct;
    scratch.release("retire sparsification coalescing scratch");
    if (test) { test->canonical.resize(distinct); copy_to_host(test->canonical.data(), canonical.get(), distinct, "download test canonical edges"); }
    backbone.allocate(distinct, "allocate sparsification backbone");
    parent.allocate(n, "allocate sparsification parents"); labels.allocate(n, "allocate sparsification labels"); best.allocate(n, "allocate sparsification choices");
    ids.allocate(distinct, "allocate sparsification canonical IDs"); ordered_ids.allocate(distinct, "allocate sparsification ordered IDs");
    cuda_check(cudaMemset(backbone.get(), 0, distinct * sizeof(unsigned)), "clear sparsification backbone");
    owned_sparsify_initialize<<<blocks_for(n), kBlock>>>(n, parent.get(), labels.get());
    owned_sparsify_ids<<<blocks_for(distinct), kBlock>>>(distinct, ids.get());
    cuda_check(cudaGetLastError(), "initialize sparsification forest and IDs");
    for (std::size_t round = 0;; ++round) {
        if (round > std::size_t(n)) throw std::logic_error("GPU sparsification forest did not converge");
        cuda_check(cudaMemset(best.get(), 0xff, std::size_t(n) * sizeof(unsigned long long)), "clear sparsification choices");
        cuda_check(cudaMemset(counts.get(), 0, 2 * sizeof(unsigned)), "clear sparsification contraction counts");
        owned_sparsify_best<<<blocks_for(distinct), kBlock>>>(canonical.get(), distinct, labels.get(), best.get());
        owned_sparsify_contract<<<blocks_for(n), kBlock>>>(n, canonical.get(), labels.get(), best.get(), parent.get(), backbone.get(), counts.get());
        cuda_check(cudaGetLastError(), "contract strict-priority sparsification forest");
        unsigned progress[2]{}; copy_to_host(progress, counts.get(), 2, "download sparsification forest progress"); result.scalar_download_bytes += sizeof(progress);
        if (progress[0] != progress[1]) throw std::logic_error("GPU sparsification choices were not a forest");
        if (!progress[0]) break;
        result.backbone_edges += progress[0];
        owned_sparsify_labels<<<blocks_for(n), kBlock>>>(n, parent.get(), labels.get());
        cuda_check(cudaGetLastError(), "refresh sparsification component labels");
    }
    parent.release("retire sparsification parents"); labels.release("retire sparsification labels"); best.release("retire sparsification choices");
    importance.allocate(distinct, "allocate sparsification importance");
    result.normalization_blocks = (distinct + 16383) / 16384;
    partials.allocate(result.normalization_blocks, "allocate sparsification ordered partials"); scale.allocate(1, "allocate sparsification scale");
    owned_sparsify_scale host_scale{}; host_scale.target = keep_probability * double(distinct - result.backbone_edges);
    copy_to_device(scale.get(), &host_scale, 1, "initialize sparsification scale");
    owned_sparsify_importance<<<blocks_for(distinct), kBlock>>>(canonical.get(), distinct, importance.get());
    owned_sparsify_partials<<<blocks_for(result.normalization_blocks), kBlock>>>(distinct, backbone.get(), importance.get(), scale.get(), partials.get(), true);
    owned_sparsify_fold<<<1, 1>>>(result.normalization_blocks, partials.get(), scale.get(), true);
    for (unsigned i = 0; i < 6; ++i) {
        owned_sparsify_partials<<<blocks_for(result.normalization_blocks), kBlock>>>(distinct, backbone.get(), importance.get(), scale.get(), partials.get(), false);
        owned_sparsify_fold<<<1, 1>>>(result.normalization_blocks, partials.get(), scale.get(), false);
    }
    cuda_check(cudaGetLastError(), "normalize sparsification importance in CPU order");
    copy_to_host(&host_scale, scale.get(), 1, "download sparsification normalization state"); result.scalar_download_bytes += sizeof(host_scale);
    result.normalization_passes = host_scale.passes;
    result.importance_scale = scale_override >= 0.0 ? scale_override : host_scale.value;
    cuda_check(cudaMemset(counts.get(), 0, 2 * sizeof(unsigned)), "clear sparsification weight validation");
    owned_sparsify_sample<<<blocks_for(distinct), kBlock>>>(canonical.get(), distinct, backbone.get(), importance.get(), scale.get(), scale_override, seed, flags.get(), counts.get());
    cuda_check(cudaGetLastError(), "sample and validate HT sparsification weights"); check_error();
    partials.release("retire sparsification normalization partials");
    diagnostic_partials.allocate(result.normalization_blocks, "allocate sparsification diagnostic partials");
    device_stats.allocate(1, "allocate sparsification diagnostics");
    copy_to_device(device_stats.get(), &result, 1, "initialize sparsification diagnostics");
    owned_sparsify_statistics_partials<<<int(result.normalization_blocks), kBlock>>>(canonical.get(), distinct, backbone.get(), importance.get(), diagnostic_partials.get());
    owned_sparsify_statistics_fold<<<1, 1>>>(result.normalization_blocks, diagnostic_partials.get(), device_stats.get());
    cuda_check(cudaGetLastError(), "compute bounded parallel sparsification diagnostics");
    copy_to_host(&result, device_stats.get(), 1, "download bounded sparsification diagnostics"); result.scalar_download_bytes += sizeof(result);
    // Stable selection preserves the canonical encounter order in the rebuild.
    const int kept_count = cub.select(ids.get(), flags.get(), ordered_ids.get(), distinct, "compact kept sparsification IDs");
    if (kept_count < 0 || std::size_t(kept_count) > distinct)
        throw std::logic_error("GPU sparsification returned invalid kept count");
    result.kept_edges = std::size_t(kept_count);
    result.scalar_download_bytes += sizeof(int);
    if (test) {
        test->kept.resize(distinct); copy_to_host(test->kept.data(), flags.get(), distinct, "download test kept flags");
        test->probabilities.resize(distinct); copy_to_host(test->probabilities.data(), importance.get(), distinct, "download test probabilities");
        owned_sparsify_backbone_bytes<<<blocks_for(distinct), kBlock>>>(backbone.get(), distinct, flags.get());
        cuda_check(cudaGetLastError(), "convert test backbone flags"); test->backbone.resize(distinct);
        copy_to_host(test->backbone.data(), flags.get(), distinct, "download test backbone flags");
    }
    diagnostic_partials.release("retire sparsification diagnostic partials"); scale.release("retire sparsification scale"); device_stats.release("retire sparsification diagnostics");
    const auto directed = 2 * result.kept_edges;
    expanded.allocate(directed, "allocate paired sparsification rebuild");
    cuda_check(cudaMemset(output_degrees, 0, std::size_t(n) * sizeof(std::uint32_t)), "clear sparsification output degrees");
    if (directed) {
        owned_sparsify_expand<<<blocks_for(result.kept_edges), kBlock>>>(canonical.get(), backbone.get(), importance.get(), ordered_ids.get(), result.kept_edges, expanded.get());
        count_residual_degrees<<<blocks_for(directed), kBlock>>>(expanded.get(), directed, output_degrees);
        build_owner_keys<<<blocks_for(directed), kBlock>>>(expanded.get(), directed, keys_a.get());
        cuda_check(cudaGetLastError(), "expand sparsification paired CSR");
        cub.sort_pairs(keys_a.get(), keys_b.get(), expanded.get(), output, directed, "sort sparsification rebuilt owner incidences");
    }
    cub.exclusive_sum(output_degrees, output_offsets, n);
    set_last_offset<<<1, 1>>>(output_offsets, n, std::uint32_t(directed));
    cuda_check(cudaGetLastError(), "finish sparsification row offsets");
    cuda_check(cudaDeviceSynchronize(), "validate sparsification spare before publication");
    result.peak_device_bytes = tracker.peak; if (test) test->stats = result; return result;
}

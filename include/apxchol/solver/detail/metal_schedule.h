#pragma once
#include "apxchol/solver/detail/gpu_dataflow_plan.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <type_traits>
#include <stdexcept>
#include <vector>

namespace apxchol::detail::metal_schedule {

inline bool in_range(double value) {
    const double magnitude = std::abs(value);
    return magnitude == 0 || (magnitude >= 0x1p-100 && magnitude <= 0x1p100);
}

struct step {
    unsigned first, last;
    bool one_batch_per_block = false;
};

struct schedule {
    gpu_dataflow::packed_plan tasks;
    std::vector<step> stages;
    unsigned levels = 0;
};

template<class Value = float, unsigned SparseLanes = 32>
inline schedule build(std::span<const unsigned> ptr, std::span<const unsigned> col,
                      std::span<const std::type_identity_t<Value>> values, bool reverse, std::span<const float> diagonals = {}) {
    if (ptr.empty() || ptr.size() > gpu_dataflow::special_bit || ptr.front() != 0 ||
        ptr.back() != col.size() || col.size() != values.size() ||
        (!diagonals.empty() && diagonals.size() + 1 != ptr.size()))
        throw std::invalid_argument("Metal factor has invalid CSR dimensions");
    const unsigned n = ptr.size() - 1;
    std::vector<unsigned> depth(n);
    unsigned height = 0;
    for (unsigned q = 0; q < n; ++q) {
        const unsigned row = reverse ? n - 1 - q : q;
        if (ptr[row] >= ptr[row + 1] || ptr[row + 1] > col.size())
            throw std::invalid_argument("Metal factor has invalid row pointers");
        const unsigned diagonal = reverse ? ptr[row] : ptr[row + 1] - 1;
        const float d = diagonals.empty() ? float(values[diagonal]) : diagonals[row];
        if (col[diagonal] != row || d == 0)
            throw std::invalid_argument("Metal factor has a missing or zero diagonal");
        for (unsigned p = ptr[row]; p < ptr[row + 1]; ++p) {
            if (!std::isfinite(p == diagonal ? d : float(values[p])) || (p == diagonal && !in_range(d)))
                throw std::invalid_argument("Metal factor has a non-finite value or unsupported diagonal");
            if (p == diagonal)
                continue;
            const unsigned other = col[p];
            if (other >= n || (reverse ? other <= row : other >= row))
                throw std::invalid_argument("Metal factor has a dependency outside its triangle");
            depth[row] = std::max(depth[row], depth[other] + 1);
        }
        height = std::max(height, depth[row] + 1);
    }
    if (height <= 1)
        return {{}, {}, height};
    std::vector<unsigned> block_ptr, rows;
    std::vector<step> stages;
    {
        // Dependencies within a stage share one component and workgroup. A row that
        // would exceed a workgroup's capacity starts a later stage instead.
        constexpr unsigned capacity = 128;
        std::vector<unsigned> parent(n), size(n, 1), stage(n), seen(n);
        std::iota(parent.begin(), parent.end(), 0);
        auto find = [&](unsigned row) {
            while (row != parent[row]) {
                parent[row] = parent[parent[row]];
                row = parent[row];
            }
            return row;
        };
        gpu_dataflow::row_order order{n, reverse};
        std::vector<unsigned> roots;
        unsigned stage_count = 0;
        for (unsigned q = 0; q < n; ++q) {
            const unsigned row = order(q);
            if (!depth[row])
                continue;
            roots.clear();
            for (auto p = ptr[row] + 1; p < ptr[row + 1]; ++p) {
                const auto e = reverse ? p : ptr[row + 1] - 1 - (p - ptr[row]);
                const unsigned other = col[e];
                if (!depth[other] || stage[other] < stage[row])
                    continue;
                if (stage[other] > stage[row]) {
                    stage[row] = stage[other];
                    size[row] = 1;
                    roots.clear();
                }
                const unsigned root = find(other);
                if (seen[root] == row + 1)
                    continue;
                seen[root] = row + 1;
                roots.push_back(root);
                size[row] += size[root];
            }
            if (size[row] > capacity) {
                ++stage[row];
                size[row] = 1;
            } else {
                const unsigned leader = roots.empty() ? row : *std::max_element(roots.begin(), roots.end(),
                    [&](unsigned a, unsigned b) { return size[a] < size[b]; });
                size[leader] = size[row];
                parent[row] = leader;
                for (unsigned root : roots)
                    parent[root] = leader;
            }
            stage_count = std::max(stage_count, stage[row] + 1);
        }
        roots.clear();
        for (unsigned row = 0; row < n; ++row)
            if (depth[row] && parent[row] == row)
                roots.push_back(row);
        auto key = [&](unsigned root) {
            return std::size_t(stage[root]) * (capacity + 1) + capacity - size[root];
        };
        std::vector<unsigned> buckets(std::size_t(stage_count) * (capacity + 1)), ordered(roots.size());
        for (unsigned root : roots)
            ++buckets[key(root)];
        std::exclusive_scan(buckets.begin(), buckets.end(), buckets.begin(), 0u);
        for (unsigned root : roots)
            ordered[buckets[key(root)]++] = root;
        auto &cursor = seen; // Visit stamps are no longer needed.
        block_ptr.push_back(0);
        rows.resize(n);
        unsigned written = 0, component = 0;
        for (unsigned st = 0; st < stage_count; ++st) {
            step launch{unsigned(block_ptr.size() - 1), 0};
            while (component < ordered.size() && stage[ordered[component]] == st) {
                const unsigned root = ordered[component++];
                if (written - block_ptr.back() + size[root] > capacity)
                    block_ptr.push_back(written);
                cursor[root] = written;
                written += size[root];
            }
            if (block_ptr.back() != written)
                block_ptr.push_back(written);
            launch.last = block_ptr.size() - 1;
            stages.push_back(launch);
        }
        rows.resize(written);
        for (unsigned q = 0; q < n; ++q) {
            const unsigned row = order(q);
            if (depth[row])
                rows[cursor[find(row)]++] = row;
        }
    }
    return {gpu_dataflow::pack<SparseLanes>(ptr.data(), {n, reverse, rows.data()}, block_ptr),
            std::move(stages), height};
}
} // namespace apxchol::detail::metal_schedule

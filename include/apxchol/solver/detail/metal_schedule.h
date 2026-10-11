#pragma once
#include "apxchol/solver/detail/gpu_dataflow_plan.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace apxchol::detail::metal_schedule {

inline bool in_range(double value) {
    const double magnitude = std::abs(value);
    return magnitude == 0 || (magnitude >= 0x1p-100 && magnitude <= 0x1p100);
}

namespace detail {
// Merge consecutive levels while every connected component fits one workgroup.
// A failed merge is rolled back; no dependency may cross workgroups in a stage.
struct step {
    unsigned first, last;
};
struct plan {
    std::vector<unsigned> block_ptr{0}, rows;
    std::vector<step> steps;
};

template <class P, class I>
inline plan partition(const P &ptr, const I &col, const std::vector<unsigned> &levels,
                      const std::vector<unsigned> &rows, const std::vector<unsigned> &depth,
                      unsigned capacity) {
    const unsigned n = rows.size();
    std::vector<unsigned> parent(n), size(n, 1);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](unsigned x) {
        while (x != parent[x])
            x = parent[x];
        return x;
    };
    std::vector<unsigned> undo;
    std::vector<int> head(n, -1), next(n);
    plan out;
    out.rows.resize(n);
    unsigned written = 0;
    out.block_ptr.reserve(n / capacity + 16);
    auto flush = [&](unsigned first, unsigned last) {
        std::vector<unsigned> roots;
        for (unsigned s = levels[first]; s < levels[last]; ++s) {
            unsigned row = rows[s], root = find(row);
            if (head[root] < 0)
                roots.push_back(root);
            next[row] = head[root];
            head[root] = row;
        }
        std::vector<unsigned> buckets(capacity + 1, 0), ordered(roots.size());
        for (unsigned root : roots)
            ++buckets[capacity - size[root]];
        std::exclusive_scan(buckets.begin(), buckets.end(), buckets.begin(), 0u);
        for (unsigned root : roots)
            ordered[buckets[capacity - size[root]]++] = root;
        roots.swap(ordered);
        step launch{unsigned(out.block_ptr.size() - 1), 0};
        for (unsigned root : roots) {
            if (written - out.block_ptr.back() + size[root] > capacity)
                out.block_ptr.push_back(written);
            written += size[root];
            auto position = written;
            for (int row = head[root]; row >= 0; row = next[row])
                out.rows[--position] = row;
            head[root] = -1;
        }
        if (out.block_ptr.back() != written)
            out.block_ptr.push_back(written);
        launch.last = out.block_ptr.size() - 1;
        out.steps.push_back(launch);
        for (unsigned s = levels[first]; s < levels[last]; ++s) {
            parent[rows[s]] = rows[s];
            size[rows[s]] = 1;
        }
    };
    for (unsigned first = 1, l = 1; l + 1 < levels.size();) {
        bool fits = true;
        for (unsigned s = levels[l]; fits && s < levels[l + 1]; ++s) {
            unsigned row = rows[s];
            for (auto e = ptr[row]; e < ptr[row + 1]; ++e) {
                unsigned other = col[e];
                if (other == row || depth[other] < first)
                    continue;
                if (depth[other] >= l)
                    throw std::runtime_error("non-topological levels");
                unsigned a = find(row), b = find(other);
                if (a == b)
                    continue;
                if (size[a] < size[b])
                    std::swap(a, b);
                undo.push_back(b);
                parent[b] = a;
                size[a] += size[b];
                if (size[a] > capacity) {
                    fits = false;
                    break;
                }
            }
        }
        if (!fits) {
            while (!undo.empty()) {
                const unsigned child = undo.back(), root = parent[child];
                undo.pop_back();
                parent[child] = child;
                size[root] -= size[child];
            }
            flush(first, l);
            first = l;
        } else {
            undo.clear();
            ++l;
            if (l + 1 == levels.size())
                flush(first, l);
        }
    }
    out.rows.resize(written);
    return out;
}
} // namespace detail

struct schedule {
    gpu_dataflow::packed_plan tasks;
    std::vector<detail::step> stages;
    unsigned levels = 0;
};

inline schedule build(std::span<const unsigned> ptr, std::span<const unsigned> col,
                      std::span<const float> values, bool reverse) {
    if (ptr.empty() || ptr.size() > gpu_dataflow::special_bit || ptr.front() != 0 ||
        ptr.back() != col.size() || col.size() != values.size())
        throw std::invalid_argument("Metal factor has invalid CSR dimensions");
    const unsigned n = ptr.size() - 1;
    std::vector<unsigned> depth(n);
    unsigned height = 0;
    for (unsigned q = 0; q < n; ++q) {
        const unsigned row = reverse ? n - 1 - q : q;
        if (ptr[row] >= ptr[row + 1] || ptr[row + 1] > col.size())
            throw std::invalid_argument("Metal factor has invalid row pointers");
        const unsigned diagonal = reverse ? ptr[row] : ptr[row + 1] - 1;
        if (col[diagonal] != row || values[diagonal] == 0)
            throw std::invalid_argument("Metal factor has a missing or zero diagonal");
        for (unsigned p = ptr[row]; p < ptr[row + 1]; ++p) {
            if (!std::isfinite(values[p]) || (p == diagonal && !in_range(values[p])))
                throw std::domain_error("Metal factor has a non-finite value or unsupported diagonal");
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
    std::vector<unsigned> levels(height + 1), rows(n);
    for (unsigned d : depth)
        ++levels[d + 1];
    std::partial_sum(levels.begin(), levels.end(), levels.begin());
    auto cursor = levels;
    for (unsigned row = 0; row < n; ++row)
        rows[cursor[depth[row]]++] = row;
    auto groups = detail::partition(ptr, col, levels, rows, depth, 128);
    return {gpu_dataflow::pack(ptr.data(), {n, reverse, groups.rows.data()}, groups.block_ptr),
            std::move(groups.steps), height};
}
} // namespace apxchol::detail::metal_schedule

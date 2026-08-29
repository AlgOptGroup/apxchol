#pragma once

// A cheap stale-synchronous SpTRSV schedule for elimination factors.
//
// Rows are visited once in factor order. A row follows a processor owning one
// of its latest parents, which minimizes its earliest legal staleness-2 step:
// same-processor dependencies may execute in the same step (the processor's
// rows stay in topological order), while dependencies owned by another
// processor must be at least two steps old. Ties go to the least-loaded
// destination slot. Roots are assigned round-robin.
// The factor scan is O(nnz+n); flattening the per-processor supersteps adds
// O(P*steps), with P bounded by the OpenMP team. There is no search/refinement
// and no external runtime.
//
// This is an independently implemented critical-parent heuristic motivated by
// the weak-barrier execution model in "Elasticity in Parallel Sparse
// Triangular Solve" (arXiv:2607.02324), not an implementation of that paper's
// GrowLocalSSP scheduler.

#include "apxchol/types.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace apxchol::detail {

enum class cpu_sptrsv_schedule_mode { automatic, levels };

inline cpu_sptrsv_schedule_mode cpu_sptrsv_schedule_from_env() {
    const char* value = std::getenv("APXCHOL_CPU_SPTRSV");
    if (value == nullptr || *value == '\0' || std::string_view(value) == "auto")
        return cpu_sptrsv_schedule_mode::automatic;
    if (std::string_view(value) == "levels")
        return cpu_sptrsv_schedule_mode::levels;
    throw std::invalid_argument(
        "APXCHOL_CPU_SPTRSV must be auto or levels");
}

struct critical_schedule_plan {
    unsigned processors = 0;
    std::size_t steps = 0;
    node_index first_row = 0;
    node_index row_count = 0;
    std::vector<std::size_t> forward_ptr;
    std::vector<node_index> forward_rows;

    std::size_t slot(unsigned processor, std::size_t step) const {
        return static_cast<std::size_t>(processor) * steps + step;
    }

    std::size_t memory_bytes() const {
        return forward_ptr.capacity() * sizeof(std::size_t) +
               forward_rows.capacity() * sizeof(node_index);
    }

    void clear() {
        processors = 0;
        steps = 0;
        first_row = 0;
        row_count = 0;
        std::vector<std::size_t>().swap(forward_ptr);
        std::vector<node_index>().swap(forward_rows);
    }
};

class critical_weak_barrier {
    static constexpr std::size_t cache_line = 128;
    struct alignas(cache_line) counter_slot {
        std::atomic<std::uint64_t> value{0};
    };

public:
    void setup(unsigned processors) {
        processors_ = processors;
        stride_ = ((processors + 15u) / 16u) * 16u;
        counters_ = std::make_unique<counter_slot[]>(processors);
        cache_.assign(static_cast<std::size_t>(processors) * stride_, 0);
    }

    void clear() {
        processors_ = 0;
        stride_ = 0;
        counters_.reset();
        std::vector<std::uint64_t>().swap(cache_);
    }

    void wait(unsigned processor, std::uint64_t staleness) const {
        std::uint64_t* local = cache_.data() +
            static_cast<std::size_t>(processor) * stride_;
        const std::uint64_t target =
            std::max(local[processor], staleness) - staleness;
        for (unsigned other = 0; other < processors_; ++other) {
            while (local[other] < target) {
                local[other] =
                    counters_[other].value.load(std::memory_order_acquire);
                if (local[other] >= target) break;
                cpu_relax();
            }
        }
    }

    void arrive(unsigned processor) const {
        const std::uint64_t value = counters_[processor].value.fetch_add(
            1, std::memory_order_release) + 1;
        cache_[static_cast<std::size_t>(processor) * stride_ + processor] =
            value;
    }

    std::size_t memory_bytes() const {
        return static_cast<std::size_t>(processors_) * sizeof(counter_slot) +
               cache_.capacity() * sizeof(std::uint64_t);
    }

private:
    static void cpu_relax() {
#if defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
        asm volatile("pause" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }

    unsigned processors_ = 0;
    unsigned stride_ = 0;
    mutable std::unique_ptr<counter_slot[]> counters_;
    mutable std::vector<std::uint64_t> cache_;
};

struct critical_assignments {
    std::vector<unsigned> processor;
    std::vector<std::size_t> step;
    std::size_t steps = 0;
};

inline critical_assignments assign_critical_schedule(
        node_index rows, const edge_index* csr_ptr,
        const node_index* csr_idx, unsigned processors,
        node_index first_row = 0) {
    if (processors == 0)
        throw std::invalid_argument("critical SpTRSV schedule: zero processors");
    if (first_row > rows)
        throw std::invalid_argument("critical SpTRSV schedule: invalid row range");

    constexpr std::size_t unset = std::numeric_limits<std::size_t>::max();
    critical_assignments out;
    const node_index row_count = rows - first_row;
    out.processor.resize(row_count);
    out.step.resize(row_count);
    std::vector<std::vector<std::uint64_t>> slot_work(processors);
    std::vector<std::size_t> max_by_processor(processors, unset);
    std::vector<std::uint64_t> total_work(processors, 0);
    std::vector<unsigned> touched;
    touched.reserve(processors);
    unsigned next_root = 0;

    for (node_index v = first_row; v < rows; ++v) {
        const node_index local_v = v - first_row;
        const edge_index begin = csr_ptr[v];
        const edge_index end = csr_ptr[v + 1] - 1;  // diagonal is last
        const std::uint64_t work =
            static_cast<std::uint64_t>(end - begin) + 1;
        unsigned chosen = 0;
        std::size_t chosen_step = 0;
        edge_index internal_begin = begin;
        while (internal_begin < end && csr_idx[internal_begin] < first_row)
            ++internal_begin;
        if (internal_begin == end) {
            chosen = next_root++ % processors;
        } else {
            std::size_t latest = 0;
            touched.clear();
            for (edge_index edge = internal_begin; edge < end; ++edge) {
                const node_index parent = csr_idx[edge];
                const node_index local_parent = parent - first_row;
                const unsigned processor = out.processor[local_parent];
                const std::size_t step = out.step[local_parent];
                if (max_by_processor[processor] == unset) {
                    touched.push_back(processor);
                    max_by_processor[processor] = step;
                } else {
                    max_by_processor[processor] =
                        std::max(max_by_processor[processor], step);
                }
                latest = std::max(latest, step);
            }

            unsigned latest_owners = 0;
            std::size_t second = unset;
            for (unsigned processor : touched) {
                const std::size_t step = max_by_processor[processor];
                if (step == latest)
                    ++latest_owners;
                else if (second == unset || step > second)
                    second = step;
            }

            std::uint64_t best_slot_work =
                std::numeric_limits<std::uint64_t>::max();
            std::uint64_t best_total_work =
                std::numeric_limits<std::uint64_t>::max();
            for (unsigned processor : touched) {
                if (max_by_processor[processor] != latest) continue;
                const std::size_t outside =
                    latest_owners > 1 ? latest : second;
                const std::size_t step = outside == unset
                    ? latest : std::max(latest, outside + 2);
                const std::uint64_t assigned = step < slot_work[processor].size()
                    ? slot_work[processor][step] : 0;
                if (assigned < best_slot_work ||
                    (assigned == best_slot_work &&
                     total_work[processor] < best_total_work) ||
                    (assigned == best_slot_work &&
                     total_work[processor] == best_total_work &&
                     processor < chosen)) {
                    chosen = processor;
                    chosen_step = step;
                    best_slot_work = assigned;
                    best_total_work = total_work[processor];
                }
            }
            for (unsigned processor : touched)
                max_by_processor[processor] = unset;
        }

        out.processor[local_v] = chosen;
        out.step[local_v] = chosen_step;
        if (slot_work[chosen].size() <= chosen_step)
            slot_work[chosen].resize(chosen_step + 1, 0);
        slot_work[chosen][chosen_step] += work;
        total_work[chosen] += work;
        out.steps = std::max(out.steps, chosen_step + 1);
    }
    return out;
}

inline bool critical_assignments_are_valid(
        const critical_assignments& schedule, node_index rows,
        const edge_index* csr_ptr, const node_index* csr_idx,
        node_index first_row = 0) {
    if (first_row > rows || schedule.processor.size() != rows - first_row ||
        schedule.step.size() != rows - first_row)
        return false;
    for (node_index v = first_row; v < rows; ++v) {
        const node_index local_v = v - first_row;
        const unsigned vp = schedule.processor[local_v];
        const std::size_t vs = schedule.step[local_v];
        for (edge_index edge = csr_ptr[v]; edge < csr_ptr[v + 1] - 1; ++edge) {
            const node_index parent = csr_idx[edge];
            if (parent < first_row) continue;  // completed bulk prefix
            const node_index local_parent = parent - first_row;
            const unsigned up = schedule.processor[local_parent];
            const std::size_t us = schedule.step[local_parent];
            const bool valid = up == vp
                ? (us < vs || (us == vs && parent < v))
                : (us + 2 <= vs);
            if (!valid) return false;
        }
    }
    return true;
}

inline critical_schedule_plan flatten_critical_schedule(
        const critical_assignments& schedule, node_index rows,
        unsigned processors, node_index first_row = 0) {
    critical_schedule_plan out;
    out.processors = processors;
    out.steps = schedule.steps;
    out.first_row = first_row;
    out.row_count = rows - first_row;
    if (out.steps != 0 && processors >
            std::numeric_limits<std::size_t>::max() / out.steps)
        throw std::overflow_error("critical SpTRSV schedule is too large");
    const std::size_t slots = static_cast<std::size_t>(processors) * out.steps;
    std::vector<std::size_t> forward_count(slots, 0);
    for (node_index v = first_row; v < rows; ++v) {
        const node_index local_v = v - first_row;
        const unsigned processor = schedule.processor[local_v];
        const std::size_t step = schedule.step[local_v];
        ++forward_count[static_cast<std::size_t>(processor) * out.steps + step];
    }

    auto prefix = [](const std::vector<std::size_t>& count) {
        std::vector<std::size_t> ptr(count.size() + 1, 0);
        for (std::size_t i = 0; i < count.size(); ++i)
            ptr[i + 1] = ptr[i] + count[i];
        return ptr;
    };
    out.forward_ptr = prefix(forward_count);
    out.forward_rows.resize(out.row_count);
    auto forward_next = out.forward_ptr;
    for (node_index v = first_row; v < rows; ++v) {
        const node_index local_v = v - first_row;
        const unsigned processor = schedule.processor[local_v];
        const std::size_t step = schedule.step[local_v];
        const std::size_t slot =
            static_cast<std::size_t>(processor) * out.steps + step;
        out.forward_rows[forward_next[slot]++] = v;
    }
    // Rows are ascending inside every forward (processor, step) slot. The
    // backward executor maps step -> steps-1-step and traverses that same slot
    // in reverse, reproducing the former backward_rows order without storing a
    // second pointer table or a second copy of every tail row.
    return out;
}

inline critical_schedule_plan build_critical_schedule(
        node_index rows, const edge_index* csr_ptr,
        const node_index* csr_idx, unsigned processors,
        node_index first_row = 0) {
    auto assignments =
        assign_critical_schedule(rows, csr_ptr, csr_idx, processors, first_row);
#ifndef NDEBUG
    if (!critical_assignments_are_valid(
            assignments, rows, csr_ptr, csr_idx, first_row))
        throw std::logic_error("critical SpTRSV schedule violates a dependency");
#endif
    return flatten_critical_schedule(assignments, rows, processors, first_row);
}

}  // namespace apxchol::detail

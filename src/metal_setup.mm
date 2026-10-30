#include "metal_setup.h"
#include "metal_setup_options.h"
#include "apxchol/solver/partition/block_greedy.h"
#include "apxchol/operator_class.h"
#include "metal_setup/finalizer.h"
#import <Foundation/Foundation.h>
#include <limits>

namespace apxchol::detail::metal_setup {
namespace {

struct input_buffers {
    id<MTLBuffer> ptr, idx, values;
    bool operator_exact = true;
    unsigned max_operator_row = 0;
    float preconditioner_scale = 1;
};

id<MTLBuffer> buffer(id<MTLDevice> device, std::size_t bytes) {
    auto out = [device newBufferWithLength:std::max(bytes, std::size_t(16))
                                   options:MTLResourceStorageModeShared];
    if (!out)
        throw std::bad_alloc();
    return out;
}

input_buffers upload(id<MTLDevice> device, const Eigen::SparseMatrix<double> &A) {
    const auto n = A.rows(), nnz = A.nonZeros();
    input_buffers out{buffer(device, (n + 1) * sizeof(unsigned)),
                      buffer(device, nnz * sizeof(unsigned)),
                      buffer(device, nnz * sizeof(metal_host::df))};
    std::memcpy(out.ptr.contents, A.outerIndexPtr(), (n + 1) * sizeof(unsigned));
    std::memcpy(out.idx.contents, A.innerIndexPtr(), nnz * sizeof(unsigned));
    auto values = static_cast<metal_host::df *>(out.values.contents);
    double min_diagonal = std::numeric_limits<double>::infinity(), max_diagonal = 0;
    bool exact = true, valid = true;
    unsigned max_row = 0;
    #pragma omp parallel reduction(&& : exact, valid) reduction(min : min_diagonal) reduction(max : max_diagonal, max_row)
    {
        const auto [first, last] = work_balanced_range(A.outerIndexPtr(), A.rows(),
            omp_get_thread_num(), omp_get_num_threads());
        for (auto col = first; col < last; ++col) {
            max_row = std::max(max_row, unsigned(A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col]));
            for (int p = A.outerIndexPtr()[col]; p < A.outerIndexPtr()[col + 1]; ++p) {
                const int row = A.innerIndexPtr()[p];
                const double value = A.valuePtr()[p];
                valid = valid && metal_schedule::in_range(value);
                values[p] = metal_host::split(value);
                if (row >= col && double(float(value)) != value) exact = false;
                if (row == col && value > 0) {
                    min_diagonal = std::min(min_diagonal, value);
                    max_diagonal = std::max(max_diagonal, value);
                }
            }
        }
    }
    if (!valid)
        throw std::domain_error("Metal device setup requires finite entries with nonzero "
                                "magnitudes in [2^-100, 2^100]");
    out.operator_exact = exact;
    out.max_operator_row = max_row;
    if (max_diagonal > 0) {
        const int exponent_sum = std::ilogb(min_diagonal) + std::ilogb(max_diagonal);
        out.preconditioner_scale = std::scalbn(1.0f, int(std::floor(exponent_sum / 4.0)));
    }
    return out;
}

// Pipelines belong to the device context, not individual solver owners.
struct pipelines {
    Engine factorizer;
    Finalizer finalizer;
    explicit pipelines(id<MTLDevice> dev) : factorizer(dev), finalizer(dev) {}
};
pipelines& pipelines_for(id<MTLDevice> device) {
    static pipelines cached(device);
    return cached;
}

} // namespace

void prepare_lane_plans(id<MTLDevice> device, const factorization& factor,
                        triangle& forward, triangle& backward) {
    const unsigned n = factor.perm.size();
    auto rounds = buffer(device, (factor.rounds.size() + 1) * sizeof(unsigned));
    auto* bounds = static_cast<unsigned*>(rounds.contents);
    bounds[0] = 0;
    for (std::size_t r = 0; r < factor.rounds.size(); ++r) {
        const auto& round = factor.rounds[r];
        require(round.active == n - bounds[r] && round.is_size <= round.active,
                "invalid resident Metal round boundaries");
        bounds[r + 1] = bounds[r] + unsigned(round.is_size);
    }
    require(bounds[factor.rounds.size()] == n, "incomplete resident Metal rounds");
    auto status = buffer(device, sizeof(unsigned));
    *static_cast<unsigned*>(status.contents) = 0;
    pipelines_for(device).finalizer.prepare_plans<8>(forward, backward, n - !factor.sddm,
        rounds, 0, unsigned(factor.rounds.size()), status);
}

resident_factor factorize(id<MTLDevice> device, const Eigen::SparseMatrix<double> &A,
                          const solve_options &options, checkpoint *cp, bool input_prechecked) {
    @autoreleasepool {
        validate_factor_options(options.factor_opts);
        if (!metal_setup_options_supported(options.factor_opts))
            throw std::invalid_argument("Metal device setup currently requires FP32 graph storage "
                                        "and default block-greedy GKS options");
        auto& cached = pipelines_for(device);
        if (cp) {
            cp->descend("setup");
            cp->tick();
        }
        std::size_t offdiagonals = 0;
        bool zeros = false;
        if (!input_prechecked && !gpu_owned_csc_layout_supported(A, offdiagonals, &zeros))
            throw std::invalid_argument(
                "Metal device setup requires compressed, sorted, unique and paired 32-bit CSC storage");
        if (!input_prechecked && !metal_setup_regularization_supported(options.factor_opts.regularization, &A))
            throw std::invalid_argument(metal_regularization_error);
        const operator_view validated(A, options.factor_opts.lump_max_mass);
        if (cp)
            (*cp)("operator_prepare");
        const bool unchanged = &validated.matrix() == &A;
        // Validation proves pairing only for the unchanged nonzero operator.
        const bool paired = input_prechecked || (unchanged && !zeros) || gpu_owned_csc_paired(A);
        if (!paired || (!unchanged && !gpu_owned_csc_supported(validated)))
            throw std::invalid_argument(
                "Metal device setup requires compressed, sorted, unique and paired 32-bit CSC storage");
        const auto input = upload(device, A);
        const auto graph = unchanged ? input : upload(device, validated.matrix());
        if (cp)
            (*cp)("metal_input");
        const float regularization = float(options.factor_opts.regularization);
        auto raw = cached.factorizer.run(
            unsigned(A.rows()), unsigned(validated.matrix().nonZeros()),
            unsigned(validated.scan().offdiag_nnz - validated.lumped()), options.factor_opts.seed,
            graph.ptr, graph.idx, graph.values, regularization,
            unsigned(unchanged && validated.scan().triangles_bit_identical), options.factor_opts,
            options.factor_opts.parallel_residual_threshold == std::numeric_limits<std::size_t>::max()
                ? block_greedy_partitioner::residual_handoff_threshold
                : options.factor_opts.parallel_residual_threshold);
        if (cp)
            (*cp)("metal_factorize");
        const bool laplacian = !raw.data().state->has_excess;
        resident_factor out;
        auto &F = out.factor;
        F.sddm = !laplacian;
        F.lumped_offdiag = validated.lumped();
        const auto data = raw.data();
        const unsigned n = data.state->nodes;
        if (options.keep_factor_values) {
            std::vector<factor_entry> entries(data.state->entries);
            for (unsigned p = 0; p < data.state->entries; ++p)
                entries[p] = {data.factors[p].node, data.factors[p].value};
            std::vector<factor_col> columns(n);
            for (unsigned row = 0; row < n; ++row) {
                const auto col = data.columns[row];
                columns[row] = {col.node, col.diagonal, entries.data() + col.begin, col.count};
            }
            build_csc(F, columns, n, nullptr);
        }
        const bool half = resolve_fp16_storage(options.factor_opts.factor_storage, false);
        auto finalized = cached.finalizer.run(raw, laplacian, half, options.factor_opts.drop_tolerance);
        if (cp)
            (*cp)("metal_factor_prep");
        out.operator_exact = input.operator_exact;
        out.max_operator_row = input.max_operator_row;
        out.preconditioner_scale = input.preconditioner_scale;
        out.operator_triangles_identical = validated.scan().triangles_bit_identical;
        out.forward = std::move(finalized.forward);
        out.backward = std::move(finalized.backward);
        out.metadata = finalized.metadata;
        out.half = half;
        if (!options.keep_factor_values) {
            F.perm.assign(static_cast<unsigned *>(finalized.permutation.contents),
                          static_cast<unsigned *>(finalized.permutation.contents) + n);
            F.L.resize(n, n);
            for (unsigned row = 0; row < n; ++row)
                F.L.outer_[row] = data.columns[row].begin + row;
            F.L.outer_[n] = data.state->entries + n;
        }
        F.rounds.reserve(data.state->round);
        for (unsigned r = 0; r < data.state->round; ++r) {
            const unsigned first = data.round_ptr[r], last = data.round_ptr[r + 1];
            F.rounds.push_back({n - first, last - first});
        }
        raw = {};
        cached.finalizer.prepare_operator(input.ptr, input.idx, input.values, finalized.permutation,
                                          n, unsigned(A.nonZeros()), out);
        if (cp) {
            (*cp)("metal_operator");
            cp->ascend();
        }
        return out;
    }
}

} // namespace apxchol::detail::metal_setup

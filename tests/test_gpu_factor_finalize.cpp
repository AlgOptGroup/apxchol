#include <gtest/gtest.h>
#include "scoped_omp_threads.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/preconditioner.h"
#include "apxchol/solver/solve.h"

#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/sptrsv/cuda.h"
#include "apxchol/solver/sptrsv/omp.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using apxchol::node_index;
struct edge { node_index u, v; double weight; };

struct complete_factor {
    apxchol::detail::gpu_round_shadow_device_state state;
    apxchol::factorization reference;
    apxchol::factorization exported;

    complete_factor(node_index n, std::span<const edge> edges,
                    std::span<const node_index> order, bool keep_factor = false) {
        apxchol::detail::gpu_round_shadow_input input;
        input.vertex_count = n;
        input.active.assign(n, 1);
        input.excess.assign(n, 1.0);
        std::vector<std::vector<std::pair<node_index, apxchol::pool_value_t>>> adjacency(n);
        for (const auto& e : edges) {
            const auto value = static_cast<apxchol::pool_value_t>(e.weight);
            adjacency[e.u].emplace_back(e.v, value);
            adjacency[e.v].emplace_back(e.u, value);
        }
        input.owner_offsets.resize(std::size_t(n) + 1);
        for (node_index v = 0; v < n; ++v) {
            input.owner_offsets[v] = input.incidences.size();
            for (const auto& [neighbor, value] : adjacency[v])
                input.incidences.push_back({v, neighbor, value});
        }
        input.owner_offsets[n] = input.incidences.size();
        auto frontend = state.initialize_owned_raw_for_test(input);
        for (std::size_t i = 0; i < order.size(); ++i) {
            frontend->select_for_test(order.subspan(i, 1));
            state.compute_owned_round_for_test(frontend->device_selection(), 42);
            if (i + 1 < order.size()) state.advance_selector(*frontend);
        }
        // Finalization cannot observe an unpublished or incomplete factor.
        std::vector<node_index> perm(n);
        for (node_index i = 0; i < n; ++i) perm[order[i]] = i;
        EXPECT_THROW(state.finalize_device_factor(perm, n), std::logic_error);
        const auto log = state.download_factor_log();
        std::vector<apxchol::detail::factor_entry> retained;
        const auto headers = state.complete_owned_factorization_for_test(
            keep_factor ? &retained : nullptr);
        if (headers.size() != n || log.columns.size() != n)
            throw std::logic_error("owned test factor did not eliminate every vertex");
        std::vector<std::vector<apxchol::detail::factor_entry>> entries(n);
        std::vector<apxchol::detail::factor_col> columns;
        for (node_index i = 0; i < n; ++i) {
            const auto& col = log.columns[i];
            for (unsigned j = 0; j < col.entry_count; ++j) {
                const auto& e = log.entries[col.entry_begin + j];
                entries[i].push_back({e.neighbor, e.value});
            }
            columns.push_back({col.vertex, col.diag, entries[i].data(), node_index(entries[i].size())});
        }
        apxchol::detail::build_csc(reference, columns, n, nullptr);
        reference.sddm = true;
        EXPECT_EQ(reference.perm, perm);
        if (keep_factor) {
            columns.clear();
            for (const auto& c : headers)
                columns.push_back({c.vertex, c.diag,
                    c.entry_count ? retained.data() + c.entry_begin : nullptr, c.entry_count});
            apxchol::detail::build_csc(exported, columns, n, nullptr);
        }
    }
};

Eigen::SparseMatrix<double> solve_matrix(int n, double shift) {
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 4.0 + shift);
        for (int offset : {1, 8}) {
            const int j = (i + offset) % n;
            entries.emplace_back(i, j, -1.0);
            entries.emplace_back(j, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

#define REQUIRE_FINALIZER_DEVICE() \
    do { \
        if (!apxchol::detail::gpu_round_shadow_runtime_available()) \
            GTEST_SKIP() << "CUDA device unavailable"; \
    } while (false)
} // namespace

TEST(GpuFactorFinalize, CompleteOwnedPermutationAndGrounding) {
    REQUIRE_FINALIZER_DEVICE();
    const apxchol::factor_options options{
        .factor_storage = apxchol::factor_storage_type::fp32, .drop_tolerance = 0};
    const std::vector<edge> edges = {{1, 0, 2.0}, {1, 3, 3.0}};
    const std::vector<node_index> order = {1, 3, 0, 2};
    complete_factor fixture(4, edges, order);
    const auto& reference = fixture.reference;
    EXPECT_EQ(reference.perm, (std::vector<node_index>{2, 0, 3, 1}));
    for (node_index m : {3u, 4u}) {
        apxchol::cuda_sptrsv ordinary, adopted;
        ordinary.setup(reference.L, m, options);
        auto factor = fixture.state.finalize_device_factor(reference.perm, m, options.factor_storage, options.drop_tolerance);
        ASSERT_FALSE(factor->empty());
        adopted.setup_adopting_device_factor_for_research(std::move(*factor), options);
        ASSERT_TRUE(factor->empty());
        EXPECT_TRUE(adopted.adopted_device_factor());
        EXPECT_LE(adopted.adoption_host_download_bytes(), 512u);
        EXPECT_EQ(adopted.stored_nnz(), ordinary.stored_nnz());
        EXPECT_EQ(adopted.drop_stats().dropped_flush, ordinary.drop_stats().dropped_flush);
        apxchol::omp_sptrsv cpu;
        cpu.setup(reference.L, m, options);
        for (int k = 0; k < 5; ++k) {
            std::vector<double> a(m), b(m), c(m), scratch(m);
            for (node_index i = 0; i < m; ++i) a[i] = b[i] = c[i] = std::sin(i + 3 * k + 0.1);
            ordinary.solve_LLt(a.data(), a.data()); adopted.solve_LLt(b.data(), b.data());
            cpu.forward_solve(c.data(), scratch.data()); cpu.transpose_solve(scratch.data(), c.data());
            for (node_index i = 0; i < m; ++i) {
                EXPECT_EQ(a[i], b[i]);
                EXPECT_NEAR(b[i], c[i], 2e-6 * std::max(1.0, std::abs(c[i])));
            }
        }
    }
}

TEST(GpuFactorFinalize, DefaultStorageMatchesHostCompensationAndHalfBits) {
    REQUIRE_FINALIZER_DEVICE();
    constexpr node_index n = 10;
    const std::vector<edge> edges = {
        {0, 1, 1}, {0, 2, 0.3}, {2, 9, 0x1p-25}, {2, 8, 0x1p-14},
        {2, 7, 0.3333333432674408}, {2, 6, 1.0}, {2, 5, 0.500244140625},
        {1, 3, 0x1p-15}, {3, 4, 0.25}, {4, 5, 0.125},
        {5, 6, 0x1p-130}, {6, 7, 0x1p-129}, {7, 8, 0.00009999}, {8, 9, 0.25}};
    const std::vector<node_index> order = {0, 2, 1, 3, 4, 5, 6, 7, 8, 9};
    complete_factor fixture(n, edges, order);
    complete_factor retained(n, edges, order, true);
    const auto& reference = fixture.reference;
    const auto& export_reference = retained.reference;
    ASSERT_EQ(retained.exported.L.outer_, export_reference.L.outer_);
    ASSERT_EQ(retained.exported.L.inner_, export_reference.L.inner_);
    ASSERT_EQ(retained.exported.perm, export_reference.perm);
    ASSERT_EQ(retained.exported.L.vals_.size(), export_reference.L.vals_.size());
    for (std::size_t i = 0; i < export_reference.L.vals_.size(); ++i)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(retained.exported.L.vals_[i]),
                  std::bit_cast<std::uint32_t>(export_reference.L.vals_[i]));
    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        for (double rel : {0.0, 0.00001, 0.0001, 0.2}) {
            const apxchol::factor_options options{.factor_storage = storage, .drop_tolerance = rel};
            for (node_index m : {n - 1, n}) {
                SCOPED_TRACE((storage == apxchol::factor_storage_type::fp16 ? "float16/" : "float32/")
                    + std::to_string(rel) + "/" + std::to_string(m));
                apxchol::cuda_sptrsv ordinary, adopted;
                ordinary.setup(reference.L, m, options);
                auto factor = fixture.state.finalize_device_factor(reference.perm, m, storage, rel);
                auto retained_factor = retained.state.finalize_device_factor(reference.perm, m, storage, rel);
                auto lt = apxchol::cuda_host::build_L11_csc_int<float>(reference.L, m);
                auto scale = apxchol::cuda_host::column_scales(lt);
                const bool half = storage == apxchol::factor_storage_type::fp16;
                if (half) {
                    for (node_index j = 0; j < m; ++j)
                        if (!std::isfinite(1.0f / scale[j]) ||
                            !std::isfinite(lt.vals[lt.ptr[j]] / scale[j])) scale[j] = 1.0f;
                }
                apxchol::cuda_host::apply_factor_drop(lt, scale, rel, half);
                std::vector<std::byte> expected;
                auto append = [&](const auto* data, std::size_t count) {
                    auto bytes = std::as_bytes(std::span(data, count));
                    expected.insert(expected.end(), bytes.begin(), bytes.end());
                };
                if (half) {
                    auto h = apxchol::cuda_host::narrow_fp16_scaled(lt, scale);
                    apxchol::cuda_host::csr_int<_Float16> lt16;
                    lt16.m = m; lt16.nnz = lt.nnz; lt16.ptr = lt.ptr;
                    lt16.idx = std::move(lt.idx); lt16.vals = std::move(h.vals);
                    auto l16 = apxchol::cuda_host::transpose_csr(lt16);
                    append(l16.ptr.data(), m + 1); append(l16.idx.get(), l16.nnz); append(l16.vals.get(), l16.nnz);
                    append(lt16.ptr.data(), m + 1); append(lt16.idx.get(), lt16.nnz); append(lt16.vals.get(), lt16.nnz);
                    append(h.diag.data(), m);
                    append(h.inv_scale2.data(), m);
                } else {
                    auto l = apxchol::cuda_host::transpose_csr(lt);
                    append(l.ptr.data(), m + 1); append(l.idx.get(), l.nnz); append(l.vals.get(), l.nnz);
                    append(lt.ptr.data(), m + 1); append(lt.idx.get(), lt.nnz); append(lt.vals.get(), lt.nnz);
                }
                EXPECT_EQ(fixture.state.encode_finalized_factor_for_test(*factor), expected);
                EXPECT_EQ(retained.state.encode_finalized_factor_for_test(*retained_factor), expected);
                adopted.setup_adopting_device_factor_for_research(std::move(*factor), options);
                EXPECT_EQ(adopted.fp16(), half);
                const auto& want = ordinary.drop_stats(); const auto& got = adopted.drop_stats();
                EXPECT_EQ(got.rel, want.rel); EXPECT_EQ(got.nnz_factor, want.nnz_factor);
                EXPECT_EQ(got.nnz_stored, want.nnz_stored); EXPECT_EQ(got.dropped, want.dropped);
                EXPECT_EQ(got.dropped_threshold, want.dropped_threshold);
                EXPECT_EQ(got.dropped_flush, want.dropped_flush);
                for (node_index basis = 0; basis < m; ++basis) {
                    std::vector<double> a(m), b(m); a[basis] = b[basis] = 1.0;
                    ordinary.solve_LLt(a.data(), a.data()); adopted.solve_LLt(b.data(), b.data());
                    EXPECT_EQ(a, b) << "basis=" << basis;
                }
            }
        }
    }
}

TEST(GpuFactorFinalize, ConsumingSolveAcceptsDefaultFp16AndCompensatedDrop) {
    REQUIRE_FINALIZER_DEVICE();
    scoped_omp_threads serial(1);
    for (double shift : {0.0, 1.0}) {
        auto A = solve_matrix(1025, shift);
        Eigen::VectorXd exact(A.rows());
        for (int i = 0; i < exact.size(); ++i) exact[i] = std::sin(i + 0.25);
        Eigen::VectorXd b = A * exact;
        apxchol::solve_options opts;
        opts.backend = apxchol::solve_backend::gpu;
        opts.tol = 1e-8; opts.max_iter = 1000;
        const auto solved = apxchol::solve(A, b, opts);
        EXPECT_EQ(solved.backend, apxchol::solve_backend::gpu);
        EXPECT_LE((A * solved.x - b).norm() / b.norm(), 1e-8);
    }
}

TEST(GpuFactorFinalize, InvalidPermutationAndExtentFailClosed) {
    REQUIRE_FINALIZER_DEVICE();
    const apxchol::factor_options options{
        .factor_storage = apxchol::factor_storage_type::fp32, .drop_tolerance = 0};
    const std::vector<edge> edges = {{0, 1, 1}, {0, 2, 2}};
    const std::vector<node_index> order = {0, 1, 2};
    complete_factor fixture(3, edges, order);
    auto& state = fixture.state;
    const auto& perm = fixture.reference.perm;
    EXPECT_THROW(state.finalize_device_factor(std::vector<node_index>{0, 1, 1}, 3, options.factor_storage, options.drop_tolerance), std::invalid_argument);
    EXPECT_THROW(state.finalize_device_factor(std::vector<node_index>{0, 1}, 3, options.factor_storage, options.drop_tolerance), std::invalid_argument);
    EXPECT_THROW(state.finalize_device_factor(std::vector<node_index>{0, 1, 3}, 3, options.factor_storage, options.drop_tolerance), std::invalid_argument);
    EXPECT_THROW(state.finalize_device_factor(perm, 0, options.factor_storage, options.drop_tolerance), std::invalid_argument);
    EXPECT_THROW(state.finalize_device_factor(perm, 4, options.factor_storage, options.drop_tolerance), std::invalid_argument);
    EXPECT_NO_THROW(state.finalize_device_factor(perm, 3, options.factor_storage, options.drop_tolerance));
    // The existing Compute Sanitizer CTest checks these repeated allocations.
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int i = 0; i < 8; ++i) EXPECT_NO_THROW(state.finalize_device_factor(perm, 3, options.factor_storage, options.drop_tolerance));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(GpuFactorFinalize, PreconditionerInstallsAndReplacesOwnedCapsules) {
    REQUIRE_FINALIZER_DEVICE();
    const apxchol::factor_options options{
        .factor_storage = apxchol::factor_storage_type::fp32, .drop_tolerance = 0};
    const std::vector<edge> edges = {{1, 0, 2.0}, {1, 3, 3.0}};
    const std::vector<node_index> order = {1, 3, 0, 2};
    complete_factor fixture(4, edges, order);
    const auto& reference = fixture.reference;
    apxchol::detail::gpu_preconditioner preconditioner;
    preconditioner.set_keep_factor(true);
    preconditioner.set_options(options);
    apxchol::omp_sptrsv cpu;
    cpu.setup(reference.L, 4, options);
    for (int repetition = 0; repetition < 2; ++repetition) {
        auto factor = reference;
        factor.research_device_factor = fixture.state.finalize_device_factor(reference.perm, 4, options.factor_storage, options.drop_tolerance);
        preconditioner.set_factor(std::move(factor));
        ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
        ASSERT_TRUE(preconditioner.factor().sddm);
        ASSERT_EQ(preconditioner.factor().L.vals_, reference.L.vals_);
        Eigen::VectorXd b(4), expected(4), tmp(4), work(4);
        for (int i = 0; i < 4; ++i) { b[i] = std::cos(i + 0.5); work[reference.perm[i]] = b[i]; }
        cpu.forward_solve(work.data(), tmp.data()); cpu.transpose_solve(tmp.data(), work.data());
        for (int i = 0; i < 4; ++i) expected[i] = work[reference.perm[i]];
        const Eigen::VectorXd observed = preconditioner.solve(b);
        EXPECT_LT((observed - expected).norm() / expected.norm(), 2e-6);
    }
    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        apxchol::factor_options options;
        options.factor_storage = storage;
        options.drop_tolerance = 0;
        preconditioner.set_options(options);
        for (double shift : {0.0, 1.0}) {
            preconditioner.compute(solve_matrix(64, shift));
            ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
            const auto& factor = preconditioner.factor();
            ASSERT_EQ(factor.L.vals_.size(), factor.L.nonZeros());
            ASSERT_EQ(factor.L.inner_.size(), factor.L.nonZeros());
            ASSERT_EQ(factor.L.rows(), 64);
            EXPECT_EQ(preconditioner.trsv().fp16(), storage == apxchol::factor_storage_type::fp16);
            if (!factor.sddm) { EXPECT_EQ(factor.L.vals_[factor.L.outer_[63]], 1.0f); }
        }
    }
}

TEST(GpuFactorFinalize, SharedCapsulesAndExplicitHostImportsPreserveOwnership) {
    REQUIRE_FINALIZER_DEVICE();
    const apxchol::factor_options options{
        .factor_storage = apxchol::factor_storage_type::fp32, .drop_tolerance = 0};
    scoped_omp_threads serial(1);
    const std::vector<edge> edges = {{1, 0, 2.0}, {1, 3, 3.0}};
    const std::vector<node_index> order = {1, 3, 0, 2};
    complete_factor fixture(4, edges, order);
    for (bool sddm : {false, true}) {
        const node_index m = sddm ? 4 : 3;
        auto reference = fixture.reference;
        reference.sddm = sddm;
        reference.research_device_factor = fixture.state.finalize_device_factor(reference.perm, m, options.factor_storage, options.drop_tolerance);
        {
            apxchol::detail::gpu_preconditioner shared_owner;
            shared_owner.set_options(options);
            EXPECT_THROW(shared_owner.set_factor(reference), std::invalid_argument);
        }
        ASSERT_FALSE(reference.research_device_factor->empty());
        ASSERT_EQ(reference.research_device_factor.use_count(), 1);
        auto unique = reference;
        unique.research_device_factor = fixture.state.finalize_device_factor(reference.perm, m, options.factor_storage, options.drop_tolerance);
        apxchol::detail::gpu_preconditioner moved;
        moved.set_options(options);
        moved.set_factor(std::move(unique));
        ASSERT_TRUE(moved.trsv().adopted_device_factor());
        EXPECT_TRUE(moved.factor().L.vals_.empty());
        EXPECT_TRUE(moved.factor().L.inner_.empty());
        EXPECT_EQ(moved.factor().L.outer_, reference.L.outer_);
        EXPECT_EQ(moved.factor().perm, reference.perm);
        std::vector<double> b(m), expected(m), observed(m);
        for (node_index i = 0; i < m; ++i) b[i] = std::sin(i + 0.25);
        apxchol::cuda_sptrsv imported;
        imported.setup(reference.L, m, options);
        EXPECT_FALSE(imported.adopted_device_factor());
        imported.solve_LLt(b.data(), expected.data());
        moved.trsv().solve_LLt(b.data(), observed.data());
        EXPECT_EQ(observed, expected);
        auto import_copy = [reference, b, m, options]() {
            apxchol::cuda_sptrsv solver;
            solver.setup(reference.L, m, options);
            std::vector<double> result(m);
            solver.solve_LLt(b.data(), result.data());
            return std::pair{solver.adopted_device_factor(), std::move(result)};
        };
        auto first = std::async(std::launch::async, import_copy);
        auto second = std::async(std::launch::async, import_copy);
        for (auto* task : {&first, &second}) {
            auto [adopted, result] = task->get();
            EXPECT_FALSE(adopted);
            EXPECT_EQ(result, expected);
        }
        EXPECT_FALSE(reference.research_device_factor->empty());
        const auto A = solve_matrix(64, sddm ? 1.0 : 0.0);
        const auto exported = apxchol::factorize(A);
        EXPECT_EQ(exported.L.vals_.size(), exported.L.nonZeros());
        EXPECT_EQ(exported.L.inner_.size(), exported.L.nonZeros());
        EXPECT_FALSE(exported.research_device_factor);
        apxchol::cuda_sptrsv host_import;
        host_import.setup(exported.L, sddm ? 64 : 63, options);
        EXPECT_TRUE(host_import.ready());
        EXPECT_FALSE(host_import.adopted_device_factor());
    }
}
#endif

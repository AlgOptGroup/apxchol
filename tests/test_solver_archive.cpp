#include <gtest/gtest.h>
#include "apxchol/solver/archive.h"
#include "apxchol/detail/csc_import.h"
#include <zip.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <barrier>

namespace {
using namespace apxchol;
using Sparse = Eigen::SparseMatrix<double>;

Sparse matrix(int n = 32) {
    Sparse A(n, n);
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 3.0);
        if (i) { entries.emplace_back(i, i - 1, -1.0); entries.emplace_back(i - 1, i, -1.0); }
    }
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

solve_options options() {
    solve_options result;
    result.backend = solve_backend::cpu;
    result.num_threads = 2;
    result.keep_factor_values = true;
    result.factor_opts.factor_storage = factor_storage_type::fp32;
    return result;
}

class SolverArchive : public ::testing::Test {
protected:
    std::filesystem::path directory;
    Sparse A = matrix();
    solver original{A, options()};
    void SetUp() override {
        directory = std::filesystem::temp_directory_path() /
            ("apxchol-archive-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ASSERT_TRUE(std::filesystem::create_directory(directory));
        save_solver(directory / "original.npz", A, original.factor());
    }
    void TearDown() override { std::filesystem::remove_all(directory); }

    std::filesystem::path change(const char* label, const char* entry,
                                const std::function<void(std::vector<unsigned char>&)>& modify) {
        const auto path = directory / (std::string(label) + ".npz");
        std::filesystem::copy_file(directory / "original.npz", path);
        int error = 0;
        zip_t* file = zip_open(path.c_str(), 0, &error);
        if (!file) throw std::runtime_error("fixture open failed");
        zip_stat_t stat;
        if (zip_stat(file, entry, 0, &stat)) throw std::runtime_error("fixture stat failed");
        std::vector<unsigned char> bytes(stat.size);
        auto* input = zip_fopen(file, entry, 0);
        if (!input || zip_fread(input, bytes.data(), bytes.size()) != static_cast<zip_int64_t>(bytes.size()))
            throw std::runtime_error("fixture read failed");
        zip_fclose(input);
        modify(bytes);
        auto* source = zip_source_buffer(file, bytes.data(), bytes.size(), 0);
        if (!source || zip_file_replace(file, stat.index, source, 0) || zip_close(file))
            throw std::runtime_error("fixture write failed");
        return path;
    }
    static std::size_t data_offset(const std::vector<unsigned char>& bytes) {
        return 10 + bytes[8] + 256 * bytes[9];
    }
};

TEST_F(SolverArchive, ExactFactorAndScalarBatchWarmSolves) {
    auto restored = load_solver(directory / "original.npz", options());
    EXPECT_EQ(restored.operator_nnz(), A.nonZeros());
    EXPECT_EQ(original.operator_nnz(), A.nonZeros());
    const auto& before = original.factor();
    const auto& after = restored.factor();
    EXPECT_EQ(before.L.outer_, after.L.outer_);
    EXPECT_EQ(before.L.inner_, after.L.inner_);
    ASSERT_EQ(before.L.vals_.size(), after.L.vals_.size());
    EXPECT_EQ(std::memcmp(before.L.vals_.data(), after.L.vals_.data(), before.L.vals_.size() * sizeof(float)), 0);
    EXPECT_EQ(before.perm, after.perm);
    EXPECT_EQ(before.sddm, after.sddm);
    EXPECT_EQ(before.lumped_offdiag, after.lumped_offdiag);
    ASSERT_EQ(before.rounds.size(), after.rounds.size());
    for (std::size_t i = 0; i < before.rounds.size(); ++i)
        EXPECT_EQ(before.rounds[i].is_size, after.rounds[i].is_size);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), -1.0, 1.0);
    auto scalar = restored.solve(b, 1e-9);
    EXPECT_TRUE(scalar.converged);
    EXPECT_LT((A * scalar.x - b).norm() / b.norm(), 1e-9);
    EXPECT_EQ(restored.apply(b), original.apply(b));
    auto warm = restored.solve(b, 1e-9, 0, &scalar.x);
    EXPECT_TRUE(warm.converged);
    EXPECT_EQ(warm.iterations, 0);
    Eigen::MatrixXd B(A.rows(), 5), X(A.rows(), 5);
    for (int col = 0; col < B.cols(); ++col) B.col(col) = (col + 1) * b;
    auto batch = restored.solve(B, X, 1e-9);
    ASSERT_EQ(batch.columns.size(), 5);
    for (int col = 0; col < B.cols(); ++col) {
        EXPECT_TRUE(batch.columns[col].converged);
        EXPECT_LT((A * X.col(col) - B.col(col)).norm() / B.col(col).norm(), 1e-9);
    }
}

TEST_F(SolverArchive, LaplacianFactorAndCenteredSolves) {
    auto L = matrix();
    for (int i = 0; i < L.rows(); ++i)
        L.coeffRef(i, i) = (i > 0) + (i + 1 < L.rows());
    solver before(L, options());
    ASSERT_FALSE(before.factor().sddm);
    const auto path = directory / "laplacian.npz";
    save_solver(path, L, before.factor());
    auto restored = load_solver(path, options());
    EXPECT_FALSE(restored.factor().sddm);
    EXPECT_EQ(restored.factor().L.outer_, before.factor().L.outer_);
    EXPECT_EQ(restored.factor().L.inner_, before.factor().L.inner_);
    EXPECT_EQ(restored.factor().L.vals_, before.factor().L.vals_);
    EXPECT_EQ(restored.factor().perm, before.factor().perm);
    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(L.rows(), -1.0, 1.0);
    const auto scalar = restored.solve(b, 1e-9);
    EXPECT_TRUE(scalar.converged);
    EXPECT_LT((L * scalar.x - b).norm() / b.norm(), 1e-9);
    EXPECT_NEAR(scalar.x.mean(), 0.0, 1e-12);
    Eigen::MatrixXd B(L.rows(), 3), X(L.rows(), 3);
    for (int col = 0; col < B.cols(); ++col) B.col(col) = (col + 1) * b;
    const auto batch = restored.solve(B, X, 1e-9);
    ASSERT_EQ(batch.columns.size(), 3);
    for (int col = 0; col < B.cols(); ++col) {
        EXPECT_TRUE(batch.columns[col].converged);
        EXPECT_LT((L * X.col(col) - B.col(col)).norm() / B.col(col).norm(), 1e-9);
        EXPECT_NEAR(X.col(col).mean(), 0.0, 1e-12);
    }
}

TEST_F(SolverArchive, AutomaticAdoptionUsesCpu) {
    auto opts = options(); opts.backend = solve_backend::automatic;
    auto restored = load_solver(directory / "original.npz", opts);
    EXPECT_EQ(restored.backend(), solve_backend::cpu);
    EXPECT_EQ(restored.factor_storage(), factor_storage_type::fp32);
}

TEST_F(SolverArchive, ExistingFilesAndReleasedFactorsAreRejected) {
    EXPECT_THROW(save_solver(directory / "original.npz", A, original.factor()), std::runtime_error);
    auto F = original.factor(); F.L.release_values();
    EXPECT_THROW(save_solver(directory / "released.npz", A, F), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(directory / "released.npz"));
    F = original.factor(); F.sddm = false;
    EXPECT_THROW(save_solver(directory / "wrong-kind.npz", A, F), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(directory / "wrong-kind.npz"));
}

TEST_F(SolverArchive, RejectsMetadataAndIndexOverflow) {
    for (auto [index, value] : std::array<std::pair<std::size_t, std::uint64_t>, 5>{{
        {0, 99}, {1, UINT64_MAX}, {2, 2}, {3, UINT64_MAX}, {4, UINT64_MAX}}}) {
        const auto path = change(("meta-" + std::to_string(index)).c_str(), "apxchol.npy", [&](auto& bytes) {
            std::memcpy(bytes.data() + data_offset(bytes) + index * 8, &value, 8);
        });
        EXPECT_THROW(load_solver(path), std::invalid_argument);
    }
    const auto bad_row = change("bad-row", "A_indices.npy", [&](auto& bytes) {
        const std::int32_t value = -1;
        std::memcpy(bytes.data() + data_offset(bytes), &value, sizeof(value));
    });
    EXPECT_THROW(load_solver(bad_row), std::invalid_argument);
    const auto bad_offset = change("bad-offset", "L_indptr.npy", [&](auto& bytes) {
        std::fill_n(bytes.data() + data_offset(bytes), sizeof(edge_index), 0xff);
    });
    EXPECT_THROW(load_solver(bad_offset), std::invalid_argument);
}

TEST_F(SolverArchive, RejectsMalformedHeaderAndTruncation) {
    const auto path = change("shape", "L_data.npy", [&](auto& bytes) {
        bytes[8] = bytes[9] = 0xff;
    });
    EXPECT_THROW(load_solver(path), std::invalid_argument);
    const auto truncated = directory / "truncated.npz";
    std::filesystem::copy_file(directory / "original.npz", truncated);
    std::filesystem::resize_file(truncated, 100);
    EXPECT_THROW(load_solver(truncated), std::runtime_error);
    const auto wrong_type = change("type", "A_data.npy", [&](auto& bytes) {
        auto it = std::search(bytes.begin(), bytes.end(), "<f8", "<f8" + 3);
        ASSERT_NE(it, bytes.end()); it[1] = 'u';
    });
    EXPECT_THROW(load_solver(wrong_type), std::invalid_argument);
}

TEST_F(SolverArchive, ReadsOppositeEndianArrays) {
    const auto path = change("big-endian", "A_data.npy", [&](auto& bytes) {
        auto it = std::search(bytes.begin(), bytes.end(), "<f8", "<f8" + 3);
        ASSERT_NE(it, bytes.end()); *it = '>';
        for (std::size_t i = data_offset(bytes); i < bytes.size(); i += 8)
            std::reverse(bytes.begin() + i, bytes.begin() + i + 8);
    });
    auto restored = load_solver(path, options());
    const Eigen::VectorXd b = Eigen::VectorXd::Ones(A.rows());
    auto result = restored.solve(b, 1e-9);
    EXPECT_LT((A * result.x - b).norm() / b.norm(), 1e-9);
}

TEST_F(SolverArchive, RejectsCorruptArrayPayload) {
    const auto path = directory / "crc.npz";
    std::filesystem::copy_file(directory / "original.npz", path);
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    file.seekg(0, std::ios::end);
    const auto size = file.tellg(); file.seekg(0);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    file.read(bytes.data(), bytes.size());
    const auto name = std::search(bytes.begin(), bytes.end(), "A_data.npy", "A_data.npy" + 10);
    ASSERT_NE(name, bytes.end());
    const auto magic = std::search(name, bytes.end(), "\x93NUMPY", "\x93NUMPY" + 6);
    ASSERT_NE(magic, bytes.end());
    const auto offset = static_cast<std::size_t>(magic - bytes.begin());
    const auto data = offset + 10 + static_cast<unsigned char>(bytes[offset + 8])
        + 256 * static_cast<unsigned char>(bytes[offset + 9]);
    file.seekp(data); file.put(static_cast<char>(bytes[data] ^ 1)); file.close();
    EXPECT_THROW(load_solver(path), std::invalid_argument);
}

TEST_F(SolverArchive, OperatorEntryCountUsesCanonicalInput) {
    const int outer[]{0, 3, 6}, rows[]{0, 0, 1, 0, 1, 1};
    const double values[]{1.5, 1.5, -1, -1, 1.5, 1.5};
    const auto canonical = detail::import_csc(2, outer, rows, values);
    ASSERT_EQ(canonical.nonZeros(), 4);
    solver direct(canonical, options());
    solver adopted(canonical, direct.factor(), options());
    EXPECT_EQ(direct.operator_nnz(), 4);
    EXPECT_EQ(adopted.operator_nnz(), 4);
    save_solver(directory / "duplicates.npz", canonical, direct.factor());
    EXPECT_EQ(load_solver(directory / "duplicates.npz").operator_nnz(), 4);
}

TEST_F(SolverArchive, CpuAndGpuFactorsCrossAdopt) {
    if (!backend_available(solve_backend::gpu)) GTEST_SKIP() << "GPU backend/device unavailable";
    for (auto storage : {factor_storage_type::fp16, factor_storage_type::fp32}) {
        auto gpu = options(); gpu.backend = solve_backend::gpu;
        gpu.factor_opts.factor_storage = storage;
        solver resident(A, gpu);
        EXPECT_EQ(resident.operator_nnz(), A.nonZeros());
        const auto path = directory / (storage == factor_storage_type::fp16 ? "gpu16.npz" : "gpu32.npz");
        save_solver(path, A, resident.factor());
        for (auto target : {solve_backend::cpu, solve_backend::gpu}) {
            auto opts = options(); opts.backend = target;
            if (target == solve_backend::gpu) opts.factor_opts.factor_storage = storage;
            auto restored = load_solver(path, opts);
            EXPECT_EQ(restored.operator_nnz(), A.nonZeros());
            Eigen::MatrixXd B(A.rows(), 4), X(A.rows(), 4);
            for (int col = 0; col < 4; ++col)
                B.col(col) = Eigen::VectorXd::LinSpaced(A.rows(), 0.1 + col, 1.0 + col);
            auto result = restored.solve(B, X, 1e-8);
            for (int col = 0; col < 4; ++col) {
                EXPECT_TRUE(result.columns[col].converged);
                EXPECT_LT((A * X.col(col) - B.col(col)).norm() / B.col(col).norm(), 1e-8);
            }
            const_matrix_view initial(X);
            Eigen::MatrixXd warm(A.rows(), 4);
            const auto warm_result = restored.solve(B, warm, 1e-8, 0, &initial);
            for (const auto& column : warm_result.columns) {
                EXPECT_TRUE(column.converged); EXPECT_EQ(column.iterations, 0);
            }
        }
        auto cpu_factor_on_gpu = load_solver(directory / "original.npz", gpu);
        const Eigen::VectorXd b = Eigen::VectorXd::Ones(A.rows());
        const auto scalar = cpu_factor_on_gpu.solve(b, 1e-8);
        EXPECT_TRUE(scalar.converged);
        EXPECT_LT((A * scalar.x - b).norm() / b.norm(), 1e-8);
    }
}

TEST_F(SolverArchive, PreservesLumpingAndRegularizationWithoutReapplyingThem) {
    Sparse positive(2, 2);
    positive.insert(0, 0) = 2; positive.insert(1, 1) = 2;
    positive.insert(0, 1) = 0.1; positive.insert(1, 0) = 0.1;
    positive.makeCompressed();
    auto construction = options();
    construction.factor_opts.lump_max_mass = 1.0;
    construction.factor_opts.regularization = 0.25;
    solver before(positive, construction);
    ASSERT_EQ(before.factor().lumped_offdiag, 2);
    const auto path = directory / "positive.npz";
    save_solver(path, positive, before.factor());
    EXPECT_THROW(load_solver(path), std::invalid_argument);
    auto admission = options(); admission.factor_opts.lump_max_mass = 1.0;
    auto restored = load_solver(path, admission);
    EXPECT_EQ(restored.factor().lumped_offdiag, 2);
    EXPECT_EQ(restored.operator_nnz(), 4);
    ASSERT_EQ(restored.factor().L.vals_.size(), before.factor().L.vals_.size());
    EXPECT_EQ(std::memcmp(restored.factor().L.vals_.data(), before.factor().L.vals_.data(),
        before.factor().L.vals_.size() * sizeof(float)), 0);
    const Eigen::VectorXd b = Eigen::VectorXd::Ones(2);
    const auto result = restored.solve(b, 1e-10);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.lumped_offdiag, 2);
    EXPECT_LT((positive * result.x - b).norm() / b.norm(), 1e-10);
}

TEST_F(SolverArchive, DoesNotReplaceSymlinksOrConcurrentWriters) {
    const auto link = directory / "dangling.npz";
    std::filesystem::create_symlink(directory / "missing", link);
    EXPECT_THROW(save_solver(link, A, original.factor()), std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_symlink(link));
    EXPECT_FALSE(std::filesystem::exists(directory / "missing"));
    const auto path = directory / "concurrent.npz";
    std::barrier start(2);
    auto write = [&] {
        start.arrive_and_wait();
        try { save_solver(path, A, original.factor()); return 1; }
        catch (const std::runtime_error&) { return 0; }
    };
    auto first = std::async(std::launch::async, write);
    auto second = std::async(std::launch::async, write);
    EXPECT_EQ(first.get() + second.get(), 1);
    EXPECT_EQ(load_solver(path).rows(), A.rows());
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        EXPECT_FALSE(entry.path().filename().string().starts_with(".apxchol-"));
}
} // namespace

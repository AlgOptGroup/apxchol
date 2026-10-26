#include <gtest/gtest.h>
#include "apxchol/solver/archive.h"
#include "apxchol/detail/csc_import.h"
#include "../src/solver_archive_internal.h"
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
#include <ranges>
#include <string_view>

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
        save_solver(directory / "original.npz", original);
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
    static std::vector<unsigned char> entry_bytes(const std::filesystem::path& path, const char* name) {
        int error = 0;
        std::unique_ptr<zip_t, decltype(&zip_discard)> file(zip_open(path.c_str(), ZIP_RDONLY, &error), &zip_discard);
        if (!file) throw std::runtime_error("fixture open failed");
        zip_stat_t stat;
        if (zip_stat(file.get(), name, 0, &stat) || stat.size > 16 * 1024 * 1024)
            throw std::runtime_error("fixture entry unavailable or oversized");
        std::vector<unsigned char> bytes(stat.size);
        std::unique_ptr<zip_file_t, decltype(&zip_fclose)> entry(zip_fopen(file.get(), name, 0), &zip_fclose);
        if (!entry || zip_fread(entry.get(), bytes.data(), bytes.size()) != static_cast<zip_int64_t>(bytes.size()))
            throw std::runtime_error("fixture entry read failed");
        return bytes;
    }
    static void expect_operator(const std::filesystem::path& path, const Sparse& expected) {
        const auto compare = [&](const char* name, const void* data, std::size_t size) {
            const auto bytes = entry_bytes(path, name);
            ASSERT_GE(bytes.size(), 10);
            const auto offset = data_offset(bytes);
            ASSERT_GE(bytes.size(), offset);
            ASSERT_EQ(bytes.size() - offset, size);
            EXPECT_EQ(std::memcmp(bytes.data() + offset, data, size), 0);
        };
        compare("A_indptr.npy", expected.outerIndexPtr(), (expected.cols() + 1) * sizeof(int));
        compare("A_indices.npy", expected.innerIndexPtr(), expected.nonZeros() * sizeof(int));
        compare("A_data.npy", expected.valuePtr(), expected.nonZeros() * sizeof(double));
    }
    static void check_solve_modes(const Sparse& input, solver& restored, bool laplacian, int& checked_rhs) {
        for (int width : {1, 4, 16}) {
            Eigen::MatrixXd B(input.rows(), width), X(input.rows(), width), cold(input.rows(), width);
            for (int col = 0; col < width; ++col) {
                for (int row = 0; row < input.rows(); ++row)
                    B(row, col) = (static_cast<double>((row * 7 + col * 3) % 19) - 9) / 16;
                if (laplacian) B.col(col).array() -= B.col(col).mean();
            }
            for (int phase = 0; phase < 3; ++phase) {
                Eigen::MatrixXd initial = phase ? Eigen::MatrixXd(cold * (phase == 1 ? 0.5 : 1.0)) : Eigen::MatrixXd::Zero(input.rows(), width);
                std::vector<solve_result> columns;
                if (width == 1) {
                    const Eigen::VectorXd b = B.col(0), x0 = initial.col(0);
                    columns.push_back(restored.solve(b, 1e-8, phase == 2 ? 0 : 200, phase ? &x0 : nullptr));
                    X.col(0) = columns.front().x;
                } else {
                    const const_matrix_view x0(initial);
                    columns = restored.solve(B, X, 1e-8, phase == 2 ? 0 : 200, phase ? &x0 : nullptr).columns;
                }
                ASSERT_EQ(columns.size(), static_cast<std::size_t>(width));
                for (int col = 0; col < width; ++col) {
                    Eigen::VectorXd residual = B.col(col);
                    for (int j = 0; j < input.cols(); ++j)
                        for (Sparse::InnerIterator entry(input, j); entry; ++entry)
                            residual[entry.row()] -= entry.value() * X(j, col);
                    EXPECT_TRUE(columns[col].converged);
                    EXPECT_LT(residual.stableNorm() / B.col(col).stableNorm(), 1e-8);
                    if (phase == 2) { EXPECT_EQ(columns[col].iterations, 0); }
                    ++checked_rhs;
                }
                if (phase == 0) cold = X;
            }
        }
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
    save_solver(path, before);
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

TEST_F(SolverArchive, ReconstructsOriginalSystemCoefficients) {
    for (int mode = 0; mode < 5; ++mode) {
        SCOPED_TRACE(mode);
        auto expected = matrix();
        if (mode == 1) expected *= 1.1; // Cannot be represented losslessly in FP32.
        Sparse input = expected;
        if (mode == 2) {
            input.insert(2, 0) = 0.0;
            expected.insert(2, 0) = 0.0;
            expected.insert(0, 2) = 0.0;
        }
        if (mode == 3) input.coeffRef(0, 1) = std::nextafter(input.coeff(0, 1), 0.0);
        if (mode == 4) input.insert(0, 2) = 0.0;
        input.makeCompressed(); expected.makeCompressed();
        solver fresh(expected, options());
        solver owner(input, fresh.factor(), options());
        const auto path = directory / ("operator-" + std::to_string(mode) + ".npz");
        save_solver(path, owner);
        expect_operator(path, expected);
        auto restored = load_solver(path, options());
        EXPECT_EQ(restored.operator_nnz(), input.nonZeros());
        EXPECT_EQ(restored.operator_nnz(), owner.operator_nnz());
        const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(expected.rows(), 0.1, 1.0);
        const auto result = restored.solve(b, 1e-10);
        EXPECT_TRUE(result.converged);
        EXPECT_LT((expected * result.x - b).norm() / b.norm(), 1e-10);
    }
}

TEST_F(SolverArchive, SharedReconstructionReversesBothCoordinatePermutations) {
    const auto expected = matrix();
    std::vector<node_index> perm(expected.rows());
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < expected.rows(); ++i) perm[i] = (i * 7) % expected.rows();
    for (int col = 0; col < expected.cols(); ++col)
        for (Sparse::InnerIterator e(expected, col); e; ++e)
            entries.emplace_back(perm[e.row()], perm[e.col()], e.value());
    Eigen::SparseMatrix<double, Eigen::RowMajor> stored(expected.rows(), expected.cols());
    stored.setFromTriplets(entries.begin(), entries.end());
    const auto restored = detail::archive_operator_csc(static_cast<int>(expected.rows()),
        stored.outerIndexPtr(), stored.innerIndexPtr(), stored.valuePtr(), perm);
    EXPECT_EQ(restored.nonZeros(), expected.nonZeros());
    EXPECT_EQ((restored - expected).norm(), 0.0);
}

TEST_F(SolverArchive, RejectsTooManyEmptyRoundsBeforeWriting) {
    auto F = original.factor();
    F.rounds.resize(A.rows() + 1);
    solver owner(A, std::move(F), options());
    const auto path = directory / "empty-rounds.npz";
    EXPECT_THROW(save_solver(path, owner), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(SolverArchive, AutomaticAdoptionUsesCpu) {
    auto opts = options(); opts.backend = solve_backend::automatic;
    auto restored = load_solver(directory / "original.npz", opts);
    EXPECT_EQ(restored.backend(), solve_backend::cpu);
    EXPECT_EQ(restored.factor_storage(), factor_storage_type::fp32);
}

TEST_F(SolverArchive, ExistingFilesAndReleasedFactorsAreRejected) {
    EXPECT_THROW(save_solver(directory / "original.npz", original), std::runtime_error);
    auto opts = options(); opts.keep_factor_values = false;
    solver released(A, opts);
    EXPECT_THROW(save_solver(directory / "released.npz", released), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(directory / "released.npz"));
}

TEST_F(SolverArchive, RejectsNullCharactersInPaths) {
    const auto existing = (directory / "original.npz").string() + std::string("\0suffix", 7);
    EXPECT_THROW(load_solver(existing), std::invalid_argument);
    const auto prefix = directory / "new.npz";
    const auto output = prefix.string() + std::string("\0suffix", 7);
    EXPECT_THROW(save_solver(output, original), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(prefix));
}

TEST_F(SolverArchive, InvalidOptionsFailBeforeOpeningAnArchive) {
    auto opts = options(); opts.num_threads = -1;
    EXPECT_THROW(load_solver(directory / "missing.npz", opts), std::invalid_argument);
    EXPECT_THROW(load_solver(directory / "missing.npz", options()), std::runtime_error);
}

TEST_F(SolverArchive, RejectsMetadataAndIndexOverflow) {
    for (auto [index, value] : std::array<std::pair<std::size_t, std::uint64_t>, 6>{{
        {0, 99}, {1, UINT64_MAX}, {2, 2}, {3, UINT64_MAX}, {4, UINT64_MAX}, {6, UINT64_MAX}}}) {
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
        auto it = std::ranges::search(bytes, std::string_view("<f8")).begin();
        ASSERT_NE(it, bytes.end()); it[1] = 'u';
    });
    EXPECT_THROW(load_solver(wrong_type), std::invalid_argument);
}

TEST_F(SolverArchive, ReadsOppositeEndianArrays) {
    const auto path = change("big-endian", "A_data.npy", [&](auto& bytes) {
        auto it = std::ranges::search(bytes, std::string_view("<f8")).begin();
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
    const auto name = std::ranges::search(bytes, std::string_view("A_data.npy")).begin();
    ASSERT_NE(name, bytes.end());
    const auto magic = std::ranges::search(std::ranges::subrange(name, bytes.end()),
                                            std::string_view("\x93NUMPY")).begin();
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
    save_solver(directory / "duplicates.npz", direct);
    EXPECT_EQ(load_solver(directory / "duplicates.npz").operator_nnz(), 4);
}

TEST_F(SolverArchive, CpuAndGpuFactorsCrossAdopt) {
    if (!backend_available(solve_backend::gpu)) GTEST_SKIP() << "GPU backend/device unavailable";
    int source_owners = 0, loaded_owners = 0, resaves = 0, checked_rhs = 0, nontrivial_permutations = 0;
    for (bool laplacian : {false, true}) for (bool exact_fp32 : {true, false}) {
        auto input = matrix();
        if (laplacian)
            for (int i = 0; i < input.rows(); ++i)
                input.coeffRef(i, i) = (i > 0) + (i + 1 < input.rows());
        if (!exact_fp32) input *= 1.1;
        for (auto storage : {factor_storage_type::fp16, factor_storage_type::fp32})
            for (auto origin : {solve_backend::cpu, solve_backend::gpu}) {
                SCOPED_TRACE(source_owners);
                auto build = options(); build.backend = origin;
                if (origin == solve_backend::gpu) build.factor_opts.factor_storage = storage;
                solver resident(input, build);
                const auto path = directory / ("source-" + std::to_string(source_owners++) + ".npz");
                save_solver(path, resident);
                expect_operator(path, input);
                const auto& F = resident.factor();
                for (std::size_t i = 0; i < F.perm.size(); ++i)
                    if (F.perm[i] != i) { ++nontrivial_permutations; break; }
                for (auto target : {solve_backend::cpu, solve_backend::gpu}) {
                    auto opts = options(); opts.backend = target;
                    if (target == solve_backend::gpu) opts.factor_opts.factor_storage = storage;
                    auto restored = load_solver(path, opts); ++loaded_owners;
                    EXPECT_EQ(restored.backend(), target);
                    EXPECT_EQ(restored.operator_nnz(), input.nonZeros());
                    EXPECT_EQ(restored.factor().L.outer_, F.L.outer_);
                    EXPECT_EQ(restored.factor().L.inner_, F.L.inner_);
                    ASSERT_EQ(restored.factor().L.vals_.size(), F.L.vals_.size());
                    EXPECT_EQ(std::memcmp(restored.factor().L.vals_.data(), F.L.vals_.data(), F.L.vals_.size() * sizeof(float)), 0);
                    EXPECT_EQ(restored.factor().perm, F.perm);
                    EXPECT_EQ(restored.factor().sddm, F.sddm);
                    ASSERT_EQ(restored.factor().rounds.size(), F.rounds.size());
                    for (std::size_t i = 0; i < F.rounds.size(); ++i)
                        EXPECT_EQ(restored.factor().rounds[i].is_size, F.rounds[i].is_size);
                    check_solve_modes(input, restored, laplacian, checked_rhs);
                    if (target == solve_backend::gpu) {
                        const auto adopted = directory / ("adopted-" + std::to_string(resaves++) + ".npz");
                        save_solver(adopted, restored); expect_operator(adopted, input);
                        auto host = load_solver(adopted, options());
                        Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(input.rows(), -1.0, 1.0);
                        const auto result = host.solve(b, 1e-8);
                        EXPECT_TRUE(result.converged);
                        EXPECT_LT((input * result.x - b).stableNorm() / b.stableNorm(), 1e-8);
                        ++checked_rhs;
                    }
                }
            }
    }
    EXPECT_EQ(source_owners, 16); EXPECT_EQ(loaded_owners, 32);
    EXPECT_EQ(resaves, 16); EXPECT_EQ(checked_rhs, 2032);
    EXPECT_GT(nontrivial_permutations, 0);
    RecordProperty("source_owners", source_owners); RecordProperty("loaded_owners", loaded_owners);
    RecordProperty("adopted_resaves", resaves); RecordProperty("independent_rhs", checked_rhs);
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
    save_solver(path, before);
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
    EXPECT_THROW(save_solver(link, original), std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_symlink(link));
    EXPECT_FALSE(std::filesystem::exists(directory / "missing"));
    const auto path = directory / "concurrent.npz";
    std::barrier start(2);
    auto write = [&] {
        start.arrive_and_wait();
        try { save_solver(path, original); return 1; }
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

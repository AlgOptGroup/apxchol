#include <gtest/gtest.h>
#include "apxchol/solver/solve.h"
#include "apxchol/detail/csc_import.h"
#include "apxchol/solver/sptrsv/omp.h"
#include "../src/crc32c.h"
#include "../src/solver_archive_internal.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <string>
#include <system_error>
#include <vector>
#include <barrier>
#include <sys/stat.h>

using csc_access = apxchol::detail::sparse_csc_access;

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
    result.factor_opts.factor_storage = factor_storage_type::fp32;
    return result;
}

// Independent test fixture encoder for the documented binary layout.
using Bytes = std::vector<unsigned char>;

// Bytewise CRC-32C, independent of the library's implementation.
std::uint32_t reference_crc32c(const unsigned char* data, std::size_t size) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t b = 0; b < 256; ++b) {
            std::uint32_t c = b;
            for (int k = 0; k < 8; ++k) c = c & 1 ? (c >> 1) ^ 0x82F63B78u : c >> 1;
            t[b] = c;
        }
        return t;
    }();
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; ++i) c = table[(c ^ data[i]) & 0xff] ^ (c >> 8);
    return ~c;
}
constexpr std::size_t prefix_size = 64 + 11 * 40;
enum ArrayId : std::size_t { metadata, a_outer, a_inner, a_values, l_outer, l_inner, l_values, permutation, round_sizes,
                             l_diagonal, l_scales };
std::uint64_t number(const Bytes& bytes, std::size_t at, unsigned width, bool little = true) {
    std::uint64_t result = 0;
    for (unsigned i = 0; i < width; ++i)
        result |= static_cast<std::uint64_t>(bytes.at(at + i)) << (8 * (little ? i : width - 1 - i));
    return result;
}
void put_number(Bytes& bytes, std::size_t at, unsigned width, std::uint64_t value, bool little = true) {
    for (unsigned i = 0; i < width; ++i)
        bytes.at(at + i) = static_cast<unsigned char>(value >> (8 * (little ? i : width - 1 - i)));
}
Bytes read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto size = input.tellg();
    if (size < 0 || size > 64 * 1024 * 1024) throw std::runtime_error("invalid test fixture size");
    Bytes bytes(static_cast<std::size_t>(size));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) throw std::runtime_error("fixture read failed");
    return bytes;
}
void write_bytes(const std::filesystem::path& path, const Bytes& bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size())) throw std::runtime_error("fixture write failed");
}
void checksum_metadata(Bytes& bytes) {
    put_number(bytes, 20, 4, 0);
    put_number(bytes, 20, 4, reference_crc32c(bytes.data(), prefix_size));
}
struct FixtureArray {
    unsigned char endian, kind, width;
    std::uint64_t count;
    Bytes data;
    std::uint64_t value(std::size_t index) const { return number(data, index * width, width, endian == '<'); }
    void value(std::size_t index, std::uint64_t bits) { put_number(data, index * width, width, bits, endian == '<'); }
};
using Fixture = std::array<FixtureArray, 11>;
Fixture read_fixture(const std::filesystem::path& path) {
    const auto bytes = read_bytes(path);
    Fixture arrays{};
    for (std::size_t i = 0; i < arrays.size(); ++i) {
        const auto d = 64 + 40 * i;
        if (number(bytes, d, 4) != i) throw std::runtime_error("unexpected test fixture array order");
        const auto offset = number(bytes, d + 16, 8), length = number(bytes, d + 24, 8);
        if (offset > bytes.size() || length > bytes.size() - offset) throw std::runtime_error("invalid test fixture extent");
        arrays[i] = {bytes.at(d + 4), bytes.at(d + 5), bytes.at(d + 6), number(bytes, d + 8, 8),
                     Bytes(bytes.begin() + offset, bytes.begin() + offset + length)};
    }
    return arrays;
}
Bytes encode_fixture(const Fixture& arrays) {
    Bytes bytes(prefix_size, 0);
    std::memcpy(bytes.data(), "APXCHB01", 8);
    put_number(bytes, 8, 4, 1); put_number(bytes, 12, 4, arrays.size()); put_number(bytes, 16, 4, 40);
    for (std::size_t i = 0; i < arrays.size(); ++i) {
        const auto& a = arrays[i];
        const auto offset = (bytes.size() + 63) & ~std::size_t(63), d = 64 + i * 40;
        bytes.resize(offset, 0);
        bytes.insert(bytes.end(), a.data.begin(), a.data.end());
        put_number(bytes, d, 4, i); bytes[d + 4] = a.endian; bytes[d + 5] = a.kind; bytes[d + 6] = a.width;
        put_number(bytes, d + 8, 8, a.count); put_number(bytes, d + 16, 8, offset);
        put_number(bytes, d + 24, 8, a.data.size());
        put_number(bytes, d + 32, 4, reference_crc32c(a.data.data(), a.data.size()));
    }
    put_number(bytes, 24, 8, bytes.size()); checksum_metadata(bytes);
    return bytes;
}
void widen(FixtureArray& a) {
    if (a.width == 8) return;
    Bytes wide(a.count * 8);
    for (std::size_t i = 0; i < a.count; ++i) put_number(wide, i * 8, 8, a.value(i), a.endian == '<');
    a.width = 8; a.data = std::move(wide);
}

class SolverArchive : public ::testing::Test {
protected:
    std::filesystem::path directory;
    Sparse A = matrix();
    solver original{A, options()};
    void SetUp() override {
        // Parallel test processes can read the same clock tick, so retry until
        // create_directory claims a fresh directory.
        const auto stem = "apxchol-archive-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-";
        int attempt = 0;
        do directory = std::filesystem::temp_directory_path() / (stem + std::to_string(attempt));
        while (!std::filesystem::create_directory(directory) && ++attempt < 1000);
        ASSERT_LT(attempt, 1000);
        original.save(directory / "original.apxchol");
    }
    void TearDown() override { std::filesystem::remove_all(directory); }

    std::filesystem::path change(const char* label, ArrayId entry,
                                const std::function<void(FixtureArray&)>& modify) {
        auto fixture = read_fixture(directory / "original.apxchol");
        modify(fixture[entry]);
        const auto path = directory / (std::string(label) + ".apxchol");
        write_bytes(path, encode_fixture(fixture));
        return path;
    }
    std::filesystem::path change_file(const char* label, const std::function<void(Bytes&)>& modify,
                                     bool repair_checksum = true) {
        auto bytes = read_bytes(directory / "original.apxchol");
        modify(bytes);
        if (repair_checksum) checksum_metadata(bytes);
        const auto path = directory / (std::string(label) + ".apxchol");
        write_bytes(path, bytes);
        return path;
    }
    static void expect_operator(const std::filesystem::path& path, const Sparse& expected) {
        const auto arrays = read_fixture(path);
        const auto compare = [&](ArrayId id, const void* data, std::size_t size) {
            ASSERT_EQ(arrays[id].data.size(), size);
            EXPECT_EQ(std::memcmp(arrays[id].data.data(), data, size), 0);
        };
        compare(a_outer, expected.outerIndexPtr(), (expected.cols() + 1) * sizeof(int));
        compare(a_inner, expected.innerIndexPtr(), expected.nonZeros() * sizeof(int));
        compare(a_values, expected.valuePtr(), expected.nonZeros() * sizeof(double));
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
    auto restored = solver::load(directory / "original.apxchol", options());
    EXPECT_EQ(restored.operator_nnz(), A.nonZeros());
    EXPECT_EQ(original.operator_nnz(), A.nonZeros());
    const auto before = original.factor();
    const auto after = restored.factor();
    EXPECT_EQ(after.storage, factor_storage_type::fp32);
    EXPECT_TRUE(after.scale.empty());
    EXPECT_EQ(csc_access::outer(before.L), csc_access::outer(after.L));
    EXPECT_EQ(csc_access::inner(before.L), csc_access::inner(after.L));
    ASSERT_EQ(csc_access::vals(before.L).size(), csc_access::vals(after.L).size());
    EXPECT_EQ(std::memcmp(csc_access::vals(before.L).data(), csc_access::vals(after.L).data(), csc_access::vals(before.L).size() * sizeof(float)), 0);
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
    const auto path = directory / "laplacian.apxchol";
    before.save(path);
    auto restored = solver::load(path, options());
    EXPECT_FALSE(restored.factor().sddm);
    EXPECT_EQ(csc_access::outer(restored.factor().L), csc_access::outer(before.factor().L));
    EXPECT_EQ(csc_access::inner(restored.factor().L), csc_access::inner(before.factor().L));
    EXPECT_EQ(csc_access::vals(restored.factor().L), csc_access::vals(before.factor().L));
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
        const auto path = directory / ("operator-" + std::to_string(mode) + ".apxchol");
        owner.save(path);
        expect_operator(path, expected);
        auto restored = solver::load(path, options());
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
    const auto path = directory / "empty-rounds.apxchol";
    EXPECT_THROW(owner.save(path), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(SolverArchive, AutomaticAdoptionUsesCpu) {
    auto opts = options(); opts.backend = solve_backend::automatic;
    auto restored = solver::load(directory / "original.apxchol", opts);
    EXPECT_EQ(restored.backend(), solve_backend::cpu);
    EXPECT_EQ(restored.factor_storage(), factor_storage_type::fp32);
}

TEST_F(SolverArchive, SavingReplacesAnExistingFile) {
    const auto path = directory / "replaced.apxchol";
    write_bytes(path, Bytes{'o', 'l', 'd'});
    original.save(path);
    EXPECT_EQ(read_bytes(path), read_bytes(directory / "original.apxchol"));
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
}

TEST_F(SolverArchive, LeftoverTemporaryFileIsReportedAndKept) {
    const auto path = directory / "busy.apxchol";
    const auto temporary = path.string() + ".tmp";
    write_bytes(temporary, Bytes{'x'});
    try {
        original.save(path);
        FAIL() << "expected the leftover temporary file to block the save";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(temporary), std::string::npos) << e.what();
    }
    EXPECT_TRUE(std::filesystem::exists(temporary));
    EXPECT_FALSE(std::filesystem::exists(path));
}

// Saving and loading reinstalls the installed factor exactly, whatever the
// load options say about storage and dropping: solves are bit-identical at a
// fixed thread count.
TEST_F(SolverArchive, ReinstallsInstalledFactorExactly) {
    if (!detail::omp_sptrsv::fp16_supported()) GTEST_SKIP() << "no FP16 conversion on this CPU";
    Sparse W = matrix(400);
    for (int k = 0; k < W.outerSize(); ++k)
        for (Sparse::InnerIterator e(W, k); e; ++e)
            if (e.row() != e.col()) e.valueRef() = -std::pow(10.0, ((e.row() + e.col()) % 7) / 2.0);
    for (int i = 0; i < W.rows(); ++i) W.coeffRef(i, i) = 0;
    for (int k = 0; k < W.outerSize(); ++k)
        for (Sparse::InnerIterator e(W, k); e; ++e)
            if (e.row() != e.col()) W.coeffRef(e.col(), e.col()) -= e.value();
    for (bool laplacian : {false, true})
        for (auto storage : {factor_storage_type::fp32, factor_storage_type::fp16})
            for (double drop : {0.0, 1e-2}) {
                SCOPED_TRACE(testing::Message() << laplacian << ' ' << int(storage) << ' ' << drop);
                Sparse input = W;
                if (!laplacian) input.diagonal().array() += 1e-3;
                auto opts = options();
                opts.factor_opts.factor_storage = storage;
                opts.factor_opts.drop_tolerance = drop;
                solver saver(input, opts);
                const auto F = saver.factor();
                EXPECT_EQ(F.storage, storage);
                EXPECT_EQ(F.scale.size(), storage == factor_storage_type::fp16 ? input.rows() : 0);
                EXPECT_EQ(F.L.nonZeros(), csc_access::vals(F.L).size());
                if (laplacian) { EXPECT_EQ(csc_access::outer(F.L)[input.rows() - 1], csc_access::outer(F.L)[input.rows()]); }
                const auto path = directory / ("exact-" + std::to_string(laplacian) +
                    std::to_string(int(storage)) + std::to_string(drop > 0) + ".apxchol");
                saver.save(path);
                auto other = options();
                other.factor_opts.factor_storage = storage == factor_storage_type::fp16
                    ? factor_storage_type::fp32 : factor_storage_type::fp16;
                other.factor_opts.drop_tolerance = drop > 0 ? 0.0 : 0.5;
                auto loaded = solver::load(path, other);
                EXPECT_EQ(loaded.factor_storage(), storage);
                const auto G = loaded.factor();
                EXPECT_EQ(csc_access::outer(G.L), csc_access::outer(F.L));
                EXPECT_EQ(csc_access::inner(G.L), csc_access::inner(F.L));
                EXPECT_EQ(std::memcmp(csc_access::vals(G.L).data(), csc_access::vals(F.L).data(), csc_access::vals(F.L).size() * sizeof(float)), 0);
                EXPECT_EQ(G.scale, F.scale);
                // Adopting the exported factor directly is exact too.
                solver adopted(input, F, other);
                Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(input.rows(), -1.0, 2.0);
                if (laplacian) b.array() -= b.mean();
                const auto x = saver.solve(b, 1e-9, 2000), y = loaded.solve(b, 1e-9, 2000),
                           z = adopted.solve(b, 1e-9, 2000);
                EXPECT_TRUE(x.converged);
                EXPECT_EQ(x.iterations, y.iterations);
                EXPECT_EQ(x.x, y.x);
                EXPECT_EQ(x.x, z.x);
                EXPECT_EQ(saver.apply(b), loaded.apply(b));
            }
}

TEST_F(SolverArchive, RejectsInvalidInstalledFactors) {
    auto F = original.factor();
    for (std::size_t col = 0; col < F.L.rows(); ++col)
        for (auto p = csc_access::outer(F.L)[col] + 1; p < csc_access::outer(F.L)[col + 1]; ++p) csc_access::vals(F.L)[p] = -0.25f;
    F.storage = factor_storage_type::fp16;
    EXPECT_THROW((void)solver(A, F, options()), std::invalid_argument);  // no scales
    F.scale.assign(A.rows(), 1.0f);
    csc_access::vals(F.L)[1] = 0.1f;  // not a binary16 value
    EXPECT_THROW((void)solver(A, F, options()), std::invalid_argument);
    csc_access::vals(F.L)[1] = -0.25f;
    F.scale[0] = 0.0f;
    EXPECT_THROW((void)solver(A, F, options()), std::invalid_argument);
    F.scale[0] = 1.0f;
    EXPECT_NO_THROW((void)solver(A, F, options()));
    auto G = original.factor();
    G.scale.assign(A.rows(), 1.0f);  // scales only belong to fp16 storage
    EXPECT_THROW((void)solver(A, G, options()), std::invalid_argument);
}

TEST_F(SolverArchive, NewArchivesFollowTheUmask) {
    struct umask_guard {
        mode_t previous = umask(022);
        ~umask_guard() { umask(previous); }
    };
    {
        const umask_guard guard;
        original.save(directory / "shared.apxchol");
    }
    struct stat info {};
    ASSERT_EQ(stat((directory / "shared.apxchol").c_str(), &info), 0);
    EXPECT_EQ(info.st_mode & 0777, 0644);
    EXPECT_FALSE(std::filesystem::exists((directory / "shared.apxchol").string() + ".tmp"));
}

TEST_F(SolverArchive, RejectsNullCharactersInPaths) {
    const auto existing = (directory / "original.apxchol").string() + std::string("\0suffix", 7);
    EXPECT_THROW(solver::load(existing), std::invalid_argument);
    const auto prefix = directory / "new.apxchol";
    const auto output = prefix.string() + std::string("\0suffix", 7);
    EXPECT_THROW(original.save(output), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(prefix));
}

TEST_F(SolverArchive, InvalidOptionsFailBeforeOpeningAnArchive) {
    auto opts = options(); opts.num_threads = -1;
    EXPECT_THROW(solver::load(directory / "missing.apxchol", opts), std::invalid_argument);
    EXPECT_THROW(solver::load(directory / "missing.apxchol", options()), std::runtime_error);
}

TEST_F(SolverArchive, ReportsDirectoryReadError) {
    EXPECT_THROW(solver::load(directory, options()), std::system_error);
}

TEST_F(SolverArchive, RejectsMetadataAndIndexOverflow) {
    for (const auto& item : std::array<std::pair<std::size_t, std::uint64_t>, 7>{{
        {0, 99}, {1, UINT64_MAX}, {2, 2}, {3, UINT64_MAX}, {4, UINT64_MAX}, {6, UINT64_MAX},
        {7, 2}}}) {
        const auto path = change(("meta-" + std::to_string(item.first)).c_str(), metadata,
                                 [&](auto& a) { a.value(item.first, item.second); });
        EXPECT_THROW(solver::load(path), std::invalid_argument);
    }
    const auto bad_row = change("bad-row", a_inner, [](auto& a) { a.value(0, UINT32_MAX); });
    EXPECT_THROW(solver::load(bad_row), std::invalid_argument);
    const auto bad_offset = change("bad-offset", l_outer, [](auto& a) { a.value(0, UINT64_MAX); });
    EXPECT_THROW(solver::load(bad_offset), std::invalid_argument);
    const auto wide_row = change("wide-row", a_inner, [](auto& a) { widen(a); a.kind = 'u'; a.value(0, UINT64_MAX); });
    EXPECT_THROW(solver::load(wide_row), std::invalid_argument);
}

TEST_F(SolverArchive, RejectsMalformedHeaderAndTruncation) {
    const auto header = change_file("version", [](auto& bytes) { put_number(bytes, 8, 4, 99); });
    EXPECT_THROW(solver::load(header), std::invalid_argument);
    const auto extent = change_file("extent", [](auto& bytes) { put_number(bytes, 64 + 40 * l_values + 8, 8, UINT64_MAX); });
    EXPECT_THROW(solver::load(extent), std::invalid_argument);
    const auto wrong_type = change("type", a_values, [](auto& a) { a.kind = 'u'; });
    EXPECT_THROW(solver::load(wrong_type), std::invalid_argument);
    const auto wrong_width = change_file("width", [](auto& bytes) { bytes.at(64 + 40 * a_values + 6) = 3; });
    EXPECT_THROW(solver::load(wrong_width), std::invalid_argument);
    const auto duplicate = change_file("duplicate", [](auto& bytes) { put_number(bytes, 64 + 40, 4, 0); });
    EXPECT_THROW(solver::load(duplicate), std::invalid_argument);
    const auto overlap = change_file("overlap", [](auto& bytes) { put_number(bytes, 64 + 40 + 16, 8, 448); });
    EXPECT_THROW(solver::load(overlap), std::invalid_argument);
    const auto truncated = change_file("truncated", [](auto& bytes) { bytes.resize(100); }, false);
    EXPECT_THROW(solver::load(truncated), std::invalid_argument);
    const auto short_payload = change_file("short-payload", [](auto& bytes) { bytes.pop_back(); }, false);
    EXPECT_THROW(solver::load(short_payload), std::invalid_argument);
    const auto trailing = change_file("trailing", [](auto& bytes) { bytes.push_back(0); put_number(bytes, 24, 8, bytes.size()); });
    EXPECT_THROW(solver::load(trailing), std::invalid_argument);
}

TEST_F(SolverArchive, RejectsNewerFormatVersion) {
    const auto path = change_file("version-2", [](auto& bytes) { put_number(bytes, 8, 4, 2); });
    try {
        solver::load(path);
        FAIL() << "a version 2 archive loaded";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(),
            "solver archive: written by a newer apxchol (format 2); upgrade to read it");
    }
}

TEST(Crc32c, KnownAnswers) {
    const std::string digits = "123456789";
    std::array<unsigned char, 32> zeros{}, ones{}, ascending{};
    ones.fill(0xff);
    for (std::size_t i = 0; i < ascending.size(); ++i) ascending[i] = static_cast<unsigned char>(i);
    std::vector<std::uint32_t (*)(std::uint32_t, const void*, std::size_t) noexcept> all{detail::crc32c};
    for (const auto& implementation : detail::crc32c_implementations()) all.push_back(implementation.checksum);
    for (auto crc : all) {
        EXPECT_EQ(crc(0, digits.data(), digits.size()), 0xE3069283u);
        EXPECT_EQ(crc(0, digits.data(), 0), 0u);
        // RFC 3720, appendix B.4.
        EXPECT_EQ(crc(0, zeros.data(), zeros.size()), 0x8A9136AAu);
        EXPECT_EQ(crc(0, ones.data(), ones.size()), 0x62A8AB43u);
        EXPECT_EQ(crc(0, ascending.data(), ascending.size()), 0x46DD794Eu);
    }
    EXPECT_EQ(reference_crc32c(reinterpret_cast<const unsigned char*>(digits.data()), digits.size()), 0xE3069283u);
}

// The folding kernel takes 32 KiB chunks on x86 and 8160-byte chunks on ARM,
// the crc instruction lanes 24 KiB and then 768-byte stretches, and the table
// 8 bytes at a time.
// Every implementation must match the bytewise reference at every alignment and
// length, and when continued.
TEST(Crc32c, EveryImplementationAgreesAtAnyAlignmentAndLength) {
    Bytes data(3 * 3 * 8192 + 1000);
    std::uint64_t state = 0x9E3779B97F4A7C15ull;
    for (auto& byte : data) { state = state * 6364136223846793005ull + 1442695040888963407ull; byte = state >> 56; }
    const auto implementations = detail::crc32c_implementations();
    ASSERT_FALSE(implementations.empty());
    std::vector<std::size_t> sizes{0, 1, 7, 8, 9, 63, 135, 136, 137, 543, 544, 545, 767, 768, 769,
                                   8159, 8160, 8161, 2 * 8160 + 7, 32767, 32768, 32769, 2 * 32768 + 7, 65536, 3 * 8192 - 1, 3 * 8192, 3 * 8192 + 5,
                                   2 * 3 * 8192 + 13};
    sizes.push_back(data.size() - 16);
    for (const auto& implementation : implementations)
        for (std::size_t offset = 0; offset < 16; ++offset)
            for (std::size_t size : sizes) {
                const auto* p = data.data() + offset;
                const auto expected = reference_crc32c(p, size);
                const auto crc = implementation.checksum;
                EXPECT_EQ(crc(0, p, size), expected) << implementation.name << " " << offset << " " << size;
                EXPECT_EQ(crc(0x12345678u, p, size), detail::crc32c_portable(0x12345678u, p, size))
                    << implementation.name << " " << offset << " " << size;
                const auto third = size / 3;
                EXPECT_EQ(crc(crc(0, p, third), p + third, size - third), expected)
                    << implementation.name << " " << offset << " " << size;
            }
}

TEST(Crc32c, CombineAcrossManyBoundaries) {
    Bytes data(200000);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<unsigned char>(i * 131 + (i >> 9));
    const auto expected = reference_crc32c(data.data(), data.size());
    for (std::size_t step : {std::size_t(1), std::size_t(7), std::size_t(4096), std::size_t(65536), std::size_t(99999)}) {
        std::uint32_t crc = 0;
        std::size_t pieces = 0;
        for (std::size_t first = 0; first < data.size(); first += step, ++pieces) {
            const auto size = std::min(step, data.size() - first);
            crc = detail::crc32c_combine(crc, detail::crc32c(0, data.data() + first, size), size);
        }
        EXPECT_EQ(crc, expected) << step << " bytes per piece, " << pieces << " pieces";
    }
    EXPECT_EQ(detail::crc32c_combine(0x12345678u, 0, 0), 0x12345678u);
    EXPECT_EQ(detail::crc32c_combine(0, 0xE3069283u, 9), 0xE3069283u);
}

// The committed fixture is tests/fixtures/path5_sddm.mtx saved with default CPU
// options (FP32 storage). Every 1.x release must keep loading it.
TEST(SolverArchiveFormat, LoadsCommittedVersion1Fixture) {
    Sparse A(5, 5);
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < 5; ++i) {
        entries.emplace_back(i, i, i == 0 || i == 4 ? 1.5 : 2.5);
        if (i) { entries.emplace_back(i, i - 1, -1.0); entries.emplace_back(i - 1, i, -1.0); }
    }
    A.setFromTriplets(entries.begin(), entries.end());
    solve_options opts;
    opts.backend = solve_backend::cpu;
    auto restored = solver::load(std::filesystem::path(APXCHOL_TEST_FIXTURES) / "path5_sddm_v1.apxchol", opts);
    EXPECT_EQ(restored.rows(), 5);
    EXPECT_EQ(restored.operator_nnz(), 13);
    EXPECT_TRUE(restored.factor().sddm);
    EXPECT_EQ(restored.factor_storage(), factor_storage_type::fp32);
    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(5, 1.0, 5.0);
    const auto result = restored.solve(b, 1e-10);
    EXPECT_TRUE(result.converged);
    EXPECT_LT((A * result.x - b).norm() / b.norm(), 1e-10);
}

TEST_F(SolverArchive, ReadsOppositeEndianArrays) {
    auto arrays = read_fixture(directory / "original.apxchol");
    for (auto& a : arrays) {
        a.endian = a.endian == '<' ? '>' : '<';
        for (std::size_t i = 0; i < a.data.size(); i += a.width)
            std::reverse(a.data.begin() + i, a.data.begin() + i + a.width);
    }
    const auto path = directory / "opposite-endian.apxchol";
    write_bytes(path, encode_fixture(arrays));
    auto restored = solver::load(path, options());
    EXPECT_EQ(csc_access::vals(restored.factor().L), csc_access::vals(original.factor().L));
    EXPECT_EQ(restored.factor().perm, original.factor().perm);
    const Eigen::VectorXd b = Eigen::VectorXd::Ones(A.rows());
    auto result = restored.solve(b, 1e-9);
    EXPECT_TRUE(result.converged);
    EXPECT_LT((A * result.x - b).norm() / b.norm(), 1e-9);
}

TEST_F(SolverArchive, ReadsWideIntegerArrays) {
    auto arrays = read_fixture(directory / "original.apxchol");
    for (auto& a : arrays) if (a.kind != 'f') widen(a);
    const auto path = directory / "wide.apxchol";
    write_bytes(path, encode_fixture(arrays));
    auto restored = solver::load(path, options());
    EXPECT_EQ(csc_access::outer(restored.factor().L), csc_access::outer(original.factor().L));
    EXPECT_EQ(csc_access::inner(restored.factor().L), csc_access::inner(original.factor().L));
    EXPECT_EQ(csc_access::vals(restored.factor().L), csc_access::vals(original.factor().L));
    EXPECT_EQ(restored.factor().perm, original.factor().perm);
}

TEST_F(SolverArchive, RejectsCorruptArrayPayload) {
    const auto path = change_file("crc", [](auto& bytes) {
        const auto offset = number(bytes, 64 + 40 * a_values + 16, 8);
        bytes.at(offset) ^= 1;
    }, false);
    EXPECT_THROW(solver::load(path), std::invalid_argument);
    const auto descriptor = change_file("metadata-crc", [](auto& bytes) { bytes.at(64 + 32) ^= 1; }, false);
    EXPECT_THROW(solver::load(descriptor), std::invalid_argument);
}

// Arrays over 4 MiB are read in blocks by several threads; the result, the
// combined checksum and the reported error must not depend on the team size.
TEST_F(SolverArchive, MultiBlockArraysLoadAtAnyTeamSize) {
    const Sparse large = matrix(700000);  // operator values span five 4 MiB blocks
    const solver source(large, options());
    const auto path = directory / "large.apxchol";
    source.save(path);
    const auto expect_error = [](const std::filesystem::path& file, int threads, const char* message) {
        auto opts = options();
        opts.num_threads = threads;
        try {
            (void)solver::load(file, opts);
            ADD_FAILURE() << "expected: " << message;
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
        }
    };
    for (int threads : {1, 4}) {
        auto opts = options();
        opts.num_threads = threads;
        const auto restored = solver::load(path, opts);
        EXPECT_EQ(csc_access::vals(restored.factor().L), csc_access::vals(source.factor().L));
        EXPECT_EQ(csc_access::inner(restored.factor().L), csc_access::inner(source.factor().L));
        EXPECT_EQ(restored.factor().perm, source.factor().perm);
        expect_operator(path, large);
    }
    constexpr std::size_t block = std::size_t(1) << 22;
    auto bytes = read_bytes(path);
    const auto values = number(bytes, 64 + 40 * a_values + 16, 8);
    const auto length = number(bytes, 64 + 40 * a_values + 24, 8);
    ASSERT_GT(length, 4 * block);
    for (const auto at : {block + 3, static_cast<std::size_t>(length) - 1}) {
        auto corrupt = bytes;
        corrupt.at(values + at) ^= 1;
        const auto file = directory / "large-corrupt.apxchol";
        write_bytes(file, corrupt);
        for (int threads : {1, 4}) expect_error(file, threads, "array checksum mismatch");
    }
    // A negative row index in a later block, with a valid checksum.
    auto fixture = read_fixture(path);
    ASSERT_GT(fixture[a_inner].data.size(), 2 * block);
    fixture[a_inner].value(block / 4 + 5, UINT32_MAX);
    const auto negative = directory / "large-negative.apxchol";
    write_bytes(negative, encode_fixture(fixture));
    for (int threads : {1, 4}) expect_error(negative, threads, "negative index");
    // A failed check in a later block is reported before a checksum mismatch
    // in an earlier one, as by a sequential reader.
    auto mixed = read_bytes(negative);
    mixed.at(number(mixed, 64 + 40 * a_inner + 16, 8)) ^= 1;
    const auto both = directory / "large-negative-and-corrupt.apxchol";
    write_bytes(both, mixed);
    for (int threads : {1, 4}) expect_error(both, threads, "negative index");
    // Rows out of order in a column far into the operator, checked in parallel.
    fixture = read_fixture(path);
    const auto at = fixture[a_inner].count - 1000;
    const auto row = fixture[a_inner].value(at);
    fixture[a_inner].value(at, fixture[a_inner].value(at + 1));
    fixture[a_inner].value(at + 1, row);
    const auto unsorted = directory / "large-unsorted.apxchol";
    write_bytes(unsorted, encode_fixture(fixture));
    for (int threads : {1, 4}) expect_error(unsorted, threads, "rows must be sorted");
}

TEST_F(SolverArchive, RejectsUnsortedAndOutOfRangeRows) {
    const auto unsorted = change("unsorted-rows", a_inner, [](auto& a) {
        const auto first = a.value(0);
        a.value(0, a.value(1));
        a.value(1, first);
    });
    EXPECT_THROW(solver::load(unsorted), std::invalid_argument);
    const auto range = change("row-range", a_inner, [](auto& a) { a.value(a.count - 1, 32); });
    EXPECT_THROW(solver::load(range), std::invalid_argument);
}

TEST_F(SolverArchive, RejectsDuplicatePermutation) {
    const auto path = change("permutation", permutation, [](auto& a) { a.value(1, a.value(0)); });
    EXPECT_THROW(solver::load(path), std::invalid_argument);
}

TEST_F(SolverArchive, AcceptsUnicodeAndExtensionlessPaths) {
    for (const auto& name : {std::filesystem::path(u8"matrice-α.apxchol"), std::filesystem::path("system")}) {
        const auto path = directory / name;
        original.save(path);
        auto restored = solver::load(path, options());
        EXPECT_EQ(csc_access::vals(restored.factor().L), csc_access::vals(original.factor().L));
        EXPECT_EQ(restored.factor().perm, original.factor().perm);
    }
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
    direct.save(directory / "duplicates.apxchol");
    EXPECT_EQ(solver::load(directory / "duplicates.apxchol").operator_nnz(), 4);
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
                const auto path = directory / ("source-" + std::to_string(source_owners++) + ".apxchol");
                resident.save(path);
                expect_operator(path, input);
                const auto F = resident.factor();
                for (std::size_t i = 0; i < F.perm.size(); ++i)
                    if (F.perm[i] != i) { ++nontrivial_permutations; break; }
                for (auto target : {solve_backend::cpu, solve_backend::gpu}) {
                    auto opts = options(); opts.backend = target;
                    if (target == solve_backend::gpu) opts.factor_opts.factor_storage = storage;
                    auto restored = solver::load(path, opts); ++loaded_owners;
                    EXPECT_EQ(restored.backend(), target);
                    EXPECT_EQ(restored.operator_nnz(), input.nonZeros());
                    EXPECT_EQ(csc_access::outer(restored.factor().L), csc_access::outer(F.L));
                    EXPECT_EQ(csc_access::inner(restored.factor().L), csc_access::inner(F.L));
                    ASSERT_EQ(csc_access::vals(restored.factor().L).size(), csc_access::vals(F.L).size());
                    EXPECT_EQ(std::memcmp(csc_access::vals(restored.factor().L).data(), csc_access::vals(F.L).data(), csc_access::vals(F.L).size() * sizeof(float)), 0);
                    EXPECT_EQ(restored.factor_storage(), F.storage);
                    EXPECT_EQ(restored.factor().scale, F.scale);
                    EXPECT_EQ(restored.factor().perm, F.perm);
                    EXPECT_EQ(restored.factor().sddm, F.sddm);
                    ASSERT_EQ(restored.factor().rounds.size(), F.rounds.size());
                    for (std::size_t i = 0; i < F.rounds.size(); ++i)
                        EXPECT_EQ(restored.factor().rounds[i].is_size, F.rounds[i].is_size);
                    check_solve_modes(input, restored, laplacian, checked_rhs);
                    if (origin == target) {
                        // A reloaded factor is the same preconditioner: bit for bit on
                        // CPU and CUDA. Metal's resident setup orders each forward row by
                        // thread timing, so a reload agrees to FP32 rounding and solves in
                        // the same number of iterations.
                        Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(input.rows(), -1.0, 2.0);
                        if (laplacian) b.array() -= b.mean();
                        const Eigen::VectorXd za = resident.apply(b), zb = restored.apply(b);
                        const auto xa = resident.solve(b, 1e-9, 500), xb = restored.solve(b, 1e-9, 500);
#if defined(APXCHOL_USE_METAL)
                        if (target == solve_backend::gpu) {
                            EXPECT_LT((za - zb).norm() / za.norm(), 1e-6);
                            EXPECT_EQ(xa.iterations, xb.iterations);
                            EXPECT_LT((xa.x - xb.x).norm() / xa.x.norm(), 1e-8);
                        } else
#endif
                        {
                            EXPECT_EQ(za, zb);
                            EXPECT_EQ(xa.x, xb.x);
                        }
                    }
                    if (target == solve_backend::gpu) {
                        const auto adopted = directory / ("adopted-" + std::to_string(resaves++) + ".apxchol");
                        restored.save(adopted); expect_operator(adopted, input);
                        auto host = solver::load(adopted, options());
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
    const auto path = directory / "positive.apxchol";
    before.save(path);
    EXPECT_THROW(solver::load(path), std::invalid_argument);
    auto admission = options(); admission.factor_opts.lump_max_mass = 1.0;
    auto restored = solver::load(path, admission);
    EXPECT_EQ(restored.factor().lumped_offdiag, 2);
    EXPECT_EQ(restored.operator_nnz(), 4);
    ASSERT_EQ(csc_access::vals(restored.factor().L).size(), csc_access::vals(before.factor().L).size());
    EXPECT_EQ(std::memcmp(csc_access::vals(restored.factor().L).data(), csc_access::vals(before.factor().L).data(),
        csc_access::vals(before.factor().L).size() * sizeof(float)), 0);
    const Eigen::VectorXd b = Eigen::VectorXd::Ones(2);
    const auto result = restored.solve(b, 1e-10);
    EXPECT_TRUE(result.converged);
    EXPECT_LT((positive * result.x - b).norm() / b.norm(), 1e-10);
}

TEST_F(SolverArchive, ReplacesSymlinksAndSerializesConcurrentWriters) {
    // A symlink destination is replaced by the archive; its target is untouched.
    const auto link = directory / "dangling.apxchol";
    std::filesystem::create_symlink(directory / "missing", link);
    original.save(link);
    EXPECT_FALSE(std::filesystem::is_symlink(link));
    EXPECT_FALSE(std::filesystem::exists(directory / "missing"));
    EXPECT_EQ(solver::load(link).rows(), A.rows());
    // Overlapping saves to one path never interleave: a save that finds the
    // other's temporary file fails and names it.
    const auto path = directory / "concurrent.apxchol";
    std::barrier start(2);
    auto write = [&] {
        start.arrive_and_wait();
        try { original.save(path); return 1; }
        catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(".tmp"), std::string::npos) << e.what();
            return 0;
        }
    };
    auto first = std::async(std::launch::async, write);
    auto second = std::async(std::launch::async, write);
    EXPECT_GE(first.get() + second.get(), 1);
    EXPECT_EQ(solver::load(path).rows(), A.rows());
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
}
} // namespace

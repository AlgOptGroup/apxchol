#include <gtest/gtest.h>
#include "apxchol/solver/archive.h"
#include "apxchol/detail/csc_import.h"
#include "../src/solver_archive_internal.h"
#include <zlib.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
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
    result.keep_factor_values = true;
    result.factor_opts.factor_storage = factor_storage_type::fp32;
    return result;
}

// Independent test fixture encoder for the documented binary layout.
using Bytes = std::vector<unsigned char>;
constexpr std::size_t prefix_size = 64 + 9 * 40;
enum ArrayId : std::size_t { metadata, a_outer, a_inner, a_values, l_outer, l_inner, l_values, permutation, round_sizes };
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
    if (size < 0 || size > 16 * 1024 * 1024) throw std::runtime_error("invalid test fixture size");
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
    put_number(bytes, 20, 4, crc32(0, bytes.data(), prefix_size));
}
struct FixtureArray {
    unsigned char endian, kind, width;
    std::uint64_t count;
    Bytes data;
    std::uint64_t value(std::size_t index) const { return number(data, index * width, width, endian == '<'); }
    void value(std::size_t index, std::uint64_t bits) { put_number(data, index * width, width, bits, endian == '<'); }
};
using Fixture = std::array<FixtureArray, 9>;
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
        put_number(bytes, d + 32, 4, crc32(0, a.data.data(), a.data.size()));
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
        directory = std::filesystem::temp_directory_path() /
            ("apxchol-archive-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ASSERT_TRUE(std::filesystem::create_directory(directory));
        save_solver(directory / "original.apxchol", original);
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
    auto restored = load_solver(directory / "original.apxchol", options());
    EXPECT_EQ(restored.operator_nnz(), A.nonZeros());
    EXPECT_EQ(original.operator_nnz(), A.nonZeros());
    const auto& before = original.factor();
    const auto& after = restored.factor();
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
    save_solver(path, before);
    auto restored = load_solver(path, options());
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
    const auto path = directory / "empty-rounds.apxchol";
    EXPECT_THROW(save_solver(path, owner), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(SolverArchive, AutomaticAdoptionUsesCpu) {
    auto opts = options(); opts.backend = solve_backend::automatic;
    auto restored = load_solver(directory / "original.apxchol", opts);
    EXPECT_EQ(restored.backend(), solve_backend::cpu);
    EXPECT_EQ(restored.factor_storage(), factor_storage_type::fp32);
}

TEST_F(SolverArchive, SavingReplacesAnExistingFile) {
    const auto path = directory / "replaced.apxchol";
    write_bytes(path, Bytes{'o', 'l', 'd'});
    save_solver(path, original);
    EXPECT_EQ(read_bytes(path), read_bytes(directory / "original.apxchol"));
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
}

TEST_F(SolverArchive, LeftoverTemporaryFileIsReportedAndKept) {
    const auto path = directory / "busy.apxchol";
    const auto temporary = path.string() + ".tmp";
    write_bytes(temporary, Bytes{'x'});
    try {
        save_solver(path, original);
        FAIL() << "expected the leftover temporary file to block the save";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(temporary), std::string::npos) << e.what();
    }
    EXPECT_TRUE(std::filesystem::exists(temporary));
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(SolverArchive, ReleasedFactorsAreRejected) {
    auto opts = options(); opts.keep_factor_values = false;
    solver released(A, opts);
    EXPECT_THROW(save_solver(directory / "released.apxchol", released), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(directory / "released.apxchol"));
}

TEST_F(SolverArchive, NewArchivesFollowTheUmask) {
    struct umask_guard {
        mode_t previous = umask(022);
        ~umask_guard() { umask(previous); }
    };
    {
        const umask_guard guard;
        save_solver(directory / "shared.apxchol", original);
    }
    struct stat info {};
    ASSERT_EQ(stat((directory / "shared.apxchol").c_str(), &info), 0);
    EXPECT_EQ(info.st_mode & 0777, 0644);
    EXPECT_FALSE(std::filesystem::exists((directory / "shared.apxchol").string() + ".tmp"));
}

TEST_F(SolverArchive, RejectsNullCharactersInPaths) {
    const auto existing = (directory / "original.apxchol").string() + std::string("\0suffix", 7);
    EXPECT_THROW(load_solver(existing), std::invalid_argument);
    const auto prefix = directory / "new.apxchol";
    const auto output = prefix.string() + std::string("\0suffix", 7);
    EXPECT_THROW(save_solver(output, original), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(prefix));
}

TEST_F(SolverArchive, InvalidOptionsFailBeforeOpeningAnArchive) {
    auto opts = options(); opts.num_threads = -1;
    EXPECT_THROW(load_solver(directory / "missing.apxchol", opts), std::invalid_argument);
    EXPECT_THROW(load_solver(directory / "missing.apxchol", options()), std::runtime_error);
}

TEST_F(SolverArchive, ReportsDirectoryReadError) {
    EXPECT_THROW(load_solver(directory, options()), std::system_error);
}

TEST_F(SolverArchive, RejectsMetadataAndIndexOverflow) {
    for (const auto& item : std::array<std::pair<std::size_t, std::uint64_t>, 6>{{
        {0, 99}, {1, UINT64_MAX}, {2, 2}, {3, UINT64_MAX}, {4, UINT64_MAX}, {6, UINT64_MAX}}}) {
        const auto path = change(("meta-" + std::to_string(item.first)).c_str(), metadata,
                                 [&](auto& a) { a.value(item.first, item.second); });
        EXPECT_THROW(load_solver(path), std::invalid_argument);
    }
    const auto bad_row = change("bad-row", a_inner, [](auto& a) { a.value(0, UINT32_MAX); });
    EXPECT_THROW(load_solver(bad_row), std::invalid_argument);
    const auto bad_offset = change("bad-offset", l_outer, [](auto& a) { a.value(0, UINT64_MAX); });
    EXPECT_THROW(load_solver(bad_offset), std::invalid_argument);
    const auto wide_row = change("wide-row", a_inner, [](auto& a) { widen(a); a.kind = 'u'; a.value(0, UINT64_MAX); });
    EXPECT_THROW(load_solver(wide_row), std::invalid_argument);
}

TEST_F(SolverArchive, RejectsMalformedHeaderAndTruncation) {
    const auto header = change_file("version", [](auto& bytes) { put_number(bytes, 8, 4, 99); });
    EXPECT_THROW(load_solver(header), std::invalid_argument);
    const auto extent = change_file("extent", [](auto& bytes) { put_number(bytes, 64 + 40 * l_values + 8, 8, UINT64_MAX); });
    EXPECT_THROW(load_solver(extent), std::invalid_argument);
    const auto wrong_type = change("type", a_values, [](auto& a) { a.kind = 'u'; });
    EXPECT_THROW(load_solver(wrong_type), std::invalid_argument);
    const auto wrong_width = change_file("width", [](auto& bytes) { bytes.at(64 + 40 * a_values + 6) = 3; });
    EXPECT_THROW(load_solver(wrong_width), std::invalid_argument);
    const auto duplicate = change_file("duplicate", [](auto& bytes) { put_number(bytes, 64 + 40, 4, 0); });
    EXPECT_THROW(load_solver(duplicate), std::invalid_argument);
    const auto overlap = change_file("overlap", [](auto& bytes) { put_number(bytes, 64 + 40 + 16, 8, 448); });
    EXPECT_THROW(load_solver(overlap), std::invalid_argument);
    const auto truncated = change_file("truncated", [](auto& bytes) { bytes.resize(100); }, false);
    EXPECT_THROW(load_solver(truncated), std::invalid_argument);
    const auto short_payload = change_file("short-payload", [](auto& bytes) { bytes.pop_back(); }, false);
    EXPECT_THROW(load_solver(short_payload), std::invalid_argument);
    const auto trailing = change_file("trailing", [](auto& bytes) { bytes.push_back(0); put_number(bytes, 24, 8, bytes.size()); });
    EXPECT_THROW(load_solver(trailing), std::invalid_argument);
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
    auto restored = load_solver(path, options());
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
    auto restored = load_solver(path, options());
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
    EXPECT_THROW(load_solver(path), std::invalid_argument);
    const auto descriptor = change_file("metadata-crc", [](auto& bytes) { bytes.at(64 + 32) ^= 1; }, false);
    EXPECT_THROW(load_solver(descriptor), std::invalid_argument);
}

TEST_F(SolverArchive, RejectsDuplicatePermutation) {
    const auto path = change("permutation", permutation, [](auto& a) { a.value(1, a.value(0)); });
    EXPECT_THROW(load_solver(path), std::invalid_argument);
}

TEST_F(SolverArchive, AcceptsUnicodeAndExtensionlessPaths) {
    for (const auto& name : {std::filesystem::path(u8"matrice-α.apxchol"), std::filesystem::path("system")}) {
        const auto path = directory / name;
        save_solver(path, original);
        auto restored = load_solver(path, options());
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
    save_solver(directory / "duplicates.apxchol", direct);
    EXPECT_EQ(load_solver(directory / "duplicates.apxchol").operator_nnz(), 4);
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
                    EXPECT_EQ(csc_access::outer(restored.factor().L), csc_access::outer(F.L));
                    EXPECT_EQ(csc_access::inner(restored.factor().L), csc_access::inner(F.L));
                    ASSERT_EQ(csc_access::vals(restored.factor().L).size(), csc_access::vals(F.L).size());
                    EXPECT_EQ(std::memcmp(csc_access::vals(restored.factor().L).data(), csc_access::vals(F.L).data(), csc_access::vals(F.L).size() * sizeof(float)), 0);
                    EXPECT_EQ(restored.factor().perm, F.perm);
                    EXPECT_EQ(restored.factor().sddm, F.sddm);
                    ASSERT_EQ(restored.factor().rounds.size(), F.rounds.size());
                    for (std::size_t i = 0; i < F.rounds.size(); ++i)
                        EXPECT_EQ(restored.factor().rounds[i].is_size, F.rounds[i].is_size);
                    check_solve_modes(input, restored, laplacian, checked_rhs);
                    if (target == solve_backend::gpu) {
                        const auto adopted = directory / ("adopted-" + std::to_string(resaves++) + ".apxchol");
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
    const auto path = directory / "positive.apxchol";
    save_solver(path, before);
    EXPECT_THROW(load_solver(path), std::invalid_argument);
    auto admission = options(); admission.factor_opts.lump_max_mass = 1.0;
    auto restored = load_solver(path, admission);
    EXPECT_EQ(restored.factor().lumped_offdiag, 2);
    EXPECT_EQ(restored.operator_nnz(), 4);
    ASSERT_EQ(csc_access::vals(restored.factor().L).size(), csc_access::vals(before.factor().L).size());
    EXPECT_EQ(std::memcmp(csc_access::vals(restored.factor().L).data(), csc_access::vals(before.factor().L).data(),
        csc_access::vals(before.factor().L).size() * sizeof(float)), 0);
    const Eigen::VectorXd b = Eigen::VectorXd::Ones(2);
    const auto result = restored.solve(b, 1e-10);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.lumped_offdiag, 2);
    EXPECT_LT((positive * result.x - b).norm() / b.norm(), 1e-10);
}

TEST_F(SolverArchive, ReplacesSymlinksAndSerializesConcurrentWriters) {
    // A symlink destination is replaced by the archive; its target is untouched.
    const auto link = directory / "dangling.apxchol";
    std::filesystem::create_symlink(directory / "missing", link);
    save_solver(link, original);
    EXPECT_FALSE(std::filesystem::is_symlink(link));
    EXPECT_FALSE(std::filesystem::exists(directory / "missing"));
    EXPECT_EQ(load_solver(link).rows(), A.rows());
    // Overlapping saves to one path never interleave: a save that finds the
    // other's temporary file fails and names it.
    const auto path = directory / "concurrent.apxchol";
    std::barrier start(2);
    auto write = [&] {
        start.arrive_and_wait();
        try { save_solver(path, original); return 1; }
        catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(".tmp"), std::string::npos) << e.what();
            return 0;
        }
    };
    auto first = std::async(std::launch::async, write);
    auto second = std::async(std::launch::async, write);
    EXPECT_GE(first.get() + second.get(), 1);
    EXPECT_EQ(load_solver(path).rows(), A.rows());
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
}
} // namespace

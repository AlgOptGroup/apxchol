#include "solver_archive_internal.h"
#include "apxchol/solver/detail/factor_validation.h"

#include <zlib.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <fcntl.h>
#include <unistd.h>

namespace apxchol {
namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("solver archive: " + message);
}

void check_path(const std::filesystem::path& path) {
    if (path.native().find('\0') != std::filesystem::path::string_type::npos)
        invalid("path contains a null character");
}

struct temporary_file {
    std::string name;
    int fd = -1;
    // open() with mode 0666 applies the caller's umask, as for any new file;
    // mkstemp would always create 0600.
    explicit temporary_file(const std::filesystem::path& destination) {
        if (std::filesystem::exists(std::filesystem::symlink_status(destination)))
            throw std::runtime_error("solver archive: destination already exists");
        std::random_device random;
        for (int attempt = 0; attempt < 100 && fd < 0; ++attempt) {
            char suffix[17];
            std::snprintf(suffix, sizeof suffix, "%08x%08x", random(), random());
            name = (destination.parent_path() / (std::string(".apxchol-") + suffix)).string();
            fd = ::open(name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
            if (fd < 0 && errno != EEXIST)
                throw std::system_error(errno, std::generic_category(), "create archive temporary");
        }
        if (fd < 0) throw std::system_error(EEXIST, std::generic_category(), "create archive temporary");
    }
    std::FILE* open() {
        auto* file = fdopen(fd, "wb+");
        if (!file) throw std::system_error(errno, std::generic_category(), "open archive temporary");
        fd = -1;
        return file;
    }
    ~temporary_file() {
        if (fd >= 0) (void)close(fd);
        (void)unlink(name.c_str());
    }
    temporary_file(const temporary_file&) = delete;
    temporary_file& operator=(const temporary_file&) = delete;
};

enum class array_id : std::uint32_t {
    metadata = 0, operator_offsets = 1, operator_indices = 2, operator_values = 3,
    factor_offsets = 4, factor_indices = 5, factor_values = 6, permutation = 7, round_sizes = 8,
    count = 9
};
constexpr auto array_count = std::to_underlying(array_id::count);
constexpr std::size_t header_bytes = 64, descriptor_bytes = 40;
constexpr std::size_t prefix_bytes = header_bytes + descriptor_bytes * array_count;
using prefix_type = std::array<unsigned char, prefix_bytes>;
struct close_file {
    void operator()(std::FILE* file) const noexcept { (void)std::fclose(file); }
};
using file_owner = std::unique_ptr<std::FILE, close_file>;

file_owner open_file(const std::filesystem::path& path, const char* mode) {
    file_owner file(std::fopen(path.c_str(), mode));
    if (!file) throw std::system_error(errno, std::generic_category(), "open solver archive");
    return file;
}

void read_exact(std::FILE* file, void* output, std::size_t size) {
    if (size && std::fread(output, 1, size, file) != size) {
        const int error = errno;
        if (std::ferror(file)) throw std::system_error(error, std::generic_category(), "read solver archive");
        invalid("truncated array");
    }
}
void write_exact(std::FILE* file, const void* input, std::size_t size) {
    if (size && std::fwrite(input, 1, size, file) != size)
        throw std::system_error(errno, std::generic_category(), "write solver archive");
}
void seek(std::FILE* file, std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) invalid("file offset overflow");
    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0)
        throw std::system_error(errno, std::generic_category(), "seek solver archive");
}
std::uint64_t align_payload(std::uint64_t offset) {
    if (offset > std::numeric_limits<std::uint64_t>::max() - 63) invalid("file offset overflow");
    return (offset + 63) & ~std::uint64_t(63);
}
template<class T> T get(const unsigned char* bytes) {
    T value;
    std::memcpy(&value, bytes, sizeof(T));
    if constexpr (std::endian::native != std::endian::little) value = std::byteswap(value);
    return value;
}
template<class T> void set(unsigned char* bytes, T value) {
    if constexpr (std::endian::native != std::endian::little) value = std::byteswap(value);
    std::memcpy(bytes, &value, sizeof(T));
}
std::uint32_t checksum(std::uint32_t crc, const void* data, std::size_t size) {
    // All callers process at most one 64 KiB chunk or the small metadata prefix.
    return static_cast<std::uint32_t>(crc32(crc, static_cast<const Bytef*>(data), static_cast<uInt>(size)));
}

struct array_info {
    std::uint64_t count, offset, length;
    std::uint32_t crc;
    char endian, kind;
    unsigned width;
};

class writer {
    std::filesystem::path destination_;
    temporary_file temporary_;
    file_owner file_;
    prefix_type prefix_{};
    std::array<bool, array_count> present_{};
    std::size_t added_ = 0;
    std::uint64_t position_ = prefix_bytes;
public:
    explicit writer(const std::filesystem::path& path)
        : destination_(std::filesystem::absolute(path)), temporary_(destination_),
          file_(temporary_.open()) {
        write_exact(file_.get(), prefix_.data(), prefix_.size());
    }

    template<class T> void add(array_id array, std::span<const T> values) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
        static_assert(sizeof(T) == 4 || sizeof(T) == 8);
        const auto id = std::to_underlying(array);
        if (id >= array_count || present_[id] || added_ == array_count) invalid("duplicate or unknown array id");
        present_[id] = true;
        const auto offset = align_payload(position_);
        const auto length = values.size_bytes();
        if (length > std::numeric_limits<std::uint64_t>::max() - offset) invalid("array length overflow");
        const std::array<unsigned char, 64> padding{};
        write_exact(file_.get(), padding.data(), static_cast<std::size_t>(offset - position_));
        std::uint32_t crc = 0;
        const auto bytes = std::as_bytes(values);
        for (std::size_t first = 0; first < bytes.size();) {
            const auto count = std::min<std::size_t>(65536, bytes.size() - first);
            crc = checksum(crc, bytes.data() + first, count);
            write_exact(file_.get(), bytes.data() + first, count);
            first += count;
        }
        auto* d = prefix_.data() + header_bytes + added_++ * descriptor_bytes;
        set<std::uint32_t>(d, id);
        d[4] = std::endian::native == std::endian::little ? '<' : '>';
        d[5] = std::is_floating_point_v<T> ? 'f' : std::is_signed_v<T> ? 'i' : 'u';
        d[6] = sizeof(T);
        set<std::uint64_t>(d + 8, values.size());
        set<std::uint64_t>(d + 16, offset);
        set<std::uint64_t>(d + 24, length);
        set<std::uint32_t>(d + 32, crc);
        position_ = offset + length;
    }

    void finish() {
        if (added_ != array_count) invalid("missing array");
        std::memcpy(prefix_.data(), "APXCHB01", 8);
        set<std::uint32_t>(prefix_.data() + 8, 1);
        set<std::uint32_t>(prefix_.data() + 12, array_count);
        set<std::uint32_t>(prefix_.data() + 16, descriptor_bytes);
        set<std::uint64_t>(prefix_.data() + 24, position_);
        set<std::uint32_t>(prefix_.data() + 20, checksum(0, prefix_.data(), prefix_.size()));
        seek(file_.get(), 0);
        write_exact(file_.get(), prefix_.data(), prefix_.size());
        if (std::fclose(file_.release()) != 0)
            throw std::system_error(errno, std::generic_category(), "finish solver archive");
        std::error_code error;
        std::filesystem::create_hard_link(temporary_.name, destination_, error);
        if (error == std::errc::file_exists)
            throw std::runtime_error("solver archive: destination already exists");
        if (error) throw std::system_error(error, "publish solver archive");
    }
};

class reader {
    file_owner file_;
    std::array<array_info, array_count> arrays_{};
public:
    explicit reader(const std::filesystem::path& path) : file_(open_file(path, "rb")) {
        prefix_type prefix;
        read_exact(file_.get(), prefix.data(), prefix.size());
        if (std::memcmp(prefix.data(), "APXCHB01", 8) || get<std::uint32_t>(prefix.data() + 8) != 1 ||
            get<std::uint32_t>(prefix.data() + 12) != array_count ||
            get<std::uint32_t>(prefix.data() + 16) != descriptor_bytes ||
            !std::all_of(prefix.begin() + 32, prefix.begin() + header_bytes, [](auto v) { return v == 0; }))
            invalid("unsupported binary header");
        const auto expected_crc = get<std::uint32_t>(prefix.data() + 20);
        set<std::uint32_t>(prefix.data() + 20, 0);
        if (checksum(0, prefix.data(), prefix.size()) != expected_crc) invalid("header checksum mismatch");
        if (fseeko(file_.get(), 0, SEEK_END) != 0)
            throw std::system_error(errno, std::generic_category(), "seek solver archive");
        const auto file_length = ftello(file_.get());
        if (file_length < 0) throw std::system_error(errno, std::generic_category(), "measure solver archive");
        if (static_cast<std::uint64_t>(file_length) != get<std::uint64_t>(prefix.data() + 24))
            invalid("file length mismatch");
        std::array<bool, array_count> present{};
        std::uint64_t position = prefix_bytes;
        for (std::size_t i = 0; i < array_count; ++i) {
            const auto* d = prefix.data() + header_bytes + i * descriptor_bytes;
            const auto id = get<std::uint32_t>(d);
            if (id >= array_count || present[id]) invalid("duplicate or unknown array id");
            present[id] = true;
            auto& a = arrays_[id];
            a = {get<std::uint64_t>(d + 8), get<std::uint64_t>(d + 16), get<std::uint64_t>(d + 24),
                 get<std::uint32_t>(d + 32), static_cast<char>(d[4]), static_cast<char>(d[5]), d[6]};
            if (d[7] || get<std::uint32_t>(d + 36) || (a.endian != '<' && a.endian != '>') ||
                (a.kind != 'f' && a.kind != 'i' && a.kind != 'u') || (a.width != 4 && a.width != 8))
                invalid("unsupported array dtype");
            if (a.count > std::numeric_limits<std::uint64_t>::max() / a.width || a.length != a.count * a.width ||
                a.offset != align_payload(position) || a.offset > static_cast<std::uint64_t>(file_length) ||
                a.length > static_cast<std::uint64_t>(file_length) - a.offset)
                invalid("invalid array extent");
            position = a.offset + a.length;
        }
        if (position != static_cast<std::uint64_t>(file_length)) invalid("trailing bytes");
    }

    array_info inspect(array_id id) const { return arrays_[std::to_underlying(id)]; }

    template<class T> void read(const array_info& info, std::span<T> values) const {
        if (info.count != values.size()) invalid("array length mismatch");
        std::size_t offset = 0;
        decode<T>(info, [&](std::span<const T> chunk) {
            std::copy(chunk.begin(), chunk.end(), values.begin() + offset);
            offset += chunk.size();
        });
    }

    template<class T> std::vector<T> read_vector(const array_info& info) const {
        std::vector<T> values;
        if (info.count > values.max_size()) invalid("array exceeds container capacity");
        values.reserve(static_cast<std::size_t>(info.count));
        decode<T>(info, [&](std::span<const T> chunk) { values.insert(values.end(), chunk.begin(), chunk.end()); });
        return values;
    }

private:
    template<class T, class Put> void decode(const array_info& info, Put put) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (info.kind != 'f' || info.width != sizeof(T)) invalid("unexpected floating-point type");
        } else if (info.kind != 'i' && info.kind != 'u') invalid("expected an integer array");
        seek(file_.get(), info.offset);
        std::array<std::uint64_t, 8192> buffer;
        std::array<T, 8192> decoded;
        const bool swap = (info.endian == '<') != (std::endian::native == std::endian::little);
        std::uint32_t crc = 0;
        std::size_t first = 0;
        while (first < info.count) {
            const auto count = std::min({static_cast<std::size_t>(info.count) - first, sizeof(buffer) / info.width, decoded.size()});
            const char target_kind = std::is_floating_point_v<T> ? 'f' : std::is_signed_v<T> ? 'i' : 'u';
            if (!swap && info.width == sizeof(T) && info.kind == target_kind) {
                read_exact(file_.get(), decoded.data(), count * sizeof(T));
                crc = checksum(crc, decoded.data(), count * sizeof(T));
                if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
                    for (std::size_t i = 0; i < count; ++i)
                        if (decoded[i] < 0) invalid("negative index");
            } else {
                read_exact(file_.get(), buffer.data(), count * info.width);
                crc = checksum(crc, buffer.data(), count * info.width);
                for (std::size_t i = 0; i < count; ++i) {
                    std::uint64_t bits = 0;
                    if (info.width == 4) {
                        std::uint32_t word;
                        std::memcpy(&word, reinterpret_cast<const std::byte*>(buffer.data()) + 4 * i, 4);
                        bits = swap ? std::byteswap(word) : word;
                    } else {
                        bits = buffer[i];
                        if (swap) bits = std::byteswap(bits);
                    }
                    if constexpr (std::is_floating_point_v<T>) {
                        if constexpr (sizeof(T) == 4) decoded[i] = std::bit_cast<float>(static_cast<std::uint32_t>(bits));
                        else decoded[i] = std::bit_cast<double>(bits);
                    } else {
                        if ((info.kind == 'i' && (bits >> (info.width * 8 - 1))) ||
                            bits > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
                            invalid("index does not fit this build");
                        decoded[i] = static_cast<T>(bits);
                    }
                }
            }
            put(std::span<const T>(decoded.data(), count));
            first += count;
        }
        if (crc != info.crc) invalid("array checksum mismatch");
    }
};

void check_operator_structure(const Eigen::SparseMatrix<double>& A) {
    if (A.rows() == 0 || A.rows() != A.cols() || !A.isCompressed()) invalid("operator must be nonempty square compressed CSC");
    const auto n = A.rows(), nnz = A.nonZeros();
    if (A.outerIndexPtr()[0] != 0 || A.outerIndexPtr()[n] != nnz) invalid("invalid operator offsets");
    for (Eigen::Index col = 0; col < n; ++col) {
        const auto first = A.outerIndexPtr()[col], last = A.outerIndexPtr()[col + 1];
        if (first < 0 || first > last || last > nnz) invalid("invalid operator offsets");
        int previous = -1;
        for (auto p = first; p < last; ++p) {
            const auto row = A.innerIndexPtr()[p];
            if (row <= previous || row >= n) invalid("operator rows must be sorted, unique and in range");
            previous = row;
        }
    }
}

template<class T> void require_extent(const array_info& info, std::uint64_t expected) {
    if (info.count != expected || expected > std::numeric_limits<std::size_t>::max() / sizeof(T))
        invalid("invalid array extent");
    if constexpr (std::is_floating_point_v<T>) {
        if (info.kind != 'f' || info.width != sizeof(T)) invalid("unexpected floating-point type");
    } else if (info.kind != 'i' && info.kind != 'u') invalid("expected integer array");
}

} // namespace

void detail::save_solver_data(const std::filesystem::path& path, const Eigen::SparseMatrix<double>& A,
                              const factorization& F, Eigen::Index input_entries) {
    if (A.rows() != static_cast<Eigen::Index>(F.L.rows())) invalid("operator/factor dimension mismatch");
    Eigen::SparseMatrix<double> compressed;
    const auto* op = &A;
    if (!A.isCompressed()) { compressed = A; compressed.makeCompressed(); op = &compressed; }
    check_operator_structure(*op);
    detail::validate_adopted_operator(*op, F, 1.0);
    const auto n = static_cast<std::size_t>(op->rows());
    const auto annz = static_cast<std::size_t>(op->nonZeros());
    if (F.lumped_offdiag < 0) invalid("negative lumped entry count");
    const std::array<std::uint64_t, 7> metadata{1, n, F.sddm, annz, F.L.vals_.size(),
        static_cast<std::uint64_t>(F.lumped_offdiag), static_cast<std::uint64_t>(input_entries)};
    std::vector<std::uint64_t> rounds;
    rounds.reserve(F.rounds.size());
    for (const auto& round : F.rounds) rounds.push_back(round.is_size);
    writer archive(path);
    archive.add(array_id::metadata, std::span<const std::uint64_t>(metadata));
    archive.add(array_id::operator_offsets, std::span<const int>(op->outerIndexPtr(), n + 1));
    archive.add(array_id::operator_indices, std::span<const int>(op->innerIndexPtr(), annz));
    archive.add(array_id::operator_values, std::span<const double>(op->valuePtr(), annz));
    archive.add(array_id::factor_offsets, std::span<const edge_index>(F.L.outer_));
    archive.add(array_id::factor_indices, std::span<const node_index>(F.L.inner_));
    archive.add(array_id::factor_values, std::span<const factor_value_t>(F.L.vals_));
    archive.add(array_id::permutation, std::span<const node_index>(F.perm));
    archive.add(array_id::round_sizes, std::span<const std::uint64_t>(rounds));
    archive.finish();
}

void save_solver(const std::filesystem::path& path, const solver& owner) {
    check_path(path);
    detail::solver_archive_access::save(path, owner);
}

solver load_solver(const std::filesystem::path& path, const solve_options& options) {
    detail::validate_solve_options(options);
    check_path(path);
    reader archive(path);
    const auto metadata_info = archive.inspect(array_id::metadata);
    require_extent<std::uint64_t>(metadata_info, 7);
    std::array<std::uint64_t, 7> metadata{};
    archive.read(metadata_info, std::span<std::uint64_t>(metadata));
    const auto [version, n, sddm, annz, lnnz, lumped, input_entries] = metadata;
    if (version != 1) invalid("unsupported schema version");
    if (n == 0 || n > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<node_index>::max() || sddm > 1 ||
        annz > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        input_entries > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        lnnz > std::numeric_limits<edge_index>::max() ||
        lumped > static_cast<std::uint64_t>(std::numeric_limits<Eigen::Index>::max())) invalid("dimensions do not fit this build");
    const auto ao = archive.inspect(array_id::operator_offsets), ai = archive.inspect(array_id::operator_indices), av = archive.inspect(array_id::operator_values);
    const auto lo = archive.inspect(array_id::factor_offsets), li = archive.inspect(array_id::factor_indices), lv = archive.inspect(array_id::factor_values);
    const auto perm = archive.inspect(array_id::permutation), rounds = archive.inspect(array_id::round_sizes);
    require_extent<int>(ao, n + 1); require_extent<int>(ai, annz); require_extent<double>(av, annz);
    require_extent<edge_index>(lo, n + 1); require_extent<node_index>(li, lnnz); require_extent<float>(lv, lnnz);
    require_extent<node_index>(perm, n);
    if (rounds.count > n) invalid("too many elimination rounds");
    require_extent<std::uint64_t>(rounds, rounds.count);
    Eigen::SparseMatrix<double> A;
    {
        auto outer = archive.read_vector<int>(ao);
        if (outer.front() != 0 || outer.back() != static_cast<int>(annz) || !std::is_sorted(outer.begin(), outer.end()))
            invalid("invalid operator offsets");
        A.resize(static_cast<int>(n), static_cast<int>(n));
        A.resizeNonZeros(static_cast<int>(annz));
        std::copy(outer.begin(), outer.end(), A.outerIndexPtr());
    }
    archive.read(ai, std::span<int>(A.innerIndexPtr(), annz));
    archive.read(av, std::span<double>(A.valuePtr(), annz));
    check_operator_structure(A);
    factorization F;
    F.L.n_ = static_cast<node_index>(n);
    F.sddm = sddm;
    F.lumped_offdiag = static_cast<Eigen::Index>(lumped);
    F.L.outer_ = archive.read_vector<edge_index>(lo);
    F.L.inner_ = archive.read_vector<node_index>(li);
    F.L.vals_ = archive.read_vector<float>(lv);
    F.perm = archive.read_vector<node_index>(perm);
    const auto sizes = archive.read_vector<std::uint64_t>(rounds);
    for (auto count : sizes) {
        if (count > n) invalid("invalid elimination round size");
        F.rounds.push_back({0, static_cast<std::size_t>(count), std::numeric_limits<double>::quiet_NaN(), 0, 0});
    }
    solver restored(A, std::move(F), options);
    detail::solver_archive_access::restore_input_entries(restored, static_cast<Eigen::Index>(input_entries));
    return restored;
}

} // namespace apxchol

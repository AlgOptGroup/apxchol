#include "solver_archive_internal.h"
#include "apxchol/solver/detail/factor_validation.h"
#include "apxchol/solver/detail/solver_route.h"

#include <zlib.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
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

// The archive is written to <destination>.tmp and renamed into place, so readers
// see either the old file or the complete new one. O_EXCL makes a second save
// to the same path fail instead of interleaving writes, and mode 0666 applies
// the caller's umask.
struct temporary_file {
    std::string name;
    int fd = -1;
    bool published = false;
    explicit temporary_file(const std::filesystem::path& destination)
        : name(destination.string() + ".tmp") {
        fd = ::open(name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
        if (fd >= 0) return;
        if (errno == EEXIST)
            throw std::runtime_error("solver archive: " + name + " already exists; another save "
                                     "to this path is running, or an interrupted save left it "
                                     "behind and it must be removed");
        throw std::system_error(errno, std::generic_category(), "create " + name);
    }
    std::FILE* open() {
        auto* file = fdopen(fd, "wb+");
        if (!file) throw std::system_error(errno, std::generic_category(), "open " + name);
        fd = -1;
        return file;
    }
    ~temporary_file() {
        if (fd >= 0) (void)close(fd);
        if (!published) (void)unlink(name.c_str());
    }
    temporary_file(const temporary_file&) = delete;
    temporary_file& operator=(const temporary_file&) = delete;
};

enum class array_id : std::uint32_t {
    metadata = 0, operator_offsets = 1, operator_indices = 2, operator_values = 3,
    factor_offsets = 4, factor_indices = 5, factor_values = 6, permutation = 7, round_sizes = 8,
    factor_diagonal = 9, factor_scales = 10, count = 11
};
constexpr std::size_t metadata_count = 8;
// Metadata storage codes.
constexpr std::uint64_t storage_fp32 = 0, storage_fp16 = 1;

// IEEE 754 binary16 bit patterns, the element type of FP16 factor values.
struct binary16 { std::uint16_t bits; };
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
        constexpr bool half = std::is_same_v<T, binary16>;
        static_assert(half || std::is_integral_v<T> || std::is_floating_point_v<T>);
        static_assert(half || sizeof(T) == 4 || sizeof(T) == 8);
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
        d[5] = half || std::is_floating_point_v<T> ? 'f' : std::is_signed_v<T> ? 'i' : 'u';
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
        if (std::fflush(file_.get()) != 0 || ::fsync(fileno(file_.get())) != 0)
            throw std::system_error(errno, std::generic_category(), "flush solver archive");
        if (std::fclose(file_.release()) != 0)
            throw std::system_error(errno, std::generic_category(), "finish solver archive");
        if (std::rename(temporary_.name.c_str(), destination_.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(), "publish solver archive");
        temporary_.published = true;
        // Sync the directory so the rename itself survives a power loss.
        const auto parent = destination_.parent_path();
        const int directory = ::open(parent.empty() ? "." : parent.c_str(),
                                     O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory < 0)
            throw std::system_error(errno, std::generic_category(), "open archive directory");
        const int synced = ::fsync(directory);
        const int error = errno;
        (void)::close(directory);
        if (synced != 0)
            throw std::system_error(error, std::generic_category(), "sync archive directory");
    }
};

class reader {
    file_owner file_;
    std::array<array_info, array_count> arrays_{};
public:
    explicit reader(const std::filesystem::path& path) : file_(open_file(path, "rb")) {
        prefix_type prefix;
        read_exact(file_.get(), prefix.data(), prefix.size());
        const auto version = get<std::uint32_t>(prefix.data() + 8);
        if (!std::memcmp(prefix.data(), "APXCHB01", 8) && version > 1)
            invalid("written by a newer apxchol (format " + std::to_string(version) + "); upgrade to read it");
        if (std::memcmp(prefix.data(), "APXCHB01", 8) || version != 1 ||
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
                (a.kind != 'f' && a.kind != 'i' && a.kind != 'u') ||
                (a.width != 4 && a.width != 8 && !(a.width == 2 && a.kind == 'f')))
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
        if constexpr (std::is_same_v<T, binary16> || std::is_floating_point_v<T>) {
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
            const char target_kind = std::is_same_v<T, binary16> || std::is_floating_point_v<T>
                ? 'f' : std::is_signed_v<T> ? 'i' : 'u';
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
                    if (info.width == 2) {
                        std::uint16_t word;
                        std::memcpy(&word, reinterpret_cast<const std::byte*>(buffer.data()) + 2 * i, 2);
                        bits = swap ? std::byteswap(word) : word;
                    } else if (info.width == 4) {
                        std::uint32_t word;
                        std::memcpy(&word, reinterpret_cast<const std::byte*>(buffer.data()) + 4 * i, 4);
                        bits = swap ? std::byteswap(word) : word;
                    } else {
                        bits = buffer[i];
                        if (swap) bits = std::byteswap(bits);
                    }
                    if constexpr (std::is_same_v<T, binary16>) {
                        decoded[i] = {static_cast<std::uint16_t>(bits)};
                    } else if constexpr (std::is_floating_point_v<T>) {
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

// nnz is the allocated entry count of the arrays behind A.
void check_operator_structure(detail::csc_ref<double> A, std::uint64_t nnz) {
    if (A.n == 0) invalid("operator must be nonempty square compressed CSC");
    const std::uint64_t n = A.n;
    if (A.ptr[0] != 0 || A.ptr[n] != nnz) invalid("invalid operator offsets");
    for (std::uint64_t col = 0; col < n; ++col) {
        const std::uint64_t first = A.ptr[col], last = A.ptr[col + 1];
        if (first > last || last > nnz) invalid("invalid operator offsets");
        for (auto p = first; p < last; ++p) {
            const std::uint64_t row = A.idx[p];
            if ((p > first && row <= A.idx[p - 1]) || row >= n)
                invalid("operator rows must be sorted, unique and in range");
        }
    }
}

// The format stores signed 32-bit operator indices (docs/serialization.md).
// 32-bit unsigned indices below 2^31 have the same bytes and are written in
// place; wider ones are narrowed into `narrowed`.
template<class Index>
std::span<const int> int32_indices(std::span<const Index> values, std::vector<int>& narrowed) {
    for (const Index value : values)
        if (value > static_cast<Index>(std::numeric_limits<int>::max()))
            invalid("operator exceeds the format's 32-bit indices");
    if constexpr (sizeof(Index) == sizeof(int)) {
        return {reinterpret_cast<const int*>(values.data()), values.size()};
    } else {
        narrowed.assign(values.begin(), values.end());
        return narrowed;
    }
}

template<class T> void require_extent(const array_info& info, std::uint64_t expected) {
    if (info.count != expected || expected > std::numeric_limits<std::size_t>::max() / sizeof(T))
        invalid("invalid array extent");
    if constexpr (std::is_same_v<T, binary16> || std::is_floating_point_v<T>) {
        if (info.kind != 'f' || info.width != sizeof(T)) invalid("unexpected floating-point type");
    } else if (info.kind != 'i' && info.kind != 'u') invalid("expected integer array");
}

} // namespace

void detail::save_solver_data(const std::filesystem::path& path, csc_ref<double> A,
                              const factorization& F, std::int64_t input_entries) {
    using access = detail::sparse_csc_access;
    if (A.n != F.L.rows()) invalid("operator/factor dimension mismatch");
    if (F.storage == factor_storage_type::automatic) invalid("only an installed factor can be saved");
    detail::validate_adopted_factor(F);
    if (F.rounds.size() > F.L.rows()) invalid("too many elimination rounds");
    check_operator_structure(A, A.nnz());
    detail::validate_adopted_operator(A, F, 1.0);
    const auto n = static_cast<std::size_t>(A.n);
    const auto annz = static_cast<std::size_t>(A.nnz());
    if (F.lumped_offdiag < 0) invalid("negative lumped entry count");
    const bool fp16 = F.storage == factor_storage_type::fp16;
    const std::array<std::uint64_t, metadata_count> metadata{1, n, F.sddm, annz, access::vals(F.L).size(),
        static_cast<std::uint64_t>(F.lumped_offdiag), static_cast<std::uint64_t>(input_entries),
        fp16 ? storage_fp16 : storage_fp32};
    std::vector<std::uint64_t> rounds;
    rounds.reserve(F.rounds.size());
    for (const auto& round : F.rounds) rounds.push_back(round.is_size);
    // FP16 factors move each diagonal d_j into its own FP32 array; the
    // binary16 diagonal slots hold zero.
    std::vector<binary16> half;
    std::vector<float> diagonal;
    if (fp16) {
        half.resize(access::vals(F.L).size());
        diagonal.assign(n, 0.0f);
        for (std::size_t col = 0; col < n; ++col)
            for (auto p = access::outer(F.L)[col]; p < access::outer(F.L)[col + 1]; ++p) {
                if (access::inner(F.L)[p] == col) diagonal[col] = access::vals(F.L)[p];
                else half[p] = {std::bit_cast<std::uint16_t>(static_cast<_Float16>(access::vals(F.L)[p]))};
            }
    }
    writer archive(path);
    archive.add(array_id::metadata, std::span<const std::uint64_t>(metadata));
    std::vector<int> narrowed_offsets, narrowed_indices;
    archive.add(array_id::operator_offsets,
                int32_indices(std::span<const edge_index>(A.ptr, n + 1), narrowed_offsets));
    archive.add(array_id::operator_indices,
                int32_indices(std::span<const node_index>(A.idx, annz), narrowed_indices));
    archive.add(array_id::operator_values, std::span<const double>(A.val, annz));
    archive.add(array_id::factor_offsets, std::span<const edge_index>(access::outer(F.L)));
    archive.add(array_id::factor_indices, std::span<const node_index>(access::inner(F.L)));
    if (fp16) archive.add(array_id::factor_values, std::span<const binary16>(half));
    else archive.add(array_id::factor_values, std::span<const factor_value_t>(access::vals(F.L)));
    archive.add(array_id::permutation, std::span<const node_index>(F.perm));
    archive.add(array_id::round_sizes, std::span<const std::uint64_t>(rounds));
    archive.add(array_id::factor_diagonal, std::span<const float>(diagonal));
    archive.add(array_id::factor_scales, std::span<const float>(F.scale));
    archive.finish();
}

void solver::save(const std::filesystem::path& path) const {
    check_path(path);
    detail::solver_access::save(path, *this);
}

solver solver::load(const std::filesystem::path& path, const solve_options& options) {
    const auto started = std::chrono::steady_clock::now();
    detail::validate_solve_options(options);
    check_path(path);
    reader archive(path);
    const auto metadata_info = archive.inspect(array_id::metadata);
    require_extent<std::uint64_t>(metadata_info, metadata_count);
    std::array<std::uint64_t, metadata_count> metadata{};
    archive.read(metadata_info, std::span<std::uint64_t>(metadata));
    const auto [version, n, sddm, annz, lnnz, lumped, input_entries, storage] = metadata;
    if (version != 1) invalid("unsupported schema version");
    if (storage != storage_fp32 && storage != storage_fp16) invalid("unknown factor storage");
    const bool fp16 = storage == storage_fp16;
    if (n == 0 || n > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<node_index>::max() || sddm > 1 ||
        annz > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        input_entries > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        lnnz > std::numeric_limits<edge_index>::max() ||
        lumped > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) invalid("dimensions do not fit this build");
    const auto ao = archive.inspect(array_id::operator_offsets), ai = archive.inspect(array_id::operator_indices), av = archive.inspect(array_id::operator_values);
    const auto lo = archive.inspect(array_id::factor_offsets), li = archive.inspect(array_id::factor_indices), lv = archive.inspect(array_id::factor_values);
    const auto perm = archive.inspect(array_id::permutation), rounds = archive.inspect(array_id::round_sizes);
    const auto ld = archive.inspect(array_id::factor_diagonal), ls = archive.inspect(array_id::factor_scales);
    require_extent<int>(ao, n + 1); require_extent<int>(ai, annz); require_extent<double>(av, annz);
    require_extent<edge_index>(lo, n + 1); require_extent<node_index>(li, lnnz);
    if (fp16) require_extent<binary16>(lv, lnnz); else require_extent<float>(lv, lnnz);
    require_extent<float>(ld, fp16 ? n : 0); require_extent<float>(ls, fp16 ? n : 0);
    require_extent<node_index>(perm, n);
    if (rounds.count > n) invalid("too many elimination rounds");
    require_extent<std::uint64_t>(rounds, rounds.count);
    detail::csc_matrix<double> A(static_cast<node_index>(n), static_cast<edge_index>(annz));
    archive.read(ao, std::span<edge_index>(A.ptr(), n + 1));
    if (A.ptr()[0] != 0 || A.ptr()[n] != annz || !std::is_sorted(A.ptr(), A.ptr() + n + 1))
        invalid("invalid operator offsets");
    archive.read(ai, std::span<node_index>(A.idx(), annz));
    archive.read(av, std::span<double>(A.val(), annz));
    check_operator_structure(A.ref(), annz);
    factorization F;
    using access = detail::sparse_csc_access;
    access::n(F.L) = static_cast<node_index>(n);
    F.sddm = sddm;
    F.lumped_offdiag = static_cast<decltype(F.lumped_offdiag)>(lumped);
    access::outer(F.L) = archive.read_vector<edge_index>(lo);
    access::inner(F.L) = archive.read_vector<node_index>(li);
    if (access::outer(F.L).front() != 0 || access::outer(F.L).back() != lnnz || !std::is_sorted(access::outer(F.L).begin(), access::outer(F.L).end()))
        invalid("invalid factor offsets");
    if (fp16) {
        F.storage = factor_storage_type::fp16;
        const auto half = archive.read_vector<binary16>(lv);
        const auto diagonal = archive.read_vector<float>(ld);
        F.scale = archive.read_vector<float>(ls);
        access::vals(F.L).resize(half.size());
        for (std::size_t col = 0; col < n; ++col)
            for (auto p = access::outer(F.L)[col]; p < access::outer(F.L)[col + 1]; ++p)
                access::vals(F.L)[p] = access::inner(F.L)[p] == col ? diagonal[col]
                    : static_cast<float>(std::bit_cast<_Float16>(half[p].bits));
    } else {
        F.storage = factor_storage_type::fp32;
        access::vals(F.L) = archive.read_vector<float>(lv);
    }
    F.perm = archive.read_vector<node_index>(perm);
    const auto sizes = archive.read_vector<std::uint64_t>(rounds);
    std::uint64_t eliminated = 0;
    for (auto count : sizes) {
        if (count > n - eliminated) invalid("invalid elimination round size");
        // A round's active count is what earlier rounds left.
        F.rounds.push_back({static_cast<std::size_t>(n - eliminated), static_cast<std::size_t>(count)});
        eliminated += count;
    }
    solver restored = detail::solver_access::make(A.ref(), std::move(F), options);
    detail::solver_access::finish_load(restored, static_cast<std::int64_t>(input_entries), started);
    return restored;
}

} // namespace apxchol

#include "apxchol/solver/archive.h"
#include "apxchol/solver/detail/factor_validation.h"

#include <zip.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <string_view>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <unistd.h>

namespace apxchol {
namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("solver archive: " + message);
}

void zip_error(zip_t* file, const std::string& operation) {
    throw std::runtime_error("solver archive: " + operation + ": " + zip_strerror(file));
}

using zip_owner = std::unique_ptr<zip_t, decltype(&zip_discard)>;
using entry_owner = std::unique_ptr<zip_file_t, decltype(&zip_fclose)>;

zip_owner open_zip(const std::filesystem::path& path, int flags) {
    int error = 0;
    zip_owner result(zip_open(path.c_str(), flags, &error), &zip_discard);
    if (!result) {
        zip_error_t detail;
        zip_error_init_with_code(&detail, error);
        const std::string message = zip_error_strerror(&detail);
        zip_error_fini(&detail);
        throw std::runtime_error("solver archive: " + message);
    }
    return result;
}

template<class T> std::string dtype() {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    return std::string(1, std::endian::native == std::endian::little ? '<' : '>')
        + (std::is_floating_point_v<T> ? 'f' : std::is_signed_v<T> ? 'i' : 'u')
        + std::to_string(sizeof(T));
}

std::string npy_header(const std::string& type, std::size_t count) {
    std::string header = "{'descr': '" + type + "', 'fortran_order': False, 'shape': ("
        + std::to_string(count) + ",), }";
    header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
    header += '\n';
    std::string result("\x93NUMPY\x01\x00", 8);
    result += static_cast<char>(header.size() & 255);
    result += static_cast<char>(header.size() >> 8);
    return result + header;
}

struct temporary_file {
    std::string name;
    explicit temporary_file(const std::filesystem::path& destination)
        : name((destination.parent_path() / ".apxchol-XXXXXX").string()) {
        if (std::filesystem::exists(std::filesystem::symlink_status(destination)))
            throw std::runtime_error("solver archive: destination already exists");
        const int fd = mkstemp(name.data());
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "create archive temporary");
        (void)close(fd);
    }
    ~temporary_file() { (void)unlink(name.c_str()); }
    temporary_file(const temporary_file&) = delete;
    temporary_file& operator=(const temporary_file&) = delete;
};

class writer {
    std::filesystem::path destination_;
    temporary_file temporary_;
    std::deque<std::string> headers_;
    zip_owner file_;
public:
    explicit writer(const std::filesystem::path& path)
        : destination_(std::filesystem::absolute(path)), temporary_(destination_),
          file_(open_zip(temporary_.name, ZIP_TRUNCATE)) {}

    template<class T> void add(const char* name, std::span<const T> values) {
        headers_.push_back(npy_header(dtype<T>(), values.size()));
        const auto& header = headers_.back();
        const zip_buffer_fragment_t fragments[] = {
            {const_cast<zip_uint8_t*>(reinterpret_cast<const zip_uint8_t*>(header.data())), header.size()},
            {const_cast<zip_uint8_t*>(reinterpret_cast<const zip_uint8_t*>(values.data())), values.size_bytes()}
        };
        zip_source_t* source = zip_source_buffer_fragment(file_.get(), fragments, 2, 0);
        if (!source) zip_error(file_.get(), "create array");
        const auto index = zip_file_add(file_.get(), name, source, ZIP_FL_ENC_UTF_8);
        if (index < 0) {
            zip_source_free(source);
            zip_error(file_.get(), "add array");
        }
        if (zip_set_file_compression(file_.get(), static_cast<zip_uint64_t>(index), ZIP_CM_STORE, 0) < 0)
            zip_error(file_.get(), "set compression");
    }

    void finish() {
        if (zip_close(file_.get()) != 0) zip_error(file_.get(), "finish writing");
        (void)file_.release();
        // Publishing a hard link fails if another writer created the destination.
        std::filesystem::create_hard_link(temporary_.name, destination_);
    }
};

void read_exact(zip_file_t* file, void* destination, std::size_t size) {
    auto* output = static_cast<std::byte*>(destination);
    while (size) {
        const auto got = zip_fread(file, output, size);
        if (got < 0) invalid(std::string("cannot read array: ") + zip_file_strerror(file));
        if (got == 0) invalid("truncated array");
        output += got;
        size -= static_cast<std::size_t>(got);
    }
}

struct array_info {
    zip_uint64_t index;
    std::uint64_t count;
    std::uint32_t header_size;
    char endian, kind;
    unsigned width;
};

class reader {
    zip_owner file_;
public:
    explicit reader(const std::filesystem::path& path) : file_(open_zip(path, ZIP_RDONLY | ZIP_CHECKCONS)) {}

    array_info inspect(const char* name) const {
        zip_int64_t index = -1;
        const auto entries = zip_get_num_entries(file_.get(), 0);
        for (zip_int64_t i = 0; i < entries; ++i) {
            const char* candidate = zip_get_name(file_.get(), i, ZIP_FL_ENC_STRICT);
            if (!candidate) zip_error(file_.get(), "read entry name");
            if (std::strcmp(candidate, name) == 0) {
                if (index >= 0) invalid(std::string("duplicate array ") + name);
                index = i;
            }
        }
        if (index < 0) invalid(std::string("missing array ") + name);
        zip_stat_t stat;
        zip_stat_init(&stat);
        if (zip_stat_index(file_.get(), index, 0, &stat) < 0) zip_error(file_.get(), "read array metadata");
        if (!(stat.valid & ZIP_STAT_SIZE) || stat.encryption_method != ZIP_EM_NONE ||
            (stat.comp_method != ZIP_CM_STORE && stat.comp_method != ZIP_CM_DEFLATE))
            invalid("unsupported ZIP entry");
        auto entry = open(index);
        std::array<unsigned char, 12> prefix{};
        read_exact(entry.get(), prefix.data(), 8);
        if (std::memcmp(prefix.data(), "\x93NUMPY", 6) ||
            (prefix[6] != 1 && prefix[6] != 2) || prefix[7] != 0)
            invalid("unsupported NPY header");
        const unsigned prefix_size = prefix[6] == 1 ? 10 : 12;
        read_exact(entry.get(), prefix.data() + 8, prefix_size - 8);
        std::uint32_t size = prefix[8] | (std::uint32_t(prefix[9]) << 8);
        if (prefix_size == 12) size |= (std::uint32_t(prefix[10]) << 16) | (std::uint32_t(prefix[11]) << 24);
        if (size == 0 || size > 1024 || stat.size < prefix_size + size)
            invalid("invalid NPY header length");
        std::string header(size, '\0');
        read_exact(entry.get(), header.data(), size);
        std::string_view fields(header);
        const auto whitespace = [&] {
            while (!fields.empty() && (fields.front() == ' ' || fields.front() == '\n' || fields.front() == '\t'))
                fields.remove_prefix(1);
        };
        const auto consume = [&](std::string_view token) {
            whitespace();
            if (!fields.starts_with(token)) invalid("invalid NPY header");
            fields.remove_prefix(token.size());
        };
        const auto quoted = [&]() {
            whitespace();
            if (fields.empty() || (fields.front() != '\'' && fields.front() != '"')) invalid("expected NPY string");
            const char quote = fields.front(); fields.remove_prefix(1);
            const auto end = fields.find(quote);
            if (end == std::string_view::npos) invalid("unterminated NPY string");
            const auto result = fields.substr(0, end); fields.remove_prefix(end + 1);
            return result;
        };
        char endian = 0, kind = 0;
        unsigned width = 0, found = 0;
        std::uint64_t count = 0;
        consume("{");
        while (true) {
            whitespace();
            if (fields.starts_with('}')) { fields.remove_prefix(1); break; }
            const auto key = quoted(); consume(":");
            unsigned bit = 0;
            if (key == "descr") {
                bit = 1; const auto type = quoted();
                if (type.size() != 3) invalid("unsupported NPY dtype");
                endian = type[0]; kind = type[1]; width = type[2] - '0';
                if ((endian != '<' && endian != '>') || (kind != 'u' && kind != 'i' && kind != 'f') ||
                    (width != 4 && width != 8)) invalid("unsupported NPY dtype");
            } else if (key == "fortran_order") {
                bit = 2; whitespace();
                if (fields.starts_with("True")) consume("True");
                else consume("False"); // Both orders are contiguous for one-dimensional arrays.
            } else if (key == "shape") {
                bit = 4; consume("("); whitespace();
                const auto parsed = std::from_chars(fields.data(), fields.data() + fields.size(), count);
                if (parsed.ec != std::errc{}) invalid("array length overflow");
                fields.remove_prefix(static_cast<std::size_t>(parsed.ptr - fields.data()));
                consume(","); consume(")");
            } else invalid("unknown NPY header key");
            if (found & bit) invalid("duplicate NPY header key");
            found |= bit; whitespace();
            if (fields.starts_with('}')) continue;
            consume(",");
        }
        whitespace();
        if (found != 7 || !fields.empty()) invalid("incomplete NPY header");
        const auto header_size = prefix_size + size;
        if (count > (std::numeric_limits<std::uint64_t>::max() - header_size) / width ||
            stat.size != header_size + count * width)
            invalid("array shape and ZIP length differ");
        return {static_cast<zip_uint64_t>(index), count, header_size,
                endian, kind, width};
    }

    entry_owner open(zip_uint64_t index) const {
        entry_owner entry(zip_fopen_index(file_.get(), index, 0), &zip_fclose);
        if (!entry) zip_error(file_.get(), "open array");
        return entry;
    }

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
        // Reserve address space without touching a forged declared payload.
        values.reserve(static_cast<std::size_t>(info.count));
        decode<T>(info, [&](std::span<const T> chunk) { values.insert(values.end(), chunk.begin(), chunk.end()); });
        return values;
    }

private:
    template<class T, class Put> void decode(const array_info& info, Put put) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (info.kind != 'f' || info.width != sizeof(T)) invalid("unexpected floating-point type");
        } else if (info.kind != 'i' && info.kind != 'u') invalid("expected an integer array");
        auto entry = open(info.index);
        std::array<std::byte, 1040> header{};
        read_exact(entry.get(), header.data(), info.header_size);
        std::array<std::uint64_t, 8192> buffer;
        std::array<T, 8192> decoded;
        const bool swap = (info.endian == '<') != (std::endian::native == std::endian::little);
        std::size_t first = 0;
        while (first < info.count) {
            const auto count = std::min({static_cast<std::size_t>(info.count) - first, sizeof(buffer) / info.width, decoded.size()});
            read_exact(entry.get(), buffer.data(), count * info.width);
            const char target_kind = std::is_floating_point_v<T> ? 'f' : std::is_signed_v<T> ? 'i' : 'u';
            if (!swap && info.width == sizeof(T) && info.kind == target_kind) {
                std::memcpy(decoded.data(), buffer.data(), count * sizeof(T));
                if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
                    for (std::size_t i = 0; i < count; ++i)
                        if (decoded[i] < 0) invalid("negative index");
            } else for (std::size_t i = 0; i < count; ++i) {
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
            put(std::span<const T>(decoded.data(), count));
            first += count;
        }
        std::byte tail;
        if (zip_fread(entry.get(), &tail, 1) != 0) invalid("array checksum or length mismatch");
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

void save_solver(const std::filesystem::path& path, const Eigen::SparseMatrix<double>& A,
                 const factorization& F) {
    detail::validate_adopted_factor(F);
    if (A.rows() != static_cast<Eigen::Index>(F.L.rows())) invalid("operator/factor dimension mismatch");
    Eigen::SparseMatrix<double> compressed;
    const auto* op = &A;
    if (!A.isCompressed()) { compressed = A; compressed.makeCompressed(); op = &compressed; }
    check_operator_structure(*op);
    detail::validate_adopted_operator(*op, F, 1.0);
    const auto n = static_cast<std::size_t>(op->rows());
    const auto annz = static_cast<std::size_t>(op->nonZeros());
    if (F.lumped_offdiag < 0) invalid("negative lumped entry count");
    const std::array<std::uint64_t, 6> metadata{1, n, F.sddm, annz, F.L.vals_.size(),
        static_cast<std::uint64_t>(F.lumped_offdiag)};
    std::vector<std::uint64_t> rounds;
    rounds.reserve(F.rounds.size());
    for (const auto& round : F.rounds) rounds.push_back(round.is_size);
    writer archive(path);
    archive.add("apxchol.npy", std::span<const std::uint64_t>(metadata));
    archive.add("A_indptr.npy", std::span<const int>(op->outerIndexPtr(), n + 1));
    archive.add("A_indices.npy", std::span<const int>(op->innerIndexPtr(), annz));
    archive.add("A_data.npy", std::span<const double>(op->valuePtr(), annz));
    archive.add("L_indptr.npy", std::span<const edge_index>(F.L.outer_));
    archive.add("L_indices.npy", std::span<const node_index>(F.L.inner_));
    archive.add("L_data.npy", std::span<const factor_value_t>(F.L.vals_));
    archive.add("permutation.npy", std::span<const node_index>(F.perm));
    archive.add("round_sizes.npy", std::span<const std::uint64_t>(rounds));
    archive.finish();
}

solver load_solver(const std::filesystem::path& path, const solve_options& options) {
    reader archive(path);
    const auto metadata_info = archive.inspect("apxchol.npy");
    require_extent<std::uint64_t>(metadata_info, 6);
    std::array<std::uint64_t, 6> metadata{};
    archive.read(metadata_info, std::span<std::uint64_t>(metadata));
    const auto [version, n, sddm, annz, lnnz, lumped] = metadata;
    if (version != 1) invalid("unsupported schema version");
    if (n == 0 || n > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<node_index>::max() || sddm > 1 ||
        annz > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        lnnz > std::numeric_limits<edge_index>::max() ||
        lumped > static_cast<std::uint64_t>(std::numeric_limits<Eigen::Index>::max())) invalid("dimensions do not fit this build");
    const auto ao = archive.inspect("A_indptr.npy"), ai = archive.inspect("A_indices.npy"), av = archive.inspect("A_data.npy");
    const auto lo = archive.inspect("L_indptr.npy"), li = archive.inspect("L_indices.npy"), lv = archive.inspect("L_data.npy");
    const auto perm = archive.inspect("permutation.npy"), rounds = archive.inspect("round_sizes.npy");
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
    return solver(A, std::move(F), options);
}

} // namespace apxchol

// MatrixMarket coordinate reader.
//
// The bytes are mapped once and turned into numbers in place: line boundaries
// come from memchr and fields from std::from_chars. The reader this replaced
// ran std::getline into a std::string, then built a FRESH std::istringstream
// per line and pulled each field through locale-aware num_get. On the 141 MB
// IPM file that was ~1.33 s, of which ~0.22 s was istringstream CONSTRUCTION
// and ~0.96 s num_get; the parse feeds every --mtx cell in the suite.
//
// Nothing else changed. Same signatures, same error strings, same order of
// checks, same |value| edge weights, dropped diagonal, symmetric expansion and
// duplicate summing. The scanners below deliberately reproduce operator>>'s
// quirks rather than a "sane" parser's, because those quirks are already baked
// into every number in results/ — see the comments on scan_double.
#include "mmio.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fstream>
#if defined(__unix__) || defined(__APPLE__)
#define MMIO_HAVE_MMAP 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define MMIO_HAVE_MMAP 0
#endif

namespace {

// What the banner + dimension lines of a coordinate MatrixMarket file say.
// Shared by both readers so the two kinds can never drift apart on the parts
// they legitimately agree about (geometry and storage), only on the part they
// are supposed to differ on: what the VALUES mean.
struct MtxHeader {
    bool symmetric = false;
    bool pattern   = false;
    int  rows = 0, cols = 0;
    long nnz  = 0;
};

// The whole file as one contiguous read-only byte range. mmap when POSIX is
// available and the target is a non-empty regular file, a plain slurp
// otherwise; the parser below only ever sees [begin(), end()) and does not care
// which it got. Anything the mapping cannot handle — a directory, a pipe, a
// zero-length file, a platform without <sys/mman.h> — falls through to the
// stream, which is what the reader this replaced used for everything.
class FileBytes {
public:
    explicit FileBytes(const std::string& path) {
#if MMIO_HAVE_MMAP
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            struct stat st {};
            if (::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
                const size_t len = static_cast<size_t>(st.st_size);
                void*        p   = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
                if (p != MAP_FAILED) {
                    map_  = static_cast<const char*>(p);
                    data_ = map_;
                    size_ = len;
                    // Advisory only, and one call each: madvise takes a single
                    // advice, not a bitmask (MADV_SEQUENTIAL | MADV_WILLNEED
                    // would silently collapse to one of them). Errors ignored.
                    (void)::madvise(const_cast<char*>(map_), size_, MADV_SEQUENTIAL);
                    (void)::madvise(const_cast<char*>(map_), size_, MADV_WILLNEED);
                }
            }
            ::close(fd);
            if (map_) return;
        }
#endif
        slurp(path);
    }

    ~FileBytes() {
#if MMIO_HAVE_MMAP
        if (map_) ::munmap(const_cast<char*>(map_), size_);
#endif
    }

    FileBytes(const FileBytes&)            = delete;
    FileBytes& operator=(const FileBytes&) = delete;

    const char* begin() const { return data_; }
    const char* end() const { return data_ + size_; }

private:
    // The stream path. The two failure messages are the ones the old reader
    // produced: `ifstream::is_open` false -> "Cannot open", and anything that
    // yields no bytes (an empty file, but also a directory, which does open)
    // -> "Empty file", because its very first getline failed.
    void slurp(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open())
            throw std::runtime_error("Cannot open " + path);
        in.seekg(0, std::ios::end);
        const std::streamoff len = in.tellg();
        in.seekg(0, std::ios::beg);
        if (len > 0) {
            buf_.resize(static_cast<size_t>(len));
            in.read(&buf_[0], len);
            buf_.resize(static_cast<size_t>(in.gcount()));
        }
        if (buf_.empty())
            throw std::runtime_error("Empty file: " + path);
        data_ = buf_.data();
        size_ = buf_.size();
    }

    const char* data_ = nullptr;
    size_t      size_ = 0;
    std::string buf_;
#if MMIO_HAVE_MMAP
    const char* map_ = nullptr;
#endif
};

inline bool is_digit(char c) { return static_cast<unsigned>(c) - '0' < 10u; }

// The classic locale's whitespace, minus '\n' — which never occurs inside a
// line, because a line is by definition what precedes one.
inline const char* skip_ws(const char* p, const char* e) {
    while (p < e && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\v' || *p == '\f')) ++p;
    return p;
}

// End of the current line: the '\n', or the end of the file for a final line
// with no trailing newline (which std::getline also accepts).
inline const char* eol(const char* p, const char* e) {
    const char* nl =
        static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(e - p)));
    return nl ? nl : e;
}

// `istream >> int`. Returns the position after the field, or nullptr when the
// extraction would have failed — which is exactly when both call sites throw,
// so the value written on failure is never observed. An out-of-range field
// counts as a failure because num_get sets failbit for it too.
inline const char* scan_int(const char* p, const char* e, int& out) {
    p = skip_ws(p, e);
    const char* q = p;
    bool neg = false;
    if (q < e && (*q == '+' || *q == '-')) {
        neg = (*q == '-');
        ++q;
    }
    const char* d0 = q;
    const unsigned long long limit = neg ? 2147483648ull : 2147483647ull;
    unsigned long long v = 0;
    bool ovf = false;
    while (q < e && is_digit(*q)) {
        const unsigned d = static_cast<unsigned>(*q - '0');
        if (!ovf) {
            if (v > (limit - d) / 10) ovf = true;
            else v = v * 10 + d;
        }
        ++q;
    }
    if (q == d0) return nullptr;
    if (ovf) {
        out = neg ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
        return nullptr;
    }
    out = neg ? -static_cast<int>(v) : static_cast<int>(v);
    return q;
}

// `istream >> long`, same contract as scan_int.
inline const char* scan_long(const char* p, const char* e, long& out) {
    p = skip_ws(p, e);
    const char* q = p;
    bool neg = false;
    if (q < e && (*q == '+' || *q == '-')) {
        neg = (*q == '-');
        ++q;
    }
    const char* d0 = q;
    const unsigned long long limit =
        neg ? static_cast<unsigned long long>(std::numeric_limits<long>::max()) + 1ull
            : static_cast<unsigned long long>(std::numeric_limits<long>::max());
    unsigned long long v = 0;
    bool ovf = false;
    while (q < e && is_digit(*q)) {
        const unsigned d = static_cast<unsigned>(*q - '0');
        if (!ovf) {
            if (v > (limit - d) / 10) ovf = true;
            else v = v * 10 + d;
        }
        ++q;
    }
    if (q == d0) return nullptr;
    if (ovf) {
        out = neg ? std::numeric_limits<long>::min() : std::numeric_limits<long>::max();
        return nullptr;
    }
    out = neg ? -static_cast<long>(v) : static_cast<long>(v);
    return q;
}

// `istream >> double`, quirks included. std::from_chars is correctly rounded
// and so is the strtod behind num_get, so every value a MatrixMarket file can
// actually carry converts to the identical bit pattern; the work here is
// reproducing the DISAGREEMENTS at the edges, because load_mtx_as_adjacency
// ignores the failure flag and keeps whatever value the failed extraction left
// behind:
//
//   nothing left on the line   operator>>'s sentry fails before num_get runs,
//                              so the caller's initialiser survives untouched
//                              (a `real` entry line with no third field is read
//                              as weight 1.0, not 0.0)
//   a non-numeric token        num_get zeroes the value
//   "inf" / "nan"              REJECTED, value zeroed — num_get's atom set has
//                              no letters but e/E, while from_chars takes both
//   a leading '+'              accepted — from_chars does not take one
//   a truncated exponent       REJECTED, value zeroed — see below
//   out of range               strtod's own answer, and a failure only when it
//                              overflowed (underflow-to-zero extracts cleanly)
inline const char* scan_double(const char* p, const char* e, double& out) {
    p = skip_ws(p, e);
    if (p == e) return nullptr;  // sentry failure: `out` keeps the caller's value

    const char* q = p;
    if (*q == '+' || *q == '-') ++q;
    if (q == e || !(is_digit(*q) || *q == '.')) {
        out = 0.0;
        return nullptr;
    }

    double     v     = 0.0;
    const auto start = (*p == '+') ? p + 1 : p;
    const auto r     = std::from_chars(start, e, v);
    if (r.ec == std::errc()) {
        // num_get is greedier than from_chars about an exponent marker: on
        // "1.5e" it swallows the 'e', hands "1.5e" to strtod and fails with a
        // zeroed value, where from_chars backs off and reports a clean 1.5.
        // Only the FIRST marker behaves that way — a second one just ends the
        // token, and "12e5e5" is 12e5 to both — so the cold branch checks
        // whether an exponent was already consumed.
        if (r.ptr < e && (*r.ptr == 'e' || *r.ptr == 'E')) {
            bool seen = false;
            for (const char* s = start; s < r.ptr; ++s)
                if (*s == 'e' || *s == 'E') { seen = true; break; }
            if (!seen) {
                out = 0.0;
                return nullptr;
            }
        }
        out = v;
        return r.ptr;
    }
    if (r.ec == std::errc::result_out_of_range) {
        // Cold path, and never reached by a file any real writer produces.
        const std::string tok(p, static_cast<size_t>(r.ptr - p));
        const double      d = std::strtod(tok.c_str(), nullptr);
        if (d == 0.0) {
            out = d;  // underflow: operator>> reports success
            return r.ptr;
        }
        out = (d > 0.0) ? std::numeric_limits<double>::max()
                        : std::numeric_limits<double>::lowest();
        return nullptr;
    }
    out = 0.0;
    return nullptr;
}

// Banner, comments and the dimension line. Returns the first byte of the body.
const char* read_header(const FileBytes& f, const std::string& path, MtxHeader& h) {
    const char* p = f.begin();
    const char* e = f.end();

    const char* nl = eol(p, e);
    const std::string line(p, static_cast<size_t>(nl - p));
    if (line.find("%%MatrixMarket") == std::string::npos)
        throw std::runtime_error("Not a MatrixMarket file: " + path);

    std::string lower_line = line;
    std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);
    h.symmetric = lower_line.find("symmetric") != std::string::npos;
    h.pattern   = lower_line.find("pattern") != std::string::npos;

    if (lower_line.find("array") != std::string::npos)
        throw std::runtime_error("Array (dense) format not supported, need coordinate format: " +
                                 path);

    // Skip comments. `while (getline) if (!line.empty() && line[0] != '%')`:
    // getline strips only '\n', so on a CRLF file a blank line arrives as "\r"
    // and is NOT empty. That is matched here rather than fixed — the old reader
    // took such a line as the dimension line and failed on it, and this commit
    // is not the place to change which files are accepted.
    p = (nl < e) ? nl + 1 : e;
    const char* dim  = nullptr;
    while (p < e) {
        const char* q = eol(p, e);
        if (q != p && *p != '%') {
            dim = p;
            break;
        }
        p = (q < e) ? q + 1 : e;
    }
    // Running out of lines lands on the same message: the old loop left `line`
    // holding the last comment (or the banner), which never parses as a
    // dimension line either.
    if (!dim) throw std::runtime_error("Bad dimension line in " + path);

    const char* lim = eol(dim, e);
    const char* q   = scan_int(dim, lim, h.rows);
    if (q) q = scan_int(q, lim, h.cols);
    if (q) q = scan_long(q, lim, h.nnz);
    if (!q) throw std::runtime_error("Bad dimension line in " + path);

    if (h.rows != h.cols)
        throw std::runtime_error("Matrix is not square in " + path);

    return (lim < e) ? lim + 1 : e;
}

} // namespace

MtxResult load_mtx_as_adjacency(const std::string& path) {
    const FileBytes f(path);

    MtxHeader   h;
    const char* p = read_header(f, path, h);
    const char* e = f.end();

    const int n = h.rows;
    std::vector<std::vector<Edge>> adj(static_cast<size_t>(n));

    // Exact per-vertex sizing. The old reader grew every list by repeated
    // push_back, which on the big `pattern` graphs is most of what is left once
    // the parse itself gets cheap. This first walk reads only the two indices —
    // for a pattern file, the whole line — and IGNORES every malformation it
    // meets: the real pass below is the one that reports errors, so the two
    // readers still fail in exactly the same place with exactly the same
    // message. Capacity is not observable in the result either way.
    {
        std::vector<int> deg(static_cast<size_t>(n), 0);
        const char*      s = p;
        for (long k = 0; k < h.nnz && s < e; ++k) {
            const char* lim = eol(s, e);
            int         i, j;
            const char* q = scan_int(s, lim, i);
            if (q) q = scan_int(q, lim, j);
            if (!q) break;
            --i; --j;
            if (i != j && i >= 0 && i < n && j >= 0 && j < n) {
                ++deg[static_cast<size_t>(i)];
                if (h.symmetric) ++deg[static_cast<size_t>(j)];
            }
            s = (lim < e) ? lim + 1 : e;
        }
        for (int v = 0; v < n; ++v)
            if (deg[static_cast<size_t>(v)])
                adj[static_cast<size_t>(v)].reserve(static_cast<size_t>(deg[static_cast<size_t>(v)]));
    }

    // Parse entries
    for (long k = 0; k < h.nnz; ++k) {
        if (p >= e)
            throw std::runtime_error("Premature end of file: " + path);
        const char* lim = eol(p, e);
        int    i, j;
        double w = 1.0;
        const char* q = scan_int(p, lim, i);
        if (q) q = scan_int(q, lim, j);
        if (!q)
            throw std::runtime_error("Bad entry line in " + path);
        // A failed value extraction is IGNORED here, exactly as the unchecked
        // `iss >> w` was: whatever it left in `w` is the weight.
        if (!h.pattern) (void)scan_double(q, lim, w);
        p = (lim < e) ? lim + 1 : e;

        --i; --j; // 1-indexed to 0-indexed

        if (i == j) continue; // skip diagonal (we build Laplacian from adjacency)

        w = std::abs(w); // off-diagonal Laplacian entries are negative; use abs as edge weight

        adj[static_cast<size_t>(i)].push_back({j, w});
        if (h.symmetric && i != j)
            adj[static_cast<size_t>(j)].push_back({i, w});
    }

    return {std::move(adj), n, static_cast<int>(h.nnz)};
}

MtxOperator load_mtx_as_operator(const std::string& path) {
    const FileBytes f(path);

    MtxHeader   h;
    const char* p = read_header(f, path, h);
    const char* e = f.end();

    if (h.pattern)
        throw std::runtime_error(
            "kind=operator was declared for " + path +
            ", but its MatrixMarket header says `pattern`: the file carries no "
            "values, so there is no published operator to solve. A pattern file "
            "can only be kind=graph (solved as L = D - A with unit weights).");

    const int n = h.rows;
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(h.symmetric ? 2 * h.nnz : h.nnz));

    for (long k = 0; k < h.nnz; ++k) {
        if (p >= e)
            throw std::runtime_error("Premature end of file: " + path);
        const char* lim = eol(p, e);
        int    i, j;
        double v;
        const char* q = scan_int(p, lim, i);
        if (q) q = scan_int(q, lim, j);
        if (q) q = scan_double(q, lim, v);
        if (!q)
            throw std::runtime_error("Bad entry line in " + path);
        p = (lim < e) ? lim + 1 : e;

        --i; --j; // 1-indexed to 0-indexed

        // Published value, sign and all — including the diagonal, which is the
        // whole point of reading the file as an operator.
        trips.emplace_back(i, j, v);
        if (h.symmetric && i != j)
            trips.emplace_back(j, i, v);
    }

    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(trips.begin(), trips.end());   // sums duplicates, as MatrixMarket specifies
    A.makeCompressed();

    return {std::move(A), n, h.nnz, h.symmetric};
}

// Host side of the Apple GPU block PCG (include/apxchol/solver/metal_solver.h):
// factorization, the dropped-factor schedules, the permuted operator, packing,
// the fp64 exit checks and the Laplacian centring. All OpenMP host work lives
// here; the device side (src/metal_device.mm) sees only plain arrays.
#include "apxchol/solver/metal_solver.h"

#include "apxchol/csc_work.h"
#include "apxchol/solver/detail/metal_host.h"
#include "apxchol/solver/pcg_cuda_host.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include "apxchol/solver/sptrsv/level_schedule.h"
#include "metal_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

namespace {

namespace mh = detail::metal_host;
namespace ls = level_schedule;
namespace dm = detail::metal;

static_assert(sizeof(mh::df) == sizeof(dm::df32));

mh::df* as_df(dm::df32* p) { return reinterpret_cast<mh::df*>(p); }
dm::df32 to_device(mh::df v) { return {v.hi, v.lo}; }

// ── Device self-test ─────────────────────────────────────────────────────────
// The double-float kernels are only exact if the Metal compiler neither
// contracts nor reassociates them. Compare the device's error-free transforms
// with the host's (metal_host.h) bit for bit, and check two_sum / two_prod
// against exact fp64 arithmetic, on deterministic inputs whose exponents keep
// every exact sum and product inside fp64.

std::uint32_t next_random(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float random_float(std::uint32_t& state, int exponent_span) {
    const std::uint32_t bits = next_random(state);
    const float mantissa = 1.0f + static_cast<float>(bits & 0x7fffff) / 8388608.0f;
    const int e = static_cast<int>((bits >> 23) % static_cast<std::uint32_t>(2 * exponent_span + 1)) - exponent_span;
    const float v = std::ldexp(mantissa, e);
    return (bits >> 31) ? -v : v;
}

std::vector<float> probe_inputs(std::size_t count) {
    std::vector<float> in(count * 4);
    std::uint32_t state = 0x9e3779b9u;
    for (std::size_t i = 0; i < count; ++i) {
        for (int k = 0; k < 2; ++k) {
            const float hi = random_float(state, 12);
            const float lo = hi * std::ldexp(random_float(state, 0) * 0.5f, -24);
            const mh::df v = mh::quick_two_sum(hi, lo);
            in[4 * i + 2 * k] = v.hi;
            in[4 * i + 2 * k + 1] = v.lo;
        }
    }
    return in;
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

std::string run_self_test() {
    constexpr std::size_t kCases = 2048;
    const std::vector<float> in = probe_inputs(kCases);
    const std::vector<float> out = dm::run_probe(in);
    for (std::size_t i = 0; i < kCases; ++i) {
        const mh::df x{in[4 * i], in[4 * i + 1]}, y{in[4 * i + 2], in[4 * i + 3]};
        const float* o = &out[16 * i];
        const mh::df ts = mh::two_sum(x.hi, y.hi), tp = mh::two_prod(x.hi, y.hi);
        const mh::df da = mh::df_add(x, y), dmul = mh::df_mul(x, y), dmf = mh::df_mul_f(x, y.hi);
        const float expect[] = {ts.hi, ts.lo, tp.hi, tp.lo, da.hi, da.lo, dmul.hi, dmul.lo,
                                dmf.hi, dmf.lo, std::fma(x.hi, y.hi, x.lo),
                                std::fma(-x.hi, y.hi, x.lo) * y.lo};
        for (int k = 0; k < 12; ++k)
            if (!same_bits(o[k], expect[k]))
                return "double-float self-test: device result " + std::to_string(k) +
                       " differs from the host's on case " + std::to_string(i);
        if (!same_bits(o[14], (x.hi - y.hi) * y.lo) || !same_bits(o[15], mh::quick_two_sum(x.hi, x.lo).lo))
            return "double-float self-test: device rounding differs on case " + std::to_string(i);
        const double sum = static_cast<double>(x.hi) + static_cast<double>(y.hi);
        const double prod = static_cast<double>(x.hi) * static_cast<double>(y.hi);
        if (static_cast<double>(o[0]) + static_cast<double>(o[1]) != sum ||
            static_cast<double>(o[2]) + static_cast<double>(o[3]) != prod)
            return "double-float self-test: two_sum / two_prod are not error-free on case " +
                   std::to_string(i);
        const double q = mh::join({o[12], o[13]});
        const double q_ref = mh::join(x) / mh::join(y);
        if (!(std::fabs(q - q_ref) <= 1e-12 * std::fabs(q_ref)))
            return "double-float self-test: division is inaccurate on case " + std::to_string(i);
    }
    return {};
}

struct availability {
    bool ok = false;
    std::string reason;
};

const availability& check_availability() noexcept {
    static const availability result = []() noexcept {
        availability a;
        try {
            const dm::device_status& st = dm::status();
            if (!st.ok) {
                a.reason = st.error;
                return a;
            }
            a.reason = run_self_test();
            a.ok = a.reason.empty();
        } catch (const std::exception& e) {
            a.reason = e.what();
        } catch (...) {
            a.reason = "unknown error";
        }
        return a;
    }();
    return result;
}

void print_banner_once(const metal_solver::statistics& st) {
    static std::once_flag flag;
    std::call_once(flag, [&] {
        if (!std::getenv("APXCHOL_VERBOSE")) return;
        std::fprintf(stderr,
                     "[apxchol] Metal block PCG on %s: %d columns per batch, fp32 factor, "
                     "double-float Krylov vectors, %s operator\n",
                     st.device.c_str(), st.block_columns,
                     st.operator_double_float ? "double-float" : "fp32-exact");
    });
}

}  // namespace

const char* to_string(metal_stop stop) noexcept {
    switch (stop) {
    case metal_stop::zero_rhs: return "zero_rhs";
    case metal_stop::initial_guess: return "initial_guess";
    case metal_stop::recursive_tolerance: return "recursive_tolerance";
    case metal_stop::max_iterations: return "max_iterations";
    case metal_stop::breakdown: return "breakdown";
    case metal_stop::stagnation: return "stagnation";
    case metal_stop::nonfinite: return "nonfinite";
    }
    return "unknown";
}

bool metal_solver::available() noexcept { return check_availability().ok; }

struct metal_solver::impl {
    solve_options opts;
    factorization F;
    std::size_t n = 0;
    std::uint32_t m = 0;
    bool laplacian = false;
    // The permuted operator A' = P A P^T (canonical lower values) in fp64:
    // the host residuals run on it.
    std::vector<int> op_ptr;
    std::unique_ptr<int[]> op_col;
    std::unique_ptr<double[]> op_val;
    std::int64_t op_nnz = 0;
    // op_ptr at the metal_host::kFoldBlock row boundaries: the exit check's
    // fold blocks are split across threads by stored entries.
    std::vector<int> fold_block_ptr;
    // Level structure only (plan_steps reads level_ptr / heavy_ptr).
    ls::level_solve fwd_levels, bwd_levels;
    std::unique_ptr<dm::engine> device;
    std::uint32_t block_columns = 0;
    std::uint32_t check_every = 1;
    std::map<std::uint32_t, std::pair<std::vector<dm::tri_step>, std::vector<dm::tri_step>>> plans;
    statistics stats;
    std::mutex mutex;

    void setup(const Eigen::SparseMatrix<double>& A, checkpoint* cp);

    const std::pair<std::vector<dm::tri_step>, std::vector<dm::tri_step>>& plan(std::uint32_t kc) {
        auto it = plans.find(kc);
        if (it != plans.end()) return it->second;
        auto flatten = [&](const ls::level_solve& s) {
            std::vector<dm::tri_step> out;
            for (const ls::level_step& st : ls::plan_steps(s, kc))
                out.push_back({static_cast<std::uint32_t>(st.kind), st.first, st.last});
            return out;
        };
        return plans.emplace(kc, std::make_pair(flatten(fwd_levels), flatten(bwd_levels))).first->second;
    }

    // y = A' x in fp64, each row summed in storage order; rows are split across
    // threads by stored entries (detail::work_balanced_range).
    void spmv(const double* x, double* y) const {
        const int* ptr = op_ptr.data();
        const int* col = op_col.get();
        const double* val = op_val.get();
        const std::ptrdiff_t rows = static_cast<std::ptrdiff_t>(n);
        #pragma omp parallel
        {
            int tid = 0, nt = 1;
#ifdef _OPENMP
            tid = omp_get_thread_num();
            nt = omp_get_num_threads();
#endif
            const auto [lo, hi] = detail::work_balanced_range(ptr, rows, tid, nt);
            for (std::ptrdiff_t i = lo; i < hi; ++i) {
                double acc = 0.0;
                for (int p = ptr[i]; p < ptr[i + 1]; ++p) acc += val[p] * x[col[p]];
                y[i] = acc;
            }
        }
    }

    // ||bp - A' xp|| (permuted vectors), thread-count independent.
    double residual_norm(const double* bp, const double* xp, std::vector<double>& work) const {
        work.resize(n);
        spmv(xp, work.data());
        const double* y = work.data();
        return std::sqrt(mh::fold_sum(n, [=](std::size_t i) {
            const double d = bp[i] - y[i];
            return d * d;
        }));
    }

    void centre(double* xp) const {
        const double mean = mh::fold_sum(n, [=](std::size_t i) { return xp[i]; }) /
                            static_cast<double>(n);
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) xp[i] -= mean;
    }
};

void metal_solver::impl::setup(const Eigen::SparseMatrix<double>& A, checkpoint* cp) {
    if (A.rows() != A.cols()) throw std::invalid_argument("metal_solver: the operator must be square");
    n = static_cast<std::size_t>(A.rows());
    if (n == 0) throw std::invalid_argument("metal_solver: empty operator");
    if (static_cast<std::size_t>(F.L.rows()) != n || F.perm.size() != n)
        throw std::invalid_argument("metal_solver: factorization dimension mismatch");
    if (F.L.nonZeros() > 0 && F.L.vals_.empty())
        throw std::invalid_argument(
            "metal_solver: factorization values were released; pass a freshly computed factorization");
    // Eigen's int StorageIndex keeps n and the operator's stored count within
    // the device's 32-bit indices; the factor and the two-triangle permuted
    // operator are checked here.
    static_assert(std::is_same_v<Eigen::SparseMatrix<double>::StorageIndex, int>);
    laplacian = !F.sddm;
    m = static_cast<std::uint32_t>(laplacian ? n - 1 : n);
    if (F.L.outerIndexPtr()[m] > static_cast<edge_index>(std::numeric_limits<int>::max()))
        throw std::length_error("metal_solver: the factor exceeds the device's 32-bit offsets");

    if (cp) { cp->descend("setup"); cp->tick(); }
    // The factor the CPU triangular solves store: L11, compacting drop at the
    // same relative threshold, fp32 values, then the two level schedules.
    ls::factor_schedules sched = ls::build_factor_schedules(F.L, m, factor_drop_rel_from_env());
    if (cp) (*cp)("metal_factor_prep");

    Eigen::SparseMatrix<double> compressed;
    const Eigen::SparseMatrix<double>* src = &A;
    if (!A.isCompressed()) {
        compressed = A;
        compressed.makeCompressed();
        src = &compressed;
    }
    // Both triangles are stored: up to twice the stored entries of one triangle.
    if (src->nonZeros() > std::numeric_limits<int>::max() / 2)
        throw std::length_error("metal_solver: operators above 2^30 stored entries exceed the "
                                "device's 32-bit offsets");
    bool exact = true;
    detail::build_permuted_full_symmetric_csr(*src, F.perm, op_ptr, op_col, op_val, op_nnz, exact);
    const std::size_t nnz = static_cast<std::size_t>(op_nnz);
    fold_block_ptr.resize((n + mh::kFoldBlock - 1) / mh::kFoldBlock + 1);
    for (std::size_t b = 0; b < fold_block_ptr.size(); ++b)
        fold_block_ptr[b] = op_ptr[std::min(n, b * mh::kFoldBlock)];
    std::vector<float> hi(nnz), lo(exact ? 0 : nnz);
    bool in_range = true;
    #pragma omp parallel for schedule(static) reduction(&& : in_range)
    for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nnz); ++p) {
        const double v = op_val[p];
        in_range = in_range && ls::in_range(v);
        const mh::df d = mh::split(v);
        hi[p] = d.hi;
        if (!exact) lo[p] = d.lo;
    }
    if (!in_range)
        throw std::domain_error("metal_solver: an operator value is non-finite or outside [2^-100, 2^100]");
    if (cp) (*cp)("metal_operator");

    const dm::device_status& st = dm::status();
    auto tri_bytes = [](const ls::level_solve& s) {
        return (s.level_ptr.size() + s.rows.size() + s.ptr.size() + s.col.size()) * 4 +
               (s.val.size() + s.dinv.size()) * 4;
    };
    mh::block_limits limits;
    limits.max_buffer_bytes = st.max_buffer_bytes;
    limits.working_set_bytes = st.working_set_bytes;
    limits.static_bytes = (n + 1) * 4 + nnz * (exact ? 8 : 12) + tri_bytes(sched.forward) +
                          tri_bytes(sched.backward);
    limits.tree_threads = st.tree_threads;
    limits.row_threads = std::min(st.row_threads, st.heavy_threads);
    block_columns = mh::choose_block_columns(n, limits);
    if (block_columns == 0)
        throw dm::device_memory_error("metal_solver: the system does not fit the Metal device (n = " +
                                      std::to_string(n) + ", operator nnz = " + std::to_string(nnz) + ")");

    static_assert(sizeof(int) == sizeof(std::uint32_t));
    dm::operator_arrays op;
    op.ptr = reinterpret_cast<const std::uint32_t*>(op_ptr.data());
    op.col = reinterpret_cast<const std::uint32_t*>(op_col.get());
    op.hi = hi.data();
    op.lo = exact ? nullptr : lo.data();
    op.n = n;
    op.nnz = nnz;
    auto arrays = [](const ls::level_solve& s) {
        dm::tri_arrays t;
        t.level_ptr = s.level_ptr.data();
        t.levels = s.levels();
        t.rows = s.rows.data();
        t.ptr = s.ptr.data();
        t.slots = s.slots();
        t.col = s.col.data();
        t.val = s.val.data();
        t.deps = s.col.size();
        t.dinv = s.dinv.data();
        return t;
    };
    device = std::make_unique<dm::engine>(op, arrays(sched.forward), arrays(sched.backward), m, laplacian);
    // Inherited from the prototype: one iteration per command buffer on large
    // systems, four on small ones (fewer host round trips).
    check_every = n > 200000 ? 1 : 4;

    fwd_levels.level_ptr = std::move(sched.forward.level_ptr);
    fwd_levels.heavy_ptr = std::move(sched.forward.heavy_ptr);
    bwd_levels.level_ptr = std::move(sched.backward.level_ptr);
    bwd_levels.heavy_ptr = std::move(sched.backward.heavy_ptr);
    stats.n = static_cast<Eigen::Index>(n);
    stats.levels_forward = fwd_levels.levels();
    stats.levels_backward = bwd_levels.levels();
    const auto& full = plan(block_columns);
    stats.steps_forward = full.first.size();
    stats.steps_backward = full.second.size();
    stats.block_columns = static_cast<int>(block_columns);
    stats.operator_double_float = !exact;
    stats.device = st.name;
    if (!opts.keep_factor_values) F.L.release_values();
    if (cp) { (*cp)("metal_upload"); cp->ascend(); }
    print_banner_once(stats);
}

metal_solver::metal_solver(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
                           checkpoint* cp)
    : impl_(std::make_unique<impl>()) {
    if (!available())
        throw std::runtime_error("apxchol::metal_solver: the Metal backend is unavailable: " +
                                 check_availability().reason);
    impl_->opts = opts;
    impl_->F = detail::factorize_for_solver(A, opts.storage, opts.factor_opts, cp, true);
    impl_->setup(A, cp);
}

metal_solver::metal_solver(const Eigen::SparseMatrix<double>& A, factorization F,
                           const solve_options& opts, checkpoint* cp)
    : impl_(std::make_unique<impl>()) {
    if (!available())
        throw std::runtime_error("apxchol::metal_solver: the Metal backend is unavailable: " +
                                 check_availability().reason);
    impl_->opts = opts;
    impl_->F = std::move(F);
    impl_->setup(A, cp);
}

metal_solver::~metal_solver() = default;
metal_solver::metal_solver(metal_solver&&) noexcept = default;
metal_solver& metal_solver::operator=(metal_solver&&) noexcept = default;

const factorization& metal_solver::factor() const { return impl_->F; }
metal_solver::statistics metal_solver::stats() const { return impl_->stats; }
Eigen::Index metal_solver::rows() const { return static_cast<Eigen::Index>(impl_->n); }

namespace {

bool all_finite(const metal_solver::block_cref& M) {
    for (Eigen::Index j = 0; j < M.cols(); ++j)
        for (Eigen::Index i = 0; i < M.rows(); ++i)
            if (!std::isfinite(M(i, j))) return false;
    return true;
}

metal_stop stop_of(std::uint32_t code) {
    switch (code) {
    case dm::kStopTolerance: return metal_stop::recursive_tolerance;
    case dm::kStopBreakdown: return metal_stop::breakdown;
    case dm::kStopStagnation: return metal_stop::stagnation;
    case dm::kStopNonfinite: return metal_stop::nonfinite;
    default: return metal_stop::max_iterations;
    }
}

}  // namespace

metal_block_result metal_solver::solve(block_cref B, Eigen::Ref<Eigen::MatrixXd> X, double tol,
                                       int max_iter, const block_cref* X0) const {
    impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    const Eigen::Index n = static_cast<Eigen::Index>(s.n);
    const Eigen::Index k = B.cols();
    if (B.rows() != n) throw std::invalid_argument("metal_solver::solve: B row count mismatch");
    if (X.rows() != n || X.cols() != k)
        throw std::invalid_argument("metal_solver::solve: output X size mismatch");
    if (X0 != nullptr && (X0->rows() != n || X0->cols() != k))
        throw std::invalid_argument("metal_solver::solve: x0 size mismatch");
    if (!all_finite(B)) throw std::invalid_argument("metal_solver::solve: b contains a non-finite value");
    if (X0 != nullptr && !all_finite(*X0))
        throw std::invalid_argument("metal_solver::solve: x0 contains a non-finite value");
    if (tol < 0.0) tol = s.opts.tol;
    if (max_iter < 0) max_iter = s.opts.max_iter;
    const std::uint32_t window = static_cast<std::uint32_t>(std::max(0, s.opts.stagnation_window));

    metal_block_result result;
    result.columns.resize(static_cast<std::size_t>(k));
    result.lumped_offdiag = s.F.lumped_offdiag;
    checkpoint& cp = result.timings;
    cp.descend("pcg");
    cp.tick();

    const node_index* perm = s.F.perm.data();
    const std::size_t nn = s.n;

    // ||b|| and max |b| in the caller's order: every (column, fold block) pair
    // in one parallel pass, then each column's block partials in block order,
    // so ||b|| has the bits of metal_host::fold_sum_squares at any k.
    const std::size_t blocks = s.fold_block_ptr.size() - 1;
    const std::size_t pairs = static_cast<std::size_t>(k) * blocks;
    std::vector<double> part(pairs), part_max(pairs);
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(pairs); ++t) {
        const std::size_t c = static_cast<std::size_t>(t) / blocks;
        const std::size_t lo = (static_cast<std::size_t>(t) % blocks) * mh::kFoldBlock;
        const std::size_t hi = std::min(nn, lo + mh::kFoldBlock);
        const double* b = B.col(static_cast<Eigen::Index>(c)).data();
        double sum = 0.0, max_abs = 0.0;
        for (std::size_t i = lo; i < hi; ++i) {
            const double sq = b[i] * b[i];
            sum += sq;
            max_abs = std::max(max_abs, std::fabs(b[i]));
        }
        part[static_cast<std::size_t>(t)] = sum;
        part_max[static_cast<std::size_t>(t)] = max_abs;
    }
    std::vector<double> bnorm(static_cast<std::size_t>(k)), bmax(static_cast<std::size_t>(k));
    for (std::size_t c = 0; c < static_cast<std::size_t>(k); ++c) {
        double sum = 0.0, max_abs = 0.0;
        for (std::size_t b = 0; b < blocks; ++b) {
            sum += part[c * blocks + b];
            max_abs = std::max(max_abs, part_max[c * blocks + b]);
        }
        bnorm[c] = std::sqrt(sum);
        bmax[c] = max_abs;
    }

    // A warm-started column resolved on the host from its permuted x (centred
    // for a Laplacian), with the fp64 residual on the original operator.
    std::vector<double> work, column;
    auto finish_host = [&](Eigen::Index c, std::vector<double>& xp, const std::vector<double>& bp,
                           Eigen::Index iterations, double recursive, metal_stop stop) {
        column.resize(nn);
        if (s.laplacian) s.centre(xp.data());
        metal_column_result& out = result.columns[static_cast<std::size_t>(c)];
        out.iterations = iterations;
        out.residual = s.residual_norm(bp.data(), xp.data(), work) / bnorm[static_cast<std::size_t>(c)];
        out.recursive_residual = recursive;
        out.converged = out.residual < tol;
        out.stop = stop;
        mh::gather(xp.data(), perm, nn, column.data());
        X.col(c) = Eigen::Map<const Eigen::VectorXd>(column.data(), n);
    };

    struct job {
        Eigen::Index c;
        double scale, initial;
        std::vector<double> x0p, w;  // warm start only: permuted x0 and b - A x0
    };
    std::vector<job> batch;

    // Packs a batch node-major, runs it on the device and checks every column
    // on the host. The block passes below touch each node's kc entries
    // contiguously; per column they perform the same fp64 operations, in the
    // same order, as the one-column definitions (pack_column, unpack_column,
    // centre, residual_norm, gather), so a column's bits do not depend on kc.
    auto run_batch = [&] {
        if (batch.empty()) return;
        const std::uint32_t kc = static_cast<std::uint32_t>(batch.size());
        s.device->reserve(kc);
        std::vector<const double*> bcol(kc), wcol(kc), x0col(kc);
        std::vector<double*> xcol(kc);
        std::vector<double> scale(kc);
        for (std::uint32_t j = 0; j < kc; ++j) {
            const job& jb = batch[j];
            bcol[j] = B.col(jb.c).data();
            xcol[j] = X.col(jb.c).data();
            wcol[j] = jb.w.empty() ? nullptr : jb.w.data();
            x0col[j] = jb.x0p.empty() ? nullptr : jb.x0p.data();
            scale[j] = jb.scale;
        }
        // r = 2^e (b - A x0), node-major (cold columns read b directly).
        mh::df* r = as_df(s.device->r());
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(nn); ++v) {
            const std::size_t i = static_cast<std::size_t>(perm[v]);
            mh::df* row = r + i * kc;
            for (std::uint32_t j = 0; j < kc; ++j)
                row[j] = mh::split(scale[j] * (wcol[j] != nullptr ? wcol[j][i] : bcol[j][v]));
        }
        dm::column_state* cs = s.device->columns();
        for (std::uint32_t j = 0; j < kc; ++j) {
            job& jb = batch[j];
            const double bn = bnorm[static_cast<std::size_t>(jb.c)];
            const double thr = tol * bn * jb.scale;
            const double ref = bn * jb.scale;
            cs[j] = dm::column_state{};
            cs[j].thr = to_device(mh::split_saturated(thr * thr));
            cs[j].prev = to_device(mh::split_saturated(ref * ref));
            cs[j].active = 1;
            std::vector<double>().swap(jb.w);
        }
        cp("pack");
        const auto& plans = s.plan(kc);
        s.device->solve(kc, plans.first, plans.second, static_cast<std::uint32_t>(max_iter), window,
                        s.check_every);
        cp("device");

        // x in fp64 (+ x0), node-major, into the dead A p buffer; b permuted
        // node-major into the dead r buffer (both hold n kc double-floats).
        static_assert(sizeof(dm::df32) == sizeof(double));
        const mh::df* xs = as_df(s.device->x());
        double* xd = reinterpret_cast<double*>(s.device->ap());
        double* bd = reinterpret_cast<double*>(s.device->r());
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(nn); ++q) {
            const std::size_t i = static_cast<std::size_t>(q);
            for (std::uint32_t j = 0; j < kc; ++j) {
                double t = mh::join(xs[i * kc + j]) / scale[j];
                if (x0col[j] != nullptr) t += x0col[j][i];
                xd[i * kc + j] = t;
            }
        }
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(nn); ++v) {
            double* row = bd + static_cast<std::size_t>(perm[v]) * kc;
            for (std::uint32_t j = 0; j < kc; ++j) row[j] = bcol[j][v];
        }
        // Per-block column sums of fp64 terms, folded as metal_host::fold_sum.
        part.assign(blocks * kc, 0.0);
        auto fold_columns = [&](std::vector<double>& sums) {
            sums.assign(kc, 0.0);
            for (std::size_t b = 0; b < blocks; ++b)
                for (std::uint32_t j = 0; j < kc; ++j) sums[j] += part[b * kc + j];
        };
        std::vector<double> sums;
        if (s.laplacian) {  // centre: x -= mean(x)
            #pragma omp parallel for schedule(static)
            for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(blocks); ++b) {
                double* pb = part.data() + static_cast<std::size_t>(b) * kc;
                const std::size_t lo = static_cast<std::size_t>(b) * mh::kFoldBlock;
                const std::size_t hi = std::min(nn, lo + mh::kFoldBlock);
                for (std::size_t i = lo; i < hi; ++i)
                    for (std::uint32_t j = 0; j < kc; ++j) pb[j] += xd[i * kc + j];
            }
            fold_columns(sums);
            for (std::uint32_t j = 0; j < kc; ++j) sums[j] /= static_cast<double>(nn);
            #pragma omp parallel for schedule(static)
            for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(nn); ++q)
                for (std::uint32_t j = 0; j < kc; ++j) xd[static_cast<std::size_t>(q) * kc + j] -= sums[j];
            part.assign(blocks * kc, 0.0);
        }
        // ||b - A' x||^2 per column: one pass over the operator for the block,
        // fold blocks split across threads by stored entries.
        {
            const int* ptr = s.op_ptr.data();
            const int* col = s.op_col.get();
            const double* val = s.op_val.get();
            #pragma omp parallel
            {
                int tid = 0, nt = 1;
#ifdef _OPENMP
                tid = omp_get_thread_num();
                nt = omp_get_num_threads();
#endif
                const auto [blo, bhi] = detail::work_balanced_range(
                    s.fold_block_ptr.data(), static_cast<std::ptrdiff_t>(blocks), tid, nt);
                double acc[mh::kMaxBlockColumns];
                for (std::ptrdiff_t b = blo; b < bhi; ++b) {
                    double* pb = part.data() + static_cast<std::size_t>(b) * kc;
                    const std::size_t lo = static_cast<std::size_t>(b) * mh::kFoldBlock;
                    const std::size_t hi = std::min(nn, lo + mh::kFoldBlock);
                    for (std::size_t i = lo; i < hi; ++i) {
                        for (std::uint32_t j = 0; j < kc; ++j) acc[j] = 0.0;
                        for (int p = ptr[i]; p < ptr[i + 1]; ++p) {
                            const double a = val[p];
                            const double* xr = xd + static_cast<std::size_t>(col[p]) * kc;
                            for (std::uint32_t j = 0; j < kc; ++j) acc[j] += a * xr[j];
                        }
                        const double* br = bd + i * kc;
                        for (std::uint32_t j = 0; j < kc; ++j) {
                            const double d = br[j] - acc[j];
                            const double sq = d * d;
                            pb[j] += sq;
                        }
                    }
                }
            }
            fold_columns(sums);
        }
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(nn); ++v) {
            const double* row = xd + static_cast<std::size_t>(perm[v]) * kc;
            for (std::uint32_t j = 0; j < kc; ++j) xcol[j][v] = row[j];
        }
        for (std::uint32_t j = 0; j < kc; ++j) {
            const job& jb = batch[j];
            const dm::column_state st = cs[j];
            const double bn = bnorm[static_cast<std::size_t>(jb.c)];
            metal_column_result& out = result.columns[static_cast<std::size_t>(jb.c)];
            out.iterations = st.iters;
            out.residual = std::sqrt(sums[j]) / bn;
            out.recursive_residual = st.iters > 0
                ? std::sqrt(std::max(0.0, mh::join({st.rr.hi, st.rr.lo}))) / (bn * jb.scale)
                : jb.initial;
            out.converged = out.residual < tol;
            out.stop = stop_of(st.stop);
        }
        cp("exit_check");
        batch.clear();
    };

    for (Eigen::Index c = 0; c < k; ++c) {
        metal_column_result& out = result.columns[static_cast<std::size_t>(c)];
        const double bn = bnorm[static_cast<std::size_t>(c)];
        if (bn == 0.0) {
            X.col(c).setZero();
            out.converged = 0.0 < tol;
            out.stop = metal_stop::zero_rhs;
            continue;
        }
        job jb;
        jb.c = c;
        double max_abs = bmax[static_cast<std::size_t>(c)];
        const bool warm = X0 != nullptr && !X0->col(c).isZero(0.0);
        if (warm) {
            std::vector<double> bp(s.n);
            mh::scatter(B.col(c).data(), perm, s.n, bp.data());
            jb.x0p.resize(s.n);
            mh::scatter(X0->col(c).data(), perm, s.n, jb.x0p.data());
            jb.w.resize(s.n);
            s.spmv(jb.x0p.data(), jb.w.data());
            #pragma omp parallel for schedule(static)
            for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(s.n); ++q)
                jb.w[q] = bp[q] - jb.w[q];
            jb.initial = std::sqrt(mh::fold_sum_squares(jb.w.data(), s.n)) / bn;
            if (jb.initial < tol || max_iter == 0) {
                finish_host(c, jb.x0p, bp, 0, jb.initial,
                            jb.initial < tol ? metal_stop::initial_guess : metal_stop::max_iterations);
                continue;
            }
            max_abs = mh::fold_max_abs(jb.w.data(), s.n);
            if (max_abs == 0.0) {  // b - A x0 vanished in fp64: x0 is exact
                finish_host(c, jb.x0p, bp, 0, 0.0, metal_stop::initial_guess);
                continue;
            }
        } else {
            jb.initial = 1.0;
            if (max_iter == 0) {
                // Honest pre-loop state: x = 0, relative residual exactly 1.
                X.col(c).setZero();
                out.residual = 1.0;
                out.recursive_residual = 1.0;
                out.converged = 1.0 < tol;
                out.stop = metal_stop::max_iterations;
                continue;
            }
        }
        jb.scale = mh::pow2_scale(max_abs);
        batch.push_back(std::move(jb));
        if (batch.size() == s.block_columns) run_batch();
    }
    run_batch();
    cp.ascend();
    return result;
}

metal_block_result metal_solver::solve(block_cref B, double tol, int max_iter,
                                       const block_cref* X0) const {
    if (B.rows() != rows()) throw std::invalid_argument("metal_solver::solve: B row count mismatch");
    Eigen::MatrixXd X(B.rows(), B.cols());
    metal_block_result result = solve(B, X, tol, max_iter, X0);
    result.X = std::move(X);
    return result;
}

solve_result metal_solver::solve(const Eigen::VectorXd& b, double tol, int max_iter,
                                 const Eigen::VectorXd* x0) const {
    if (b.size() != rows()) throw std::invalid_argument("metal_solver::solve: b length mismatch");
    if (x0 != nullptr && x0->size() != rows())
        throw std::invalid_argument("metal_solver::solve: x0 length mismatch");
    Eigen::MatrixXd X(b.size(), 1);
    const block_cref B(b);
    std::optional<block_cref> X0;
    if (x0 != nullptr) X0.emplace(*x0);
    metal_block_result block = solve(B, X, tol, max_iter, X0 ? &*X0 : nullptr);
    solve_result res;
    res.x = X.col(0);
    res.iterations = block.columns[0].iterations;
    res.residual = block.columns[0].residual;
    res.lumped_offdiag = block.lumped_offdiag;
    res.timings = std::move(block.timings);
    return res;
}

Eigen::VectorXd metal_solver::apply(const Eigen::VectorXd& r) const {
    impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (r.size() != static_cast<Eigen::Index>(s.n))
        throw std::invalid_argument("metal_solver::apply: r length mismatch");
    if (!r.allFinite()) throw std::invalid_argument("metal_solver::apply: r contains a non-finite value");
    const node_index* perm = s.F.perm.data();
    std::vector<double> rp(s.n);
    mh::scatter(r.data(), perm, s.n, rp.data());
    Eigen::VectorXd z = Eigen::VectorXd::Zero(r.size());
    const double max_abs = mh::fold_max_abs(rp.data(), s.n);
    if (max_abs == 0.0) return z;
    const double scale = mh::pow2_scale(max_abs);
    s.device->reserve(1);
    mh::pack_column(rp.data(), s.n, scale, 1, 0, as_df(s.device->r()));
    dm::column_state& cs = s.device->columns()[0];
    cs = dm::column_state{};
    cs.active = 1;
    const auto& plans = s.plan(1);
    s.device->apply(1, plans.first, plans.second);
    const float* p = s.device->p();
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(s.n); ++q) rp[q] = static_cast<double>(p[q]) / scale;
    mh::gather(rp.data(), perm, s.n, z.data());
    return z;
}

}  // namespace apxchol

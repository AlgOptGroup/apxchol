// Host side of the Apple GPU PCG (include/apxchol/solver/metal_solver.h):
// factorization, the dropped-factor schedules, the permuted operator, vector conversion,
// the fp64 exit checks and the Laplacian centring. All OpenMP host work lives
// here; the device side (src/metal_device.mm) sees only plain arrays.
#include "apxchol/solver/metal_solver.h"

#include "apxchol/csc_work.h"
#include "apxchol/solver/detail/metal_host.h"
#include "apxchol/solver/detail/permuted_operator.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include "apxchol/solver/detail/metal_schedule.h"
#include "metal_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

namespace apxchol {

namespace {

namespace mh = detail::metal_host;
namespace ls = detail::metal_schedule;
namespace dm = detail::metal;

void validate_options(const solve_options& opts) {
    if (opts.backend == solve_backend::cpu)
        throw std::invalid_argument("metal_solver cannot execute an explicit CPU request");
    if (detail::resolve_fp16_storage(opts.factor_opts.factor_storage, false))
        throw std::invalid_argument("Metal requires float32 factor storage");
}

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
                     "[apxchol] Metal PCG on %s: fp32 factor, "
                     "double-float Krylov vectors, %s operator\n",
                     st.device.c_str(),
                     st.operator_double_float ? "double-float" : "fp32-exact");
    });
}

}  // namespace

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
    std::unique_ptr<dm::engine> device;
    std::vector<dm::tri_step> forward, backward;
    std::uint32_t check_every = 1;
    statistics stats;

    void setup(const Eigen::SparseMatrix<double>& A, checkpoint* cp);

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
            tid = omp_get_thread_num();
            nt = omp_get_num_threads();
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

    auto plan = [](const ls::level_solve& schedule) {
        std::vector<dm::tri_step> out;
        for (const auto& step : ls::plan_steps(schedule))
            out.push_back({static_cast<std::uint32_t>(step.kind), step.first, step.last});
        return out;
    };
    forward = plan(sched.forward);
    backward = plan(sched.backward);
    stats.n = static_cast<Eigen::Index>(n);
    stats.levels_forward = sched.forward.levels();
    stats.levels_backward = sched.backward.levels();
    stats.steps_forward = forward.size();
    stats.steps_backward = backward.size();
    stats.operator_double_float = !exact;
    stats.device = st.name;
    if (!opts.keep_factor_values) F.L.release_values();
    if (cp) { (*cp)("metal_upload"); cp->ascend(); }
    print_banner_once(stats);
}

metal_solver::metal_solver(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
                           checkpoint* cp)
    : impl_(std::make_unique<impl>()) {
    validate_options(opts);
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
    validate_options(opts);
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

solve_result metal_solver::solve(const Eigen::VectorXd& b, double tol, int max_iter,
                                 const Eigen::VectorXd* x0) const {
    solve_result result;
    solve(b, result, tol, max_iter, x0);
    return result;
}

void metal_solver::solve(const Eigen::VectorXd& b, solve_result& result, double tol,
                         int max_iter, const Eigen::VectorXd* x0) const {
    impl& s = *impl_;
    if (b.size() != rows()) throw std::invalid_argument("metal_solver::solve: RHS length mismatch");
    if (x0 != nullptr && x0->size() != rows())
        throw std::invalid_argument("metal_solver::solve: x0 length mismatch");
    if (!b.allFinite() || (x0 != nullptr && !x0->allFinite()))
        throw std::invalid_argument("metal_solver::solve: non-finite RHS or initial guess");
    if (tol < 0.0) tol = s.opts.tol;
    if (max_iter < 0) max_iter = s.opts.max_iter;
    const auto window = static_cast<std::uint32_t>(std::max(0, s.opts.stagnation_window));

    result.iterations = 0;
    result.residual = 0.0;
    result.backend = solve_backend::gpu;
    result.lumped_offdiag = s.F.lumped_offdiag;
    result.x = Eigen::VectorXd::Zero(rows());
    checkpoint& cp = result.timings;
    cp.descend("pcg");
    cp.tick();
    const double bnorm = std::sqrt(mh::fold_sum_squares(b.data(), s.n));
    if (bnorm == 0.0) { cp.ascend(); return; }
    if (max_iter == 0 && (x0 == nullptr || x0->isZero(0.0))) {
        result.residual = 1.0;
        cp.ascend();
        return;
    }

    const node_index* perm = s.F.perm.data();
    std::vector<double> bp(s.n), xp(s.n, 0.0), work(s.n);
    mh::scatter(b.data(), perm, s.n, bp.data());
    auto finish = [&] {
        if (s.laplacian) s.centre(xp.data());
        result.residual = s.residual_norm(bp.data(), xp.data(), work) / bnorm;
        mh::gather(xp.data(), perm, s.n, result.x.data());
        cp("exit_check");
        cp.ascend();
    };

    if (x0 != nullptr) {
        mh::scatter(x0->data(), perm, s.n, xp.data());
        s.spmv(xp.data(), work.data());
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(s.n); ++i)
            work[i] = bp[i] - work[i];
    } else {
        work = bp;
    }
    const double initial = std::sqrt(mh::fold_sum_squares(work.data(), s.n)) / bnorm;
    if (initial < tol || max_iter == 0) { finish(); return; }
    const double max_abs = mh::fold_max_abs(work.data(), s.n);
    if (max_abs == 0.0) { finish(); return; }
    const double scale = mh::pow2_scale(max_abs);
    mh::pack(work.data(), s.n, scale, as_df(s.device->r()));
    auto& state = s.device->state();
    state = {};
    const double threshold = tol * bnorm * scale;
    const double reference = bnorm * scale;
    state.thr = to_device(mh::split_saturated(threshold * threshold));
    state.prev = to_device(mh::split_saturated(reference * reference));
    state.active = 1;
    cp("pack");
    s.device->solve(s.forward, s.backward, static_cast<std::uint32_t>(max_iter), window, s.check_every);
    cp("device");
    mh::unpack(as_df(s.device->x()), s.n, scale, work.data());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(s.n); ++i) xp[i] += work[i];
    result.iterations = state.iters;
    finish();
    return;
}

Eigen::VectorXd metal_solver::apply(const Eigen::VectorXd& r) const {
    impl& s = *impl_;
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
    mh::pack(rp.data(), s.n, scale, as_df(s.device->r()));
    auto& cs = s.device->state();
    cs = {};
    cs.active = 1;
    s.device->apply(s.forward, s.backward);
    const float* p = s.device->p();
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(s.n); ++q) rp[q] = static_cast<double>(p[q]) / scale;
    mh::gather(rp.data(), perm, s.n, z.data());
    return z;
}

}  // namespace apxchol

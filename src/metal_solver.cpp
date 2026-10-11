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
#include "apxchol/solver/sptrsv/cuda_host.h"
#include "metal_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

dm::df32 to_device(mh::df v) { return {v.hi, v.lo}; }

// Error-free transforms need round-to-nearest-even, separately rounded
// intermediates and an actual fused multiply-add.
std::string run_arithmetic_probe() {
    const auto out = dm::run_probe({
        0x1p0f, 0x1.8p-24f, 0x1.8p-24f, 0.0f, // Sum rounds up, leaving -2^-25.
        0x1p0f, 0.0f, 0x1p-24f, 0.0f,         // Halfway sum rounds to the even mantissa.
        0x1.000002p0f, 0x1p-24f, 0x1.000002p0f, 0.0f,
    });
    // The last quick_two_sum starts with an odd mantissa, so its tie rounds up.
    if (out[0] != 0x1.000002p0f || out[1] != -0x1p-25f || out[15] != -0x1p-25f ||
        out[16] != 0x1p0f || out[17] != 0x1p-24f || out[47] != -0x1p-24f)
        return "Metal double-float transforms require round-to-nearest-even without reassociation";
    // (1 + 2^-23)^2 = 1 + 2^-22 + 2^-46; only FMA retains the product residual.
    if (out[34] != 0x1.000004p0f || out[35] != 0x1p-46f)
        return "Metal double-float transforms require a fused multiply-add";
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
            a.reason = run_arithmetic_probe();
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
    bool laplacian = false;
    // The permuted operator A' = P A P^T (canonical lower values) in fp64:
    // the host residuals run on it.
    std::vector<int> op_ptr;
    std::unique_ptr<int[]> op_col;
    std::unique_ptr<double[]> op_val;
    std::unique_ptr<dm::engine> device;
    std::uint32_t check_every = 1;
    statistics stats;

    explicit impl(const solve_options& options) : opts(options) {
        if (opts.backend == solve_backend::cpu)
            throw std::invalid_argument("metal_solver cannot execute an explicit CPU request");
        if (detail::resolve_fp16_storage(opts.factor_opts.factor_storage, false))
            throw std::invalid_argument("Metal requires float32 factor storage");
        if (!metal_solver::available())
            throw std::runtime_error("apxchol::metal_solver: the Metal backend is unavailable: " +
                                     check_availability().reason);
    }

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
            const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
            const auto [lo, hi] = detail::work_balanced_range(ptr, rows, tid, nt);
            for (std::ptrdiff_t i = lo; i < hi; ++i) {
                double acc = 0.0;
                for (int p = ptr[i]; p < ptr[i + 1]; ++p) acc += val[p] * x[col[p]];
                y[i] = acc;
            }
        }
    }

    double residual_norm(const Eigen::VectorXd& bp, const Eigen::VectorXd& xp,
                         Eigen::VectorXd& work) const {
        spmv(xp.data(), work.data());
        return (bp - work).stableNorm();
    }

    void centre(Eigen::VectorXd& x) const { x.array() -= x.mean(); }
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
    const auto m = static_cast<std::uint32_t>(laplacian ? n - 1 : n);
    if (F.L.outerIndexPtr()[m] > static_cast<edge_index>(std::numeric_limits<int>::max()))
        throw std::length_error("metal_solver: the factor exceeds the device's 32-bit offsets");

    if (cp) { cp->descend("setup"); cp->tick(); }
    // Keep compensated dropping in the shared FP64 host implementation.
    // Both backends consume the CSR factor without a level-permuted value copy.
    auto backward = cuda_host::build_L11_csc_int<float>(F.L, m);
    const auto scales = cuda_host::column_scales(backward);
    cuda_host::apply_factor_drop(backward, scales, factor_drop_rel_from_env(), false);
    const auto forward = cuda_host::transpose_csr(backward);
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
    std::int64_t op_nnz = 0;
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
    auto arrays = [](const cuda_host::csr_int<float>& s) {
        dm::tri_arrays t;
        t.ptr = reinterpret_cast<const std::uint32_t*>(s.ptr.data());
        t.col = reinterpret_cast<const std::uint32_t*>(s.idx.get());
        t.val = s.vals.get();
        t.nnz = static_cast<std::size_t>(s.nnz);
        return t;
    };
    device = std::make_unique<dm::engine>(op, arrays(forward), arrays(backward), m);
    // Group four iterations on small systems to amortize command submission;
    // check large solves after each iteration.
    check_every = n > 200000 ? 1 : 4;

    const auto prepared = device->factor_stats();
    stats.n = static_cast<Eigen::Index>(n);
    stats.levels_forward = prepared.levels_forward;
    stats.levels_backward = prepared.levels_backward;
    stats.steps_forward = prepared.steps_forward;
    stats.steps_backward = prepared.steps_backward;
    stats.operator_double_float = !exact;
    stats.device = st.name;
    if (!opts.keep_factor_values) F.L.release_values();
    if (cp) { (*cp)("metal_upload"); cp->ascend(); }
    print_banner_once(stats);
}

metal_solver::metal_solver(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
                           checkpoint* cp)
    : impl_(std::make_unique<impl>(opts)) {
    impl_->F = detail::factorize_for_solver(A, opts.storage, opts.factor_opts, cp, true);
    impl_->setup(A, cp);
}

metal_solver::metal_solver(const Eigen::SparseMatrix<double>& A, factorization F,
                           const solve_options& opts, checkpoint* cp)
    : impl_(std::make_unique<impl>(opts)) {
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
    const double bnorm = b.stableNorm();
    if (bnorm == 0.0) { cp.ascend(); return; }
    if (max_iter == 0 && (x0 == nullptr || x0->isZero(0.0))) {
        result.residual = 1.0;
        cp.ascend();
        return;
    }

    const node_index* perm = s.F.perm.data();
    Eigen::VectorXd bp(s.n), work(s.n);
    auto& xp = result.x;
    mh::scatter(b.data(), perm, s.n, bp.data());
    auto finish = [&] {
        if (s.laplacian) s.centre(xp);
        result.residual = s.residual_norm(bp, xp, work) / bnorm;
        mh::gather(xp.data(), perm, s.n, work.data());
        result.x.swap(work);
        cp("exit_check");
        cp.ascend();
    };

    if (x0 != nullptr) {
        mh::scatter(x0->data(), perm, s.n, xp.data());
        s.spmv(xp.data(), work.data());
        work = bp - work;
    } else {
        work = bp;
    }
    const double initial = work.stableNorm() / bnorm;
    if (initial < tol || max_iter == 0) { finish(); return; }
    if (s.laplacian) s.centre(work);
    const double max_abs = work.cwiseAbs().maxCoeff();
    if (max_abs == 0.0) { finish(); return; }
    const double scale = mh::pow2_scale(max_abs);
    mh::pack(work.data(), s.n, scale, s.device->r());
    auto& state = s.device->state();
    state = {};
    const double reference = bnorm * scale;
    const double threshold = tol * reference;
    state.thr = to_device(mh::split_saturated(threshold * threshold));
    state.prev = to_device(mh::split_saturated(reference * reference));
    state.active = 1;
    cp("pack");
    s.device->solve(static_cast<std::uint32_t>(max_iter), window, s.check_every);
    cp("device");
    mh::add_unpacked(s.device->x(), s.n, scale, xp.data());
    result.iterations = state.iters;
    finish();
}

Eigen::VectorXd metal_solver::apply(const Eigen::VectorXd& r) const {
    impl& s = *impl_;
    if (r.size() != static_cast<Eigen::Index>(s.n))
        throw std::invalid_argument("metal_solver::apply: r length mismatch");
    if (!r.allFinite()) throw std::invalid_argument("metal_solver::apply: r contains a non-finite value");
    const node_index* perm = s.F.perm.data();
    const double mean = s.laplacian ? r.mean() : 0;
    const double max_abs = (r.array() - mean).abs().maxCoeff();
    Eigen::VectorXd z = Eigen::VectorXd::Zero(r.size());
    if (max_abs == 0.0) return z;
    const double scale = mh::pow2_scale(max_abs);
    auto* packed = s.device->r();
    #pragma omp parallel for schedule(static)
    for (Eigen::Index i = 0; i < r.size(); ++i)
        packed[perm[i]] = to_device(mh::split(scale * (r[i] - mean)));
    auto& cs = s.device->state();
    cs = {};
    cs.active = 1;
    const float* applied = s.device->apply();
    const double mean_z = s.laplacian
        ? Eigen::Map<const Eigen::VectorXf>(applied, s.n).cast<double>().mean() : 0;
    #pragma omp parallel for schedule(static)
    for (Eigen::Index i = 0; i < r.size(); ++i)
        z[i] = (double(applied[perm[i]]) - mean_z) / scale;
    return z;
}

}  // namespace apxchol

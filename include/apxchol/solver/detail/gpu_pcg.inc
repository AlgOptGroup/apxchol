APXCHOL_GPU_CODE(
namespace apxchol { namespace detail { namespace gpu_pcg {
enum finalize_mode : unsigned {
    kFinalizeMuR, kFinalizeRzInit, kFinalizeMuZ, kFinalizeAlpha, kFinalizeCheck, kFinalizeRz
};
enum reduce_mode : unsigned { kReduceSumR, kReduceSumZ, kReduceRz };
enum stop_reason : unsigned { kStopRunning, kStopTolerance, kStopBreakdown, kStopStagnation, kStopNonfinite };

template<class Wide, class Scalar>
struct solve_state {
    Wide rz, rr, thr, prev, pap;
    Scalar alpha, beta, mu_r, mu_z;
    unsigned active, iters, stop, pad;
};

// The arithmetic policy preserves each backend's precision and stopping
// comparison. The recurrence and its state transitions have one implementation.
template<class Math, class State, class Wide>
inline State advance(Math math, State st, unsigned mode, Wide t,
                     unsigned iteration, unsigned window = 0) {
    if (mode == kFinalizeAlpha) {
        st.pap = t;
        if (!math.positive(t)) { st.active = 0; st.stop = kStopBreakdown; }
        else {
            st.alpha = math.ratio(st.rz, t);
            if (!math.accept_ratio(st.alpha)) { st.active = 0; st.stop = kStopNonfinite; }
        }
    } else if (mode == kFinalizeCheck) {
        st.rr = t;
        st.iters = iteration;
        if (!math.valid_residual(t)) { st.active = 0; st.stop = kStopNonfinite; }
        else if (math.converged(t, st.thr)) { st.active = 0; st.stop = kStopTolerance; }
        else if (window && iteration % window == 0) {
            if (math.stagnated(st.prev, t)) { st.active = 0; st.stop = kStopStagnation; }
            else st.prev = t;
        }
    } else if (mode == kFinalizeRz) {
        if (!math.valid_rz(t)) { st.active = 0; st.stop = kStopNonfinite; }
        else {
            const auto beta = math.ratio(t, st.rz);
            if (!math.accept_ratio(beta)) { st.active = 0; st.stop = kStopNonfinite; }
            else { st.beta = beta; st.rz = t; }
        }
    }
    return st;
}

// Keep each backend's index width and fixed group bounds. limit clips the
// final partial group; all lanes still participate in the row reduction.
template<class Index>
struct pcg_work {
    Index first, last, stride, offset, limit;
};

template<class Wide>
struct pcg_sums { Wide norm, sum; };

template<int Lanes, class Math, class Index, class Entry, class Offset>
APXCHOL_GPU_FUNCTION typename Math::wide pcg_spmv(
    Math math, APXCHOL_GPU_BUFFER(const Offset) ptr, APXCHOL_GPU_BUFFER(const Offset) col,
    APXCHOL_GPU_BUFFER(const Entry) hi, APXCHOL_GPU_BUFFER(const typename Math::scalar) p,
    APXCHOL_GPU_BUFFER(typename Math::wide) ap, pcg_work<Index> work, unsigned lane,
    bool active = true, APXCHOL_GPU_BUFFER(const Entry) lo = nullptr) {
    using Wide = typename Math::wide;
    Wide pap = Wide(0);
    if (active) {
        for (auto base = work.first; base < work.last; base += work.stride) {
            const auto row = base + work.offset;
            Wide sum = Wide(0);
            if (row < work.limit) {
                auto e = ptr[row];
                e += lane;
                const auto end = ptr[row + 1];
                for (; e < end; e += Lanes)
                    sum = math.add_entry(sum, hi, lo, e, p[col[e]]);
            }
            sum = math.sum_row(sum);
            if (row < work.limit && lane == 0) {
                ap[row] = sum;
                pap = math.add_product(pap, sum, p[row]);
            }
        }
    }
    return pap;
}

template<class Math, class Index>
APXCHOL_GPU_FUNCTION pcg_sums<typename Math::wide> pcg_update_xr(
    Math math, APXCHOL_GPU_BUFFER(typename Math::wide) x,
    APXCHOL_GPU_BUFFER(const typename Math::scalar) p, APXCHOL_GPU_BUFFER(typename Math::wide) r,
    APXCHOL_GPU_BUFFER(const typename Math::wide) ap, typename Math::scalar alpha,
    pcg_work<Index> work, bool sum_residual = false, bool active = true) {
    using Wide = typename Math::wide;
    pcg_sums<Wide> sums{Wide(0), Wide(0)};
    if (active)
        for (auto i = work.first + work.offset; i < work.last; i += work.stride) {
            if (i >= work.limit) break;
            x[i] = math.add_product(x[i], alpha, p[i]);
            const auto value = math.subtract_product(r[i], ap[i], alpha);
            r[i] = value;
            sums.norm = math.add_product(sums.norm, value, value);
            if (sum_residual) sums.sum = math.add(sums.sum, value);
        }
    return sums;
}

template<class Math, class Index>
APXCHOL_GPU_FUNCTION typename Math::wide pcg_reduce(
    Math math, APXCHOL_GPU_BUFFER(const typename Math::wide) r,
    APXCHOL_GPU_BUFFER(const typename Math::scalar) z, pcg_work<Index> work,
    unsigned mode = kReduceRz, typename Math::scalar mean_z = 0, bool active = true) {
    using Wide = typename Math::wide;
    Wide sum = Wide(0);
    if (active)
        for (auto i = work.first + work.offset; i < work.last; i += work.stride) {
            if (i >= work.limit) break;
            if (mode == kReduceSumR) sum = math.add(sum, r[i]);
            else if (mode == kReduceSumZ) sum = math.add(sum, math.from_scalar(z[i]));
            else sum = math.add_product(sum, r[i], z[i] - mean_z);
        }
    return sum;
}

template<class Math, class Index>
APXCHOL_GPU_FUNCTION void pcg_update_p(
    Math math, APXCHOL_GPU_BUFFER(typename Math::scalar) p,
    APXCHOL_GPU_BUFFER(const typename Math::scalar) z, typename Math::scalar beta,
    pcg_work<Index> work, typename Math::scalar mean_z = 0, bool active = true) {
    if (!active) return;
    for (auto i = work.first + work.offset; i < work.last; i += work.stride) {
        if (i >= work.limit) break;
        p[i] = math.direction(z[i] - mean_z, p[i], beta);
    }
}

} } }
)

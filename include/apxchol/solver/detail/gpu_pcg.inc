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
inline State advance(Math math, State st, unsigned mode, Wide t, Wide sum_r,
                     unsigned iteration, unsigned window, bool laplacian) {
    if (mode == kFinalizeMuR) {
        st.mu_r = laplacian ? math.mean(t) : 0;
    } else if (mode == kFinalizeMuZ) {
        st.mu_z = laplacian ? math.mean(t) : 0;
    } else if (mode == kFinalizeRzInit) {
        if (!math.valid_initial(t)) { st.active = 0; st.stop = kStopNonfinite; }
        else { st.rz = t; st.beta = 0; }
    } else if (mode == kFinalizeAlpha) {
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
        if (st.active && laplacian) st.mu_r = math.mean(sum_r);
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

template<class Wide, class Direction, class Entry, class Index = unsigned>
struct pcg_arrays {
    APXCHOL_GPU_BUFFER(Wide) x = nullptr;
    APXCHOL_GPU_BUFFER(const Wide) r = nullptr;
    APXCHOL_GPU_BUFFER(Wide) r_out = nullptr;
    APXCHOL_GPU_BUFFER(const Direction) p = nullptr;
    APXCHOL_GPU_BUFFER(Direction) p_out = nullptr;
    APXCHOL_GPU_BUFFER(const Wide) ap = nullptr;
    APXCHOL_GPU_BUFFER(Wide) ap_out = nullptr;
    APXCHOL_GPU_BUFFER(const Direction) z = nullptr;
    APXCHOL_GPU_BUFFER(const Index) ptr = nullptr;
    APXCHOL_GPU_BUFFER(const Index) col = nullptr;
    APXCHOL_GPU_BUFFER(const Entry) hi = nullptr;
    APXCHOL_GPU_BUFFER(const Entry) lo = nullptr;
    APXCHOL_GPU_BUFFER(Wide) partial = nullptr;
    unsigned row_lane = 0, row_lanes = 1;
};
// A uniform loop range for all threads participating in a reduction. offset
// selects a row/lane; limit clips a partial group without changing its fixed
// loop bound. The adapter owns the reduction tree.
template<class Index>
struct pcg_work {
    Index first, last, stride, offset, limit;
};

template<class Memory, class Index>
APXCHOL_GPU_FUNCTION void pcg_spmv(Memory m, pcg_work<Index> work) {
    using Wide = typename Memory::wide;
    const auto a = m.arrays;
    Wide pap = Wide(0);
    if (m.active()) {
        for (auto base = work.first; base < work.last; base += work.stride) {
            const auto row = base + work.offset;
            Wide sum = Wide(0);
            if (row < work.limit) {
                auto e = a.ptr[row];
                e += a.row_lane;
                const auto end = a.ptr[row + 1];
                for (; e < end; e += a.row_lanes)
                    sum = m.add_entry(sum, e, a.p[a.col[e]]);
            }
            sum = m.sum_row(sum);
            if (row < work.limit && a.row_lane == 0) {
                a.ap_out[row] = sum;
                pap = m.add_product(pap, sum, a.p[row]);
            }
        }
    }
    m.publish(pap, Wide(0));
}

template<class Memory, class Index>
APXCHOL_GPU_FUNCTION void pcg_update_xr(Memory m, pcg_work<Index> work) {
    using Wide = typename Memory::wide;
    const auto a = m.arrays;
    Wide rr = Wide(0), rs = Wide(0);
    if (m.active()) {
        const auto alpha = m.alpha();
        for (auto i = work.first + work.offset; i < work.last; i += work.stride) {
            if (i >= work.limit) break;
            a.x[i] = m.add_product(a.x[i], alpha, a.p[i]);
            const auto r = m.subtract_product(a.r[i], a.ap[i], alpha);
            a.r_out[i] = r;
            rr = m.add_product(rr, r, r);
            rs = m.residual_sum(rs, r);
        }
    }
    m.publish(rr, rs);
}

template<class Memory, class Index>
APXCHOL_GPU_FUNCTION void pcg_reduce(Memory m, pcg_work<Index> work) {
    using Wide = typename Memory::wide;
    const auto a = m.arrays;
    Wide sum = Wide(0);
    if (m.active())
        for (auto i = work.first + work.offset; i < work.last; i += work.stride) {
            if (i >= work.limit) break;
            sum = m.add(sum, m.reduction_term(i));
        }
    m.publish(sum, Wide(0));
}

template<class Memory, class Index>
APXCHOL_GPU_FUNCTION void pcg_update_p(Memory m, pcg_work<Index> work) {
    if (!m.active()) return;
    const auto a = m.arrays;
    const auto beta = m.beta();
    for (auto i = work.first + work.offset; i < work.last; i += work.stride) {
        if (i >= work.limit) break;
        a.p_out[i] = m.direction(a.z[i], a.p[i], beta);
    }
}

} } }
)

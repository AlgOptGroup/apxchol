APXCHOL_GPU_CODE(
namespace apxchol { namespace detail { namespace gpu_pcg {
enum finalize_mode : unsigned {
    kFinalizeRzInit, kFinalizeAlpha, kFinalizeCheck, kFinalizeRz
};
enum class operation : unsigned { multiply, residual, dot, direction };

// All lanes participate; the complete sum is returned in lane zero.
template<unsigned Lanes, class Math>
APXCHOL_GPU_FUNCTION typename Math::wide sum_lanes(Math math, typename Math::wide value) {
    for (unsigned offset = Lanes / 2; offset; offset /= 2)
        value = math.add(value, math.shuffle_down(value, offset, Lanes));
    return value;
}

template<class Wide, class Scalar>
struct solve_state {
    Wide rz, rr, thr, prev;
    Scalar alpha, beta;
    unsigned active, iters;
};

template<class Wide, class Scalar>
struct workspace {
    using state_type = solve_state<Wide, Scalar>;
    using wide_pointer = APXCHOL_GPU_BUFFER(Wide);
    using scalar_pointer = APXCHOL_GPU_BUFFER(Scalar);
    wide_pointer x, r, ap;
    scalar_pointer p, z;
    wide_pointer partial;
    APXCHOL_GPU_BUFFER(state_type) state;
};

// The arithmetic policy preserves each backend's precision and stopping
// comparison. The recurrence and its state transitions have one implementation.
template<class Math, class State, class Wide>
APXCHOL_GPU_FUNCTION State advance(Math math, State st, unsigned mode, Wide t,
                     unsigned iteration, unsigned window = 0) {
    if (mode == kFinalizeRzInit) {
        if (!math.valid_initial(t)) st.active = 0;
        else { st.rz = t; st.beta = 0; }
    } else if (mode == kFinalizeAlpha) {
        if (!math.positive(t)) st.active = 0;
        else {
            st.alpha = math.ratio(st.rz, t);
            if (!math.accept_ratio(st.alpha)) st.active = 0;
        }
    } else if (mode == kFinalizeCheck) {
        st.rr = t;
        st.iters = iteration;
        if (!math.valid_residual(t)) st.active = 0;
        else if (math.converged(t, st.thr)) st.active = 0;
        else if (window && iteration % window == 0) {
            if (math.stagnated(st.prev, t)) st.active = 0;
            else st.prev = t;
        }
    } else if (mode == kFinalizeRz) {
        if (!math.valid_rz(t)) st.active = 0;
        else {
            const auto beta = math.ratio(t, st.rz);
            if (!math.accept_ratio(beta)) st.active = 0;
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

// Each backend specializes op before launching. The shared loop preserves its
// row assignment and reduction order while using one vector workspace.
template<int Lanes, class Math, class Index, class Entry, class Offset>
APXCHOL_GPU_FUNCTION typename Math::wide step(
    operation op, Math math, workspace<typename Math::wide, typename Math::scalar> v,
    APXCHOL_GPU_BUFFER(const Offset) ptr, APXCHOL_GPU_BUFFER(const Offset) col,
    APXCHOL_GPU_BUFFER(const Entry) hi, APXCHOL_GPU_BUFFER(const Entry) lo,
    pcg_work<Index> work, unsigned lane) {
    using Wide = typename Math::wide;
    Wide total{};
    if (!v.state->active) return total;
    for (auto base = work.first; base < work.last; base += work.stride) {
        const auto row = base + work.offset;
        if (op == operation::multiply) {
            Wide sum{};
            if (row < work.limit) {
                auto e = ptr[row];
                e += lane;
                const auto end = ptr[row + 1];
                for (; e < end; e += Lanes)
                    sum = math.add_entry(sum, hi, lo, e, v.p[col[e]]);
            }
            sum = math.sum_row(sum);
            if (row < work.limit && lane == 0) {
                v.ap[row] = sum;
                total = math.add_product(total, sum, v.p[row]);
            }
        } else if (row < work.limit) {
            if (op == operation::residual) {
                v.x[row] = math.add_product(v.x[row], v.state->alpha, v.p[row]);
                const auto value = math.subtract_product(v.r[row], v.ap[row], v.state->alpha);
                v.r[row] = value;
                total = math.add_product(total, value, value);
            } else if (op == operation::dot) {
                total = math.add_product(total, v.r[row], v.z[row]);
            } else {
                const auto beta = v.state->beta;
                v.p[row] = beta == 0 ? v.z[row] : math.direction(v.z[row], v.p[row], beta);
            }
        }
    }
    return total;
}

template<class Math, class Index>
APXCHOL_GPU_FUNCTION typename Math::wide sum_partials(
    Math math, APXCHOL_GPU_BUFFER(const typename Math::wide) partial, pcg_work<Index> work) {
    using Wide = typename Math::wide;
    Wide sum{};
    for (auto i = work.first + work.offset; i < work.last; i += work.stride)
        sum = math.add(sum, partial[i]);
    return sum;
}

} } }
)

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
    if (mode == kFinalizeRzInit || mode == kFinalizeRz) {
        if (!math.positive(t)) st.active = 0;
        else {
            const auto beta = mode == kFinalizeRzInit ? 0 : math.ratio(t, st.rz);
            if (!math.accept_ratio(beta)) st.active = 0;
            else { st.beta = beta; st.rz = t; }
        }
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
    }
    return st;
}

// Keep each backend's index width and fixed group bounds. limit clips the
// final partial group; all lanes still participate in the row reduction.
template<class Index>
struct pcg_work {
    Index first, last, stride, offset, limit;
};

template<class Value, unsigned Columns>
struct column_sums { Value values[Columns]; };

// RHS entries are adjacent; the factor/operator traversal is shared while
// each column keeps its own recurrence state and reduction order.
template<int Lanes, unsigned Columns, class Math, class Index, class Entry, class Offset>
APXCHOL_GPU_FUNCTION column_sums<typename Math::wide, Columns> step_columns(
    operation op, Math math, workspace<typename Math::wide, typename Math::scalar> v,
    APXCHOL_GPU_BUFFER(const Offset) ptr, APXCHOL_GPU_BUFFER(const Offset) col,
    APXCHOL_GPU_BUFFER(const Entry) hi, APXCHOL_GPU_BUFFER(const Entry) lo,
    pcg_work<Index> work, unsigned lane) {
    using Wide = typename Math::wide;
    column_sums<Wide, Columns> total{};
    bool active = false;
    for (unsigned q = 0; q < Columns; ++q) active |= v.state[q].active != 0;
    if (!active) return total;
    for (auto base = work.first; base < work.last; base += work.stride) {
        const auto row = base + work.offset;
        if (op == operation::multiply) {
            Wide sum[Columns]{};
            if (row < work.limit) {
                auto e = ptr[row];
                e += lane;
                const auto end = ptr[row + 1];
                for (; e < end; e += Lanes) {
                    const auto position = col[e] * Columns;
                    for (unsigned q = 0; q < Columns; ++q)
                        if (v.state[q].active)
                            sum[q] = math.add_entry(sum[q], hi, lo, e, v.p[position + q]);
                }
            }
            for (unsigned q = 0; q < Columns; ++q) {
                sum[q] = math.sum_row(sum[q]);
                if (row < work.limit && lane == 0 && v.state[q].active) {
                    v.ap[row * Columns + q] = sum[q];
                    total.values[q] = math.add_product(total.values[q], sum[q], v.p[row * Columns + q]);
                }
            }
        } else if (row < work.limit) {
            for (unsigned q = 0; q < Columns; ++q) {
                if (!v.state[q].active) continue;
                const auto i = row * Columns + q;
                if (op == operation::residual) {
                    v.x[i] = math.update_solution(v.x[i], v.state[q].alpha, v.p[i]);
                    const auto value = math.subtract_product(v.r[i], v.ap[i], v.state[q].alpha);
                    v.r[i] = value;
                    total.values[q] = math.add_product(total.values[q], value, value);
                } else if (op == operation::dot) {
                    total.values[q] = math.add_product(total.values[q], v.r[i], math.preconditioned(v.z[i]));
                } else {
                    const auto beta = v.state[q].beta;
                    const auto z = math.preconditioned(v.z[i]);
                    v.p[i] = beta == 0 ? z : math.direction(z, v.p[i], beta);
                }
            }
        }
    }
    return total;
}

template<unsigned Columns, class Math, class Index>
APXCHOL_GPU_FUNCTION column_sums<typename Math::wide, Columns> sum_partials_columns(
    Math math, APXCHOL_GPU_BUFFER(const typename Math::wide) partial, pcg_work<Index> work) {
    column_sums<typename Math::wide, Columns> sum{};
    for (auto i = work.first + work.offset; i < work.last; i += work.stride)
        for (unsigned q = 0; q < Columns; ++q)
            sum.values[q] = math.add(sum.values[q], partial[i * Columns + q]);
    return sum;
}

} } }
)

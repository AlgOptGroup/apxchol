APXCHOL_GPU_CODE(
namespace apxchol { namespace detail { namespace metal_host {
/// A double-float: the unevaluated sum hi + lo of two fp32 values, normalized
/// so that |lo| <= ulp(hi) / 2 (about 48 significant bits).
struct df {
    float hi, lo;
};

inline df two_sum(float a, float b) {
    const float s = a + b;
    const float bb = s - a;
    const float e = (a - (s - bb)) + (b - bb);
    return {s, e};
}

inline df quick_two_sum(float a, float b) {
    const float s = a + b;
    const float e = b - (s - a);
    return {s, e};
}

inline df df_add(df a, df b) {
    df s = two_sum(a.hi, b.hi);
    const df t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}

inline df two_prod(float a, float b) {
    const float p = a * b;
    const float e = fma(a, b, -p);
    return {p, e};
}

inline df df_mul(df a, df b) {
    const float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e);
    e = fma(a.lo, b.hi, e);
    return quick_two_sum(p, e);
}

inline df df_mul_f(df a, float b) {
    const float p = a.hi * b;
    float e = fma(a.hi, b, -p);
    e = fma(a.lo, b, e);
    return quick_two_sum(p, e);
}

inline df df_neg(df a) { return {-a.hi, -a.lo}; }
inline df df_div(df a, df b) {
    const float q1 = a.hi / b.hi;
    const df r = df_add(a, df_neg(df_mul_f(b, q1)));
    return quick_two_sum(q1, r.hi / b.hi);
}
inline bool df_lt(df a, df b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
} } }
)

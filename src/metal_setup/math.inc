APXCHOL_GPU_CODE(
namespace sparse_round {
struct compensated_math {
    using wide = apxchol::detail::metal_host::df;
    wide from_float(float x) const { return {x, 0.0f}; }
    float to_float(wide x) const { return x.hi + x.lo; }
    bool finite(wide x) const { return isfinite(x.hi) && isfinite(x.lo); }
    bool less(wide x, wide y) const { return apxchol::detail::metal_host::df_lt(x, y); }
    wide add(wide x, wide y) const { return apxchol::detail::metal_host::df_add(x, y); }
    wide subtract(wide x, wide y) const {
        return add(x, apxchol::detail::metal_host::df_neg(y));
    }
    wide multiply(wide x, wide y) const { return apxchol::detail::metal_host::df_mul(x, y); }
    wide divide(wide x, wide y) const { return apxchol::detail::metal_host::df_div(x, y); }
    wide reweight(wide weight, wide probability) const { return divide(weight, probability); }
    wide root(wide x) const {
        if (x.hi == 0 && x.lo == 0) return x;
        const float y = sqrt(x.hi);
        const auto residual = subtract(x, apxchol::detail::metal_host::two_prod(y, y));
        return add(from_float(y), divide(residual, from_float(2.0f * y)));
    }
    // Rounds the same 53 random bits to the two-float representation. Keeping
    // the signed integer residual also prevents a near-one draw rounding to one.
    wide uniform(seed_type bits) const {
        const float rounded = float(bits);
        const auto high = seed_type(rounded);
        const float residual = bits >= high ? float(bits - high) : -float(high - bits);
        return {rounded * 0x1p-53f, residual * 0x1p-53f};
    }
    // Both nonnegative operands are at most the degree. Divide the larger one
    // first: avoid product overflow without needlessly underflowing a tiny ratio.
    wide multiply_divide(wide x, wide y, wide z) const {
        return less(x, y) ? multiply(x, divide(y, z)) : multiply(y, divide(x, z));
    }
};
}
)

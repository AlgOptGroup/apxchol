#pragma once
// Low-precision STORAGE variants for the SpTRSV factor values, selected at
// configure time by the single CMake cache variable
//
//     APXCHOL_SPTRSV_LOWPREC = OFF | BF16 | BF16_SCALED | FP16_SCALED | FP24
//
// which defines exactly one of APXCHOL_SPTRSV_LOWPREC_{BF16, BF16_SCALED,
// FP16_SCALED, FP24} (the old boolean APXCHOL_SPTRSV_BF16 is accepted as an
// alias of _BF16). All variants share the same shape:
//
//   * ONLY the off-diagonals of the SpTRSV's CSR/CSC value arrays are stored
//     narrow; the DIAGONAL is kept fp32 in omp_sptrsv::diag_ (an 8-bit
//     diagonal was the dominant iteration-count damage of the first all-bf16
//     variant): L_jj itself, or under the *_SCALED variants the scaled
//     L_jj / s_j (omp_sptrsv::stored_diag). FP16_SCALED can be asked to read
//     the fp16 diagonal slot instead (env APXCHOL_FP16_DIAG=1, refused unless
//     every slot is a normal fp16 -- omp.h file header).
//   * The factor itself (sparse_csc::vals_, factor_value_t) is fp32; the
//     narrowing happens once, in omp_sptrsv::setup, through
//     omp_sptrsv::narrow_value (a pure function of the entry, so the CSR
//     transpose and the CSC copy agree bit-for-bit).
//   * Every read in the solve kernels widens to fp64 in registers via widen();
//     the arithmetic is unchanged and the kernels are one source for every
//     storage type (the fat-level kernels of the 16-bit types are SIMD:
//     _mm256_cvtph_ps / bf16 shift, vector gather + FMA -- env
//     APXCHOL_FP16_GATHER=simd|scalar picks the gather flavour). This is a
//     preconditioner-QUALITY knob (PCG iteration count), never a
//     residual-floor one.
//
// The variants differ in what per-entry rounding they apply (all RNE unless
// noted), which is what the T=1 iteration-count table in the commit history
// discriminates:
//
//   BF16         bfloat16 (8 significant bits, 2^-8 relative, fp32 range).
//                Env APXCHOL_BF16_STOCHASTIC=1 selects unbiased stochastic
//                rounding (bf16.h). The original variant.
//   BF16_SCALED  bf16 of L_ij / s_j with a per-COLUMN scale s_j = max_i
//                |L_ij| over column j's off-diagonals (stored fp32 in
//                omp_sptrsv::scale_; 1.0f if the column has no nonzero
//                off-diagonal). Widen: bf16 -> fp32 -> fp64. The scale is
//                NOT multiplied back by the kernels: it is folded into the
//                vectors (forward_solve returns D y, transpose_solve takes it
//                and scales its input by D^-2 -- omp.h "FOLDED INTO THE
//                VECTORS"). Isolates the effect of the scaling alone (bf16
//                does not need it for range) versus the mantissa width,
//                against FP16_SCALED.
//   FP16_SCALED  IEEE binary16 (11 significant bits, 2^-11 relative in the
//                normal range) of L_ij / s_j, same s_j. The scaling maps every
//                column's largest off-diagonal to +-1.0 exactly, so no entry
//                overflows; entries below 2^-14 of their column max fall into
//                fp16's SUBNORMAL range (absolute precision 2^-24 in the scaled
//                units, i.e. progressively fewer significant bits) and entries
//                below 2^-25 of it FLUSH TO ZERO under RNE (2^-25 itself ties
//                to even -> 0). setup() counts both and reports them under
//                APXCHOL_VERBOSE (omp_sptrsv::lowprec_stats()) -- the flushed
//                fraction is the direct test of "is dropping tiny entries
//                relative to their column harmless".
//   FP24         the top 24 bits of the fp32 pattern (sign, 8 exponent, 15
//                explicit mantissa bits -> 16 significant, 2^-16 relative,
//                fp32 range), RNE on the dropped 8 bits, packed in 3 bytes.
//                No scaling. 3 B/entry instead of 4.
//
// FP16_SCALED additionally flushes stored fp16 SUBNORMALS to (signed) zero at
// storage time by default (env APXCHOL_FP16_KEEP_SUBNORMAL=1 keeps them);
// and every build runs the COMPACTING drop (APXCHOL_FACTOR_DROP=<rel>, ON by
// default at kFactorDropRelDefault = 1e-4 with column-sum compensation:
// off-diagonals with |L_ij| < rel * s_j, plus whatever the storage format
// would store as zero, are REMOVED from the CSR/CSC at setup and their mass
// folded back into the kept entries of the column). Both live in omp_sptrsv
// (solver/sptrsv/omp.h, file header).
//
// This header holds the variant-selection macros, the fp16_t / fp24_t
// storage types (bf16_t lives in bf16.h) and their widen() overloads. The
// omp SpTRSV backend is the only consumer; the CUDA backend is fp32/fp64
// only (configure errors out on the combination).
#include "apxchol/bf16.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>
#if defined(__F16C__)
#include <immintrin.h>   // _cvtsh_ss: the one-instruction fp16 -> fp32 widen (vcvtph2ps)
#endif

// ── Variant selection ───────────────────────────────────────────────────
// Legacy alias: the boolean -DAPXCHOL_SPTRSV_BF16 means the unscaled bf16
// variant.
#if defined(APXCHOL_SPTRSV_BF16) && !defined(APXCHOL_SPTRSV_LOWPREC_BF16) && \
    !defined(APXCHOL_SPTRSV_LOWPREC_BF16_SCALED) && \
    !defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED) && !defined(APXCHOL_SPTRSV_LOWPREC_FP24)
#  define APXCHOL_SPTRSV_LOWPREC_BF16 1
#endif
#if defined(APXCHOL_SPTRSV_LOWPREC_BF16) + defined(APXCHOL_SPTRSV_LOWPREC_BF16_SCALED) + \
    defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED) + defined(APXCHOL_SPTRSV_LOWPREC_FP24) > 1
#  error "At most one APXCHOL_SPTRSV_LOWPREC_* variant may be defined (set the CMake variable APXCHOL_SPTRSV_LOWPREC)."
#endif
#if defined(APXCHOL_SPTRSV_BF16) && !defined(APXCHOL_SPTRSV_LOWPREC_BF16)
#  error "APXCHOL_SPTRSV_BF16 (alias of APXCHOL_SPTRSV_LOWPREC=BF16) conflicts with another APXCHOL_SPTRSV_LOWPREC_* variant."
#endif
// Derived: any low-precision variant (fp32 factor, separate fp32 diagonal
// array); any per-column-scaled variant (scale_ / inv_scale_ arrays, scaled
// diagonal, the D y / D^-2 pair contract).
#if defined(APXCHOL_SPTRSV_LOWPREC_BF16) || defined(APXCHOL_SPTRSV_LOWPREC_BF16_SCALED) || \
    defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED) || defined(APXCHOL_SPTRSV_LOWPREC_FP24)
#  define APXCHOL_SPTRSV_LOWPREC_ANY 1
#endif
#if defined(APXCHOL_SPTRSV_LOWPREC_BF16_SCALED) || defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
#  define APXCHOL_SPTRSV_LOWPREC_SCALED 1
#endif
#if defined(APXCHOL_SPTRSV_LOWPREC_ANY) && defined(APXCHOL_USE_CUDA)
#  error "The APXCHOL_SPTRSV_LOWPREC variants are implemented for the CPU/omp SpTRSV only; the CUDA backend has no low-precision path."
#endif

namespace apxchol {

// The selected variant as a string, for banners / tests ("OFF" on the
// fp32/fp64 builds).
inline constexpr const char* sptrsv_lowprec_variant =
#if defined(APXCHOL_SPTRSV_LOWPREC_BF16)
    "BF16";
#elif defined(APXCHOL_SPTRSV_LOWPREC_BF16_SCALED)
    "BF16_SCALED";
#elif defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
    "FP16_SCALED";
#elif defined(APXCHOL_SPTRSV_LOWPREC_FP24)
    "FP24";
#else
    "OFF";
#endif

// ── fp16_t: IEEE-754 binary16 storage ───────────────────────────────────
// 1 sign, 5 exponent (bias 15), 10 explicit mantissa bits (11 significant ->
// RNE relative error <= 2^-11 in the normal range [2^-14, 65504]). Values
// below 2^-14 are SUBNORMAL (m * 2^-24, m = 1..1023: absolute error <= 2^-25,
// relative precision degrades towards 1 bit at 2^-24); |x| < 2^-25 rounds to
// (signed) ZERO under RNE, |x| == 2^-25 exactly ties to even -> zero; |x| >=
// 65520 (the midpoint above 65504) rounds to +-inf; NaN stays NaN.
//
// Storage type only: the converting constructor narrows fp32 -> fp16 with a
// bit-level RNE (round_bits: same result as the compiler's _Float16 cast, which
// a unit test cross-checks exhaustively where _Float16 exists), and to_float()
// widens exactly -- via the F16C intrinsic _cvtsh_ss (one vcvtph2ps) where the
// target has it (-march=native on any x86 since Ivy Bridge / Zen), else the
// compiler-native _Float16, else the bit-level widen_bits(). The intrinsic is
// used deliberately instead of `(float)(_Float16)`: GCC folds the subsequent
// float -> double promotion into a direct half -> double conversion, which has
// no hardware instruction and becomes a libgcc __extendhfdf2 CALL in the SpTRSV
// inner loop (measured 3x solve slowdown before the switch). Widening is
// EXPLICIT (like bf16_t) so no read can silently do fp16 arithmetic; the
// SpTRSV kernels route through widen().
struct fp16_t {
    std::uint16_t bits;

    fp16_t() = default;   // trivially default-constructible, like float / bf16_t

    template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr fp16_t(T v) : bits(round_bits(std::bit_cast<std::uint32_t>(static_cast<float>(v)))) {}

    float to_float() const {
#if defined(__F16C__)
        return _cvtsh_ss(bits);
#elif defined(__FLT16_MANT_DIG__)
        return static_cast<float>(std::bit_cast<_Float16>(bits));
#else
        return widen_bits(bits);
#endif
    }
    explicit operator float()  const { return to_float(); }
    explicit operator double() const { return static_cast<double>(to_float()); }

    static constexpr fp16_t from_bits(std::uint16_t b) { fp16_t r; r.bits = b; return r; }

    // Classification on the bit pattern (what setup()'s flush statistics use).
    static constexpr bool is_zero(std::uint16_t h)      { return (h & 0x7fffu) == 0; }
    static constexpr bool is_subnormal(std::uint16_t h) { return (h & 0x7c00u) == 0 && (h & 0x03ffu) != 0; }
    static constexpr bool is_inf_or_nan(std::uint16_t h){ return (h & 0x7c00u) == 0x7c00u; }

    // fp32 bit pattern -> fp16 bit pattern, round-to-nearest-even, IEEE
    // semantics for subnormals / overflow / NaN. Pure integer arithmetic,
    // constexpr.
    static constexpr std::uint16_t round_bits(std::uint32_t u) {
        const std::uint32_t sign = (u >> 16) & 0x8000u;
        const std::uint32_t a    = u & 0x7fffffffu;            // |x| pattern
        if (a >= 0x7f800000u)                                  // inf or NaN
            return static_cast<std::uint16_t>(sign | 0x7c00u | (a > 0x7f800000u ? 0x0200u : 0u));
        if (a >= 0x477ff000u)                                  // >= 65520: RNE overflows to inf
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        if (a >= 0x38800000u) {                                // normal fp16 range: |x| >= 2^-14
            // Rebias the exponent (127 -> 15 == subtract 112 << 23), then drop
            // the low 13 mantissa bits with the half-way-tie-to-even bias; a
            // mantissa carry propagates into the exponent by plain addition.
            const std::uint32_t m   = a - (112u << 23);
            const std::uint32_t lsb = (m >> 13) & 1u;
            return static_cast<std::uint16_t>(sign | ((m + 0xfffu + lsb) >> 13));
        }
        if (a >= 0x33000000u) {                                // subnormal fp16 range: 2^-25 <= |x| < 2^-14
            // fp16 subnormal value = r * 2^-24. With the hidden bit restored the
            // fp32 significand M (24 bits) represents M * 2^(e-150) (e = biased
            // exponent), so r = round(M * 2^(e-126)) = RNE(M >> (126 - e)),
            // shift in [14, 24]. r may round up to 0x400 == the smallest normal,
            // which is the correct encoding.
            const std::uint32_t M     = (a & 0x7fffffu) | 0x800000u;
            const int           shift = 126 - static_cast<int>(a >> 23);
            const std::uint32_t half  = 1u << (shift - 1);
            const std::uint32_t lsb   = (M >> shift) & 1u;
            return static_cast<std::uint16_t>(sign | ((M + half - 1u + lsb) >> shift));
        }
        return static_cast<std::uint16_t>(sign);               // |x| < 2^-25: flush to signed zero
    }

    // fp16 bit pattern -> fp32 (exact), bit-level reference implementation.
    static constexpr float widen_bits(std::uint16_t h) {
        const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
        const std::uint32_t e    = (h >> 10) & 0x1fu;
        const std::uint32_t m    = h & 0x3ffu;
        if (e == 0) {
            if (m == 0) return std::bit_cast<float>(sign);                    // +-0
            const float f = static_cast<float>(m) * 0x1p-24f;                 // subnormal: exact
            return std::bit_cast<float>(sign | std::bit_cast<std::uint32_t>(f));
        }
        if (e == 31) return std::bit_cast<float>(sign | 0x7f800000u | (m << 13)); // inf / NaN
        return std::bit_cast<float>(sign | ((e + 112u) << 23) | (m << 13));
    }

    friend constexpr bool operator==(fp16_t a, fp16_t b) { return a.bits == b.bits; }
    friend constexpr bool operator!=(fp16_t a, fp16_t b) { return a.bits != b.bits; }
};
static_assert(sizeof(fp16_t) == 2, "fp16_t must be exactly 16 bits");
static_assert(std::is_trivially_copyable_v<fp16_t>);

// ── fp24_t: the top 24 bits of an fp32 ──────────────────────────────────
// Sign, 8 exponent, 15 explicit mantissa bits (16 significant -> RNE relative
// error <= 2^-16; same range as fp32, subnormals included). Packed in 3 bytes
// (little-endian byte order of the 24-bit pattern u >> 8), so an array of them
// is a dense 3 B/entry stream; loads assemble the 3 bytes into a uint32 and
// shift back into a float pattern (exact). Rounding on store is RNE on the
// dropped 8 bits (tie-to-even bias 0x7f + lsb, carry into the exponent by plain
// addition, NaN kept a NaN); values within half an fp24 ulp of FLT_MAX overflow
// to inf like any IEEE rounding would.
struct fp24_t {
    std::uint8_t b[3];   // b[0] = bits 0..7 of the 24-bit pattern, b[2] = bits 16..23

    fp24_t() = default;

    template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr fp24_t(T v) : b{} {
        set_bits(round_bits(std::bit_cast<std::uint32_t>(static_cast<float>(v))));
    }

    constexpr std::uint32_t bits() const {
        return static_cast<std::uint32_t>(b[0]) |
               (static_cast<std::uint32_t>(b[1]) << 8) |
               (static_cast<std::uint32_t>(b[2]) << 16);
    }
    constexpr void set_bits(std::uint32_t u24) {
        b[0] = static_cast<std::uint8_t>(u24 & 0xffu);
        b[1] = static_cast<std::uint8_t>((u24 >> 8) & 0xffu);
        b[2] = static_cast<std::uint8_t>((u24 >> 16) & 0xffu);
    }
    constexpr float to_float() const { return std::bit_cast<float>(bits() << 8); }
    explicit constexpr operator float()  const { return to_float(); }
    explicit constexpr operator double() const { return static_cast<double>(to_float()); }

    static constexpr fp24_t from_bits(std::uint32_t u24) { fp24_t r{}; r.set_bits(u24 & 0xffffffu); return r; }

    static constexpr bool is_zero(std::uint32_t u24)      { return (u24 & 0x7fffffu) == 0; }
    static constexpr bool is_subnormal(std::uint32_t u24) { return (u24 & 0x7f8000u) == 0 && (u24 & 0x007fffu) != 0; }

    // fp32 bit pattern -> 24-bit pattern (== the top 24 bits after RNE on the
    // dropped low 8).
    static constexpr std::uint32_t round_bits(std::uint32_t u) {
        if ((u & 0x7fffffffu) > 0x7f800000u)              // NaN: keep it a NaN
            return ((u >> 8) | 0x4000u) & 0xffffffu;
        const std::uint32_t lsb  = (u >> 8) & 1u;
        const std::uint32_t bias = 0x7fu + lsb;
        return ((u + bias) >> 8) & 0xffffffu;
    }

    friend constexpr bool operator==(fp24_t x, fp24_t y) { return x.bits() == y.bits(); }
    friend constexpr bool operator!=(fp24_t x, fp24_t y) { return x.bits() != y.bits(); }
};
static_assert(sizeof(fp24_t) == 3, "fp24_t must be exactly 3 bytes");
static_assert(alignof(fp24_t) == 1);
static_assert(std::is_trivially_copyable_v<fp24_t>);

/// widen(): stored value -> double for compute (see bf16.h for the
/// float/double/bf16 overloads).
inline double           widen(fp16_t v) { return static_cast<double>(v.to_float()); }
inline constexpr double widen(fp24_t v) { return static_cast<double>(v.to_float()); }

/// Subnormal-ness of a STORED value, per storage type (setup()'s flush
/// statistics; only fp16 can realistically be subnormal for a factor entry).
inline bool is_stored_subnormal(float v)  { return std::fpclassify(v) == FP_SUBNORMAL; }
inline bool is_stored_subnormal(double v) { return std::fpclassify(v) == FP_SUBNORMAL; }
inline constexpr bool is_stored_subnormal(bf16_t v) { return (v.bits & 0x7f80u) == 0 && (v.bits & 0x007fu) != 0; }
inline constexpr bool is_stored_subnormal(fp16_t v) { return fp16_t::is_subnormal(v.bits); }
inline constexpr bool is_stored_subnormal(fp24_t v) { return fp24_t::is_subnormal(v.bits()); }

} // namespace apxchol

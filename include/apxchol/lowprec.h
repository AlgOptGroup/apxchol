#pragma once
// Low-precision STORAGE variant for the SpTRSV factor values, selected at
// configure time by the single CMake cache variable
//
//     APXCHOL_SPTRSV_LOWPREC = OFF | FP16_SCALED
//
// which defines APXCHOL_SPTRSV_LOWPREC_FP16_SCALED (or nothing). The shape:
//
//   * ONLY the off-diagonals of the SpTRSV's CSR/CSC value arrays are stored
//     narrow; the DIAGONAL is kept fp32 in omp_sptrsv::diag_ (a narrow
//     diagonal was the dominant iteration-count damage of the first all-bf16
//     variant): the scaled L_jj / s_j (omp_sptrsv::stored_diag). FP16_SCALED
//     can be asked to read the fp16 diagonal slot instead (env
//     APXCHOL_FP16_DIAG=1, refused unless every slot is a normal fp16 --
//     omp.h file header).
//   * The factor itself (sparse_csc::vals_, factor_value_t) is fp32; the
//     narrowing happens once, in omp_sptrsv::setup, through
//     omp_sptrsv::narrow_value (a pure function of the entry, so the CSR
//     transpose and the CSC copy agree bit-for-bit).
//   * Every read in the solve kernels widens to fp64 in registers via widen();
//     the arithmetic is unchanged and the kernels are one source for every
//     storage type (the fat-level kernels of the 16-bit storage are SIMD:
//     _mm256_cvtph_ps, vector gather + FMA -- env APXCHOL_FP16_GATHER=
//     simd|scalar picks the gather flavour). This is a preconditioner-QUALITY
//     knob (PCG iteration count), never a residual-floor one.
//
//   FP16_SCALED  IEEE binary16 (11 significant bits, 2^-11 relative in the
//                normal range) of L_ij / s_j with a per-COLUMN scale s_j =
//                max_i |L_ij| over column j's off-diagonals (stored fp32 in
//                omp_sptrsv::scale_; 1.0f if the column has no nonzero
//                off-diagonal). The scaling maps every column's largest
//                off-diagonal to +-1.0 exactly, so no entry overflows; entries
//                below 2^-14 of their column max fall into fp16's SUBNORMAL
//                range (absolute precision 2^-24 in the scaled units, i.e.
//                progressively fewer significant bits) and entries below
//                2^-25 of it FLUSH TO ZERO under RNE (2^-25 itself ties to
//                even -> 0). setup() counts both and reports them under
//                APXCHOL_VERBOSE (omp_sptrsv::lowprec_stats()) -- the flushed
//                fraction is the direct test of "is dropping tiny entries
//                relative to their column harmless". The scale is NOT
//                multiplied back by the kernels: it is folded into the vectors
//                (forward_solve returns D y, transpose_solve takes it and
//                scales its input by D^-2 -- omp.h "FOLDED INTO THE VECTORS").
//
// (The bf16 / bf16-scaled / fp24 siblings that were measured against it --
// 8-bit mantissa 3-6x the PCG iterations on IPM, fp24 marginal -- were removed
// 2026-08-18; only fp16 with per-column folded scaling + fp32 diagonal +
// column-sum compensation stayed, default OFF.)
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
// This header holds the variant-selection macro, the fp16_t storage type and
// the widen() overloads (float / double / fp16_t). The omp SpTRSV backend is
// the compile-time consumer; the CUDA backend is fp32/fp64 at compile time
// (its fp16 storage is the runtime opt-in APXCHOL_GPU_SPTRSV_FP16=1, which
// narrows on the host through this fp16_t): with APXCHOL_USE_CUDA=ON the
// CMake variable is treated as OFF for that build (fp32 CPU storage, no
// APXCHOL_SPTRSV_LOWPREC_* macro, a STATUS line) so a non-OFF default can
// never break a CUDA build; the #error below only fires if the macro is
// defined by hand next to APXCHOL_USE_CUDA.
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>
#if defined(__F16C__)
#include <immintrin.h>   // _cvtsh_ss: the one-instruction fp16 -> fp32 widen (vcvtph2ps)
#endif

// ── Variant selection ───────────────────────────────────────────────────
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED) && defined(APXCHOL_USE_CUDA)
#  error "APXCHOL_SPTRSV_LOWPREC=FP16_SCALED is implemented for the CPU/omp SpTRSV only; the CUDA backend has no compile-time low-precision path (CMake treats APXCHOL_SPTRSV_LOWPREC as OFF under APXCHOL_USE_CUDA -- this macro was defined by hand)."
#endif

namespace apxchol {

// The selected variant as a string, for banners / tests ("OFF" on the
// fp32/fp64 builds).
inline constexpr const char* sptrsv_lowprec_variant =
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
    "FP16_SCALED";
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
// EXPLICIT so no read can silently do fp16 arithmetic; the SpTRSV kernels
// route through widen().
struct fp16_t {
    std::uint16_t bits;

    // No default member initializer, deliberately: fp16_t stays trivially
    // default-constructible like float, so `new fp16_t[n]` is uninitialized
    // (the transpose's transient bucket relies on that) while
    // std::vector::resize still value-initializes to zero bits (== 0.0f).
    fp16_t() = default;

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

/// widen(): read a stored factor value into a double for compute. Identity
/// (modulo the promotion the arithmetic would do anyway) for the fp32/fp64
/// builds, fp16 -> fp32 -> double for the FP16_SCALED build. The SpTRSV
/// kernels route every factor read through this.
inline constexpr double widen(double v) { return v; }
inline constexpr double widen(float v)  { return static_cast<double>(v); }
inline double           widen(fp16_t v) { return static_cast<double>(v.to_float()); }

/// Subnormal-ness of a STORED value, per storage type (setup()'s flush
/// statistics; only fp16 can realistically be subnormal for a factor entry).
inline bool is_stored_subnormal(float v)  { return std::fpclassify(v) == FP_SUBNORMAL; }
inline bool is_stored_subnormal(double v) { return std::fpclassify(v) == FP_SUBNORMAL; }
inline constexpr bool is_stored_subnormal(fp16_t v) { return fp16_t::is_subnormal(v.bits); }

} // namespace apxchol

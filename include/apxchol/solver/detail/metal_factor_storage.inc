APXCHOL_GPU_CODE(
namespace apxchol { namespace detail { namespace metal_host {
// diagonal = fp32(d_j s_j) is what the kernels divide by; scaled_diagonal
// keeps d_j itself for factor export, which cannot recover it from the product.
struct factor_metadata {
    float scale, diagonal, scaled_diagonal;
};
static_assert(sizeof(factor_metadata) == 12, "Matches host metadata");
} } }
)

APXCHOL_GPU_CODE(
namespace apxchol { namespace detail { namespace metal_host {
struct factor_metadata {
    float scale, diagonal;
};
static_assert(sizeof(factor_metadata) == 8, "Matches host metadata");
} } }
)

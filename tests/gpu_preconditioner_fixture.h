#pragma once
#include "apxchol/solver/preconditioner.h"

#if defined(APXCHOL_USE_CUDA)
namespace apxchol::test {
// Test-only access to individual setup stages, including deliberate host
// import/export. Production apx_cholesky is always CPU; production GPU solves
// require fully device-owned setup through detail::gpu_preconditioner.
using diagnostic_gpu_preconditioner = detail::basic_apx_cholesky<
    cuda_sptrsv, detail::setup_route::diagnostic>;
}
#endif

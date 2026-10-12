#pragma once

namespace apxchol::detail {

// The solver selects one complete setup route before factorization.
enum class setup_route { cpu, gpu };

} // namespace apxchol::detail

#pragma once

namespace apxchol::detail {

// Production solvers pass a concrete route through setup. Only low-level
// diagnostic entry points consult the legacy per-stage environment flags.
// This is a value, with no thread-local or process-wide dispatch state.
enum class setup_route { diagnostic, cpu, gpu };

} // namespace apxchol::detail

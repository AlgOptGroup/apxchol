#pragma once
#include <cmath>
namespace apxchol::detail {
inline bool pcg_restart(bool tolerance_stop, bool converged, double residual,
                        long long progress, long long remaining) {
    return tolerance_stop && !converged && std::isfinite(residual)
        && progress > 0 && remaining > 0;
}
}

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace bench_stop {
inline constexpr const char* contract = "original-v1";

// Policy only: setup lifetime, vectors, native iteration and residual evaluation
// belong to the adapter. The caller times the entire run, including checks.
// Bounded retries also handle native solves that return zero iterations. Never
// infer impossibility from a short plateau of the original residual.
struct result {
    int iterations = 0;
    int passes = 0;
    double residual = std::numeric_limits<double>::infinity();
    double check_seconds = 0;
};

template<class Pass, class Check>
result run(double target, int budget, Pass&& pass, Check&& check) {
    if (!(target > 0) || !std::isfinite(target) || budget < 0)
        throw std::invalid_argument("invalid original-residual stopping budget");
    result out;
    double native_tol = target;
    constexpr int max_passes = 8;
    for (;;) {
        if (out.iterations < budget) {
            const int remaining = budget - out.iterations;
            const int used = pass(native_tol, remaining, out.passes != 0);
            if (used < 0 || used > remaining)
                throw std::runtime_error("native solver exceeded remaining iteration budget");
            out.iterations += used;
            ++out.passes;
        }
        const auto begin = std::chrono::steady_clock::now();
        out.residual = check();
        out.check_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        if (!std::isfinite(out.residual) || out.residual <= target ||
            out.iterations >= budget || out.passes >= max_passes)
            return out;
        const double next = std::max(std::numeric_limits<double>::epsilon(), native_tol * 0.1);
        if (next >= native_tol) return out;
        native_tol = next;
    }
}
} // namespace bench_stop

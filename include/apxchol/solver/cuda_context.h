#pragma once
/// One-time CUDA device initialization, hoisted OFF the critical path.
///
/// Creating the CUDA primary context is a fixed per-process cost that the
/// runtime pays LAZILY, inside whichever CUDA call happens to come first. In
/// this solver that call used to land in the middle of the timed setup (the GPU
/// SpTRSV's first allocation), so the cost was both ON the critical path and
/// CHARGED to `sptrsv_setup`. It is constant in n, in the thread count and in
/// the factor size: measured ~100-135 ms on an RTX 4090 Laptop and ~715 ms on a
/// GH200 (Grace) -- there, 55% of the whole reported setup.
///
/// `prewarm()` starts a helper thread that forces the context into existence
/// while the host-side elimination (~0.5 s) runs; `ensure_context()` blocks
/// until it exists and returns the seconds the caller spent waiting -- which is
/// the honest cost of device init as seen by the solve, and goes to its own
/// `cuda_init` checkpoint label (see `apx_cholesky::install_factor`).
///
/// WHAT THE OVERLAP IS WORTH. Context creation is CPU work, not a sleep, so it
/// competes with the OpenMP elimination and the net saving is bounded by
/// min(host setup time, context time). Measured on an RTX 4090 Laptop, T=16,
/// `bg+tree[vec_pool]`, first setup of the process, library prewarm only (no
/// harness warmup), medians of 3: iter0040 901 -> 803 ms, grid_2000 1266 ->
/// 1188 ms, but grid_500 154 -> 156 ms -- a WASH, because grid_500's whole host
/// setup (~55 ms) is shorter than the ~90 ms context creation, so the cost just
/// moves into make_graph / find_partition / eliminate as contention instead of
/// disappearing. It is never a loss, and it is largest exactly where setup time
/// matters. On a GH200 (context ~715 ms, host setup ~570 ms) the overlap hides
/// the host-setup share and leaves the rest.
///
/// ORDERING GUARANTEE: every CUDA call the solver makes is preceded, on the
/// same thread, by an `ensure_context()`; the context therefore always exists
/// before the first real call. Correctness does not depend on the prewarm at
/// all -- the primary context is reference-counted and its creation is
/// serialized inside the driver, so a helper thread racing the main thread's
/// first CUDA call is well defined (whichever arrives first creates it, the
/// other blocks on the driver's own lock). The prewarm only moves WHEN the
/// cost is paid; `ensure_context()` alone is enough to make it VISIBLE.
///
/// Without `APXCHOL_USE_CUDA` everything here compiles to empty inline
/// functions, so call sites need no `#if` of their own.

#include <chrono>
#include <mutex>

#if defined(APXCHOL_USE_CUDA)
#include <cuda_runtime.h>
#include <thread>
#endif

namespace apxchol::cuda_ctx {

#if defined(APXCHOL_USE_CUDA)

namespace detail {

struct prewarm_state {
    std::mutex mu;
    std::thread worker;          // joined by ensure_context(), or by ~prewarm_state
    bool   started      = false; // a worker was launched (possibly already joined)
    double ctx_seconds  = 0.0;   // wall time the establishing call itself took

    /// Joining (rather than detaching) bounds the helper's lifetime. This
    /// static is constructed on the first prewarm() call, i.e. after libcudart
    /// is loaded, so its destructor runs BEFORE the CUDA runtime's own
    /// teardown -- the helper can never be left running into it. It also means
    /// a process that starts a prewarm and then never touches the GPU still
    /// terminates cleanly (no joinable std::thread destroyed = no terminate()).
    ~prewarm_state() { if (worker.joinable()) worker.join(); }
};

inline prewarm_state& state() { static prewarm_state s; return s; }

/// The canonical "create the primary context now" call, timed. Errors are
/// swallowed on purpose: when the device is unusable the solver's first real
/// CUDA call reports it through APXCHOL_CUDA_CHECK, with a proper message --
/// this helper must not turn a device problem into a mystery exception thrown
/// from a background thread.
inline double force_context() {
    const auto t0 = std::chrono::steady_clock::now();
    (void)cudaFree(nullptr);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace detail

/// Start CUDA context creation on a helper thread, once per process. Safe from
/// any thread, any number of times, from any number of solver instances: later
/// calls are no-ops. Never blocks.
inline void prewarm() {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.started) return;
    s.started = true;
    try {
        s.worker = std::thread([&s] { s.ctx_seconds = detail::force_context(); });
    } catch (...) {
        // Out of threads: fall back to lazy creation. ensure_context() still
        // establishes the context (and still reports its cost); we just do not
        // get the overlap. Never let a prewarm failure fail a factorization.
        s.started = false;
    }
}

/// Block until the CUDA primary context exists, and return the seconds spent
/// waiting. ~0 when a prewarm() had time to complete; the full context cost
/// when prewarm() was never called (e.g. set_factor(), which has no host work
/// to overlap with). Idempotent; a few microseconds once the context is up.
inline double ensure_context() {
    auto& s = detail::state();
    const auto t0 = std::chrono::steady_clock::now();
    bool had_worker = false;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        had_worker = s.started;
        if (s.worker.joinable()) s.worker.join();   // happens-after the worker's write
    }
    // Retain the primary context on THIS thread too: a no-op costing
    // microseconds when the helper already created it, the full cost when
    // nothing prewarmed it.
    const double own = detail::force_context();
    const double waited =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!had_worker) {
        std::lock_guard<std::mutex> lk(s.mu);
        if (s.ctx_seconds == 0.0) s.ctx_seconds = own;   // nobody prewarmed: we paid it
    }
    return waited;
}

/// Seconds the context-creating call itself took (0 before it has run). The
/// raw platform cost, reportable next to the (much smaller, once the prewarm
/// works) wait time ensure_context() returns.
inline double context_seconds() {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lk(s.mu);
    return s.ctx_seconds;
}

#else   // no CUDA in this build: everything above is a no-op.

inline void   prewarm()         {}
inline double ensure_context()  { return 0.0; }
inline double context_seconds() { return 0.0; }

#endif

} // namespace apxchol::cuda_ctx

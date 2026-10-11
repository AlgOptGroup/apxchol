#pragma once
// Internal boundary between the C++ host side of the Metal PCG
// (src/metal_solver.cpp: factor preparation, vector conversion, exit checks, OpenMP) and
// its Objective-C++ device side (src/metal_device.mm: device, run-time MSL
// compilation, buffers, command buffers). Standard headers only: the .mm is
// compiled as OBJCXX, without Eigen and without the OpenMP flags, and must
// not see either. The structs below mirror the MSL structs in
// src/metal_kernels.inc byte for byte (4-byte fields, double-float as a pair
// of floats); the static_asserts here and in the MSL source pin the layout.
#include "apxchol/solver/detail/gpu_pcg.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace apxchol::detail::metal {

struct df32 {
    float hi;
    float lo;
};
static_assert(sizeof(df32) == 8);

struct params {
    std::uint32_t n;
    std::uint32_t groups;
    std::uint32_t offset;
    std::uint32_t count;
    std::uint32_t mode;
    std::uint32_t iter;
    std::uint32_t window;
    std::uint32_t m;
};
static_assert(sizeof(params) == 32);
static_assert(offsetof(params, mode) == 16);
static_assert(offsetof(params, m) == 28);

using solve_state = gpu_pcg::solve_state<df32, float>;
static_assert(sizeof(solve_state) == 48);
static_assert(offsetof(solve_state, thr) == 16);
static_assert(offsetof(solve_state, alpha) == 32);
static_assert(offsetof(solve_state, active) == 40);

/// The device cannot hold a buffer. A std::bad_alloc, so callers that map
/// out-of-memory report it as one; what() names the buffer.
class device_memory_error : public std::bad_alloc {
public:
    explicit device_memory_error(std::string what) : what_(std::move(what)) {}
    const char* what() const noexcept override { return what_.c_str(); }

private:
    std::string what_;
};

struct tri_arrays {
    const std::uint32_t* ptr = nullptr;  // m + 1, CSR with diagonal
    const std::uint32_t* col = nullptr;
    const float* val = nullptr;
    std::size_t nnz = 0;
};

struct factor_statistics {
    std::size_t levels_forward, levels_backward;
    std::size_t steps_forward, steps_backward;
};

struct operator_arrays {
    const std::uint32_t* ptr = nullptr;  // n + 1
    const std::uint32_t* col = nullptr;  // nnz
    const float* hi = nullptr;           // nnz
    const float* lo = nullptr;           // nnz, nullptr when every value is fp32-exact
    std::size_t n = 0;
    std::size_t nnz = 0;
};

/// The process-wide device: created, compiled and checked once (thread-safe).
struct device_status {
    bool ok = false;
    std::string error;             // why ok is false
    std::string name;
    bool contract_pragma = false;  // "#pragma METAL fp contract(off)" accepted
    std::uint32_t tree_threads = 0;    // min max-threads of the 16-lane tree kernels
    std::uint32_t row_threads = 0;     // min max-threads of the row kernels
};
const device_status& status() noexcept;

/// Runs the df_probe kernel on `inputs` (4 floats per case) and returns 16
/// floats per case. Throws std::runtime_error on a device error.
std::vector<float> run_probe(const std::vector<float>& inputs);

// Owns the operator, factor and one RHS workspace on the device.
class engine {
public:
    engine(const operator_arrays& op, const tri_arrays& fwd, const tri_arrays& bwd,
           std::uint32_t m, bool laplacian);
    ~engine();
    engine(const engine&) = delete;
    engine& operator=(const engine&) = delete;

    // Shared-memory views, touched only between device calls.
    df32* r() noexcept;
    df32* x() noexcept;
    float* p() noexcept;
    solve_state& state() noexcept;

    factor_statistics factor_stats() const;
    void solve(std::uint32_t max_iter,
               std::uint32_t window, std::uint32_t check_every);
    void apply();
private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace apxchol::detail::metal

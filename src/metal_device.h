#pragma once
// Internal boundary between the C++ host side of the Metal block PCG
// (src/metal_solver.cpp: factor preparation, packing, exit checks, OpenMP) and
// its Objective-C++ device side (src/metal_device.mm: device, run-time MSL
// compilation, buffers, command buffers). Standard headers only: the .mm is
// compiled as OBJCXX, without Eigen and without the OpenMP flags, and must
// not see either. The structs below mirror the MSL structs in
// src/metal_kernels.inc byte for byte (4-byte fields, double-float as a pair
// of floats); the static_asserts here and in the MSL source pin the layout.
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
    std::uint32_t kc;
    std::uint32_t groups;
    std::uint32_t offset;
    std::uint32_t count;
    std::uint32_t mode;
    std::uint32_t iter;
    std::uint32_t window;
    std::uint32_t lap;
    std::uint32_t m;
    float inv_n_hi;
    float inv_n_lo;
};
static_assert(sizeof(params) == 48);
static_assert(offsetof(params, mode) == 20);
static_assert(offsetof(params, m) == 36);
static_assert(offsetof(params, inv_n_hi) == 40);

struct column_state {
    df32 rz;
    df32 rr;
    df32 thr;
    df32 prev;
    df32 pap;
    float alpha;
    float beta;
    float mu_r;
    float mu_z;
    std::uint32_t active;
    std::uint32_t iters;
    std::uint32_t stop;
    std::uint32_t pad;
};
static_assert(sizeof(column_state) == 72);
static_assert(offsetof(column_state, thr) == 16);
static_assert(offsetof(column_state, pap) == 32);
static_assert(offsetof(column_state, alpha) == 40);
static_assert(offsetof(column_state, mu_z) == 52);
static_assert(offsetof(column_state, active) == 56);
static_assert(offsetof(column_state, stop) == 64);

// column_state::stop codes written by the device (0 = still running).
inline constexpr std::uint32_t kStopRunning = 0;
inline constexpr std::uint32_t kStopTolerance = 1;
inline constexpr std::uint32_t kStopBreakdown = 2;
inline constexpr std::uint32_t kStopStagnation = 3;
inline constexpr std::uint32_t kStopNonfinite = 4;

/// The device cannot hold a buffer. A std::bad_alloc, so callers that map
/// out-of-memory report it as one; what() names the buffer.
class device_memory_error : public std::bad_alloc {
public:
    explicit device_memory_error(std::string what) : what_(std::move(what)) {}
    const char* what() const noexcept override { return what_.c_str(); }

private:
    std::string what_;
};

// One step of a triangular solve (level_schedule::level_step, flattened):
// kind 0 = light rows [first, last), 1 = heavy rows [first, last),
// 2 = narrow levels [first, last) in one threadgroup.
struct tri_step {
    std::uint32_t kind;
    std::uint32_t first;
    std::uint32_t last;
};

struct tri_arrays {
    const std::uint32_t* level_ptr = nullptr;  // levels + 1
    std::size_t levels = 0;
    const std::uint32_t* rows = nullptr;       // slots
    const std::uint32_t* ptr = nullptr;        // slots + 1
    std::size_t slots = 0;
    const std::uint32_t* col = nullptr;        // deps
    const float* val = nullptr;                // deps
    std::size_t deps = 0;
    const float* dinv = nullptr;               // slots
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
    std::uint64_t max_buffer_bytes = 0;
    std::uint64_t working_set_bytes = 0;
    std::uint32_t tree_threads = 0;    // min max-threads of the 16-lane tree kernels
    std::uint32_t row_threads = 0;     // min max-threads of the row kernels
    std::uint32_t heavy_threads = 0;   // level_heavy (both directions)
    std::uint32_t narrow_threads = 0;  // levels_narrow (both directions)
};
const device_status& status() noexcept;

/// Runs the df_probe kernel on `inputs` (4 floats per case) and returns 16
/// floats per case. Throws std::runtime_error on a device error.
std::vector<float> run_probe(const std::vector<float>& inputs);

/// Device-resident operator, triangular schedules and block work vectors of
/// one metal_solver. Not thread-safe: the owner serializes calls.
class engine {
public:
    engine(const operator_arrays& op, const tri_arrays& fwd, const tri_arrays& bwd,
           std::uint32_t m, bool laplacian);
    ~engine();
    engine(const engine&) = delete;
    engine& operator=(const engine&) = delete;

    /// Ensures the block buffers hold `kc` columns (grows, never shrinks).
    void reserve(std::uint32_t kc);
    std::uint32_t capacity() const noexcept;

    /// Host views of the shared buffers, laid out with the batch's kc as the
    /// stride. Valid until the next reserve(); only touched between calls.
    df32* r() noexcept;
    df32* x() noexcept;
    float* p() noexcept;
    float* z() noexcept;
    column_state* columns() noexcept;

    /// Block PCG on the kc columns loaded into r() / columns(): x is zeroed on
    /// the device, then the initial direction and up to max_iter iterations
    /// run, check_every per command buffer, until no column is active.
    void solve(std::uint32_t kc, const std::vector<tri_step>& fwd_plan,
               const std::vector<tri_step>& bwd_plan, std::uint32_t max_iter,
               std::uint32_t window, std::uint32_t check_every);
    /// One preconditioner application of r() into p() (columns()[c].beta must be 0).
    void apply(std::uint32_t kc, const std::vector<tri_step>& fwd_plan,
               const std::vector<tri_step>& bwd_plan);

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace apxchol::detail::metal

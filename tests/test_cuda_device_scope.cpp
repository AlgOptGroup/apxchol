#include <gtest/gtest.h>

#ifdef APXCHOL_USE_CUDA
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

namespace {
struct DeviceCalls {
    int current = 2;
    int gets = 0;
    cudaError_t get_status = cudaSuccess;
    cudaError_t set_status = cudaSuccess;
    std::vector<int> sets;
} calls;

cudaError_t test_get_device(int* device) {
    ++calls.gets;
    *device = calls.current;
    return calls.get_status;
}
cudaError_t test_set_device(int device) {
    calls.sets.push_back(device);
    if (calls.set_status == cudaSuccess) calls.current = device;
    return calls.set_status;
}
} // namespace

// Exercise the real header with deterministic CUDA failures, even on a
// single-GPU machine. Rename the test class to avoid changing an inline
// production definition across translation units. No production hook is needed.
#define cudaGetDevice test_get_device
#define cudaSetDevice test_set_device
#define cuda_device_scope tested_cuda_device_scope
#include "apxchol/solver/detail/cuda_device_scope.h"
#undef cuda_device_scope
#undef cudaSetDevice
#undef cudaGetDevice

using Scope = apxchol::detail::tested_cuda_device_scope;
static_assert(!std::is_copy_constructible_v<Scope>);
static_assert(!std::is_move_constructible_v<Scope>);
static_assert(std::is_nothrow_constructible_v<Scope, int, bool>);
static_assert(std::is_nothrow_destructible_v<Scope>);
static_assert(!std::is_nothrow_constructible_v<Scope, int, Scope::checked>);

class CudaDeviceScope : public ::testing::Test {
    void SetUp() override { calls = {}; }
};

TEST_F(CudaDeviceScope, DisabledDoesNotTouchCuda) {
    { Scope scope(-1, false); }
    EXPECT_EQ(calls.gets, 0);
    EXPECT_TRUE(calls.sets.empty());
}
TEST_F(CudaDeviceScope, CurrentDeviceDoesNotSwitch) {
    { Scope scope(2); }
    EXPECT_EQ(calls.gets, 1);
    EXPECT_TRUE(calls.sets.empty());
}
TEST_F(CudaDeviceScope, RestoresAfterCleanup) {
    { Scope scope(3); EXPECT_EQ(calls.current, 3); }
    EXPECT_EQ(calls.current, 2);
    EXPECT_EQ(calls.sets, (std::vector<int>{3, 2}));
}
TEST_F(CudaDeviceScope, FailedQueryDoesNotSwitch) {
    calls.get_status = cudaErrorUnknown;
    { Scope scope(3); }
    EXPECT_TRUE(calls.sets.empty());
}
TEST_F(CudaDeviceScope, FailedSwitchDoesNotRestore) {
    calls.set_status = cudaErrorInvalidDevice;
    { Scope scope(3); }
    EXPECT_EQ(calls.sets, (std::vector<int>{3}));
}
TEST_F(CudaDeviceScope, FailedRestorationDoesNotThrow) {
    { Scope scope(3); calls.set_status = cudaErrorUnknown; }
    EXPECT_EQ(calls.sets, (std::vector<int>{3, 2}));
}
TEST_F(CudaDeviceScope, NestedScopesRestoreTheirCallers) {
    {
        Scope outer(3);
        { Scope inner(4); EXPECT_EQ(calls.current, 4); }
        EXPECT_EQ(calls.current, 3);
    }
    EXPECT_EQ(calls.current, 2);
    EXPECT_EQ(calls.sets, (std::vector<int>{3, 4, 3, 2}));
}
TEST_F(CudaDeviceScope, CapturesDeviceBeforeOwnerStateChanges) {
    int owner_device = 3;
    { Scope scope(owner_device); owner_device = -1; }
    EXPECT_EQ(calls.sets, (std::vector<int>{3, 2}));
}
TEST_F(CudaDeviceScope, CheckedQueryFailureThrowsBeforeSwitching) {
    calls.get_status = cudaErrorUnknown;
    EXPECT_THROW((Scope{3, Scope::checked{}}), std::runtime_error);
    EXPECT_EQ(calls.gets, 1);
    EXPECT_TRUE(calls.sets.empty());
}
TEST_F(CudaDeviceScope, CheckedSwitchFailureThrowsWithoutRestoring) {
    calls.set_status = cudaErrorInvalidDevice;
    EXPECT_THROW((Scope{3, Scope::checked{}}), std::runtime_error);
    EXPECT_EQ(calls.current, 2);
    EXPECT_EQ(calls.sets, (std::vector<int>{3}));
}
TEST_F(CudaDeviceScope, CheckedScopeRestoresAfterException) {
    EXPECT_THROW(([] {
        Scope scope(3, Scope::checked{});
        EXPECT_EQ(calls.current, 3);
        throw std::runtime_error("operation failed");
    }()), std::runtime_error);
    EXPECT_EQ(calls.current, 2);
    EXPECT_EQ(calls.sets, (std::vector<int>{3, 2}));
}

namespace {
thread_local int context_device = 0;
struct ContextCalls {
    int gets = 0;
    cudaError_t get_status = cudaSuccess;
    cudaError_t set_status = cudaSuccess;
    std::vector<int> sets;
    std::vector<int> frees;
} context_calls;

cudaError_t test_context_get_device(int* device) {
    ++context_calls.gets;
    *device = context_device;
    return context_calls.get_status;
}
cudaError_t test_context_set_device(int device) {
    context_calls.sets.push_back(device);
    if (context_calls.set_status == cudaSuccess) context_device = device;
    return context_calls.set_status;
}
cudaError_t test_context_free(void*) {
    context_calls.frees.push_back(context_device);
    return cudaSuccess;
}
} // namespace

#define cudaGetDevice test_context_get_device
#define cudaSetDevice test_context_set_device
#define cudaFree test_context_free
#define cuda_ctx tested_cuda_context
#include "apxchol/solver/cuda_context.h"
#undef cuda_ctx
#undef cudaFree
#undef cudaSetDevice
#undef cudaGetDevice

namespace context = apxchol::tested_cuda_context;

class CudaPrewarm : public ::testing::Test {
    void SetUp() override {
        auto& state = context::detail::state();
        state.started = false;
        state.ctx_seconds = 0;
        context_calls = {};
        context_device = 3;
    }
    void TearDown() override {
        auto& worker = context::detail::state().worker;
        if (worker.joinable()) worker.join();
    }
};

TEST_F(CudaPrewarm, UsesCallersDeviceAndJoinsWorker) {
    context::prewarm();
    context::ensure_context();
    EXPECT_EQ(context_calls.gets, 1);
    EXPECT_EQ(context_calls.sets, (std::vector<int>{3}));
    EXPECT_EQ(context_calls.frees, (std::vector<int>{3, 3}));
    EXPECT_EQ(context_device, 3);
    EXPECT_FALSE(context::detail::state().worker.joinable());
}

TEST_F(CudaPrewarm, FailedQueryDoesNotStartWorker) {
    context_calls.get_status = cudaErrorUnknown;
    context::prewarm();
    EXPECT_FALSE(context::detail::state().started);
    EXPECT_FALSE(context::detail::state().worker.joinable());
    EXPECT_TRUE(context_calls.sets.empty());
    EXPECT_TRUE(context_calls.frees.empty());
}

TEST_F(CudaPrewarm, FailedSelectionDoesNotInitializeDefaultDevice) {
    context_calls.set_status = cudaErrorInvalidDevice;
    context::prewarm();
    context::ensure_context();
    EXPECT_EQ(context_calls.sets, (std::vector<int>{3}));
    EXPECT_EQ(context_calls.frees, (std::vector<int>{3}));
    EXPECT_EQ(context_device, 3);
}

TEST_F(CudaPrewarm, RepeatedPrewarmDoesNotRestartWorker) {
    context::prewarm();
    context::ensure_context();
    context::prewarm();
    context::ensure_context();
    EXPECT_EQ(context_calls.gets, 1);
    EXPECT_EQ(context_calls.sets, (std::vector<int>{3}));
    EXPECT_EQ(context_calls.frees, (std::vector<int>{3, 3, 3}));
}
#endif

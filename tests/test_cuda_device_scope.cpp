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
#endif

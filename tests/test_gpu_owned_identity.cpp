// The CUDA-owned setup must install the same factor bit for bit when only
// host synchronization or buffer lifetime changes. Digests were recorded
// with notes/2026-11-13-cuda-round-syncs/golden (apxchol-research-org) after
// the dense block grew to 2048 vertices at a quarter density. A deliberate
// change to the sampler, selection or numerical order must re-record them.
#include <gtest/gtest.h>
#include "gpu_owned_identity_fixture.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"

#if defined(APXCHOL_USE_CUDA) && !defined(APXCHOL_64BIT_EDGE_INDICES)
namespace {

struct identity_case {
    double shift;
    int trace;
    unsigned seed;
    std::uint64_t structure, values;
    std::size_t rounds;
};

constexpr identity_case kCases[] = {
    {0, 0, 1, 0xe6c49607b485e6cfULL, 0xf6aada0b72671f6aULL, 124},
    {0, 0, 2, 0xa279c7e75041e59fULL, 0x6b66662e11d87a3dULL, 120},
    {0, 1, 1, 0x02ea21dc9b2385d2ULL, 0xed15238cb3ce67d2ULL, 167},
    {0, 1, 2, 0x16f701567696767aULL, 0x7f4b3cee4a8b00c7ULL, 197},
    {0.001, 0, 1, 0xd078a336597dde8cULL, 0x1627029dba27ecb4ULL, 132},
    {0.001, 0, 2, 0xbbd023261097474dULL, 0x4adeba3af469d632ULL, 121},
    {0.001, 1, 1, 0x488715570a8d43bfULL, 0xa17520a62bdb4c6bULL, 181},
    {0.001, 1, 2, 0xcc70319dad78c49bULL, 0xaa3941679aafac3fULL, 194},
};

} // namespace
#endif

TEST(GpuOwnedIdentity, FactorDigestsMatchRecordedUpstreamFactors) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_EDGE_INDICES)
    // The recorded digests hash 32-bit column pointers and row indices.
    GTEST_SKIP() << "32-bit-index CUDA build required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    for (const auto& c : kCases) {
        SCOPED_TRACE(testing::Message() << "shift=" << c.shift << " trace=" << c.trace << " seed=" << c.seed);
        const auto d = apxchol_identity::owned_factor_digest(c.shift, c.trace, c.seed);
        EXPECT_EQ(d.rounds, c.rounds);
        EXPECT_EQ(d.structure, c.structure);
        EXPECT_EQ(d.values, c.values);
    }
#endif
}

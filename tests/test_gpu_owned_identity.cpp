// The CUDA-owned setup must install the same factor bit for bit when only
// host synchronization or buffer lifetime changes. Digests were recorded
// from upstream main 6f9918fc (notes: apxchol-research-org
// notes/2026-11-13-cuda-round-syncs/golden). A deliberate change to the
// sampler, selection or numerical order must re-record them.
#include <gtest/gtest.h>
#include "gpu_owned_identity_fixture.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"

namespace {

struct identity_case {
    double shift;
    int trace;
    unsigned seed;
    std::uint64_t structure, values;
    std::size_t rounds;
};

constexpr identity_case kCases[] = {
    {0, 0, 1, 0x89fa5afcb5e73a51ULL, 0x323e0270bd3a0aacULL, 74},
    {0, 0, 2, 0x01e2e8f1c0371f95ULL, 0x435565b14b07b576ULL, 70},
    {0, 1, 1, 0x73a834acd3462b0dULL, 0x84bcb1b5fdbc9c1fULL, 110},
    {0, 1, 2, 0x2da9dc4b7b58e980ULL, 0x22f0bd82767ea128ULL, 124},
    {0.001, 0, 1, 0x90f9fd3e666a6909ULL, 0x37989197c19e7cafULL, 77},
    {0.001, 0, 2, 0x7f1487aa86d0a959ULL, 0x662a4979e7bca611ULL, 71},
    {0.001, 1, 1, 0xa4b715399a744f24ULL, 0x12e01973ce77ab35ULL, 119},
    {0.001, 1, 2, 0x13666cfdbbcf620cULL, 0x85f66c71d1ef0a7fULL, 128},
};

} // namespace

TEST(GpuOwnedIdentity, FactorDigestsMatchRecordedUpstreamFactors) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA build required";
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

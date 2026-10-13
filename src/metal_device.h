#pragma once
// Internal Metal availability and arithmetic-probe seam for device tests.
#include <cstdint>
#include <string>
#include <vector>

namespace apxchol::detail::metal {
struct device_status {
    bool ok = false;
    std::string error;             // why ok is false
    std::string name;
    bool contract_pragma = false;  // "#pragma METAL fp contract(off)" accepted
    std::uint32_t row_threads = 0;     // min max-threads of the row kernels
};
const device_status& status() noexcept;

/// Runs the df_probe kernel on `inputs` (4 floats per case) and returns 16
/// floats per case. Throws std::runtime_error on a device error.
std::vector<float> run_probe(const std::vector<float>& inputs);

} // namespace apxchol::detail::metal

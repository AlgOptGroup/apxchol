#pragma once
#include "apxchol/solver/solve.h"
#include <optional>
#include <string>

namespace apxchol {

struct run_config {
    std::string input_path;
    std::optional<std::string> rhs_path;     // --rhs file
    bool random_rhs = false;                 // --random-rhs (explicit opt-in)
    std::optional<std::string> output_path;  // nullopt → don't write solution

    solve_options solve_opts;
};

/// Parse CLI arguments.  Calls std::exit() on --help or parse error.
/// Also configures spdlog level based on verbosity flags.
run_config parse_args(int argc, char* argv[]);

} // namespace apxchol

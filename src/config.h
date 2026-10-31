#pragma once
#include "apxchol/solver/solve.h"
#include "mtx_input.h"
#include <optional>
#include <string>

namespace apxchol {

struct run_config {
    std::string input_path;
    std::optional<std::string> rhs_path;     // --rhs file
    std::optional<std::string> x0_path;      // --x0 file; nullopt → start from zero
    bool random_rhs = false;                 // --random-rhs (explicit opt-in)
    std::optional<std::string> output_path;  // nullopt → don't write solution
    bool convert_input = false;              // --convert-input
    double tol = default_tol;                // --tol
    int max_iter = default_max_iter;         // --max-iter

    solve_options solve_opts;
};

/// Parse CLI arguments.  Calls std::exit() on --help or parse error.
/// Also configures spdlog level based on verbosity flags.
run_config parse_args(int argc, char* argv[]);

} // namespace apxchol

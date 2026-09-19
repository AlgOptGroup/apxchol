#include "../src/original_stop.h"
#include <iostream>
#include <vector>

static void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
int main() {
    int tests = 0;
    // A native success is a single call, even with abundant remaining budget.
    int calls = 0;
    auto first = bench_stop::run(1e-8, 100,
        [&](double t, int n, bool warm) {
            require(t == 1e-8 && n == 100 && !warm, "first call changed");
            ++calls; return 7;
        }, [] { return 4e-9; });
    require(calls == 1 && first.iterations == 7 && first.passes == 1, "unnecessary retry"); ++tests;
    // Recoverable original-residual plateau: no early-stall inference; requests
    // consume one TOTAL budget and preserve the same adapter object.
    std::vector<int> budgets;
    int pass = 0;
    auto plateau = bench_stop::run(1e-8, 10,
        [&](double, int left, bool warm) {
            require(warm == (pass > 0), "warm-start flag");
            budgets.push_back(left); ++pass; return 3;
        }, [&] { return pass < 3 ? 2e-8 : 2e-10; });
    require(budgets == std::vector<int>({10,7,4}) && plateau.iterations == 9 && plateau.passes == 3,
            "plateau or total budget"); ++tests;
    pass = 0;
    auto capped = bench_stop::run(1e-8, 5,
        [&](double, int left, bool) { ++pass; return std::min(3,left); },
        [] { return 2e-8; });
    require(pass == 2 && capped.iterations == 5 && capped.residual > 1e-8, "cap fabricated convergence"); ++tests;
    auto zero_budget = bench_stop::run(1e-8, 0,
        [](double, int, bool) -> int { throw std::runtime_error("called at zero budget"); },
        [] { return 1.0; });
    require(zero_budget.iterations == 0 && zero_budget.passes == 0 && zero_budget.residual == 1, "zero budget"); ++tests;
    calls = 0;
    auto nan = bench_stop::run(1e-8, 100,
        [&](double, int, bool) { ++calls; return 2; },
        [] { return std::numeric_limits<double>::quiet_NaN(); });
    require(calls == 1 && !std::isfinite(nan.residual), "NaN retried or accepted"); ++tests;
    auto no_progress = bench_stop::run(1e-8, 100,
        [](double, int, bool) { return 0; }, [] { return 1.0; });
    require(no_progress.passes == 8 && no_progress.iterations == 0 && no_progress.residual == 1,
            "unbounded zero-iteration returns"); ++tests;
    bool rejected = false;
    try { bench_stop::run(1e-8, 5, [](double, int, bool) { return 6; }, [] { return 0.; }); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "native budget violation hidden"); ++tests;
    auto zero_rhs = bench_stop::run(1e-8, 100,
        [](double, int, bool) { return 0; }, [] { return 0.; });
    require(zero_rhs.passes == 1 && zero_rhs.iterations == 0 && zero_rhs.residual == 0, "zero RHS"); ++tests;
    // A pinned star: the free equations already satisfy the native tolerance,
    // but their errors add up in the omitted center equation. Grade all rows.
    constexpr int leaves = 99;
    double error = 0;
    auto star = bench_stop::run(1e-8, 20,
        [&](double request, int, bool) {
            error = 0.9 * request / std::sqrt(double(leaves));
            require(std::sqrt(leaves * error * error) <= request, "native star tolerance");
            return 5;
        }, [&] {
            double pin_residual = 0, norm2 = 0;
            for (int i = 0; i < leaves; ++i) {
                pin_residual += error;
                norm2 += error * error;
            }
            return std::sqrt(norm2 + pin_residual * pin_residual) / std::sqrt(2.0);
        });
    require(star.passes == 2 && star.iterations == 10 && star.residual <= 1e-8,
            "pinned norm substituted for original norm"); ++tests;
    std::cout << tests << "/9 original stopping policy cases passed\n";
}

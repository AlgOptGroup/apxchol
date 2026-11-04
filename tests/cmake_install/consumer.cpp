#include <apxchol/solver/solve.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string route = argc == 2 ? argv[1] : "cpu";
    if (argc > 2 || (route != "cpu" && route != "gpu")) return 2;
    apxchol::solve_options options;
    options.backend = route == "cpu" ? apxchol::solve_backend::cpu : apxchol::solve_backend::gpu;
    options.num_threads = 2;
    options.factor_opts.factor_storage = apxchol::factor_storage_type::fp32;
    constexpr int n = 17;
    // A tridiagonal SDDM operator in CSC arrays, rows sorted within each column.
    std::vector<std::int32_t> col_ptr{0}, row_idx;
    std::vector<double> values;
    for (int j = 0; j < n; ++j) {
        for (int i = std::max(j - 1, 0); i <= std::min(j + 1, n - 1); ++i) {
            row_idx.push_back(i);
            values.push_back(i == j ? 3.0 : -1.0);
        }
        col_ptr.push_back(static_cast<std::int32_t>(row_idx.size()));
    }
    const apxchol::csc_view A{n, col_ptr, row_idx, values};
    const auto multiply = [&](const std::vector<double>& x) {
        std::vector<double> y(n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) y[row_idx[p]] += values[p] * x[j];
        return y;
    };
    std::vector<double> exact(n);
    for (int i = 0; i < n; ++i) exact[i] = -1.0 + 2.0 * i / (n - 1);
    const std::vector<double> rhs = multiply(exact);
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
        ("apxchol-sdk-" + std::to_string(stamp));
    if (!std::filesystem::create_directory(directory))
        throw std::runtime_error("could not create SDK test directory");
    const auto path = directory / "solver.apxchol";
    {
        apxchol::solver source(A, options);
        source.save(path);
    }
    auto restored = apxchol::solver::load(path, options);
    const auto result = restored.solve(rhs, 1e-7, 200);
    const auto Ax = multiply(result.x);
    double error = 0.0, norm = 0.0;
    for (int i = 0; i < n; ++i) {
        error += (Ax[i] - rhs[i]) * (Ax[i] - rhs[i]);
        norm += rhs[i] * rhs[i];
    }
    const double residual = std::sqrt(error / norm);
    if (result.backend != options.backend || !result.converged ||
        !std::isfinite(residual) || residual > 1e-7)
        throw std::runtime_error("installed SDK archive solve failed");
    std::cout << "SDK " << route << " archive round trip: residual=" << residual
              << ", file=" << path << '\n';
}

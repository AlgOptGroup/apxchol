#include <apxchol/solver/solve.h>
#include <chrono>
#include <cmath>
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
    options.keep_factor_values = true;
    options.num_threads = 2;
    options.factor_opts.factor_storage = apxchol::factor_storage_type::fp32;
    constexpr int n = 17;
    Eigen::SparseMatrix<double> A(n, n);
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 3.0);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    A.setFromTriplets(entries.begin(), entries.end());
    const Eigen::VectorXd exact = Eigen::VectorXd::LinSpaced(n, -1.0, 1.0);
    const Eigen::VectorXd rhs = A * exact;
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
    const double residual = (A * result.x - rhs).norm() / rhs.norm();
    if (result.backend != options.backend || !result.converged ||
        !std::isfinite(residual) || residual > 1e-7)
        throw std::runtime_error("installed SDK archive solve failed");
    std::cout << "SDK " << route << " archive round trip: residual=" << residual
              << ", file=" << path << '\n';
}

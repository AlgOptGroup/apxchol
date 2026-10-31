#pragma once
#include "apxchol/solver/factorization.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <array>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>
#include <memory>

namespace apxchol {
namespace detail { struct checkpoint; struct solver_access; }

inline constexpr double default_tol      = 1e-8;
inline constexpr int    default_max_iter = 200;

enum class solve_backend { automatic, cpu, gpu };

/// CPU/automatic are always available without querying a device. GPU requires
/// a usable device and compatible compiled kernels; options may still be unsupported.
bool backend_available(solve_backend backend) noexcept;

struct solve_options {
    /// Default relative residual tolerance; must be finite and positive.
    double tol       = default_tol;
    /// Nonnegative default iteration budget; zero only checks the initial guess.
    int    max_iter  = default_max_iter;
    /// Retain FP32 host factor values for export or adoption by another solver.
    /// Costs an additional host copy; false keeps the permutation but releases
    /// row/value arrays. Python factorize() enables retention by default.
    bool keep_factor = false;
    /// Factor construction and triangular-solve storage controls.
    factor_options factor_opts{};
    /// Select a complete setup/solve route; execution errors never change it.
    solve_backend backend = solve_backend::automatic;
    /// CPU thread ceiling for setup and every solve/apply. Positive values
    /// override the calling OpenMP task's setting; 0 inherits it.
    /// Small workloads may use fewer threads.
    int num_threads = 0;
};

namespace detail {
void validate_solve_parameters(double tol, int max_iter);
void validate_solve_options(const solve_options& opts);
/// Omitted values take the owner's defaults; explicit values must be valid.
std::pair<double, int> resolve_solve_parameters(std::optional<double> tol,
    std::optional<int> max_iter, double default_tolerance, int default_iterations);
}

struct solve_result {
    Eigen::VectorXd x;
    Eigen::Index iterations = 0;
    /// Relative residual checked against the original operator and RHS.
    double residual      = 0.0;
    /// Whether the checked residual is below the requested tolerance.
    bool converged = false;
    /// Wall-clock seconds spent in the solve call. Zero in batch columns,
    /// which share the batch total.
    double solve_seconds = 0.0;
    solve_backend backend = solve_backend::cpu;
};

using matrix_view = Eigen::Ref<Eigen::MatrixXd, 0,
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
using const_matrix_view = Eigen::Ref<const Eigen::MatrixXd, 0,
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

struct batch_solve_result {
    // Solutions are written to caller storage; each column's x remains empty.
    std::vector<solve_result> columns;
    /// Wall-clock seconds spent solving all columns.
    double solve_seconds = 0.0;
};

/// Reusable solver owning one CPU or GPU backend, selected once at setup.
/// solve_backend::cpu keeps setup and solves on CPU, also in GPU builds.
/// Owns its input data. solve() and apply() use this object's workspace, so
/// they are non-const. Concurrent non-const calls on one object are not allowed:
/// use a copy per thread, or batch the right-hand sides
/// (see "Thread safety" in docs/usage.md). An overlapping call throws.
/// Copies of a CPU solver share its factor and operator; GPU solvers cannot be copied.
/// Every solve verifies the returned residual against the original operator/RHS.
/// Supplied factors use CPU in automatic mode; an explicit GPU request uploads them.
class solver {
    friend struct detail::solver_access;
public:
    explicit solver(const Eigen::SparseMatrix<double>& A, const solve_options& opts = {});
    /// Adopt F without rebuilding or regularizing it. lump_max_mass still bounds
    /// operator validation; factor storage, dropping and solve options apply.
    solver(const Eigen::SparseMatrix<double>& A, factorization F, const solve_options& opts = {});
    ~solver();
    solver(solver&&) noexcept;
    solver& operator=(solver&&) noexcept;
    solver(const solver& other);
    solver& operator=(const solver& other);

    /// Solve Ax=b. Omitted tol/max_iter use the constructor's solve_options;
    /// explicit values must be valid. x0 supplies an optional initial guess;
    /// nullptr starts from zero. Inspect converged in the result even when
    /// the iteration budget is exhausted.
    solve_result solve(const Eigen::VectorXd& b, std::optional<double> tol = std::nullopt,
                       std::optional<int> max_iter = std::nullopt,
                       const Eigen::VectorXd* x0 = nullptr);
    /// Write into caller storage; the returned result leaves x empty.
    solve_result solve(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                       std::optional<double> tol = std::nullopt,
                       std::optional<int> max_iter = std::nullopt,
                       const Eigen::VectorXd* x0 = nullptr);
    /// Apply the preconditioner to finite scalar or matrix input.
    Eigen::VectorXd apply(const Eigen::VectorXd& r);
    /// Independent RHS columns, with bounded panel workspace. Output must not
    /// overlap the RHS or warm start. tol/max_iter apply to each column.
    template<class Input, class Output>
        requires (Input::ColsAtCompileTime != 1
            && (Input::Flags & Eigen::DirectAccessBit) != 0
            && (Output::Flags & Eigen::DirectAccessBit) != 0)
    batch_solve_result solve(const Eigen::MatrixBase<Input>& B, Eigen::MatrixBase<Output>& X,
                             std::optional<double> tol = std::nullopt,
                             std::optional<int> max_iter = std::nullopt,
                             const const_matrix_view* X0 = nullptr) {
        using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
        const Eigen::Map<const Eigen::MatrixXd, 0, Stride> rhs(B.derived().data(),
            B.rows(), B.cols(), Stride(B.colStride(), B.rowStride()));
        Eigen::Map<Eigen::MatrixXd, 0, Stride> out(X.derived().data(),
            X.rows(), X.cols(), Stride(X.colStride(), X.rowStride()));
        return solve_batch(const_matrix_view(rhs), matrix_view(out), tol, max_iter, X0);
    }
    void apply(const_matrix_view R, matrix_view Z);

    /// Factor metadata; L's row/value arrays require keep_factor=true.
    /// The returned reference is owned by this solver.
    const factorization& factor() const;
    /// Operator dimension.
    Eigen::Index rows() const;
    /// Stored entries in the original operator, after input canonicalization
    /// and before preconditioner lumping or dropping.
    Eigen::Index operator_nnz() const;
    /// Selected route: cpu or gpu, never automatic.
    solve_backend backend() const;
    /// Installed triangular-solve storage: fp16 or fp32, never automatic.
    factor_storage_type factor_storage() const;
    /// Wall-clock seconds spent constructing this solver, or in load()
    /// including reading the file. Copies keep the original's value.
    double setup_seconds() const;

    /// Save the original operator and retained FP32 factor in a portable binary
    /// file. Creates a new file; an existing destination is an error. Requires
    /// keep_factor=true at construction.
    void save(const std::filesystem::path& path) const;
    /// Restore a saved solver without factorizing again. Backend, storage and
    /// solve options apply as with solver(A, F, opts); automatic adoption uses CPU.
    static solver load(const std::filesystem::path& path, const solve_options& opts = {});

private:
    solver(const Eigen::SparseMatrix<double>& A, factorization* F,
           const solve_options& opts, detail::checkpoint* timings);
    batch_solve_result solve_batch(const_matrix_view B, matrix_view X,
                                   std::optional<double> tol, std::optional<int> max_iter,
                                   const const_matrix_view* X0);
    struct impl;
    std::unique_ptr<impl> impl_;
};

/// One-shot solve: factorize L, then solve Lx = b via PCG.
/// Automatic selects GPU when the compiled backend is available and compatible,
/// otherwise CPU. Explicit GPU selects device-owned CUDA or Metal setup and solve,
/// according to the build.
/// Unsupported inputs and execution failures are errors, never a CPU retry.
solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts = {});



} // namespace apxchol

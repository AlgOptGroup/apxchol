#pragma once
#include "apxchol/solver/solve.h"
#include "apxchol/detail/checkpoint.h"
#include "apxchol/solver/sptrsv/omp.h"
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace apxchol::detail {
// CPU route of solver: approximate Cholesky factor, triangular solves and PCG.
// Setup and solves stay on CPU, including in GPU-enabled builds.
// solver validates options and arguments, selects the route and rejects
// overlapping calls; the route trusts its arguments.
class cpu_route {
public:
    // Panel widths solve_columns and apply_columns accept, widest first.
    static constexpr std::array<int, 4> solve_widths{8, 4, 2, 1};
    static constexpr std::array<int, 4> apply_widths{8, 4, 2, 1};

    /// Factorizes A, or installs F when it is not null. Construction options
    /// (including regularization/lumping) are not reapplied to F; factor
    /// storage, dropping, scheduling and solve options still apply. A is the
    /// PCG operator, read through its lower triangle. cp receives setup
    /// timings and is not retained.
    cpu_route(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
              checkpoint* cp, factorization* F);

    /// Copies share the factor, triangular-solve structures and operator, and
    /// get their own workspace, so each thread can solve with its own copy.
    cpu_route(const cpu_route& other);
    cpu_route& operator=(const cpu_route&) = delete;

    /// Solves the columns of B, a panel of one of solve_widths, into X.
    /// timings, when not null, receives the PCG checkpoints.
    void solve_columns(const_matrix_view B, matrix_view X, const const_matrix_view* X0,
                       std::span<solve_result> results, checkpoint* timings,
                       double tol, int max_iter) const;
    /// Z = M^{-1} R for a panel of one of apply_widths.
    void apply_columns(const_matrix_view R, matrix_view Z) const;

    const factorization& factor() const { return installed_->F; }
    factor_storage_type factor_storage() const;
    const omp_sptrsv& trsv() const { return installed_->trsv; }
    /// The original operator in FP64 CSC, rebuilt from the owned SpMV operator.
    Eigen::SparseMatrix<double> original_operator() const;

private:
    // Read-only once installed; copies share it.
    struct installed {
        factorization F;
        omp_sptrsv trsv;
    };

    /// SpTRSV setup from F, then publish it as this route's installed state.
    void install_factor(factorization F, checkpoint* cp);
    /// Builds the owned row-major full-symmetric SpMV operator (fp32 when
    /// lossless). A full symmetric CSC is copied in its already-compatible CSR
    /// array layout; other stored patterns use the general lower-reflection
    /// fallback.
    void build_operator(const Eigen::SparseMatrix<double>& L, checkpoint* cp);

    template<int Columns, class Run>
    void with_packed_panel(const_matrix_view B, matrix_view X, const const_matrix_view* X0,
                           checkpoint* timings, Run run) const;
    template<int Columns>
    void solve_panel(const double* b, double* x, solve_result* results, checkpoint* timings,
                     double tol, int max_iter, const double* x0) const;
    template<int Columns>
    void finish_solution(const_matrix_view B, double* x, solve_result* results,
                         checkpoint* timings, double tol, int max_iter) const;
    template<int Columns>
    void check_solution(const_matrix_view B, const double* x, solve_result* results,
                        checkpoint* timings, double tol) const;

    // Preconditioner application z = M^{-1} b on contiguous RHS lanes.
    // Laplacian path (rank n-1): center → P → L^{-1} → L^{-T} → P^{-1} → center.
    // SDDM path     (rank n):   P → L^{-1} → L^{-T} → P^{-1}.
    // apply_panel always centers Laplacian input; PCG's apply_fused reuses
    // r_sum and corrects drift every ten applications, and gathers r·z.
    template<int Columns>
    void apply_panel(const double* b, double* z) const;
    template<int Columns>
    std::array<double, Columns> apply_fused(const double* r, const std::array<double, Columns>& r_sum,
                                            double* z, checkpoint* cp) const;
    template<int Columns>
    void apply_core(const double* b, const std::array<double, Columns>& b_sum, bool center,
                    double* z, std::array<double, Columns>* rz_out, checkpoint* cp) const;

    // Owned row-major FULL symmetric operator for the parallel SpMV; exactly
    // one matrix is populated (fp32 when every value round-trips float
    // losslessly). It cannot map constructor input: a reusable solver may
    // outlive it. Read-only after setup; copies share it.
    struct pcg_operator {
        Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm;
        Eigen::SparseMatrix<float, Eigen::RowMajor>  Lrm_f;
        bool fp32 = false;
    };

    solve_options opts_;
    std::shared_ptr<const installed> installed_;
    std::shared_ptr<const pcg_operator> op_;
    Eigen::Index n_ = 0;
    // Per-copy workspace, resized when the panel width changes; solver
    // rejects overlapping calls on one instance. scratch2_ receives the
    // transpose-solve output so the final unpermute gathers straight from it.
    mutable Eigen::VectorXd r_, z_, p_, Ap_, panel_x_, scratch_, scratch2_;
    // Per-thread partial sums of the deterministic reductions (one cache line
    // per thread; sized to the max team on use).
    mutable std::vector<double> part_;
    // Weak-barrier state of the triangular solves.
    mutable omp_sptrsv::solve_workspace barrier_;
    // Preconditioner applications in the current PCG solve.
    mutable std::uint64_t apply_count_ = 0;
};

} // namespace apxchol::detail

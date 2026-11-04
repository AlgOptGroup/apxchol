/// apxchol as the preconditioner of Eigen's conjugate gradient. The adapter is
/// an example outside apxchol's stable API; copy it into your code.
#include "apxchol.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <cstdio>
#include <vector>

class apxchol_preconditioner
    : public Eigen::SparseSolverBase<apxchol_preconditioner> {
public:
    using Scalar = double;
    using StorageIndex = int;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

    void attach(apxchol::solver& s) { solver_ = &s; m_isInitialized = true; }

    // The attached solver already holds the factor.
    template<class Matrix> apxchol_preconditioner& analyzePattern(const Matrix&) { return *this; }
    template<class Matrix> apxchol_preconditioner& factorize(const Matrix&) { return *this; }
    template<class Matrix> apxchol_preconditioner& compute(const Matrix&) { return *this; }

    Eigen::Index rows() const { return solver_->rows(); }
    Eigen::Index cols() const { return solver_->rows(); }
    Eigen::ComputationInfo info() const { return Eigen::Success; }

    template<class Rhs, class Dest>
    void _solve_impl(const Rhs& b, Dest& x) const {
        const Eigen::VectorXd r = b;
        const std::vector<double> z =
            solver_->apply({r.data(), static_cast<std::size_t>(r.size())});
        x = Eigen::Map<const Eigen::VectorXd>(z.data(), static_cast<Eigen::Index>(z.size()));
    }

private:
    apxchol::solver* solver_ = nullptr;
};

int main() {
    // An SDDM matrix: the path 0-1-2-3 with unit weights, plus 0.1 on the diagonal.
    std::vector<Eigen::Triplet<double>> entries;
    for (int u = 0; u < 4; ++u) entries.emplace_back(u, u, 0.1 + (u > 0) + (u < 3));
    for (int u = 0; u < 3; ++u) {
        entries.emplace_back(u + 1, u, -1.0);
        entries.emplace_back(u, u + 1, -1.0);
    }
    Eigen::SparseMatrix<double> A(4, 4);
    A.setFromTriplets(entries.begin(), entries.end());  // compressed and sorted

    // The solver borrows A's arrays while it is built, then keeps its own copy.
    const auto n = static_cast<std::size_t>(A.rows()), nnz = static_cast<std::size_t>(A.nonZeros());
    apxchol::solver s(apxchol::csc_view{A.rows(), {A.outerIndexPtr(), n + 1},
                                        {A.innerIndexPtr(), nnz}, {A.valuePtr(), nnz}});

    // Eigen default-constructs the preconditioner, so the adapter holds a
    // pointer set by attach; the solver must outlive cg.
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper,
                             apxchol_preconditioner> cg;
    cg.preconditioner().attach(s);
    cg.setTolerance(1e-10);
    cg.compute(A);
    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(4, 1.0, 2.0);
    const Eigen::VectorXd x = cg.solve(b);
    const double residual = (A * x - b).norm() / b.norm();
    std::printf("Eigen CG with apxchol: %ld iterations, residual %.1e\n",
                static_cast<long>(cg.iterations()), residual);
    return cg.info() == Eigen::Success && residual < 1e-9 ? 0 : 1;
}

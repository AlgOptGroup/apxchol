#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <limits>
#include <random>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include "../include/graphs.h"
#include "../include/simple_solver.h"
#include "../include/solver.h"

class lap_preconditioner {
public:
  using Scalar = double;
  using RealScalar = double;
  using StorageIndex = int;

  lap_preconditioner() = default;
  explicit lap_preconditioner(lap_solver& s) : s_(&s) {}

  template <class MatrixType>
  lap_preconditioner& compute(const MatrixType& A) {
    n_ = A.rows();
    info_ = (s_ != nullptr) ? Eigen::Success : Eigen::NumericalIssue;
    return *this;
  }

  Eigen::Index rows() const { return n_; }
  Eigen::Index cols() const { return n_; }
  Eigen::ComputationInfo info() const { return info_; }

  template <class Rhs>
  Eigen::VectorXd solve(const Rhs& b) const {
    Eigen::VectorXd bb = b;
    bb.array() -= bb.mean();

    std::vector<double> bv((size_t)bb.size());
    for (Eigen::Index i = 0; i < bb.size(); ++i) bv[(size_t)i] = bb[i];

    auto xv = s_->solve(bv);

    Eigen::VectorXd x(bb.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) x[i] = xv[(size_t)i];
    x.array() -= x.mean();
    return x;
  }

private:
  lap_solver* s_ = nullptr;
  Eigen::Index n_ = 0;
  Eigen::ComputationInfo info_ = Eigen::Success;
};

namespace Eigen { namespace internal {
  template <>
  struct traits<lap_preconditioner> : traits<Eigen::SparseMatrix<double>> {};
}}

Eigen::SparseMatrix<double>
laplacian_from_adjacencylist(const std::vector<std::vector<Edge>>& adj)
{
  using Triplet = Eigen::Triplet<double>;
  int n = (int)adj.size();
  std::vector<Triplet> triplets;

  for (int i = 0; i < n; ++i) {
    double degree = 0.0;
    for (const auto& e : adj[(size_t)i]) {
      degree += e.w;
      triplets.emplace_back(i, e.to, -e.w / 2);
      triplets.emplace_back(e.to, i, -e.w / 2);
    }
    triplets.emplace_back(i, i, degree);
  }

  Eigen::SparseMatrix<double> L(n, n);
  L.setFromTriplets(triplets.begin(), triplets.end());
  return L;
}

int main()
{
  int n = 1000;
  auto adj = grid_graph_checkerboard(n, n, 1000.0, 1.0, 4);

  auto t0 = std::chrono::high_resolution_clock::now();
  simple_solver solver(adj);
  auto t1 = std::chrono::high_resolution_clock::now();
  double setup_time = std::chrono::duration<double>(t1 - t0).count();

  std::cout << "setup time: " << setup_time << " s\n";

  Eigen::SparseMatrix<double> L = laplacian_from_adjacencylist(adj);
  int nnz = (int)L.nonZeros();
  std::cout << "nnz = " << nnz << "\n";

  std::mt19937_64 rng(1234567);
  std::normal_distribution<double> N(0.0, 1.0);

  Eigen::VectorXd g(adj.size());
  for (Eigen::Index i = 0; i < g.size(); ++i) g[i] = N(rng);

  Eigen::VectorXd bv = L * g;
  bv.array() -= bv.mean();
  double nrm = bv.norm();
  if (nrm > 0.0) bv /= nrm;

  lap_preconditioner M(solver);

  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper, lap_preconditioner> cg;

  cg.setMaxIterations(200);
  cg.setTolerance(1e-8);
  cg.preconditioner() = M;

  auto pcg_start = std::chrono::high_resolution_clock::now();
  cg.compute(L);
  Eigen::VectorXd xv = cg.solve(bv);
  auto pcg_end = std::chrono::high_resolution_clock::now();

  double pcg_time = std::chrono::duration<double>(pcg_end - pcg_start).count();

  xv.array() -= xv.mean();
  Eigen::VectorXd r = bv - L * xv;
  r.array() -= r.mean();

  double rel_res = r.norm() / (bv.norm() > 0 ? bv.norm() : 1.0);
  int iters = (int)cg.iterations();

  const double target = 1e-8;
  const double t_star  = 1e4 * target;
  const double t_star2 = 1e8 * target;

  std::string status;
  if (rel_res <= target) status = "OK";
  else if (rel_res <= t_star) status = "*";
  else if (rel_res <= t_star2) status = "**";
  else status = "INF";
  std::cout << "nnz factorization = " << solver.num_nonzeros() * 2.0 << "\n";
  std::cout << "fill-in = " << solver.num_nonzeros() * 2.0 / nnz << "\n";
  std::cout << "mu s/nnz = " << (pcg_time + setup_time) / nnz * 1000000 << "\n";
  std::cout << "pcg time = " << pcg_time << " s\n";
  std::cout << "iters = " << iters
            << "  rel_res = " << rel_res
            << "  status = " << status << "\n";
  return 0;
}

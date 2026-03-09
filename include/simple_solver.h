#pragma once
#include "solver.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

class simple_solver final : public lap_solver {
public:
  explicit simple_solver(const std::vector<std::vector<Edge>>& graph);
  std::vector<double> solve(const std::vector<double>& b) override;
  void solve(const Eigen::Ref<const Eigen::VectorXd>& b, Eigen::Ref<Eigen::VectorXd> x);
  int num_nonzeros() const;

private:
  Eigen::SparseMatrix<double> L;
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic,int> P;
  int m_ = 0;
  Eigen::VectorXd y_;
  Eigen::VectorXd z1_;
  Eigen::VectorXd w1_;
  Eigen::VectorXd w_;
};

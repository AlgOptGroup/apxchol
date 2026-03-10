#pragma once
#include <vector>
#include <Eigen/Sparse>
#include "graphs.h"

class lap_solver {
public:
  explicit lap_solver(const std::vector<std::vector<Edge>>& graph) : G(graph) {}
  virtual ~lap_solver() = default;
  virtual std::vector<double> solve(const std::vector<double>& b) = 0;

protected:
  const std::vector<std::vector<Edge>>& G;
};

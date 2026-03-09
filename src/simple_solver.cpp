#include "../include/simple_solver.h"
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <random>
#include <limits>
#include <cmath>
#include <chrono>
#include <unordered_set>
#include <set>

static inline void sort_and_sum(std::vector<Edge>& row){
  std::sort(row.begin(), row.end(),
            [](auto& a, auto& b){ return a.to < b.to; });

  int w = 0;
  for (int i = 0; i < (int)row.size();) {
    int to = row[i].to;
    double sum = 0.0;
    while (i < (int)row.size() && row[i].to == to)
      sum += row[i++].w;
    row[w++] = {to, sum};
  }
  row.resize(w);
}

static std::tuple<std::vector<int>,int>
find_independent_set(std::vector<std::vector<Edge>>& adj,
                     std::vector<bool>& active)
{
  int n = adj.size();
  std::vector<char> chosen(n, 0);

  double avg_degree = 0;
  int num_active = 0;
  for (int i = 0; i < (int)adj.size(); ++i) {
    if (active[i]) {
      avg_degree += (int)adj[i].size();
      num_active++;
    }
  }

  avg_degree /= num_active;

  for (int i = 0; i < n; ++i) {
    if (!active[i] || adj[i].size() > 3 * avg_degree) continue;
    bool ok = true;
    for (auto& e : adj[i])
      if (chosen[e.to]) { ok = false; break; }
    chosen[i] = ok;
  }

  std::vector<int> is;
  for (int i = 0; i < n; ++i)
    if (chosen[i]) { is.push_back(i); active[i] = false; }

  return {is, (int)is.size()};
}

static std::vector<std::pair<int, double>> intersection(const std::vector<std::pair<int, double>>& a, const std::vector<std::pair<int, double>>& b)
{
  std::vector<std::pair<int, double>> result;
  for (auto [x, w1] : a)
    for (auto [y, w2] : b)
      if (x == y) {
        result.push_back({x, w1 * w2});
        break;
      }
  return result;
}


static std::vector<std::vector<std::tuple<int, int, double>>> get_existing_edges(const std::vector<int>& is, std::vector<std::vector<Edge>>& adj, const std::vector<bool>& active) {
  std::vector<std::vector<std::tuple<int, int, double>>> existing_edges(adj.size()); // from, to, future_value
  std::vector<std::vector<std::pair<int, double>>> adjacent_is_vertices(adj.size());
  for (int u : is) {
    for (Edge &e : adj[u]) {
      adjacent_is_vertices[e.to].push_back({u, e.w});
    }
  }

  for (int i = 0; i < adj.size(); i++) {
    if (!active[i]) continue;
    for (int j = 0; j < adj[i].size(); j++) {
      auto e = adj[i][j];
      if (!active[adj[i][j].to] || i > adj[i][j].to) continue;
      std::vector<std::pair<int, double>> closed_triangle_vertices = intersection(adjacent_is_vertices[i], adjacent_is_vertices[e.to]);
      for (auto [k, val] : closed_triangle_vertices) {
        existing_edges[k].push_back({i, adj[i][j].to ,val});
      }
    }
  }

  return existing_edges;
}

static std::unordered_map<int, double> get_weight_lookup_from_edges(std::vector<Edge>& edges) {
  std::unordered_map<int, double> weight_lookup;
  weight_lookup.reserve(edges.size());
  for (std::size_t i = 0; i < edges.size(); ++i) weight_lookup[edges[i].to] = edges[i].w;
  return weight_lookup;
}

static void eliminate_set(std::vector<std::vector<Edge>>& adj,
                          const std::vector<int>& is,
                          const std::vector<bool>& active,
                          std::vector<std::vector<Edge>>& factor)
{
  for (int u : is) {
    sort_and_sum(adj[u]);
  }
  int n = adj.size();
  std::vector<std::vector<Edge>> add(n);

  static std::mt19937 rng{std::random_device{}()};
  std::vector<std::vector<std::tuple<int, int, double>>> existing_edges = get_existing_edges(is, adj, active); // at v in is it holds neighbor pairs with an edge
  size_t total_ex = 0;
  for (auto& e : existing_edges) {
      total_ex += e.size();
  }
  std::cout << "Previously existing edges: " << total_ex << std::endl;

  for (int u : is) {

    sort_and_sum(adj[u]);

    std::vector<Edge> valid;
    double deg = 0.0;

    for (auto& e : adj[u])
      if (active[e.to]) {
        valid.push_back(e);
        deg += e.w;
      }

    std::sort(valid.begin(), valid.end(),
          [](const Edge& a, const Edge& b) {
              return a.w < b.w;
          });

    double sqrt_deg = std::sqrt(deg);
    for (auto& e : valid) e.w /= sqrt_deg;

    factor.push_back(valid);

    for (auto& e : valid) e.w *= sqrt_deg; // This is of course bad, but easiest to think about now

    std::set<std::pair<int,int>> added_edges;

    std::vector<double> prefix(valid.size());
    if (!valid.empty()) {
      prefix[0] = valid[0].w;
      for (int i = 1; i < (int)valid.size(); ++i)
        prefix[i] = prefix[i - 1] + valid[i].w;
    }

    for (int i = 0; i + 1 < (int)valid.size(); ++i) {
      double suffix_sum = prefix.back() - prefix[i];
      if (suffix_sum <= 0.0)
        continue;

      // Should we make sure that we reattach the last edge too?
      std::uniform_real_distribution<double> U(0.0, suffix_sum);
      double r = U(rng);

      auto it = std::upper_bound(
          prefix.begin() + i + 1,
          prefix.end(),
          prefix[i] + r
      );

      int j = it - prefix.begin();

      auto a = valid[i];
      auto b = valid[j];
      double w = a.w * b.w / (a.w + b.w);
      added_edges.insert({a.to, b.to});

      add[a.to].push_back({b.to, w});
      add[b.to].push_back({a.to, w});
    }

  }

  for (int i = 0; i < n; ++i)
    if (!add[i].empty()) {
      adj[i].insert(adj[i].end(), add[i].begin(), add[i].end());
      sort_and_sum(adj[i]);
    }
}

static Eigen::SparseMatrix<double>
build_factor(const std::vector<std::vector<Edge>>& adj)
{
  using T = Eigen::Triplet<double>;
  int n = adj.size();
  std::vector<T> trip;

  for (int i = 0; i < n; ++i) {
    double d = 0;
    for (auto& e : adj[i]) {
      d += e.w;
      trip.emplace_back(e.to, i, -e.w);
    }
    trip.emplace_back(i, i, d);
  }

  Eigen::SparseMatrix<double> L(n,n);
  L.setFromTriplets(trip.begin(), trip.end());

  return L;
}

/* ==================== constructor ==================== */

simple_solver::simple_solver(const std::vector<std::vector<Edge>>& graph)
  : lap_solver(graph)
{
  auto adj = graph;
  int n = adj.size();

  std::vector<bool> active(n, true);
  std::vector<std::vector<Edge>> factor;
  std::vector<int> order;

  int alive = n;
  while (alive > 1) {
    auto [is, sz] = find_independent_set(adj, active);
    order.insert(order.end(), is.begin(), is.end());
    alive -= sz;
    eliminate_set(adj, is, active, factor);
  }

  for (int i = 0; i < n; ++i)
    if (active[i]) order.push_back(i);

  std::vector<int> map(n);
  for (int i = 0; i < n; ++i) map[order[i]] = i;

  std::vector<std::vector<Edge>> reordered(n);
  for (int i = 0; i < (int)factor.size(); ++i)
    for (auto& e : factor[i])
      reordered[map[order[i]]].push_back({map[e.to], e.w});

  L = build_factor(reordered);
  L.makeCompressed();

  P.resize(n);
  P.indices() = Eigen::Map<Eigen::VectorXi>(map.data(), n);

  m_ = n - 1;
  y_.resize(n);
  z1_.resize(m_);
  w1_.resize(m_);
  w_.resize(n);
}

/* ==================== solve ==================== */

void simple_solver::solve(const Eigen::Ref<const Eigen::VectorXd>& b,
                          Eigen::Ref<Eigen::VectorXd> x)
{
  y_.noalias() = P * b;

  const int m = m_;
  auto y1 = y_.head(m);
  auto L11 = L.topLeftCorner(m, m);

  z1_.noalias() = L11.template triangularView<Eigen::Lower>().solve(y1);
  w1_.noalias() = L11.transpose().template triangularView<Eigen::Upper>().solve(z1_);

  w_.head(m) = w1_;
  w_(m) = 0.0;

  x.noalias() = P.transpose() * w_;
  x.array() -= x.mean();
}

std::vector<double>
simple_solver::solve(const std::vector<double>& b)
{
  Eigen::VectorXd rhs =
    Eigen::Map<const Eigen::VectorXd>(b.data(), b.size());
  rhs.array() -= rhs.mean();

  Eigen::VectorXd x(rhs.size());
  solve(rhs, x);
  return {x.data(), x.data() + x.size()};
}

int simple_solver::num_nonzeros() const {
  return L.nonZeros();
}

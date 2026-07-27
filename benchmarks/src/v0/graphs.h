#pragma once
#include <vector>

struct Edge {
    int to;
    double w;
};

std::vector<std::vector<Edge>> grid_graph(int rows, int cols,
                                          double a_left = 1.0,
                                          double a_right = 1e-2,
                                          int jump_col = -1);
// 3D 7-point (face-adjacent) grid Laplacian on an n x n x n cube, uniform weight w.
std::vector<std::vector<Edge>> grid_graph_3d(int n, double w = 1.0);
std::vector<std::vector<Edge>> grid_graph_checkerboard(int rows, int cols,
                                                       double a_high = 1.0,
                                                       double a_low = 1e-2,
                                                       int tile = 1);
std::vector<std::vector<Edge>> erdos_renyi_graph(int n, double p, unsigned seed = 1, double w = 1.0);

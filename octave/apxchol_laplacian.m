function L = apxchol_laplacian(A)
  % APXCHOL_LAPLACIAN  Graph Laplacian L = D - A of a weighted ADJACENCY matrix.
  %
  %   L = apxchol_laplacian(A);        % A: sparse weighted adjacency matrix
  %   res = apxchol_solve(L, b);
  %
  % APXCHOL_SOLVER takes the ASSEMBLED operator, never the adjacency matrix of
  % the graph; this is the explicit one-call conversion. Self-loops contribute
  % nothing, D is the weighted-degree diagonal, and isolated vertices get an
  % all-zero row. Edge weights are taken as abs(A(i,j)), matching the CLI's
  % `--input-kind adjacency` and the benchmark suite's reading of a .mtx file,
  % so a graph stored with negative weights is read as an undirected graph
  % rather than silently made indefinite. Same as apxchol_operator(A, 'adjacency').
  L = apxchol_operator(A, 'adjacency');
end

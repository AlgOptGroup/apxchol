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
  % rather than silently made indefinite.
  %
  % Handing an adjacency matrix straight to APXCHOL_SOLVER raises
  % `apxchol:adjacencyInput` rather than returning a wrong answer quietly.
  if ~issparse(A)
    error('apxchol:badMatrix', 'A must be sparse (use sparse(A))');
  end
  [m, n] = size(A);
  if m ~= n
    error('apxchol:badMatrix', 'A must be square (got %dx%d)', m, n);
  end
  if ~isreal(A)
    error('apxchol:badMatrix', 'A must be real; apxchol solves real symmetric systems');
  end
  W = abs(A);
  W = W - spdiags(full(diag(W)), 0, n, n);   % self-loops cancel in D - A
  L = spdiags(full(sum(W, 2)), 0, n, n) - W;
end

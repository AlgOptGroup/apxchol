function A = apxchol_operator(M, kind)
  % APXCHOL_OPERATOR  The matrix to pass as A for input M of the given kind.
  %
  %   A = apxchol_operator(M);               % kind = 'auto'
  %   A = apxchol_operator(M, 'adjacency');  % or 'operator', 'auto'
  %   res = apxchol_solve(A, b);
  %
  % 'operator' returns M itself. 'adjacency' reads M as a weighted graph and
  % returns its Laplacian L = D - W, with W(i,j) = abs(M(i,j)) and self-loops
  % dropped (see APXCHOL_LAPLACIAN). 'auto' returns the Laplacian only when M
  % fails the solver's operator validation and is an adjacency matrix
  % (symmetric, nonnegative off-diagonals, zero diagonal) or a logical
  % pattern matrix; otherwise it returns M, so a valid operator is never
  % reinterpreted. The CLI's --input-kind and Python's apxchol.operator use
  % the same rule.
  if nargin < 2, kind = 'auto'; end
  if ~issparse(M)
    error('apxchol:badMatrix', 'M must be sparse (use sparse(M))');
  end
  A = apxchol_mex('operator', double(M), kind, islogical(M));
end

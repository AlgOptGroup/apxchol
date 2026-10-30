function A = apxchol_as_operator(M)
  % APXCHOL_AS_OPERATOR  M as an operator for APXCHOL_SOLVER and APXCHOL_SOLVE.
  %
  %   A = apxchol_as_operator(M);   % M: sparse matrix
  %   res = apxchol_solve(A, b);
  %
  % M must be square, finite and symmetric; otherwise this raises an error
  % saying why. If M has off-diagonal entries and none of them is negative,
  % it is a graph adjacency matrix and A is its Laplacian L = D - M, where D
  % holds the row sums of M; a diagonal entry (self-loop) adds to D and is
  % subtracted again, so it cancels. A logical (pattern) matrix has unit
  % weights. Otherwise A = M, and the solver checks it as an operator. The
  % CLI's --convert-input and Python's apxchol.as_operator apply the same rule.
  if ~issparse(M)
    error('apxchol:badMatrix', 'M must be sparse (use sparse(M))');
  end
  A = apxchol_mex('as_operator', double(M));
end

function A = apxchol_as_operator(M)
  % APXCHOL_AS_OPERATOR  M as an operator for APXCHOL_SOLVER and APXCHOL_SOLVE.
  %
  %   A = apxchol_as_operator(M);   % M: sparse matrix
  %   res = apxchol_solve(A, b);
  %
  % A valid input is returned unchanged. If M passes the solver's operator
  % check (a symmetric Laplacian or SDDM matrix, possibly nearly SDDM), A = M.
  % Otherwise, if M has no negative off-diagonal entry, it is a graph
  % adjacency matrix and A is its Laplacian L = D - M, where D holds the row
  % sums of M; a diagonal entry (self-loop) adds to D and is subtracted
  % again, so it cancels. A logical (pattern) matrix has unit weights.
  % Anything else raises an error explaining why M is neither. The CLI's
  % --convert-input and Python's apxchol.as_operator apply the same rule.
  if ~issparse(M)
    error('apxchol:badMatrix', 'M must be sparse (use sparse(M))');
  end
  A = apxchol_mex('as_operator', double(M));
end

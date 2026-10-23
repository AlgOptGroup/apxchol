function res = apxchol_solve(A, b, rtol, maxiter, options)
  % APXCHOL_SOLVE  One-shot convenience: factorize A, then solve A x = b.
  %
  %   res = apxchol_solve(A, b);                % res.x, res.iters, res.residual, res.converged
  %   res = apxchol_solve(A, b, 1e-10, 1000);   % custom tol / maxiter
  %
  % For repeated solves against the same A, build the factor once with
  % APXCHOL_SOLVER instead:  s = apxchol_solver(A); res = s.solve(b);
  if nargin < 3, rtol = []; end
  if nargin < 4, maxiter = []; end
  if nargin < 5, options = struct(); end
  s = apxchol_solver(A, options);
  res = s.solve(b, rtol, maxiter);
end

function res = apxchol_solve(A, b, rtol, maxiter, options, x0)
  % APXCHOL_SOLVE  One-shot convenience: factorize A, then solve A x = b.
  %
  %   res = apxchol_solve(A, b);                % res.x, res.iters, res.residual, res.converged,
  %                                             % res.solve_seconds
  %   res = apxchol_solve(A, b, 1e-10, 1000);   % custom tol / maxiter
  %
  % b may contain several RHS columns; optional x0 supplies matching initial guesses.
  %
  % For repeated solves against the same A, build the factor once with
  % APXCHOL_SOLVER instead:  s = apxchol_solver(A); res = s.solve(b);
  if nargin < 3, rtol = []; end
  if nargin < 4, maxiter = []; end
  if nargin < 5, options = struct(); end
  if nargin < 6, x0 = []; end
  s = apxchol_solver(A, options);
  res = s.solve(b, rtol, maxiter, x0);
end

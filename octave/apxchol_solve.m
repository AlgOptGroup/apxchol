function res = apxchol_solve(A, b, rtol, maxiter, x0, options)
  % APXCHOL_SOLVE  One-shot convenience: factorize A, then solve A x = b.
  %
  %   res = apxchol_solve(A, b);                % res.x, res.iterations, res.residual,
  %                                             % res.converged, res.solve_seconds
  %   res = apxchol_solve(A, b, 1e-10, 1000);   % custom rtol / maxiter
  %   res = apxchol_solve(A, b, [], [], x0, struct('num_threads', 4));
  %
  % rtol, maxiter and x0 mean the same as in s.solve(b, rtol, maxiter, x0);
  % options is the setup struct of APXCHOL_SOLVER. b may contain several RHS
  % columns; optional x0 supplies matching initial guesses.
  %
  % For repeated solves against the same A, build the factor once with
  % APXCHOL_SOLVER instead:  s = apxchol_solver(A); res = s.solve(b);
  if nargin < 3, rtol = []; end
  if nargin < 4, maxiter = []; end
  if nargin < 5, x0 = []; end
  if nargin < 6, options = struct(); end
  s = apxchol_solver(A, options);
  res = s.solve(b, rtol, maxiter, x0);
end

classdef apxchol_solver < handle
  % APXCHOL_SOLVER  Reusable approximate-Cholesky solver: factor once, solve many b.
  %
  %   s   = apxchol_solver(A, options);   % optional scalar options struct
  %   res = s.solve(b);                   % res.x, res.iters, res.residual, res.converged
  %   res = s.solve(b2, 1e-10, 1000);     % per-solve tol / maxiter
  %   z   = s.apply(r);                   % one preconditioner application, z = M\r
  %
  % Laplacian vs SDDM is auto-detected. A Laplacian with k connected components
  % has rank n-k and uses componentwise centering. The constructor builds the
  % factor once; clearing the object frees it automatically.
  %
  % Omitted or [] solve options use the C++ defaults. rtol must be finite
  % and positive; maxiter must be an integer in [0, 2^31-1].
  % Setup options: num_threads, seed, factor_storage, cpu_schedule,
  % drop_tolerance, regularization, lump_max_mass, verbose. See README.md.

  properties (Access = private)
    h = uint64(0)
  end

  methods
    function obj = apxchol_solver(A, options)
      if ~issparse(A)
        error('apxchol:badMatrix', 'A must be sparse (use sparse(A))');
      end
      if nargin < 2, options = struct(); end
      obj.h = apxchol_mex('factorize', A, options);
    end

    function res = solve(obj, b, rtol, maxiter)
      if nargin < 3, rtol = []; end
      if nargin < 4, maxiter = []; end
      [x, iters, residual, converged] = apxchol_mex('solve', obj.h, full(b), rtol, maxiter);
      res = struct('x', x, 'iters', iters, 'residual', residual, 'converged', converged);
    end

    function z = apply(obj, r)
      z = apxchol_mex('apply', obj.h, full(r));
    end

    function delete(obj)
      if obj.h ~= 0
        apxchol_mex('free', obj.h);
        obj.h = uint64(0);
      end
    end
  end
end

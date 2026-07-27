classdef apxchol_solver < handle
  % APXCHOL_SOLVER  Reusable approximate-Cholesky solver: factor once, solve many b.
  %
  %   s   = apxchol_solver(A);            % A: sparse SPD Laplacian or SDDM matrix
  %   res = s.solve(b);                   % res.x, res.iters, res.residual, res.converged
  %   res = s.solve(b2, 1e-10, 1000);     % per-solve tol / maxiter
  %   z   = s.apply(r);                   % one preconditioner application, z = M\r
  %
  % Laplacian vs SDDM is auto-detected (singular Laplacians get a rank-(n-1)
  % factor with null-space centering). The factor is built once in the
  % constructor and freed automatically when the object is cleared.

  properties (Access = private)
    h = uint64(0)
    n = 0
  end

  methods
    function obj = apxchol_solver(A)
      if ~issparse(A)
        error('apxchol:badMatrix', 'A must be sparse (use sparse(A))');
      end
      obj.h = apxchol_mex('factorize', A);
      obj.n = size(A, 1);
    end

    function res = solve(obj, b, tol, maxiter)
      if nargin < 3 || isempty(tol),     tol = 1e-8;   end
      if nargin < 4 || isempty(maxiter), maxiter = 500; end
      [x, iters, residual, converged] = apxchol_mex('solve', obj.h, full(b(:)), tol, maxiter);
      res = struct('x', x, 'iters', iters, 'residual', residual, 'converged', converged);
    end

    function z = apply(obj, r)
      z = apxchol_mex('apply', obj.h, full(r(:)));
    end

    function delete(obj)
      if obj.h ~= 0
        apxchol_mex('free', obj.h);
        obj.h = uint64(0);
      end
    end
  end
end

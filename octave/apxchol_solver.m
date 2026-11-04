classdef apxchol_solver < handle
  % APXCHOL_SOLVER  Reusable approximate-Cholesky solver: factor once, solve many b.
  %
  %   s   = apxchol_solver(A, options);   % optional scalar options struct
  %   res = s.solve(b);                   % res.x, res.iterations, res.residual, res.converged,
  %                                       % res.solve_seconds
  %   res = s.solve(b2, 1e-10, 1000);     % per-solve tol / maxiter
  %   s.save('system.apxchol');            % the installed factor and A
  %   s   = apxchol_solver.load('system.apxchol', options);
  %   z   = s.apply(r);                   % one preconditioner application, z = M\r
  %
  % Laplacian vs SDDM is auto-detected. A Laplacian with k connected components
  % has rank n-k; its RHS must sum to zero within each component. The constructor
  % builds the factor once; clearing the object frees it automatically.
  %
  % b and r may be vectors or n-by-k matrices of independent RHS columns.
  % Matrix solves return 1-by-k status arrays. x0 is an optional initial guess
  % with the same shape as b; omitted or [] starts from zero.
  %
  % s.setup_seconds is the native solver setup time (or load time);
  % compare it with res.solve_seconds to judge whether reusing the factor pays off.
  %
  % Omitted or [] solve options use the C++ defaults. rtol must be finite
  % and positive; maxiter must be an integer in [0, 2^31-1].
  % Setup options: num_threads, seed, partitioner, sampler, factor_storage,
  % degree_quantile, degree_multiplier, drop_tolerance, regularization,
  % lump_max_mass, verbose. See README.md.

  properties (SetAccess = private)
    setup_seconds = 0
  end

  properties (Access = private)
    h = uint64(0)
  end

  methods
    function obj = apxchol_solver(A, options)
      % Empty construction is used internally by load before adopting its handle.
      if nargin == 0, return; end
      if ~issparse(A)
        error('apxchol:badInput', 'A must be sparse (use sparse(A))');
      end
      if nargin < 2, options = struct(); end
      [obj.h, obj.setup_seconds] = apxchol_mex('factorize', A, options);
    end

    function res = solve(obj, b, rtol, maxiter, x0)
      if nargin < 3, rtol = []; end
      if nargin < 4, maxiter = []; end
      if nargin < 5, x0 = []; end
      [x, iterations, residual, converged, seconds] = ...
          apxchol_mex('solve', obj.h, full(b), rtol, maxiter, full(x0));
      res = struct('x', x, 'iterations', iterations, 'residual', residual, 'converged', converged, ...
                   'solve_seconds', seconds);
    end

    function z = apply(obj, r)
      z = apxchol_mex('apply', obj.h, full(r));
    end

    function save(obj, path)
      % Save the original matrix and installed factor to a new portable file.
      apxchol_mex('save', obj.h, path);
    end

    function delete(obj)
      if obj.h ~= 0
        apxchol_mex('free', obj.h);
        obj.h = uint64(0);
      end
    end
  end

  methods (Static)
    function obj = load(path, options)
      % Restore a solver with the same setup options accepted by the constructor.
      if nargin < 2, options = struct(); end
      obj = apxchol_solver();
      [obj.h, obj.setup_seconds] = apxchol_mex('load', path, options);
    end
  end
end

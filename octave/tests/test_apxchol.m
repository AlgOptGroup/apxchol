% Test suite for the apxchol Octave package. Run from the octave/ dir:
%   octave --no-gui --eval "addpath(pwd); run('tests/test_apxchol.m')"
% Exits non-zero on the first failure (assert errors abort the script).

function test_apxchol()
  ok = 0;

  % ── helpers ──
  function L = grid2d_laplacian(m)
    % 2-D m×m grid graph Laplacian (singular, rank n-1).
    n = m * m;
    idx = @(r, c) (r - 1) * m + c;
    I = []; J = [];
    for r = 1:m
      for c = 1:m
        v = idx(r, c);
        if c < m, I(end+1) = v; J(end+1) = idx(r, c+1); end
        if r < m, I(end+1) = v; J(end+1) = idx(r+1, c); end
      end
    end
    A = sparse([I, J], [J, I], 1, n, n);
    L = spdiags(sum(A, 2), 0, n, n) - A;
  end

  function expect_bad_option(invoke, option, label)
    caught = false; id = ''; msg = '';
    try, unused = invoke(); catch e, caught = true; id = e.identifier; msg = e.message; end
    assert(caught, '%s: option must be rejected', label);
    assert(strcmp(id, 'apxchol:badInput'), '%s: wrong identifier %s', label, id);
    assert(~isempty(strfind(msg, option)), '%s: error does not name %s', label, option);
  end

  function assert_same_solve(actual, expected, label)
    assert(actual.iters == expected.iters && actual.converged == expected.converged, ...
           '%s: solve status differs', label);
    assert(norm(actual.x - expected.x) <= 1e-14 && ...
           abs(actual.residual - expected.residual) <= 1e-14, ...
           '%s: solve result differs', label);
  end

  % ── 1. Laplacian solve converges ──
  L = grid2d_laplacian(50);
  rng_b = randn(size(L, 1), 1);  b = rng_b - mean(rng_b);   % consistent RHS
  res = apxchol_solve(L, b, 1e-8, 500);
  assert(res.converged, 'laplacian: not converged');
  assert(res.residual <= 1e-8, 'laplacian: residual %g > tol', res.residual);
  assert(norm(L * res.x - b) / norm(b) <= 1e-6, 'laplacian: true residual too big');
  ok = ok + 1; fprintf('ok %d  laplacian solve converges (it=%d)\n', ok, res.iters);

  % ── 2. SDDM solve converges ──
  n = 400;
  M = sprand(n, n, 0.1);  A = M + M';  A = A - spdiags(diag(A), 0, n, n);
  Ls = spdiags(sum(abs(A), 2) + 1.0, 0, n, n) - A;          % strictly diagonally dominant
  bs = randn(n, 1);
  res = apxchol_solve(Ls, bs);
  assert(res.converged && res.residual <= 1e-8, 'sddm: not converged to tol');
  ok = ok + 1; fprintf('ok %d  sddm solve converges (it=%d)\n', ok, res.iters);

  % ── 3. factor reused across many b ──
  L = grid2d_laplacian(40);
  s = apxchol_solver(L);
  for k = 1:4
    bk = randn(size(L, 1), 1);  bk = bk - mean(bk);
    rk = s.solve(bk);
    assert(rk.converged && rk.residual <= 1e-8, 'reuse: solve %d failed', k);
  end
  ok = ok + 1; fprintf('ok %d  factor reused across 4 rhs\n', ok);

  % ── 4. apply() = one preconditioner application ──
  z = s.apply(randn(size(L, 1), 1));
  assert(numel(z) == size(L, 1) && all(isfinite(z)) && norm(z) > 0, 'apply: bad output');
  ok = ok + 1; fprintf('ok %d  apply() sane\n', ok);

  % ── 5. apply() works as a pcg preconditioner ──
  b5 = randn(size(L, 1), 1);  b5 = b5 - mean(b5);
  [x5, flag5, ~, it5] = pcg(L, b5, 1e-6, 2000, @(r) s.apply(r));
  [~, flag5u, ~, it5u] = pcg(L, b5, 1e-6, it5 + 1);          % unpreconditioned, capped
  assert(flag5 == 0, 'pcg with apxchol M did not converge');
  assert(flag5u ~= 0, 'unpreconditioned pcg converged within preconditioned iters?!');
  ok = ok + 1; fprintf('ok %d  pcg M-integration (it=%d, unprec cap hit)\n', ok, it5);
  clear s;                                                    % exercises the destructor

  % ── 6. error paths ──
  err = false;
  try, apxchol_solver(sparse(ones(3, 4))); catch, err = true; end
  assert(err, 'non-square A must error');
  s6 = apxchol_solver(grid2d_laplacian(10));
  err = false;
  try, s6.solve(ones(size(L, 1) + 7, 1)); catch, err = true; end
  assert(err, 'wrong-length b must error');
  ok = ok + 1; fprintf('ok %d  error paths raise\n', ok);

  % ── 7. an adjacency matrix is REJECTED, not silently mis-solved ──
  % Without the operator contract the solver negates every edge weight, factors
  % to zero fill and reports 0 iterations / residual 1 with no diagnosis. An
  % adjacency matrix carries no positive diagonal anywhere, which is a hard PSD
  % violation, so the positive-diagonal condition catches it.
  n7 = 10;  L7 = grid2d_laplacian(n7);
  A7 = spdiags(diag(L7), 0, n7*n7, n7*n7) - L7;      % the adjacency matrix
  err = false;  msg = '';  id = '';
  try, apxchol_solver(A7); catch e, err = true; id = e.identifier; msg = e.message; end
  assert(err, 'adjacency input must error');
  assert(strcmp(id, 'apxchol:badInput'), 'wrong identifier: %s', id);
  assert(~isempty(strfind(msg, 'Condition FAILED: positive diagonal')), ...
         'message: does not name the failed condition');
  assert(~isempty(strfind(msg, sprintf('Not one of the %d non-empty rows', n7*n7))), ...
         'message: no row count');
  assert(~isempty(strfind(msg, sprintf('%d off-diagonal entries are positive', nnz(A7)))), ...
         'message: no off-diagonal count');
  assert(~isempty(strfind(msg, 'apxchol solves symmetric SDDM / Laplacian operators')), ...
         'message: does not say what is wanted');
  assert(~isempty(strfind(msg, 'ADJACENCY matrix')), 'message: no detection');
  assert(~isempty(strfind(msg, 'apxchol_laplacian(A)')), 'message: no fix');
  ok = ok + 1; fprintf('ok %d  adjacency input rejected with a diagnosis\n', ok);

  % ── 8. positive off-diagonals on a positive diagonal are LUMPED, not refused ──
  % A nearly-SDDM SPD operator (mixed-sign FEM/structural matrices look like
  % this) is repaired when the preconditioner is built and must keep solving.
  n8 = 10;  m8 = n8 * n8;
  L8 = grid2d_laplacian(n8);
  L8 = L8 + 0.5 * speye(m8);          % SDDM
  L8(1, 4) = +0.25;  L8(4, 1) = +0.25;   % ... with positive off-diagonals
  b8 = randn(m8, 1);
  r8 = apxchol_solve(L8, b8);
  assert(r8.converged, 'mixed-sign operator with a positive diagonal must solve');
  % the residual is against the TRUE operator, lumping or not
  assert(norm(b8 - L8 * r8.x) / norm(b8) <= 1e-7, 'residual is not against A');
  % ... while a SINGLE bad diagonal row is still a hard refusal
  A8 = L8;  A8(2, 2) = -1.0;
  err = false;  msg = '';
  try, apxchol_solver(A8); catch e, err = true; msg = e.message; end
  assert(err, 'a non-positive diagonal must error');
  assert(~isempty(strfind(msg, 'first at row 1')), 'message: no witness row');
  ok = ok + 1; fprintf('ok %d  positive off-diagonals lumped, bad diagonal refused\n', ok);

  % ── 9. apxchol_laplacian assembles L = D - A ──
  L9 = apxchol_laplacian(A7);
  assert(norm(L9 - L7, 1) == 0, 'laplacian: does not match D - A');
  b9 = randn(n7*n7, 1);  b9 = b9 - mean(b9);
  r9 = apxchol_solve(L9, b9);
  assert(r9.converged && r9.residual <= 1e-8, 'laplacian: converted solve failed');

  % self-loops contribute nothing; isolated vertices give all-zero rows
  As = A7;  As(1, 1) = 5.0;
  assert(norm(apxchol_laplacian(As) - L7, 1) == 0, 'laplacian: self-loop leaked');
  Ai = sparse(5, 5);  Ai(2, 3) = 1;  Ai(3, 2) = 1;
  Li = apxchol_laplacian(Ai);
  assert(isequal(full(diag(Li))', [0 1 1 0 0]), 'laplacian: isolated vertices wrong');
  assert(max(abs(sum(Li, 2))) == 0, 'laplacian: rows do not sum to zero');

  % |A_ij|, so a negatively-stored edge is read as an undirected edge
  An = sparse([1 2], [2 1], [-2 -2], 2, 2);
  assert(isequal(full(apxchol_laplacian(An)), [2 -2; -2 2]), 'laplacian: sign handling');

  % a weighted symmetric adjacency matrix, and the input is left alone
  Mw = sprand(40, 40, 0.1);  Aw = Mw + Mw';
  Aw = Aw - spdiags(diag(Aw), 0, 40, 40);
  before = Aw;
  Lw = apxchol_laplacian(Aw);
  assert(isequal(Aw, before), 'laplacian: mutated its input');
  assert(max(abs(sum(Lw, 2))) < 1e-12, 'laplacian: weighted rows do not sum to zero');
  err = false;
  try, apxchol_laplacian(sparse(ones(3, 4))); catch, err = true; end
  assert(err, 'laplacian: non-square must error');
  ok = ok + 1; fprintf('ok %d  apxchol_laplacian assembles L = D - A\n', ok);

  % ── 10. malformed options reject before scalar/integer conversion ──
  % 32 common rejections + 9 maxiter rejections + 6 raw-empty rejections
  % + 1 successful solve reusing the same object after all rejected calls.
  Aopts = sparse([2 -1; -1 2]); bopts = [1; -1]; zopts = zeros(2, 1);
  sopts = apxchol_solver(Aopts);
  bad_scalar = {
    'NaN', NaN;
    'positive infinity', Inf;
    'negative infinity', -Inf;
    'complex', 1 + 1i;
    'row', [1 2];
    'column', [1; 2];
    'matrix', ones(2);
    'N-D array', ones(1, 1, 2);
    'character', '1';
    'cell', {1};
    'cell containing empty', {[]};
    'struct', struct('value', 1);
    'function', @sin;
    'sparse row', sparse([1 2]);
    'logical row', [true false];
    'sparse NaN', sparse(NaN)
  };
  rejected = 0;
  for k = 1:size(bad_scalar, 1)
    value = bad_scalar{k, 2}; label = bad_scalar{k, 1};
    expect_bad_option(@() sopts.solve(bopts, value, 20), 'tol', label);
    expect_bad_option(@() sopts.solve(bopts, 1e-8, value), 'maxiter', label);
    rejected = rejected + 2;
  end
  bad_maxiter = {
    'positive fraction', 0.5;
    'negative fraction', -0.5;
    'upper double overflow', double(intmax('int32')) + 1;
    'lower double overflow', double(intmin('int32')) - 1;
    'upper integer overflow', int64(intmax('int32')) + int64(1);
    'lower integer overflow', int64(intmin('int32')) - int64(1);
    'wide unsigned overflow', intmax('uint64');
    'huge finite value', realmax;
    'single rounds above INT_MAX', single(intmax('int32'))
  };
  for k = 1:size(bad_maxiter, 1)
    value = bad_maxiter{k, 2};
    expect_bad_option(@() sopts.solve(bopts, 1e-8, value), 'maxiter', bad_maxiter{k, 1});
    rejected = rejected + 1;
  end
  % Wrappers intentionally replace empties with defaults; test gateway rejection
  % separately so mxGetScalar never receives an empty raw option.
  raw_handle = apxchol_mex('factorize', Aopts);
  raw_cleanup = onCleanup(@() apxchol_mex('free', raw_handle));
  raw_empty = {[], {}, ''};
  for k = 1:numel(raw_empty)
    value = raw_empty{k};
    expect_bad_option(@() apxchol_mex('solve', raw_handle, bopts, value, 20), ...
                      'tol', 'raw empty');
    expect_bad_option(@() apxchol_mex('solve', raw_handle, bopts, 1e-8, value), ...
                      'maxiter', 'raw empty');
    rejected = rejected + 2;
  end
  clear raw_cleanup;
  assert(rejected == 47, 'malformed option case denominator changed');
  reused = sopts.solve(bopts);
  assert(reused.converged && norm(Aopts * reused.x - bopts) / norm(bopts) < 1e-8, ...
         'solver did not remain usable after rejected options');
  ok = ok + 1; fprintf('ok %d  malformed scalar options (48 cases)\n', ok);

  % ── 11. accepted scalar representations and int boundaries ──
  % Zero RHS makes even INT_MAX bounded. Logical and sparse scalar options
  % deliberately retain mxGetScalar compatibility, including unstored zeros.
  valid_options = {
    'double', 1e-8, 10, true;
    'single', single(1e-8), single(10), true;
    'int8', int8(1), int8(1), true;
    'uint8', uint8(1), uint8(1), true;
    'int16', int16(1), int16(1), true;
    'uint16', uint16(1), uint16(1), true;
    'int32', int32(1), int32(1), true;
    'uint32', uint32(1), uint32(1), true;
    'int64', int64(1), int64(1), true;
    'uint64', uint64(1), uint64(1), true;
    'logical true', true, true, true;
    'logical false', false, false, false;
    'sparse double', sparse(1e-8), sparse(10), true;
    'sparse logical true', sparse(true), sparse(true), true;
    'sparse logical false', sparse(false), sparse(false), false;
    'largest positive tolerance', realmax, 0, true;
    'largest negative tolerance', -realmax, 0, true;
    'INT_MAX double', 1e-8, double(intmax('int32')), true;
    'INT_MIN double', 1e-8, double(intmin('int32')), true;
    'INT_MAX integer', 1e-8, intmax('int32'), true;
    'INT_MIN integer', 1e-8, intmin('int32'), true;
    'single below INT_MAX', 1e-8, single(2147483520), true;
    'unsigned INT_MAX', 1e-8, uint32(intmax('int32')), true;
    'negative fractional tolerance', -0.5, -7, true
  };
  assert(size(valid_options, 1) == 24, 'accepted option case denominator changed');
  for k = 1:size(valid_options, 1)
    r = sopts.solve(zopts, valid_options{k, 2}, valid_options{k, 3});
    assert(r.iters == 0 && r.residual == 0 && all(r.x == 0) && ...
           r.converged == valid_options{k, 4}, '%s: wrong zero-RHS result', valid_options{k, 1});
  end
  ok = ok + 1; fprintf('ok %d  accepted scalar options (24 cases)\n', ok);

  % ── 12. wrapper defaults, core sentinels and strict convergence ──
  % Tiny solves establish routing/equivalence; they do not measure whether the
  % iteration cap is 200 or 500. Those distinct caps remain in the source.
  wrapper_default = sopts.solve(bopts, 1e-8, 500);
  default_calls = {
    @() sopts.solve(bopts),
    @() sopts.solve(bopts, 1e-8),
    @() sopts.solve(bopts, [], []),
    @() sopts.solve(bopts, [], 500),
    @() sopts.solve(bopts, 1e-8, [])
  };
  default_cases = 0;
  for k = 1:numel(default_calls)
    invoke = default_calls{k};
    assert_same_solve(invoke(), wrapper_default, 'omitted/empty defaults');
    default_cases = default_cases + 1;
  end
  other_empty = {'', {}, zeros(0, 2), false(0, 2), sparse([], [], [], 0, 2), single([])};
  for k = 1:numel(other_empty)
    value = other_empty{k};
    assert_same_solve(sopts.solve(bopts, value, value), wrapper_default, 'empty type defaults');
    default_cases = default_cases + 1;
  end
  core_default = sopts.solve(bopts, 1e-8, 200);
  negative_options = {-1, -1; -0.5, -7; -realmax, intmin('int32')};
  for k = 1:size(negative_options, 1)
    assert_same_solve(sopts.solve(bopts, negative_options{k, 1}, negative_options{k, 2}), ...
                      core_default, 'negative core defaults');
    default_cases = default_cases + 1;
  end
  strict_options = {zopts, 0, false, 0; zopts, -1, true, 0; ...
                    bopts, 1, false, 1; bopts, 2, true, 1};
  for k = 1:size(strict_options, 1)
    r = sopts.solve(strict_options{k, 1}, strict_options{k, 2}, 0);
    assert(r.iters == 0 && r.residual == strict_options{k, 4} && all(r.x == 0) && ...
           r.converged == strict_options{k, 3}, 'strict tolerance/zero-iteration behavior changed');
    default_cases = default_cases + 1;
  end
  one_shot = apxchol_solve(Aopts, zopts);
  one_shot_empty = apxchol_solve(Aopts, zopts, [], []);
  assert(one_shot.converged && one_shot.iters == 0 && one_shot.residual == 0 && ...
         all(one_shot.x == 0), 'one-shot defaults failed');
  assert_same_solve(one_shot_empty, one_shot, 'one-shot empty defaults');
  default_cases = default_cases + 2;
  assert(default_cases == 20, 'default option case denominator changed');
  ok = ok + 1; fprintf('ok %d  scalar defaults and convergence (20 cases)\n', ok);

  fprintf('ALL %d TESTS PASSED\n', ok);
end

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
    assert(~isempty(strfind(strrep(msg, '_', ''), strrep(option, '_', ''))), ...
           '%s: error does not name %s', label, option);
  end

  function assert_same_solve(actual, expected, label)
    assert(actual.iterations == expected.iterations && actual.converged == expected.converged, ...
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
  ok = ok + 1; fprintf('ok %d  laplacian solve converges (it=%d)\n', ok, res.iterations);

  % ── 2. SDDM solve converges ──
  n = 400;
  M = sprand(n, n, 0.1);  A = M + M';  A = A - spdiags(diag(A), 0, n, n);
  Ls = spdiags(sum(abs(A), 2) + 1.0, 0, n, n) - A;          % strictly diagonally dominant
  bs = randn(n, 1);
  res = apxchol_solve(Ls, bs);
  assert(res.converged && res.residual <= 1e-8, 'sddm: not converged to tol');
  ok = ok + 1; fprintf('ok %d  sddm solve converges (it=%d)\n', ok, res.iterations);

  % ── 3. factor reused across many b ──
  L = grid2d_laplacian(40);
  s = apxchol_solver(L);
  for k = 1:4
    bk = randn(size(L, 1), 1);  bk = bk - mean(bk);
    rk = s.solve(bk);
    assert(rk.converged && rk.residual <= 1e-8, 'reuse: solve %d failed', k);
  end
  ok = ok + 1; fprintf('ok %d  factor reused across 4 rhs\n', ok);

  % ── 3b. setup and solve seconds ──
  setup = s.setup_seconds;
  assert(isscalar(setup) && isfinite(setup) && setup > 0, 'setup_seconds: %g', setup);
  bt = randn(size(L, 1), 1);  bt = bt - mean(bt);
  for rhs = {bt, [bt, -bt]}
    rt = s.solve(rhs{1});
    assert(isscalar(rt.solve_seconds) && isfinite(rt.solve_seconds) && rt.solve_seconds > 0, ...
           'solve_seconds: %g', rt.solve_seconds);
  end
  assert(s.setup_seconds == setup, 'setup_seconds changed across solves');
  r1 = apxchol_solve(L, bt);
  assert(isfinite(r1.solve_seconds) && r1.solve_seconds > 0, 'one-shot solve_seconds');
  caught = false;
  try, s.setup_seconds = 1; catch, caught = true; end
  assert(caught, 'setup_seconds must be read-only');
  ok = ok + 1; fprintf('ok %d  setup_seconds and solve_seconds reported\n', ok);

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
  assert(~isempty(strfind(msg, sprintf('%d of %d non-empty rows', n7*n7, n7*n7))), ...
         'message: no row count');
  assert(~isempty(strfind(msg, 'apxchol solves symmetric SDDM / Laplacian operators')), ...
         'message: does not say what is wanted');
  assert(~isempty(strfind(msg, 'ADJACENCY matrix')), 'message: no detection');
  assert(~isempty(strfind(msg, 'apxchol_as_operator(M)')), 'message: no fix');
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

  % ── 9. apxchol_as_operator: the rule shared with the CLI's --convert-input ──
  % valid operators come back unchanged
  S9 = L7 + 0.1 * speye(n7*n7);
  assert(isequal(apxchol_as_operator(L7), L7), 'as_operator: Laplacian changed');
  assert(isequal(apxchol_as_operator(S9), S9), 'as_operator: SDDM changed');
  % an adjacency matrix becomes L = D - A ...
  L9 = apxchol_as_operator(A7);
  assert(norm(L9 - L7, 1) == 0, 'as_operator: does not match D - A');
  b9 = randn(n7*n7, 1);  b9 = b9 - mean(b9);
  r9 = apxchol_solve(L9, b9);
  assert(r9.converged && r9.residual <= 1e-8, 'as_operator: converted solve failed');
  % ... with self-loops cancelling, also for a logical (pattern) matrix
  As = A7;  As(1, 1) = 5.0;
  Ds = spdiags(full(sum(As, 2)), 0, n7*n7, n7*n7) - As;
  assert(norm(apxchol_as_operator(As) - Ds, 1) == 0, 'as_operator: not D - A with a self-loop');
  assert(norm(apxchol_as_operator(As) - L7, 1) == 0, 'as_operator: self-loop leaked');
  P9 = A7 ~= 0;  P9(1, 1) = true;
  assert(norm(apxchol_as_operator(P9) - L7, 1) == 0, 'as_operator: pattern not converted');
  % isolated vertices give all-zero rows
  Ai = sparse(5, 5);  Ai(2, 3) = 1;  Ai(3, 2) = 1;
  Li = apxchol_as_operator(Ai);
  assert(isequal(full(diag(Li))', [0 1 1 0 0]), 'as_operator: isolated vertices wrong');
  assert(max(abs(sum(Li, 2))) == 0, 'as_operator: rows do not sum to zero');
  % mixed signs are left to the solver; an asymmetric matrix raises, saying why
  Am = sparse([1 2 2 3], [2 1 3 2], [1 1 -1 -1], 3, 3);
  assert(isequal(apxchol_as_operator(Am), Am), 'as_operator: mixed signs must pass unchanged');
  err = false;  msg = '';
  try, apxchol_as_operator(sparse([1 2], [2 3], [1 1], 3, 3)); catch e, err = true; msg = e.message; end
  assert(err, 'as_operator: an asymmetric matrix must error');
  assert(~isempty(strfind(msg, 'transpose partner')), 'as_operator: no reason');
  % a weighted symmetric adjacency matrix, and the input is left alone
  Mw = sprand(40, 40, 0.1);  Aw = Mw + Mw';
  Aw = Aw - spdiags(diag(Aw), 0, 40, 40);
  before = Aw;
  Lw = apxchol_as_operator(Aw);
  assert(isequal(Aw, before), 'as_operator: mutated its input');
  assert(max(abs(sum(Lw, 2))) < 1e-12, 'as_operator: weighted rows do not sum to zero');
  err = false;
  try, apxchol_as_operator(sparse(ones(3, 4))); catch, err = true; end
  assert(err, 'as_operator: non-square must error');
  ok = ok + 1; fprintf('ok %d  apxchol_as_operator keeps operators and converts graphs\n', ok);

  % ── 10. malformed options reject before scalar/integer conversion ──
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
    'negative integer', -1;
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
  for value = [0 -1]
    expect_bad_option(@() sopts.solve(bopts, value, 20), 'tol', 'nonpositive');
    rejected = rejected + 1;
  end
  % Only numeric [] selects defaults; other empty types are invalid.
  typed_empty = {{}, ''};
  for k = 1:numel(typed_empty)
    value = typed_empty{k};
    expect_bad_option(@() sopts.solve(bopts, value, 20), 'tol', 'typed empty');
    expect_bad_option(@() sopts.solve(bopts, 1e-8, value), 'maxiter', 'typed empty');
    rejected = rejected + 2;
  end
  assert(rejected == 48, 'malformed option case denominator changed');
  reused = sopts.solve(bopts);
  assert(reused.converged && norm(Aopts * reused.x - bopts) / norm(bopts) < 1e-8, ...
         'solver did not remain usable after rejected options');
  ok = ok + 1; fprintf('ok %d  malformed scalar options (49 cases)\n', ok);

  % ── 11. accepted scalar representations and integer boundaries ──
  valid_options = {
    1e-8, 10;
    single(1e-8), single(10);
    int8(1), int8(1);
    uint8(1), uint8(1);
    int16(1), int16(1);
    uint16(1), uint16(1);
    int32(1), int32(1);
    uint32(1), uint32(1);
    int64(1), int64(1);
    uint64(1), uint64(1);
    true, true;
    sparse(1e-8), sparse(10);
    sparse(true), sparse(true);
    realmax, 0;
    1e-8, double(intmax('int32'));
    1e-8, intmax('int32');
    1e-8, single(2147483520);
    1e-8, uint32(intmax('int32'))
  };
  for k = 1:size(valid_options, 1)
    r = sopts.solve(zopts, valid_options{k, 1}, valid_options{k, 2});
    assert(r.iterations == 0 && r.residual == 0 && all(r.x == 0) && r.converged);
  end
  ok = ok + 1; fprintf('ok %d  accepted scalar options (%d cases)\n', ok, size(valid_options, 1));

  % ── 12. shared defaults and strict convergence ──
  core_default = sopts.solve(bopts, 1e-8, 200);
  default_calls = {
    @() sopts.solve(bopts),
    @() sopts.solve(bopts, 1e-8),
    @() sopts.solve(bopts, [], []),
    @() sopts.solve(bopts, [], 200),
    @() sopts.solve(bopts, 1e-8, [])
  };
  for k = 1:numel(default_calls)
    invoke = default_calls{k};
    assert_same_solve(invoke(), core_default, 'omitted/empty C++ defaults');
  end
  r = sopts.solve(bopts, 1, 0);
  assert(r.iterations == 0 && r.residual == 1 && ~r.converged && all(r.x == 0));
  r = sopts.solve(bopts, 2, 0);
  assert(r.iterations == 0 && r.residual == 1 && r.converged && all(r.x == 0));
  one_shot = apxchol_solve(Aopts, zopts);
  assert(one_shot.converged && one_shot.iterations == 0 && one_shot.residual == 0);
  assert_same_solve(apxchol_solve(Aopts, zopts, [], []), one_shot, 'one-shot defaults');
  ok = ok + 1; fprintf('ok %d  shared defaults and convergence\n', ok);

  % ── 13. explicit setup controls and input shape ──
  options = struct('num_threads', 1, 'seed', 7, 'partitioner', 'priority_greedy', ...
                   'sampler', 'trace_cycle', 'factor_storage', 'float32', ...
                   'degree_quantile', 0, 'degree_multiplier', 3, 'drop_tolerance', 0, ...
                   'regularization', 0, 'lump_max_mass', 0.25, 'verbose', false);
  for drop = [0 1]
    options.drop_tolerance = drop;
    controlled = apxchol_solver(Aopts, options);
    r = controlled.solve(bopts);
    assert(r.converged && norm(Aopts * r.x - bopts) / norm(bopts) < 1e-8);
    assert_same_solve(apxchol_solve(Aopts, bopts, [], [], [], options), r, 'one-shot setup options');
  end
  bad_options = {struct('num_threads', 1.5), struct('num_threads', -1), ...
                 struct('seed', 1.5), struct('partitioner', 'unknown'), ...
                 struct('sampler', 'unknown'), struct('degree_quantile', 1), ...
                 struct('degree_multiplier', 0), struct('cpu_schedule', 'levels'), ...
                 struct('factor_storage', 'float64'), struct('drop_tolerance', -1), ...
                 struct('drop_tolerance', 1 + eps(1)), struct('drop_tolerance', 2), ...
                 struct('drop_tolerance', NaN), struct('drop_tolerance', Inf), ...
                 struct('drop_tolerance', -Inf), ...
                 struct('regularization', -1), struct('lump_max_mass', -0.1), ...
                 struct('verbose', 2), struct('unknown', 1)};
  for k = 1:numel(bad_options)
    rejected_option = false;
    try, apxchol_solver(Aopts, bad_options{k}); catch, rejected_option = true; end
    assert(rejected_option, 'invalid setup option was accepted');
  end
  shaped = apxchol_solver(speye(4));
  rejected_shape = false;
  try, shaped.solve(ones(2)); catch, rejected_shape = true; end
  assert(rejected_shape, 'matrix RHS was silently flattened');
  ok = ok + 1; fprintf('ok %d  explicit setup controls\n', ok);

  % ── 14. native matrix solves, apply and warm starts ──
  rng(17);
  operators = {grid2d_laplacian(7) + 0.5 * speye(49), ...
               blkdiag(grid2d_laplacian(4), grid2d_laplacian(3))};
  for a = 1:numel(operators)
    A = operators{a}; n = size(A, 1);
    batch_solver = apxchol_solver(A, struct('num_threads', 1));
    for width = [0 1 4 5 16]
      exact = randn(n, width);
      if width > 1, exact(:, 2) = 0; end
      B = A * exact; before = B;
      result = batch_solver.solve(B, 1e-9, 500);
      assert(isequal(size(result.x), [n width]));
      assert(isequal(size(result.iterations), [1 width]));
      assert(isequal(size(result.residual), [1 width]));
      assert(isequal(size(result.converged), [1 width]) && islogical(result.converged));
      assert(all(result.converged));
      errors = sqrt(sum((A * result.x - B).^2, 1));
      assert(all(errors <= 1e-9 * sqrt(sum(B.^2, 1)) + 1e-12));
      Z = batch_solver.apply(B);
      assert(isequal(size(Z), [n width]));
      for j = 1:width
        scalar_z = batch_solver.apply(B(:, j));
        assert(norm(Z(:, j) - scalar_z) <= 1e-11 * max(1, norm(scalar_z)));
      end
      warm = batch_solver.solve(B, 1e-9, 0, exact);
      assert(all(warm.converged) && all(warm.iterations == 0));
      limited = batch_solver.solve(B, 1e-14, 0);
      assert(all(limited.iterations == 0));
      assert(all(limited.converged == all(B == 0, 1)));
      assert(isequal(B, before), 'matrix input was changed');
    end
  end
  ok = ok + 1; fprintf('ok %d  native matrix solves/apply/warm starts (10 shapes)\n', ok);

  % ── 15. scalar systems, row vectors and one-shot warm starts ──
  scalar_solver = apxchol_solver(sparse(2));
  for width = [0 1 4]
    B = 1:width;
    result = scalar_solver.solve(B);
    assert(isequal(size(result.x), [1 width]) && all(result.converged));
    assert(norm(result.x - B / 2) < 1e-14);
    assert(isequal(size(scalar_solver.apply(B)), [1 width]));
  end
  row = sopts.solve(bopts');
  assert_same_solve(row, sopts.solve(bopts), 'legacy row vector');
  assert(isequal(size(sopts.apply(bopts')), [2 1]));
  exact = [1 2 3 4; -2 -3 -4 -5]; B = Aopts * exact;
  warm = apxchol_solve(Aopts, B, 1e-9, 0, exact);
  assert(all(warm.converged) && all(warm.iterations == 0));
  assert(norm(warm.x - exact) < 1e-14);
  row_warm = sopts.solve(B(:, 1)', 1e-9, 0, exact(:, 1)');
  assert(row_warm.converged && row_warm.iterations == 0);
  ok = ok + 1; fprintf('ok %d  scalar/row shapes and one-shot warm starts\n', ok);

  % ── 16. matrix and initial-guess errors leave the owner usable ──
  bad_rhs = {ones(3, 2), ones(2, 2, 2), single(ones(2, 2)), ...
             complex(ones(2, 2), 1), [NaN 1; 1 1], [Inf 1; 1 1]};
  rejected = 0;
  for k = 1:numel(bad_rhs)
    value = bad_rhs{k};
    calls = {@() sopts.solve(value), @() sopts.apply(value), ...
             @() sopts.solve(ones(2, 2), [], [], value)};
    for j = 1:numel(calls)
      caught = false;
      try, invoke = calls{j}; unused = invoke(); catch, caught = true; end
      assert(caught, 'invalid RHS or initial guess was accepted');
      rejected = rejected + 1;
    end
  end
  for guess = {ones(2, 1), ones(2, 3)}
    caught = false;
    try, sopts.solve(ones(2, 2), [], [], guess{1}); catch, caught = true; end
    assert(caught, 'wrong initial-guess width was accepted');
    rejected = rejected + 1;
  end
  assert(rejected == 20);
  reused = sopts.solve(bopts);
  assert(reused.converged && norm(Aopts * reused.x - bopts) / norm(bopts) < 1e-8);
  ok = ok + 1; fprintf('ok %d  matrix/initial-guess validation (20 errors)\n', ok);

  test_archive();
  fprintf('ALL %d TESTS PASSED (plus archive tests)\n', ok);
end

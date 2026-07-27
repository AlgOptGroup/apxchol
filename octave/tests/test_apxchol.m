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

  fprintf('ALL %d TESTS PASSED\n', ok);
end

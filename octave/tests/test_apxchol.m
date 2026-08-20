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

  % ── 7. an adjacency matrix is REJECTED, not silently mis-solved ──
  % Without this check the solver negates every edge weight, factors to zero
  % fill and reports 0 iterations / residual 1 with no diagnosis.
  n7 = 10;  L7 = grid2d_laplacian(n7);
  A7 = spdiags(diag(L7), 0, n7*n7, n7*n7) - L7;      % the adjacency matrix
  err = false;  msg = '';  id = '';
  try, apxchol_solver(A7); catch e, err = true; id = e.identifier; msg = e.message; end
  assert(err, 'adjacency input must error');
  assert(strcmp(id, 'apxchol:adjacencyInput'), 'wrong identifier: %s', id);
  assert(~isempty(strfind(msg, 'adjacency matrix')), 'message: no detection');
  assert(~isempty(strfind(msg, sprintf('none of the %d rows', n7*n7))), ...
         'message: no row count');
  assert(~isempty(strfind(msg, sprintf('%d off-diagonal entries are positive', nnz(A7)))), ...
         'message: no off-diagonal count');
  assert(~isempty(strfind(msg, 'assembled Laplacian/SDDM operator')), ...
         'message: does not say what is wanted');
  assert(~isempty(strfind(msg, 'apxchol_laplacian(A)')), 'message: no fix');
  ok = ok + 1; fprintf('ok %d  adjacency input rejected with a diagnosis\n', ok);

  % ── 8. the check is NARROW: one positive diagonal disables it ──
  % Mixed-sign operators (positive diagonal, some positive off-diagonals) are
  % ambiguous and must keep working exactly as before.
  n8 = 10;  m8 = n8 * n8;
  L8 = grid2d_laplacian(n8);
  L8 = L8 + 0.5 * speye(m8);          % SDDM
  L8(1, 4) = +0.25;  L8(4, 1) = +0.25;   % ... with positive off-diagonals
  b8 = randn(m8, 1);
  r8 = apxchol_solve(L8, b8);
  assert(r8.converged, 'mixed-sign operator with a positive diagonal must solve');
  A8 = A7;  A8(1, 1) = 1.0;           % one diagonal entry => ambiguous => allowed
  s8 = apxchol_solver(A8);  clear s8;
  ok = ok + 1; fprintf('ok %d  check is narrow (positive diagonal never rejected)\n', ok);

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

  fprintf('ALL %d TESTS PASSED\n', ok);
end

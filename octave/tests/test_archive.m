% Portable solver archive tests shared by MATLAB and Octave.
% Run from octave/: addpath(pwd); run('tests/test_archive.m')
function test_archive()
  rejected = 0;
  function expect_error(invoke, id, fragment)
    caught = false;
    try
      invoke();
    catch e
      caught = true;
      assert(strcmp(e.identifier, id), 'wrong archive error identifier: %s', e.identifier);
      assert(~isempty(strfind(e.message, fragment)), 'wrong archive error: %s', e.message);
    end
    assert(caught, 'expected archive error containing: %s', fragment);
    rejected = rejected + 1;
  end

  function bytes = read_bytes(path)
    fid = fopen(path, 'rb');
    assert(fid >= 0);
    close_file = onCleanup(@() fclose(fid));
    bytes = fread(fid, Inf, '*uint8');
  end

  % Temporary files are kept together, including on a failed assertion.
  directory = tempname();
  mkdir(directory);
  fprintf('archive test files: %s\n', directory);
  path = fullfile(directory, 'system.npz');
  laplacian = sparse([2 -1 -1; -1 1 0; -1 0 1]);
  operators = {laplacian + 0.5 * speye(3), blkdiag(laplacian, laplacian)};
  checked = 0;
  for a = 1:numel(operators)
    A = operators{a}; n = size(A, 1);
    exact = reshape(sin(1:n*5), n, 5);
    exact(:, 2) = 0;
    B = A * exact;
    archive = fullfile(directory, sprintf('operator-%d.npz', a));
    source = apxchol_solver(A, struct('num_threads', 1, 'keep_factor', true, ...
                                    'factor_storage', 'float32'));
    reference = source.apply(B);
    source.save(archive);
    clear source;
    for storage = {'float32', 'float16'}
      restored = apxchol_solver.load(archive, ...
          struct('num_threads', 1, 'factor_storage', storage{1}));
      scalar = restored.solve(B(:, 1), 1e-9, 200);
      assert(scalar.converged && norm(A * scalar.x - B(:, 1)) <= 1e-9 * norm(B(:, 1)));
      for width = [0 1 5]
        rhs = B(:, 1:width);
        result = restored.solve(rhs, 1e-9, 200);
        assert(isequal(size(result.x), [n width]));
        assert(isequal(size(result.iters), [1 width]));
        assert(isequal(size(result.residual), [1 width]));
        assert(isequal(size(result.converged), [1 width]) && islogical(result.converged));
        assert(all(result.converged));
        errors = sqrt(sum((A * result.x - rhs).^2, 1));
        assert(all(errors <= 1e-9 * sqrt(sum(rhs.^2, 1)) + 1e-12));
        warm = restored.solve(rhs, 1e-9, 0, exact(:, 1:width));
        assert(all(warm.converged) && all(warm.iters == 0));
        applied = restored.apply(rhs);
        assert(isequal(size(applied), [n width]));
        for j = 1:width
          assert(norm(applied(:, j) - restored.apply(rhs(:, j))) < 1e-12);
        end
        checked = checked + 1;
      end
      if strcmp(storage{1}, 'float32')
        assert(norm(restored.apply(B) - reference, 'fro') <= 1e-14);
      end
      expect_error(@() restored.save(path), 'apxchol:badInput', 'released');
    end
  end
  assert(checked == 12);
  fprintf('ok archive round trips: %d/12 operator/storage/RHS combinations\n', checked);

  % Loading with retention allows another save; the default does not retain.
  original = apxchol_solver(sparse([2 -1; -1 2]), struct('keep_factor', true));
  original.save(path);
  default_loaded = apxchol_solver.load(path);
  result = default_loaded.solve([1; -1]);
  assert(result.converged);
  retained = apxchol_solver.load(path, struct('keep_factor', true));
  retained.save(fullfile(directory, 'resaved.npz'));
  again = apxchol_solver.load(fullfile(directory, 'resaved.npz'));
  assert(norm(again.apply([1; -1]) - retained.apply([1; -1])) < 1e-14);
  before = read_bytes(path);
  expect_error(@() retained.save(path), 'apxchol:error', 'exists');
  assert(isequal(read_bytes(path), before), 'existing archive changed after rejected save');
  no_factor = apxchol_solver(speye(2));
  expect_error(@() no_factor.save(fullfile(directory, 'discarded.npz')), ...
               'apxchol:badInput', 'released');
  assert(exist(fullfile(directory, 'discarded.npz'), 'file') == 0);
  expect_error(@() apxchol_solver.load(fullfile(directory, 'missing.npz')), ...
               'apxchol:error', 'solver archive');
  malformed = fullfile(directory, 'malformed.npz');
  fid = fopen(malformed, 'wb'); fwrite(fid, 'not an archive'); fclose(fid);
  expect_error(@() apxchol_solver.load(malformed), 'apxchol:error', 'solver archive');
  for value = {42, {'path'}, ['ab'; 'cd']}
    expect_error(@() apxchol_solver.load(value{1}), 'apxchol:badInput', 'path');
    expect_error(@() retained.save(value{1}), 'apxchol:badInput', 'path');
  end
  expect_error(@() apxchol_solver.load([path char(0) 'ignored']), ...
               'apxchol:badInput', 'null');
  expect_error(@() apxchol_solver.load(path, struct('keep_factor', 2)), ...
               'apxchol:badInput', 'keep_factor');
  expect_error(@() apxchol_solver.load(path, struct('factor_storage', 'invalid')), ...
               'apxchol:badInput', 'factor storage');
  expect_error(@() apxchol_solver.load(path, struct('unknown', true)), ...
               'apxchol:badInput', 'unknown');
  result = retained.solve([1; -1]);
  assert(result.converged, 'owner unusable after archive errors');
  assert(rejected == 18);
  fprintf('ok archive retention, re-save and file errors: %d/18 rejected\n', rejected);
end

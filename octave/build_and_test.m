% MATLAB MEX build + test harness; Octave users should use ./build.sh.
%   matlab -batch "run('/abs/path/to/octave/build_and_test.m')"
here = fileparts(mfilename('fullpath'));
cd(here);
try
  % Preserve MATLAB's selected compiler pair when building the native core.
  c = mex.getCompilerConfigurations('C', 'Selected');
  cxx = mex.getCompilerConfigurations('C++', 'Selected');
  if isempty(c) || isempty(cxx)
    error('apxchol:compiler', 'Run mex -setup C and mex -setup C++ before building.');
  end
  build = fullfile(here, 'build-matlab');
  configure = sprintf(['cmake -S %s -B %s -DCMAKE_BUILD_TYPE=Release ' ...
                       '-DAPXCHOL_NATIVE_ARCH=OFF -DCMAKE_CXX_FLAGS=-march=x86-64-v2 ' ...
                       '-DMatlab_ROOT_DIR=%s -DCMAKE_C_COMPILER=%s -DCMAKE_CXX_COMPILER=%s'], ...
      shell_quote(here), shell_quote(build), shell_quote(matlabroot), ...
      shell_quote(c.Details.CompilerExecutable), shell_quote(cxx.Details.CompilerExecutable));
  if system(configure) ~= 0, error('apxchol:build', 'CMake configuration failed.'); end
  jobs = getenv('CMAKE_BUILD_PARALLEL_LEVEL');
  if isempty(jobs), jobs = '2'; end
  if system(sprintf('cmake --build %s --target apxchol_mex --parallel %s', ...
                    shell_quote(build), shell_quote(jobs))) ~= 0
    error('apxchol:build', 'MEX build failed.');
  end
  disp('=== MEX BUILT ===');
  addpath(here);
  run('tests/test_apxchol.m');
catch e
  disp('BUILD/TEST ERROR:');
  disp(getReport(e));
  exit(1);
end
exit(0);

function result = shell_quote(value)
  result = ['''' strrep(char(value), '''', '''"''"''') ''''];
end

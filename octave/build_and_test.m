% MATLAB MEX build + test harness (run from anywhere; cd's to octave/).
% Octave users should use ./build.sh instead — this is the MATLAB path.
%
%   matlab -batch "run('/abs/path/to/octave/build_and_test.m')"
%
% Tested on Linux x86-64 with MATLAB R2026b, GCC 13, OpenMP and Eigen.
here = fileparts(mfilename('fullpath'));
cd(here);
try
  mex('-R2018a', 'apxchol_mex.cpp', '../src/factorization.cpp', ...
      '../src/operator_class.cpp', '../src/solve.cpp', ...
      '-I../include', '-I../src', '-I/usr/include/eigen3', ...
      '-DAPXCHOL_POOL_FP32', ...
      'CXXFLAGS=$CXXFLAGS -std=c++23 -march=x86-64-v2 -fopenmp -O3 -fPIC', ...
      'LDFLAGS=$LDFLAGS -fopenmp', '-lgomp');
  disp('=== MEX BUILT ===');
  addpath(here);
  run('tests/test_apxchol.m');
catch e
  disp('BUILD/TEST ERROR:');
  disp(getReport(e));
  exit(1);
end
exit(0);

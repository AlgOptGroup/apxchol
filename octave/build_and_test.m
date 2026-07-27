% MATLAB MEX build + test harness (run from anywhere; cd's to octave/).
% Octave users should use ./build.sh instead — this is the MATLAB path.
%
%   matlab -batch "run('/abs/path/to/octave/build_and_test.m')"
%
% Needs GCC >= 14 (the core uses C++23 "deducing this"). On a modern Linux where MATLAB won't
% launch or the MEX won't load, run inside mathworks/matlab-deps (see README).
here = fileparts(mfilename('fullpath'));
cd(here);
try
  mex('-R2018a', 'apxchol_mex.cpp', '../src/factorization.cpp', '../src/solve.cpp', ...
      '-I../include', '-I/usr/include/eigen3', ...
      '-DAPXCHOL_SPTRSV_FP32', '-DAPXCHOL_POOL_FP32', ...
      'CXXFLAGS=$CXXFLAGS -std=c++23 -fopenmp -O3 -fPIC', ...
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

% MATLAB/Octave sparse matrices: sparse() sorts entries and sums duplicates.
% Laplacian of the path 1-2-3-4 with edge weights 1, 2, 1, from its adjacency.
u = [1 2 3]; v = [2 3 4]; w = [1 2 1];
adjacency = sparse([u v], [v u], [w w], 4, 4);
A = apxchol_as_operator(adjacency);   % forms L = D - adjacency
b = [1; 0; 0; -1];                    % a Laplacian RHS sums to zero
res = apxchol_solve(A, b, 1e-10);
fprintf('MATLAB/Octave: residual %.1e\n', res.residual);
assert(res.converged);

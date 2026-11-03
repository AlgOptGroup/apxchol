"""SciPy sparse input, and plain NumPy CSC arrays."""
import numpy as np
import scipy.sparse as sp

import apxchol

# Laplacian of the path 0-1-2-3 with edge weights 1, 2, 1, from its adjacency.
u, v, w = np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([1.0, 2.0, 1.0])
adjacency = sp.coo_array((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(4, 4))
A = apxchol.as_operator(adjacency)  # forms L = D - adjacency
b = np.array([1.0, 0.0, 0.0, -1.0])  # a Laplacian RHS sums to zero
result = apxchol.solve(A, b, rtol=1e-10)
print(f"SciPy: residual {result.residual:.1e}")

# Without SciPy: zero-based CSC arrays with 32- or 64-bit indices.
csc = sp.csc_array(A)
solver = apxchol.Solver.from_csc(csc.data, csc.indices.astype(np.int64),
                                 csc.indptr.astype(np.int64))
assert result.converged and solver.solve(b, rtol=1e-10).converged

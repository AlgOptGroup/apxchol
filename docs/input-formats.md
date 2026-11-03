# Input formats

apxchol takes a square sparse matrix in compressed sparse column (CSC) form.
Python and MATLAB/Octave accept their own sparse matrices and convert them.
C++ borrows the caller's CSC arrays as they are, so they must already meet
the requirements below. Each example on this page solves the Laplacian of a
4-vertex path and prints the residual; CI builds and runs all of them.

## C++ requirements

A C++ caller passes an `apxchol::csc_view` (`std::int32_t` indices) or
`apxchol::csc_view64` (`std::int64_t` indices) from `apxchol/views.h`:

- `n`, then `col_ptr` with `n + 1` offsets starting at 0, then `row_idx` and
  `values` with `col_ptr[n]` entries each.
- Both triangles of the symmetric matrix. Sources that store one triangle,
  such as CHOLMOD with `stype != 0` or a `symmetric` Matrix Market file, must
  be expanded first.
- Rows sorted and unique within each column. C++ does not sort or sum
  entries; an unsorted or repeated row throws `std::invalid_argument` naming
  the column.
- Because the matrix is symmetric, its CSR arrays (row offsets, column
  indices, values) are also its CSC arrays and can be passed unchanged.

Index arrays as wide as the build's indices are read in place: a `csc_view`
in the default 32-bit build, a `csc_view64` with `APXCHOL_INDEX_WIDTH=64`.
Other widths are converted once during setup. The solver copies what it needs,
so the arrays only have to outlive the call that receives the view.

## Your own CSR or CSC arrays

Arrays from your own code, or exported from another library, are passed
directly. PETSc's sequential AIJ matrices give their CSR arrays through
`MatGetRowIJ` and `MatSeqAIJGetArrayRead`, with `PetscInt` indices of 32 or 64
bits depending on how PETSc was configured. A distributed matrix must first be
gathered on one process. If the offsets and indices have different widths,
convert one of them to the other's width.

```{literalinclude} ../examples/input_formats/raw_csc.cpp
:language: cpp
```

## Eigen

`Eigen::SparseMatrix<double>` is column-major with `int` indices, so it is
passed as a `csc_view`; `SparseMatrix<double, Eigen::ColMajor, std::int64_t>`
is passed as a `csc_view64`. A row-major matrix is passed the same way, as
CSR. Call `makeCompressed()` first: an uncompressed matrix leaves gaps between
columns. `setFromTriplets` produces sorted columns with duplicates summed, and
`insert` keeps each column sorted but must not repeat an entry. Eigen is needed
only by this example; apxchol does not depend on it.

```{literalinclude} ../examples/input_formats/eigen.cpp
:language: cpp
```

To use apxchol as the preconditioner of Eigen's conjugate gradient, see
[using another iterative solver](usage.md#using-another-iterative-solver).

## SuiteSparse / CHOLMOD

A `cholmod_sparse` with `itype` `CHOLMOD_INT` is passed as a `csc_view`, and one
with `CHOLMOD_LONG` (the `cholmod_l_*` functions) as a `csc_view64`. It must be
real double precision, packed (`A->packed`), sorted (`A->sorted`, or call
`cholmod_sort`) and store both triangles (`stype == 0`). CHOLMOD usually stores
symmetric matrices as one triangle; `cholmod_copy(A, 0, 1, &c)` expands them.
`cholmod_triplet_to_sparse` sums duplicate triplets.

```{literalinclude} ../examples/input_formats/cholmod.cpp
:language: cpp
```

## SciPy

Pass any `scipy.sparse` matrix or array. apxchol converts a copy to CSC, sorts
its indices and sums duplicates, so `A.sort_indices()` and
`A.sum_duplicates()` are not needed and the input is never modified. Supply
both triangles; format conversion does not add missing symmetric entries.
`scipy.io.mmread(path)` reads a Matrix Market file, including `symmetric`
ones, into a matrix that can be passed directly. Without SciPy,
`Solver.from_csc` takes the three CSC arrays with 32- or 64-bit indices and
also sorts and sums them.

```{literalinclude} ../examples/input_formats/scipy_input.py
:language: python
```

## MATLAB / Octave

MATLAB and Octave sparse matrices are already sorted CSC, and `sparse(i, j, v)`
sums duplicates, so pass them directly with both triangles.

```{literalinclude} ../examples/input_formats/octave_input.m
:language: matlab
```

## Matrix Market

The [command-line solver](cli.md) reads Matrix Market files itself. In C++,
read the file into triplets, for example with
[fast_matrix_market](https://github.com/alugowski/fast_matrix_market), and
assemble them into sorted columns with duplicates summed. Its reader expands
`symmetric` files to both triangles and adds an explicit zero for each
diagonal entry, which the summing merges.

```{literalinclude} ../examples/input_formats/matrix_market.cpp
:language: cpp
```

## Building the examples

The C++ examples are a separate CMake project, so that Eigen, CHOLMOD and
fast_matrix_market are needed only there. On Ubuntu:

```bash
sudo apt-get install libeigen3-dev libsuitesparse-dev
cmake -S examples/input_formats -B build-input-formats -DCMAKE_BUILD_TYPE=Release
cmake --build build-input-formats
ctest --test-dir build-input-formats
```

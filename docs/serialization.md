# Saving a solver

Save the matrix and the installed factor, the preconditioner the solves actually use, then load them on a CPU or GPU without repeating factorization:

```cpp
#include <apxchol/solver/solve.h>

apxchol::solver prepared(A);
prepared.save("system.apxchol");

auto restored = apxchol::solver::load("system.apxchol");
auto result = restored.solve(b);
```

MATLAB and Octave use the same archive API through their reusable solver:

```matlab
s = apxchol_solver(A);
s.save('system.apxchol');
restored = apxchol_solver.load('system.apxchol', struct('num_threads', 4));
result = restored.solve(b);
```

Saving writes `<path>.tmp`, flushes it to disk, renames it onto `<path>` and syncs the directory, so an existing file is replaced, readers never see a partial archive and the new file survives a power loss. If `<path>.tmp` already exists, another save to the same path is running or an earlier save was interrupted; saving fails and names that file. Remove it only when no save to that path is running. The `.apxchol` extension is a convention; loading checks the file contents.

The original operator is reconstructed from the solver’s owned data when saving, without keeping another permanent matrix copy. A GPU solver downloads the operator and its installed factor to the host. The file preserves the coefficients used by the solver; duplicate entries and tolerated differences between the input’s two triangles have already been normalized.

## What loading reproduces

The file stores the factor after dropping and storage rounding, with its storage precision. Loading reinstalls it exactly: no dropping, rescaling or rounding is applied again. These load options apply:

- `backend`: automatic adoption selects CPU; `apxchol::solve_backend::gpu` uploads to the compiled GPU backend.
- `num_threads` and `cpu_schedule`.
- `lump_max_mass`: an operator admitted with a larger value still needs it when loading.

`factor_storage` and `drop_tolerance` have no effect: the file fixes both. Construction options such as the seed, partitioner, regularization and lumping are already represented in the saved factor.

A loaded solver is as precise and as fast as the one that saved it: it solves in the same number of iterations and in the same time. On CPU and CUDA it gives bit-identical solves at the same thread count. On Metal, setup orders each forward-solve row by thread timing, so a reloaded solver agrees with the original to FP32 rounding: preconditioner applications agree to 3e-7 relative on large grids, ecology2, G3_circuit and com-Youtube, and solutions to 5e-10. CPU, CUDA and Metal use one stored form, so a file saved by one backend loads on another; their kernels round differently, so solutions agree to the solve tolerance rather than bit for bit.

## Stored preconditioner

The file stores the `factorization` that `solver::factor()` returns. With the permutation matrix P, the lower-triangular factor L and S = diag(s),

    M = Pᵀ L S² Lᵀ P,

where S = I for FP32 storage. For a Laplacian, L's last row and column are empty; the solve applies M's pseudo-inverse on the leading n−1 rows and columns and centers its input and output.

- FP32 storage: L holds the FP32 values the triangular solves read, diagonal included.
- FP16 storage: each off-diagonal of L is an IEEE 754 binary16 value, each diagonal d_j = L_jj is FP32, and each column has a positive FP32 scale s_j. CPU and CUDA apply S⁻² as fp32(1/s_j)²; Metal forms fp32(L_ij s_j) and fp32(d_j s_j).

## File format

The uncompressed binary file stores eleven typed arrays:

| Array | Content |
|---|---|
| Metadata | Schema version, dimension, SDDM flag, operator and factor entry counts, lumped entry count, original stored entry count, factor storage |
| Operator offsets, rows and values | Original matrix in CSC, with FP64 values |
| Factor offsets, rows and values | L in CSC, each column's diagonal first; FP32 values, or binary16 values for FP16 storage |
| Permutation | Original vertex to elimination position |
| Round sizes | Elimination-round sizes; may be empty |
| Factor diagonal | FP16 storage: d_j per column, FP32; empty for FP32 storage |
| Factor scales | FP16 storage: s_j per column, FP32; empty for FP32 storage |

The header identifies format version 1. Array descriptors record the element type, width, byte order, count, file offset and CRC32 checksum. Header fields use little-endian integers; array values use their recorded byte order. The header and descriptors have a separate checksum, and each array starts at a 64-byte-aligned file offset.

Loading checks file lengths, dimensions, index ranges, conversion to the current build’s index widths and the factor's structure. The format stores the original operator with signed 32-bit indices; wider builds do not remove that limit. CRC checks detect accidental corruption but do not authenticate a file.

Only canonical CSC is stored. Loading copies the arrays and rebuilds triangular-solve layouts and backend schedules; it does not memory-map the file. Level sets, transposed copies, GPU buffers, an active iterative solve and construction diagnostics are omitted.

The original stored entry count is kept separately for fill-ratio reporting. It can differ from the saved matrix’s count when normalization reflects or removes unmatched explicit zeros.

The file format version is independent of the package version. Saving always writes the newest format version. Any 1.x release reads every format version written by earlier 1.x releases. A minor 1.x release may introduce a new format version, as long as it keeps reading the older ones. An older apxchol given a newer archive fails with a message that names the format version and asks for an upgrade; that message is not a promise to reject every future format. Each format version has one committed fixture, such as `tests/fixtures/path5_sddm_v1.apxchol` for version 1, which the unit tests load and solve in CI.

<details>
<summary>Binary layout</summary>

All offsets below are in bytes. Header and descriptor integers are unsigned little-endian. The 64-byte header is followed by eleven 40-byte descriptors.

| Header offset | Field |
|---|---|
| 0 | Eight ASCII bytes: `APXCHB01` |
| 8, 12, 16 | Three `uint32`: format version `1`, array count `11`, descriptor size `40` |
| 20 | `uint32`: header CRC32 |
| 24 | `uint64`: exact file size |
| 32 | 32 reserved bytes, all zero |

| Descriptor offset | Field |
|---|---|
| 0 | `uint32`: array ID |
| 4 | ASCII byte-order marker: `<` for little-endian or `>` for big-endian |
| 5 | ASCII element kind: `i`, `u` or `f` |
| 6 | Numeric byte value `2`, `4` or `8`: element width in bytes; `2` only for `f` |
| 7 | One reserved zero byte |
| 8, 16, 24 | Three `uint64`: element count, absolute payload offset, payload byte length |
| 32 | `uint32`: payload CRC32 |
| 36 | Four reserved zero bytes |

Array IDs `0` through `10` identify metadata, operator offsets, operator rows, operator values, factor offsets, factor rows, factor values, permutation, round sizes, factor diagonal and factor scales, respectively. Each ID occurs once; descriptors follow payload order. The eight metadata values are `[1, n, sddm, operator_nnz, factor_nnz, lumped_nnz, original_nnz, storage]`, with `sddm` equal to `0` or `1` and `storage` equal to `0` (FP32) or `1` (FP16).

With FP16 storage, factor values are binary16 and the diagonal slots hold zero; readers take each diagonal from the factor diagonal array. The diagonal and scale arrays have `n` entries; a Laplacian's empty last column has diagonal `0` and scale `1`. With FP32 storage, both arrays are empty.

Payloads contain contiguous numeric values in their declared byte order: `i` means two’s-complement signed integers, `u` unsigned integers, and `f` IEEE 754 floating point. Operator values are FP64 and factor values FP32 or binary16. The writer uses unsigned 64-bit metadata and round sizes, signed 32-bit operator indices, and the build's factor-index widths. The reader accepts 32- or 64-bit integers when every value fits its destination.

CRC32 uses zlib's CRC32 convention with initial value zero. The header checksum covers all 504 header/descriptor bytes with bytes 20–23 set to zero; each payload checksum covers exactly its declared bytes. Empty payloads have checksum zero.

The first payload starts at byte `512`. Each later payload starts at the first multiple of 64 at or after the preceding payload’s end. Payload length must equal count × width. Writers zero-fill alignment padding; readers ignore its contents and checksums exclude it. There are no trailing bytes after the final payload.

</details>

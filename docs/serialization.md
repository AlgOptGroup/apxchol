# Saving a solver

Save the matrix and retained factor together, then load them on a CPU or GPU without repeating factorization:

```cpp
#include <apxchol/solver/solve.h>

apxchol::solve_options options;
options.keep_factor_values = true;
apxchol::solver prepared(A, options);
prepared.save("system.apxchol");

auto restored = apxchol::solver::load("system.apxchol", options);
auto result = restored.solve(b);
```

MATLAB and Octave use the same archive API through their reusable solver:

```matlab
s = apxchol_solver(A, struct('keep_factor', true));
s.save('system.apxchol');
restored = apxchol_solver.load('system.apxchol', struct('factor_storage', 'float16'));
result = restored.solve(b);
```

The MATLAB/Octave loader accepts the normal constructor options and uses CPU,
as with ordinary construction. Set `keep_factor=true` when loading to save the
restored solver again.

Saving requires retained factor values. It writes `<path>.tmp`, flushes it to disk, renames it onto `<path>` and syncs the directory, so an existing file is replaced, readers never see a partial archive and the new file survives a power loss. If `<path>.tmp` already exists, another save to the same path is running or an earlier save was interrupted; saving fails and names that file. Remove it only when no save to that path is running. The `.apxchol` extension is a convention; loading checks the file contents.

The original operator is reconstructed from the solver’s owned data when saving, without keeping another permanent matrix copy. CUDA saving downloads that operator to the host. The file preserves the coefficients used by the solver; duplicate entries and tolerated differences between the input’s two triangles have already been normalized.

Loading validates the data and prepares the requested backend. Automatic factor adoption selects CPU; set `options.backend = apxchol::solve_backend::gpu` to upload to the compiled GPU backend. Storage, dropping and solve options are chosen when loading, as with the factor-adoption constructor.

An operator admitted with a larger `lump_max_mass` still needs that setting when loading. Regularization and lumping are already represented in the saved factor and are not applied again.

## File format

The uncompressed binary file stores nine typed arrays:

| Array | Content |
|---|---|
| Metadata | Schema version, dimension, SDDM flag, operator and factor entry counts, lumped entry count, original stored entry count |
| Operator offsets, rows and values | Original matrix in CSC, with FP64 values |
| Factor offsets, rows and values | Lower-triangular factor in CSC, with FP32 values |
| Permutation | Original vertex to elimination position |
| Round sizes | Elimination-round sizes; may be empty |

The header identifies format version 1. Array descriptors record the element type, 32- or 64-bit width, byte order, count, file offset and CRC32 checksum. Header fields use little-endian integers; array values use their recorded byte order. The header and descriptors have a separate checksum, and each array starts at a 64-byte-aligned file offset.

Loading checks file lengths, dimensions, index ranges and conversion to the current build’s index widths. The original operator uses Eigen’s signed 32-bit sparse indices; wider factor indices do not remove that limit. CRC checks detect accidental corruption but do not authenticate a file.

Only canonical CSC is stored. Loading copies the arrays and rebuilds triangular-solve layouts and backend schedules; it does not memory-map the file. GPU buffers, an active iterative solve, construction diagnostics and backend-specific FP16 representations are omitted.

The original stored entry count is kept separately for fill-ratio reporting. It can differ from the saved matrix’s count when normalization reflects or removes unmatched explicit zeros.

The file format version is independent of the package version. Its compatibility contract is under review before release.

<details>
<summary>Binary layout</summary>

All offsets below are in bytes. Header and descriptor integers are unsigned little-endian. The 64-byte header is followed by nine 40-byte descriptors.

| Header offset | Field |
|---|---|
| 0 | Eight ASCII bytes: `APXCHB01` |
| 8, 12, 16 | Three `uint32`: format version `1`, array count `9`, descriptor size `40` |
| 20 | `uint32`: header CRC32 |
| 24 | `uint64`: exact file size |
| 32 | 32 reserved bytes, all zero |

| Descriptor offset | Field |
|---|---|
| 0 | `uint32`: array ID |
| 4 | ASCII byte-order marker: `<` for little-endian or `>` for big-endian |
| 5 | ASCII element kind: `i`, `u` or `f` |
| 6 | Numeric byte value `4` or `8`: element width in bytes |
| 7 | One reserved zero byte |
| 8, 16, 24 | Three `uint64`: element count, absolute payload offset, payload byte length |
| 32 | `uint32`: payload CRC32 |
| 36 | Four reserved zero bytes |

Array IDs `0` through `8` identify metadata, operator offsets, operator rows, operator values, factor offsets, factor rows, factor values, permutation and round sizes, respectively. Each ID occurs once; descriptors follow payload order. The seven metadata values are `[1, n, sddm, operator_nnz, factor_nnz, lumped_nnz, original_nnz]`, with `sddm` equal to `0` or `1`.

Payloads contain contiguous numeric values in their declared byte order: `i` means two’s-complement signed integers, `u` unsigned integers, and `f` IEEE 754 floating point. Operator values are FP64 and factor values FP32. The writer uses unsigned 64-bit metadata and round sizes, signed 32-bit operator indices, and the build's factor-index widths. The reader accepts 32- or 64-bit integers when every value fits its destination.

CRC32 uses zlib's CRC32 convention with initial value zero. The header checksum covers all 424 header/descriptor bytes with bytes 20–23 set to zero; each payload checksum covers exactly its declared bytes. Empty payloads have checksum zero.

The first payload starts at byte `448`. Each later payload starts at the first multiple of 64 at or after the preceding payload’s end. Payload length must equal count × width. Writers zero-fill alignment padding; readers ignore its contents and checksums exclude it. There are no trailing bytes after the final payload.

</details>

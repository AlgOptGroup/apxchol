# Saving a solver

Save the original matrix and retained factor together, then load them on a CPU or GPU without repeating factorization:

```cpp
#include <apxchol/solver/archive.h>

apxchol::solve_options options;
options.keep_factor_values = true;
apxchol::solver prepared(A, options);
apxchol::save_solver("system.npz", A, prepared.factor());

auto restored = apxchol::load_solver("system.npz", options);
auto result = restored.solve(b);
```

Saving requires the original operator `A` and retained canonical factor values. It creates a new file and rejects an existing destination. Passing `A` explicitly avoids keeping another operator copy solely for saving.

The file contains numerical data, not GPU buffers or an active iterative solve. Loading validates the data and prepares the requested backend. Automatic factor adoption selects CPU; set `options.backend = apxchol::solve_backend::gpu` to upload to the compiled GPU backend. Storage, dropping and solve options are chosen when loading, as with the existing factor-adoption constructor.

An operator admitted with a larger `lump_max_mass` still needs that admission setting when loading. Regularization and lumping are already represented in the saved factor and are not applied again.

## File format

The file is an NPZ archive of one-dimensional numeric NPY arrays. It can be inspected with NumPy using `np.load(path, allow_pickle=False)`. The writer stores entries without compression for faster saving and loading. The reader also accepts DEFLATE-compressed NPZ files.

| Array | Content |
|---|---|
| `apxchol` | Schema version, dimension, SDDM flag, operator entry count, factor entry count, lumped entry count |
| `A_indptr`, `A_indices`, `A_data` | Original operator in CSC, with FP64 values |
| `L_indptr`, `L_indices`, `L_data` | Lower-triangular factor in CSC, with FP32 values |
| `permutation` | Original vertex to elimination position |
| `round_sizes` | Elimination-round sizes; may be empty |

Each NPY header specifies its integer width, floating-point width and byte order. Loading checks dimensions and narrowing against the current build. The original operator uses Eigen's signed 32-bit sparse indices; wider factor indices do not remove that operator limit.

Only canonical CSC is stored. Loading reconstructs triangular-solve layouts and backend schedules. Construction diagnostics, addresses and backend-specific FP16 representations are omitted. CRC checks detect accidental corruption, not authenticity.

The proposed file schema is version 1, independently of the package version. Its final compatibility contract is under review before release.

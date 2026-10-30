# Command line

The `apxchol` executable solves one system, or a batch of right-hand sides,
read from Matrix Market files. Get it from a [CLI archive](install.md#cli-archives)
or a [source build](building.md).

```bash
apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
apxchol graph.mtx --convert-input --random-rhs --tol 1e-8
apxchol --help
```

The input file is read as the operator `A`. If it is a graph adjacency matrix,
`--convert-input` solves its Laplacian instead; see
[converting input](usage.md#converting-input-to-an-operator). Without it, an
adjacency matrix is rejected with an error that suggests the conversion.

Pass exactly one of:

- `--rhs FILE`: a dense Matrix Market vector, or an `n` by `k` matrix whose
  columns are solved as one batch;
- `--random-rhs`: a random right-hand side for experiments, compatible with
  each connected component of a Laplacian.

`-o FILE` writes the solution as a dense `n` by `k` Matrix Market array.
The CLI prints the selected backend, iterations and residual for each column,
and a timing breakdown; `-q` prints errors only. The exit status is 0 when
every column converges, 1 when a column misses the tolerance (the solution is
still written), and 2 when the input or an option value is rejected or the
solution cannot be written. Options the parser rejects, such as an unknown
value or a missing input file, exit with a nonzero CLI11 error code.

Common options:

| Option | Default | Meaning |
|---|---|---|
| `--tol` | `1e-8` | Relative residual tolerance |
| `--maxiter` | `200` | PCG iteration budget |
| `--backend` | `auto` | `auto`, `cpu` or `gpu`; see [choosing CPU or GPU](usage.md#choosing-cpu-or-gpu) |
| `--threads` | `0` | CPU thread ceiling; `0` uses the OpenMP default |
| `--seed` | `42` | Seed for factorization and `--random-rhs` |

The [command-line reference](reference/cli.md) lists every option. It is
generated from `--help` of this documentation's revision.

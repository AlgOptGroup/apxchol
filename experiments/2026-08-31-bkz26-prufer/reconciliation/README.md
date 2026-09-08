# Evidence and reproduction

The [main findings](../README.md) link the experimental conclusions; this
directory preserves their evidence without requiring private paths.

With Python3.11 or newer, reproduce the tables and exact checks using only
the standard library:

```bash
python3 reproduce.py
python3 reproduce_optimality.py
python3 reproduce_q_screen.py
```

- `spielman.json`: eight factors, 2000 recorded solve rows and 392 captured
  triangles. The first script checks historical identities and enumerates
  local laws exactly. It does not recompute residuals from unbundled vectors.
- `optimality/`: eight complete rational certificates, all 1072 supports,
  their original verifier and named-law comparisons. No optimizer is run.
- `q-screen.json`: 30 projected raw observations; its script reconstructs
  six blocks, 42 controls and 28 ratios. The native packet is not bundled.
- `four-inputs.json` and `star-k50.tsv`: explicitly summary-level evidence
  for the crossed-order study and star sentinel; cohorts remain separate.
- `SOURCE.json`, `public-source.toml` and `SHA256SUMS`: source and byte bindings.

To repeat the **Spielman native experiment**, use Julia1.12.7 and the
[public SDDM collection](https://rjkyng.github.io/SDDM2023/) matrix
`spielman.k100.low0.25.up1.0e-6.i1.mm`, SHA-256
`b9e30aaf5271329ae75738573afe27cdefdfc84dfc19da5f979779f6ec2db3b4`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --threads=1 --project=. gate.jl
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --threads=1 --project=. run_case.jl \
  /path/to/spielman.k100.low0.25.up1.0e-6.i1.mm \
  b9e30aaf5271329ae75738573afe27cdefdfc84dfc19da5f979779f6ec2db3b4 \
  /path/to/new-output-directory
```

The executed Julia numerical sources and upstream license are preserved.
Historical fixtures retain the five fields used for identity checks; their
original hashes are recorded. The driver checks eight retained factors and
250 RHSs each, excluding five identity builds and eight warmups. Its
exact-clique intervention aborts above degree16. The original run took3m08s
on one GH200 CPU core; another environment may fail strict historical bitwise
identities. Record that difference rather than replacing reference hashes.

[Factor representation](MODEL.md) and the
[detailed experimental appendix](../appendix/reconciliation-details.md)
state interpretation and limitations. No timing promotion follows from
these reproduction commands.

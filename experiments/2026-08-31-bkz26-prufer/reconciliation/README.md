# Reconciliation evidence and reproduction

This directory supports the self-contained [research account](../RECONCILIATION.md). It contains public-matrix observations, exact small-star certificates, and the numerical Julia code used for the bounded Spielman experiment. Cluster/account paths and scheduler wrappers are omitted.

## Reproduce the reported tables from saved evidence

Python3.11 or newer is sufficient; both commands use only the standard library:

```bash
python3 reproduce.py
python3 reproduce_optimality.py
```

`reproduce.py` checks the manifest, eight factors,2,000 recorded solve rows,750 historical control-row identities, and392 captured triangle records. It enumerates784 ideal local laws/1,960 outcomes with exact rational arithmetic and prints the Spielman and four-input tables plus the star-k50 three-factor medians. This is saved-data verification: it cannot recompute residuals from vectors that are not included.

`reproduce_optimality.py` verifies all eight exact lower/upper certificates, covering1,072 connected simple supports, then independently enumerates GKS, weighted-Prüfer/CAST-1 and relative-trace cycle laws. It uses the original generic rational certificate verifier; it runs no optimizer, eigensolver or random sampler. The published tree and `d`-edge bounds remain separate. Full certificate bounds, support-dependent probabilities/weights and verification metadata are under `optimality/`.

## Repeat the bounded native Spielman experiment

Requirements: Julia **1.12.7**, the package versions in `Project.toml`, and the public matrix `spielman.k100.low0.25.up1.0e-6.i1.mm` from the [SDDM collection](https://rjkyng.github.io/SDDM2023/). Its SHA-256 is

```text
b9e30aaf5271329ae75738573afe27cdefdfc84dfc19da5f979779f6ec2db3b4
```

Install the Julia environment, then run the source/numerical gates and experiment, using a new output directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --threads=1 --project=. gate.jl
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --threads=1 --project=. run_case.jl \
  /path/to/spielman.k100.low0.25.up1.0e-6.i1.mm \
  b9e30aaf5271329ae75738573afe27cdefdfc84dfc19da5f979779f6ec2db3b4 \
  /path/to/new-output-directory
```

`public-source.toml` pins six relevant Laplacians.jl source files. `adapter.jl`, `reference_adapter.jl`, source checks, driver and native fixtures are byte-for-byte copies of the executed numerical payloads; their hashes are recorded in `SOURCE.json`. The two historical TOMLs retain only the five field groups used by the driver's equality checks. Their original receipt hashes are also recorded. The accompanying permissive upstream license applies to adapted public Laplacians code.

The driver retains eight factor/250-RHS arms, preceded by five actual-input identity builds and one excluded solve warmup per arm. The native gate separately includes nine dense exact-tail checks and nine degree-two factor/RNG-state checks. The exact-tail intervention refuses a pivot above degree16 rather than changing its semantics. No timing promotion is claimed; setup includes the observer and has no timing controls.

The archived execution used one GH200 CPU core. Historical bitwise identities are deliberately strict: a different Julia/package source or floating-point execution environment may fail them even if mathematical behavior remains valid. Record that difference; do not silently replace the reference hashes. The original bounded execution finished in3m08s under a15-minute ceiling.

## What each artifact establishes

| Artifact | Evidence level |
|---|---|
| `spielman.json` | Full projected factor/census metadata and2,000 original/solver residuals, iterations, RHS hashes and solution hashes; original receipt hashes preserve provenance. No raw solution vectors. |
| `four-inputs.json` | Summary-level projection of32 factors/8,000 recorded solves from the completed crossed-order study. It reproduces the displayed means and mirror-factor checks, but does not contain its8,000 raw vectors or journals. |
| `star-k50.tsv` | The nine-row table from a committed prior reconstruction, with250 RHSs per row and recorded maximum true residuals. This is summary-level evidence. |
| `optimality/*/certificate.json` | Complete rational positive-primal/lower-bound certificates for eight global tiny-star comparisons. |
| `optimality/comparison.json` | Exact named-law objectives, intervals, matched-budget ratios and support descriptions derived from those certificates. |
| `SHA256SUMS`, `SOURCE.json`, `public-source.toml` | Byte-level bindings and source/protocol provenance; hashes do not replace mathematical or residual checks. |

The recorded37-file execution inventory was fully collected with no exclusions. This public projection reorganizes its relevant evidence; it does not claim that its local filenames are the original inventory. The four-input and star-sentinel cohorts remain separate from the Spielman intervention denominator.

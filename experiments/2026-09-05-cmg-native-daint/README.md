# Native CMG packed benchmark on Daint

The fixed denominator is the existing ParAC portable campaign's 14 input files,
T=72 and16, with one warmup and three retained processes per cell:28cells and
84retained solves (112calls including warmup). Input SHA-256 hashes are copied
unchanged from that campaign's879ef9f6source. A first invalid/unsupported solve
stops that cell and retains explicit missing repetitions; unrelated cells run.

The native port runs the original unshifted operator and compatible normalized
b=A*g with NumPy PCG64(42), preserving graph D-|A| assembly and physics diagonal.
RHS sequences differ from ParAC's Julia/C++ generator; do not call these equal
RHS timings. Every solution is checked against the original operator by SciPy,
independently of the driver's Eigen residual. All three retained solves must
pass1e-8. The port is serial; requested CPU affinity16/72is not effective
parallelism and cells record effective_threads=1. Native and canonical MATLAB
CMG stay separately labelled; shallow_water1's known canonical setup exception
is recorded in each cell. Generated sources remain private and unchanged.

Build and pilot:one exclusive debug node,72task CPUs,200GiB,10minutes. Four
pilot cells (two actual matrices and both affinity settings),one warmup and one
retained solve each. Production reuses the pilot binary after SHAverification.

Production:one four-node debug job,one72CPU/200GiB task per node,30minutes.
Assign the28cells round-robin into four fixed7cell shards; complete cell cap
210seconds including warmup, three solves, and result verification leaves
5.5minutes for common input preparation. Every node reserves288CPUs and4GPUs,
so production ceiling is2node-hours,576allocated CPU-hours,8reserved GPU-hours.
The code uses no GPU. The runner preserves timeout caps and all failures rather
than selecting only passing matrices. A Slurm timeout must be reconciled against
the frozen28cell plan, including cells never reached.

Use an immutable package with MANIFEST.sha256 and SOURCE_COMMIT, submit with
absolute `--chdir` and output paths. `job.sbatch smoke` uses its default shape;
production overrides `--nodes=4 --ntasks=4 --time=00:30:00` and receives the
verified pilot executable path and SHA256. Run `sbatch --test-only` first.
Cancel only if authorized, with `ssh daint 'scancel JOB_ID'`; resubmission of
the preserved package is recovery. Local shutdown is safe after job submission.

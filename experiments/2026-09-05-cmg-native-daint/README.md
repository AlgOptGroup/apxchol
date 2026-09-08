# Native CMG Daint benchmark protocol

This experiment prepares an explicitly labelled native CMG comparison. Its
private generated core is serial: requested affinity T=16/72 does not mean
16/72 effective solver threads. Keep it separate from canonical MATLAB CMG
and from our algorithms. Generated proprietary sources must remain external.

The planned denominator is fourteen inputs at two affinities: 28 cells,
three retained repetitions per cell (84), plus 28 warmups. This README records
the plan, not proof that every invocation completed or converged.

Grade against the original, unshifted operator. The experiment generates
$b=Ag$ using NumPy PCG64 seed 42, normalizes the right-hand side, and checks
every returned solution independently against relative residual tolerance
$10^{-8}$. Its generated RHS protocol differs from ParAC's; do not claim
matched right-hand sides. A representative repetition selects timing only;
all retained residuals must pass.

The [matrix plan](matrices.json) and [Slurm script](job.sbatch) describe the
campaign. The pilot bounds are one node, 72 task CPUs, 200 GiB and ten minutes.
The production plan uses four nodes, 200 GiB per node and thirty minutes,
with a 210-second whole-cell watchdog. Exclusive-node allocations also reserve
unused GPUs; count those resources even though CMG performs no GPU work.

Use the existing [native CMG runner](../../benchmarks/cmg_native_runner.py)
and [benchmark guide](../../benchmarks/README.md). The Slurm script depends on
external private sources, executable paths, and hashes: the public files alone
do not reproduce the generated binary. Validate those prerequisites and use
`sbatch --test-only` before submission. The [full historical protocol](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/experiments/2026-09-05-cmg-native-daint/README.md)
preserves the original package instructions and canonical-port exception.

# Bounded stable adjacency pool probe (2026-08-26)

## Mechanism

The experiment replaced the relocating `std::vector` used by the pooled
incidence backends with stable mmap-backed segments. A vertex slab never
crosses a segment, so the existing compact integer-offset metadata stays
valid. During bulk edge application each worker claims and copies its own
growth ranges; only the final publication barrier remains. The old path
performed a team metadata pass, a barrier, a serial prefix/resize (including
value-initializing headroom), another barrier, parallel copies, and a final
barrier.

Segments contain 2^27 slots (1 GiB for the default 8-byte directed record,
512 MiB for a 4-byte indexed incidence). Only used segments are mapped, so the
design works under an ordinary RLIMIT_AS; it does not reserve one giant virtual
range. A worker whose aggregate round growth exceeds one segment splits its
growth list into maximal segment-sized claims. This fixed the first Orkut gate,
which had thrown `std::bad_alloc` at about 9.25 GiB RSS.

## Correctness and capacity gates

- 301/301 CPU tests pass.
- The 64-bit-edge-index build completes under `ulimit -v 16777216`.
- Full solves on grid_500, com-Amazon, and thermal2 matched current main in
  factor size, PCG iterations, and residual for three seeds.
- The fixed unsigned-32-bit candidate completes com-Orkut at about 22.9 GiB
  RSS and produces the same one-step residual as current main. Its elimination
  is 11.94 s versus 12.37--12.57 s on two controls, while total setup is neutral
  (36.96 s versus 36.80--36.97 s).

## T=16 laptop result

Four interleaved pairs, maxiter=1, current main `522514c`, Clang Release,
12 matrices:

| aggregate | candidate / main |
|---|---:|
| elimination | **0.9099** |
| full setup | **0.9788** |
| total (one PCG iteration) | **0.9795** |
| max RSS | **0.9810** |

Large setup wins include grid_2000 0.8875, thermal2 0.9367,
G3_circuit 0.9494, and com-LiveJournal 0.9702. The three apparent losses in
the four-pair sweep did not reproduce in a 12-pair recheck: grid_500 setup
0.9584, com-Amazon 0.9910, and com-Youtube 0.9912.

Perf-stat on grid_2000 (three repetitions) reports about 4.3% fewer
instructions and 5.1% fewer cycles, with cache misses and minor faults nearly
unchanged. This supports the intended mechanism: less synchronization and
less headroom initialization, not a cache-miss miracle.

## Scaling

At T=1/4/8/16 on four matrices, elimination strong scaling improved notably on
grid_2000 (4.71x -> 5.21x) and thermal2 (5.12x -> 6.29x), slightly on Amazon
(3.27x -> 3.31x), and regressed on grid_500 (4.15x -> 3.95x because its T=1
case also got faster). Full-setup scaling is correspondingly mixed.

T=32 on this 16-core laptop is slower than T=16 for both arms and is not a
physical-core scaling proxy. The candidate is mixed there.

## Daint decision gate

Daint job 4545868 checked 18/18 full-solve cells: nine matrices at T=36/72,
with every candidate bracketed by two byte-identical controls. The candidate
factor, stored factor, iteration count and residual matched its controls in
18/18 cells; every residual was below `1e-8`. Candidate / geometric control:

| scope | elimination | setup | PCG | one-RHS total | peak RSS |
|---|---:|---:|---:|---:|---:|
| all 18 cells | **0.8592** | **0.9592** | 1.0008 | **0.9732** | 0.8012 |
| T=36 | 0.8583 | 0.9687 | 0.9992 | 0.9814 | 0.7608 |
| T=72 | 0.8601 | **0.9498** | 1.0024 | **0.9651** | 0.8437 |

Eight of nine per-matrix total geomeans improve. Kron is the exception at
1.0287; the largest wins are iter0040 0.9365, grid_500 0.9509, YouTube 0.9573
and Orkut 0.9628. The peak-RSS geomean is driven by small graphs; Orkut is
0.9869 and LiveJournal is 1.0231, so this is not a claim of a universal 20%
large-graph memory reduction.

The change improves the 36-to-72-thread setup speedup from 1.0831x to 1.1047x,
but elimination scaling itself is unchanged (1.0035x to 1.0014x). It is a
strong elimination constant and allocation simplification, not the missing
setup-scaling breakthrough.

**Decision: ship unconditionally.** The production form removes the compile
definition and the relocating implementation instead of retaining two storage
modes. It also removes the retired slab-alignment knobs and adds direct tests
for segment crossings and concurrent claims. Downloaded evidence:
`/tmp/apxchol-residual-segmented-daint-r2` (432/432 result checksums verified).

## Refuted adjacent variants

- One giant lazy virtual reservation: fast, but fails ordinary RLIMIT_AS.
- Per-thread pointer arenas: larger wins, but roughly 8--16% extra RSS.
- An extra cached slab pointer while retaining the offset: mixed and slightly
  worse overall.
- Replacing the offset with a direct pointer: helped only grid_500 in the
  controlled pass; regressed setup 1--5% elsewhere and added 2--7% RSS.

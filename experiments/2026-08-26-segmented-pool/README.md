# Bounded stable adjacency pool probe (2026-08-26)

## Mechanism

Compiling with the `APXCHOL_EXPERIMENT_VIRTUAL_POOL` definition replaces the
relocating `std::vector` used by the pooled incidence backends with stable
mmap-backed segments. This is not a runtime environment switch: every A/B uses
two source-identical binaries, one built with the definition and one without.
A vertex
slab never crosses a segment, so the production integer-offset metadata stays
valid. During bulk edge application each worker claims and copies its own
growth ranges; only the final publication barrier remains. The production path
otherwise performs a team metadata pass, a barrier, a serial prefix/resize
(including value-initializing headroom), another barrier, parallel copies, and
a final barrier.

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

## Scaling and decision

At T=1/4/8/16 on four matrices, elimination strong scaling improved notably on
grid_2000 (4.71x -> 5.21x) and thermal2 (5.12x -> 6.29x), slightly on Amazon
(3.27x -> 3.31x), and regressed on grid_500 (4.15x -> 3.95x because its T=1
case also got faster). Full-setup scaling is correspondingly mixed.

T=32 on this 16-core laptop is slower than T=16 for both arms and is not a
physical-core scaling proxy. The candidate is mixed there.

**Do not merge yet.** The 2.1% laptop setup geomean does not by itself justify
the new allocator. The decisive next gate is a current-main 36/72-core Daint
campaign, where removing three team rendezvous per elimination round should
have the largest payoff. If that is not clearly larger and broad, drop the
probe rather than retaining an optional storage mode.

## Refuted adjacent variants

- One giant lazy virtual reservation: fast, but fails ordinary RLIMIT_AS.
- Per-thread pointer arenas: larger wins, but roughly 8--16% extra RSS.
- An extra cached slab pointer while retaining the offset: mixed and slightly
  worse overall.
- Replacing the offset with a direct pointer: helped only grid_500 in the
  controlled pass; regressed setup 1--5% elsewhere and added 2--7% RSS.

# Bottleneck investigation — where apxchol is memory-bound, and what would unblock it

*2026-08-04 measurement campaign (locked-frequency), verified 2026-08-04/15. Machine: AMD Ryzen 9
7945HX laptop (16C/32T, 2 CCDs × 32 MB L3, dual-channel DDR5-5200). HEAD = 877f3b9 (v0.2.2),
config `bg+tree[vec_pool]`, workloads iter0040 (n=524k, nnz 7.87M, factor 9.05M nnz, 158 levels)
and grid_2000 (n=4M, nnz 20.0M, factor 21.9M nnz, 92 levels). Every headline below survived an
adversarial verification pass that re-derived numbers from raw data; several first-pass claims
did not survive and are listed at the end.*

## TL;DR

1. **The 2–3× scaling cap is real, explained, and mostly physics.** At T=1 the code runs at
   13–36% of the DRAM roof (not enough memory-level parallelism to hide ~130–140 ns DRAM
   latency); threads add concurrent misses until aggregate traffic hits the **~40 GB/s
   (write-allocate-corrected) DRAM roof**, which happens around T=4–8. On top of that sits a
   measured **480 ms (iter0040) / 1451 ms (grid) of non-scaling serial stages** — 20–25% of
   total. Solve is bandwidth/latency-capped; setup's parallel part is DRAM-*latency*-capped;
   even the purely core-clocked part of setup reaches only 2.8–3.3× on 16 cores (≈20%
   parallel efficiency) — that residual is our own serial floor + barriers, not the machine.
2. **Which memory-bound: cache↔RAM, decisively — and it splits by phase.** IBS: 73–76% of
   sampled load-miss latency is DRAM, +6% peer-cache → ~80% off-core. L1/L2 is minor. Within
   that: PCG kernels are *bandwidth*-bound streams with zero temporal reuse (no blocking can
   help; only byte reduction), `prune`/`select` are *latency*-bound random gathers,
   `compute+apply_fused` is largely core-clocked (~2% of DRAM weight despite 23% of runtime).
3. **Hardware: don't buy bandwidth.** Many-channel TR/EPYC is a likely net *loss* on IPM
   (clock ↓, only-saturated-fraction converts). Ranking for the IPM ladder: desktop Zen 5
   (~1.3×) → add X3D + single-CCD pinning (~1.45–1.6×, model-based) → GPU-resident solve
   (measured 1.09× iter0040 / 1.50× grid, Amdahl-capped by CPU factorization). Nothing
   purchasable exceeds ~1.5× on the IPM ladder.
4. **Software headroom is comparable to hardware headroom**: top-5 verified levers total
   **−13…−16% (grid) / −9…−12% (iter0040)**, all S/M effort, most bit-identical.

## Methodology (what makes these numbers trustworthy)

- Campaign 1: all timing at **hard-locked 2.5 GHz** (boost off, acpi-cpufreq P-state),
  cooldown-gated, taskset-pinned, 3–5 reps/cell, medians; freq verified from /proc/cpuinfo.
  Campaign 2 (addendum) added 1.5 / 1.0 / 0.6 GHz clamps (amd-pstate) for the four-point fit.
- **Two-frequency discriminator**: reran key cells locked at 1.5 GHz. Per-stage time ratio
  t(1.5)/t(2.5) bands: ≈1.01 pure DRAM-bandwidth (STREAM T=16 ratio), ≈1.30–1.41 DRAM-latency
  chase (measured pointer-chase band; latency is partly core-clocked), ≈1.667 pure core-clock.
- **IBS (`perf mem`, ibs_op)** for per-load data source + latency; DSO/symbol cycle profiles;
  generic counters with a repeat-3-minus-repeat-1 subtraction to remove the mtx-load prefix.
- STREAM-style roof + pointer-chase latency ladder at both frequencies, plus boost-on.
- Caveats that survive: `membench` counts logical bytes — the *actual traffic* roof is
  ~40–42 GB/s (write-allocate ×4/3 on triad; two kernels agree); `ls_any_fills` counters
  cannot measure DRAM bytes (prefetcher-staged lines are attributed to L2; 145× undercount in
  the streaming limit); the locked-P-state L3 is anomalously slow (16 MB chase 72 ns locked vs
  14.5 ns boost), so the discriminator *understates* real-hardware upside of non-bandwidth
  stages; APXCHOL_PROFILE observer effect not measured; T changes the factor (partitioner
  keys on omp_get_max_threads: iters 44@T8 vs 46@T16, +17% instructions at T=16).

## The machine envelope

| fact | value |
|---|---|
| DRAM roof (actual traffic, locked 2.5 GHz) | **~40.3 GB/s** (triad 30.22 GB/s counted ×4/3; copy agrees at 41.5) |
| single core vs roof (2.5 GHz) | **93–96%** of it — streaming code gains almost nothing from threads |
| roof vs frequency | flat: 1.5 GHz T=16 −1.2%; boost T=16 slightly *worse* |
| roof from one CCD | 31.24 GB/s ≥ both CCDs (30.22) — pinning to one CCD costs zero bandwidth |
| DRAM latency (idle chase) | 136–146 ns locked (≈350 cycles); 128 ns boost |
| L2 / L3(4MB) latency | 5.6 / 21 ns locked |
| theoretical DDR5-5200 2ch | 83 GB/s → platform delivers ~50% |

## Per-stage bottleneck map (T=16, locked 2.5 GHz, medians)

ρ = t(1.5GHz)/t(2.5GHz). Verdicts: BW = bandwidth-saturated, LAT = DRAM-latency, CORE =
core-clocked, SER = serial/non-scaling.

| stage | iter0040 ms | grid ms | T1→T16 | ρ (iter/grid) | verdict |
|---|---|---|---|---|---|
| pcg/spmv | 175 | 457 | 2.5×/2.9× | 1.07/1.07 | **BW** (iter 71% of roof — gather-limited; grid 90%) |
| pcg/solve fwd | 305 | 607 | 2.2×/6.2× | 1.27/1.18 | LAT+levels (iter 43% of roof) / BW-ish on grid |
| pcg/solve back | 172 | 500 | 3.2×/4.6× | 1.14/1.14 | BW-leaning |
| pcg vector ops (Eigen, serial) | 75 | 721 | ~0.9× | 1.26 | **SER — but already at single-core roof (25–33 GB/s); threading them is worth ≈0 at 2.5 GHz** |
| permute+unpermute | 45 | 372 | 1.4–2.4× | — | mixed; serial `scratch_=x` inside |
| find_partition/prune | 426 | 338 | 2.8×/2.8× | 1.31/1.36 | **LAT** (random gathers; re-walks live edges every round — 26% of iter0040 setup) |
| find_partition/select | 66 | 96 | 3.5×/3.2× | 1.14/1.13 | LAT→BW at T=16 |
| eliminate/compute+apply_fused | 510 | 1301 | 3.1×/2.4× | 1.45/1.39 | **CORE** (RNG/hash/atomics; only ~2% of DRAM weight) |
| eliminate/merge_is (serial tail rounds, IS≤2000) | 144 | 16 | **0.79×** | 1.52 | SER, worsens with T |
| sptrsv_setup | 142 | 425 | 1.2×/1.0× | 1.39/1.33 | SER — **71–76% is the O(threads·nnz) CSC→CSR transpose (defect)** |
| make_graph | 119 | 305 | 1.3×/1.6× | 1.56/1.43 | CORE+SER |
| assembly | 63 | 197 | 3.7×/2.0× | 1.64/1.38 | CORE |
| **setup total** | **1540** | **2756** | 2.46×/2.19× | 1.42/1.36 | latency + serial floor |
| **solve total** | **872** | **3003** | 2.12×/2.94× | ~1.2/1.15 | bandwidth + serial tail |
| **total** | **2422** | **5882** | 2.31×/2.53× | 1.33/1.23 | |

Bucket summary (T=16): bandwidth-saturated 7–8%, latency/dependency 33–42%, core-clocked
31–43%, of which non-scaling serial **20% (iter0040) / 25% (grid)**.

Key decomposition results:
- Setup's core-clocked component scales 2.76×/3.31×; its memory-clocked component scales
  1.95×/0.91× — **the setup cap is memory-latency + serial stages, not barriers**. (Setup
  scales *better* at 1.5 GHz: 2.48× vs 2.19× on grid — impossible under a core-clock cap.)
- libgomp holds 46–58% of cycles but ~0.04% of DRAM weight — spin-wait, costing ~no wall time
  at locked frequency. **Not a lever** (passive wait would be net worse: 14.5k barriers/solve).
- Rounds: **~85–101 total** on iter0040 (nondeterministic tail), zero Baumann-Kyng residual
  rounds; per-round serial trivia (incoming() memset, erase_if, fork-joins) ≤ ~2% of setup.
  The real per-round cost is `prune`'s live-edge re-walk (Σ|active| ≈ 9.8n).
- CCD experiment (iter0040): 8T pinned to one CCD beats 8T floating by **13.0%** total
  (7.6% CCD locality + 5.8% placement stability, disjoint ranges); vs T=16 the pinned-8T
  advantage is ~3% after iteration-normalizing. Grid: all placement deltas within noise.
  IBS peer-cache (cross-CCD) share ~6% of load-miss latency.

## Which kind of memory-bound (Rasmus/Yves question)

**Cache↔RAM.** Latency-renormalized IBS (zero-weight samples excluded): DRAM 72.9% /
75.9% of load-miss latency (iter0040/grid), remote-CCD cache 6.3/6.1%, L2 13.4/15.1%,
L3 7.4/3.0%. CPU↔L1/L2 is not the constraint anywhere. Mean DRAM sample ≈ 600 cycles
(tail-heavy: p90 ≈ 570 ns — loaded queueing, not idle latency).

**Can cache utilization/blocking help?** Mostly no, by structure:
- The PCG streams (operator 96–256 MB, factor 145/350 MB read as CSR *and* CSC per iteration)
  have **zero temporal reuse** — nothing to block. Levers are byte-cutting only: fp32 factor
  (already default), fusion (−96 MB/iter), single-structure factor read (halves factor bytes;
  algorithmically awkward), sparsification (fewer nnz), index compression (sub-proportional,
  fwd solve is dependency-bound).
- The reusable data (PCG vectors: 25 MB iter0040) already fits in L3 — that, plus L3 keeping
  parts of the streams warm across iterations, is why measured DRAM traffic is below the
  naive stream model. grid's 192 MB of vectors fit nothing; column-blocked SpMV could win
  ~10–20% of grid SpMV only.
- Where cache-friendliness *does* matter: `prune`/`select`'s latency-bound gathers (slab
  compaction/locality — but note the incr-degree negative result), and CCD-aware placement
  (keep the working set in one L3), which is measured at +13% for 8T IPM.

## Hardware ranking (for the IPM ladder; grid in parens)

| # | option | expected vs this laptop @boost | notes |
|---|---|---|---|
| 1 | Desktop Zen 5 (9950X-class, DDR5-6000) | **~1.32×** (1.28×) | clocks+IPC+latency; helps every band; scaling stays ~2.3–2.8× |
| 2 | + X3D, 8T pinned single CCD (9800X3D/9950X3D) | **~1.45–1.6×** (~1.4×) | converts ~27pp of iter0040 DRAM weight to L3-speed; C/S hit model unvalidated — measure via AMD CAT L3-partitioning before buying |
| 3 | GPU-resident solve (built, fc192fd) | 1.09× measured (1.50×) | capped ≤1.57× by CPU factorization (Amdahl); ladder realistic 1.1–1.35× |
| 4 | single-CCD topology per se | 1.08× (~1.0) | subsumed by #2 |
| 5 | Strix-Halo-class APU | 1.05–1.15× (1.2–1.35×) | CPU path weak; interesting as unified-memory GPU host |
| 6 | **TR/EPYC 8–12 channel** | **≤1.0× (wash)** | only saturated fraction (~8%) converts; −25% clocks; serial floor unchanged. **Do not buy bandwidth** |

Boost vs locked corpus: this laptop at boost is already 1.81× (iter0040) / 1.52× (grid) the
locked numbers — core+L3-side, not bandwidth (roof identical at boost).

## Software roadmap (verified top-5; combined −13…16% grid, −9…12% iter0040)

1. **Delete three serial full-vector copies in the PCG loop** — `std::copy` at
   `sptrsv/omp.h:416,:487` (fold x into the solve recurrence) and `scratch_ = x` at
   `preconditioner.h:208`. grid −345…360 ms (5.9% of total), iter0040 −32…50 ms.
   Bit-identical. S effort.
2. **O(nnz) CSC→CSR transpose in sptrsv_setup** (`omp.h:168–215`; currently every thread
   rescans all nnz). grid −250…285 ms (4.4%), iter0040 −70…80 ms (3.0%). The clearest pure
   implementation defect found. M effort, bit-identical with stable sort.
3. **PCG fusion — the byte-elimination half only** (cherry-pick the fusion from
   `pcg-outer-loop-archive` @3fb5fb4: fold p·Ap into SpMV, ‖r‖ into the r-update, r·z into the
   preconditioner tail: −96…128 MB/iter). grid −215 ms firm (3.7%), iter0040 −20 ms.
   *Threading* the vector ops is worth ≈0 on this machine — they already run at the
   single-core roof (would help on multi-channel hardware). Needs deterministic reductions.
4. **CCD-pinned 8T auto-policy for IPM-class matrices** (proper pinning, not just a thread
   cap — the unpinned clamp is a known prior negative). iter0040 −72…139 ms (3–5.7%), grid 0.
   Changes the factor (iters shift) → score on iters × per-iter cost.
5. **Parallelize make_graph + assembly residual serial parts** (the genuinely core-clocked
   serial stages). grid −145…200 ms, iter0040 −40…60 ms. M effort.

Runners-up: hoist per-round `incoming()` (small), contiguous level ranges (−4 B/row/level),
merge_is tail (iter0040 144 ms, negative scaling — investigate why it *worsens* with T),
grid dTLB (40.5% miss rate despite THP=always — check hugepage coverage of the big arrays).
Explicitly not levers: libgomp spin share, software prefetch, MKL, SAIT/FSAI, metis_nd (all
prior negatives), passive OMP wait.

## Open items before circulating to Yves/Rasmus as final

1. **Nothing was A/B-tested** — all levers are inference; each top-5 item needs a paired
   locked-frequency A/B (~20 min each).
2. **Competitors were not run under this lock** — the "nobody exceeds 3× on this machine"
   line is from the boost-on prior sweep; a same-lock BoomerAMG/ParAC run would make the
   "machine, not us" argument airtight.
3. **Both workloads took the SDDM branch** — the Laplacian/centering path has zero coverage
   here, and grid_2000 classifying as SDDM looks *spurious* (exact-zero row sums should be
   Laplacian; `conversions.h:180-182` excess filter) — 5-line check; if it's a bug, the
   centering path is dead in all benchmarking.
4. Generalization: n=2 workloads; 90 IPM iterates + ~20 SuiteSparse matrices are on disk
   unused. A 3-matrix spot-check (com-Amazon, G3_circuit, iter0010) would confirm the maps.
5. libgomp internals unresolved (56% of cycles under raw addresses): `debuginfod-find
   debuginfo /usr/lib/libgomp.so.1` + re-report the archived perf.data (~2 min, no re-run).
6. membench needs read-only and NT-store kernels (the true roof for read-dominated kernels
   is likely *higher* → our "% of roof" figures are upper bounds on saturation).

## Corrections vs earlier in-session claims (superseded)

- "~30 GB/s roof; SpMV at 91–95% of it" → roof is ~40 GB/s actual-traffic; iter0040 SpMV
  ~71% (has gather headroom), grid ~90%; grid SpMV byte model was 272 MB → **352 MB (operator
  is fp64 on BOTH workloads** — grid weights 1e-2 are not fp32-exact).
- "Serial PCG tail = 44% of grid solve; re-landing fused PCG ≈ −30% grid solve" → tail is
  ~31.6% of pcg, runs at single-core roof; realistic fusion win ~6–13% of pcg.
- "Setup is core-clock/sync-bound at T=16" → inverted; it's DRAM-latency + serial stages.
- "~1500 elimination rounds; per-round memset is a prime suspect" → 85–101 rounds, zero BK
  rounds, memset ≤2% of setup.
- "8T beats T=16 on both workloads" → only *pinned* 8T beats T=16 on iter0040 (~3%
  iteration-normalized); grid placement deltas are noise.
- Old memory "STREAM peak 37 GB/s / kernel at 94% of DRAM peak" (May) → superseded by the
  convention-corrected roof and per-stage map above.

## Addendum — campaign 2 (2026-08-15): confirmation/falsification battery + lever A/Bs

All numbers below survived a second adversarial verification pass (4 independent re-derivations
from raw logs in `~/.cache/apxchol_prof/results2/`). Binary: fresh g++ 15.3 build — absolute
times moved −20…−38% vs the Aug-04 corpus (compiler upgrade), so only within-campaign
comparisons are valid. Interleaved paired A/Bs; an accidental A/A control (stale binary)
bounded the paired per-iter noise floor at ±1.3%.

### The memory-bound verdict, now on five independent instruments

1. **Causal (bandwidth-hog dose-response, solver 8T CCD0 vs hog CCD1)**: a saturating hog
   raises iter0040 solve +109…128% and setup +45…47%; grid solve +83%, setup +68%. The effect
   saturates at 2 hog threads. Hog sustains 41-42 GB/s during iter0040 co-runs, 30-33 during
   grid (the grid solver claims more bandwidth itself).
2. **Frequency scaling**: T1→T16 total scaling improves when cores slow (iter0040 2.63×→2.91×,
   grid 3.28×→3.77×; iter-normalized 2.68→2.88 / 3.15→3.63) — impossible under a core/sync cap.
3. **Top-down counters (solver-only, repeat-diff)**: backend_bound_memory = 63.5%/62.1% of
   pipeline slots at T=16 vs 29.3%/27.0% at T=1 — the per-thread memory-stall share doubles as
   threads queue on the controller.
4. **IBS load latencies** (campaign 1): 73-76% of load-miss latency from DRAM.
5. **Cache-resident falsification test**: grid_300/500 (working set inside L3, memory free)
   still cap at 2.25×/2.70× (T=8) and REGRESS at T=16 — the sync/coherence/serial cap is real
   and independent of memory. Both caps bind: sync dominates at small n, memory at production n.

### Corrected machine envelope

Two roofs (locked 2.5 GHz, actual-traffic conventions): **mixed read/write ~42-43 GB/s**
(triad/copy/NT agree), **pure read ~50 GB/s**. A single core streams 35-37 GB/s NT
(~83-88% of the mixed roof — the earlier "96%" was a counted-bytes artifact). The read-only
T=1 kernel (12.7 GB/s) is FP-chain-limited, not a roof.

### Memory-free counterfactual (VALIDATED four-point C/f+M model)

The machine switched to amd-pstate-epp at the Aug-15 reboot (kernel update; another
baseline-identity item), enabling arbitrary frequency clamps. A four-point sweep
(2.5/1.5/1.0/0.6 GHz, hard clamps verified) validates the linear model t = C/f + M with
R² ≥ 0.97 per stage (totals 0.991-0.9995; ≤2-4% residual at 0.6 GHz). Iteration-normalized:

- **If DRAM were free: iter0040 T=16 ≈ 1.17 s (scaling cap 3.50×), grid ≈ 1.64 s (cap 5.80×).**
- **The memory component M is nearly thread-invariant**: iter0040 675→622 ms and grid
  1989→1737 ms from T=1 to T=16 — fifteen extra threads recover only 8-13% of memory time,
  while the core component scales 3.5×/5.8×. All scaling lives on the compute side;
  t(T,f) ≈ C₁/(s(T)·f) + M.
- iter0040 forward SpTRSV at T=16 fits as ~pure core-clock (M≈6 of 278 ms): dependency chain
  + barrier spin, not traffic. Grid forward is ~half memory. SpMV M-share: 71%/92%.
- Caveat for very low clocks: the T=16 roof itself degrades at 0.6 GHz (42→33-40 GB/s),
  bounding how far the downclock method can be pushed.

The rest of the 16× ideal is barriers + serial stages + imbalance, cross-validated by the
cache-resident cells. Note: the previously-reported "T=16 regresses vs T=8" was an
iteration-count artifact — per-iteration T=16 is genuinely ~7-9% worse than T=8 at both
frequencies, but totals are equal within iteration luck.

### Where the time goes, and what bounds each stage (four-point fit, T=16, fresh binary)

"memory share" = fraction of the stage's T=16 time that is frequency-invariant (M/t);
"memory-free scaling" = T1→T16 scaling of the stage's compute component C — what the stage
would scale like if DRAM traffic cost nothing. Read the two together: high memory share +
high memory-free scaling = a bandwidth wall (only byte-cutting or more bandwidth helps);
low memory share + low memory-free scaling = our own serialization/barriers (code fix).

**iter0040 — total 1828 ms (T=1: 4917 ms; 2.7×)**

| stage | ms | % total | T1→T16 | memory share | memory-free scaling | verdict |
|---|---|---|---|---|---|---|
| prune | 350 | 19% | 3.0× | 30% | 3.2× | latency gathers + serial per-round passes |
| compute+apply_fused | 290 | 16% | 4.5× | 39% | 5.4× | mixed; best-scaling big stage |
| SpTRSV forward | 278 | 15% | 2.4× | **2%** | **2.7×** | **dependency/barrier-bound, not memory** (158 narrow levels) |
| SpMV | 150 | 8% | 2.8× | **71%** | 8.8× | **bandwidth wall** |
| SpTRSV backward | 136 | 7% | 3.7× | 34% | 6.4× | mixed |
| merge_is (serial tail rounds) | 127 | 7% | 0.8× | 4% | 0.8× | **serial floor** (IS ≤ 2000 → single thread) |
| PCG vector ops | 87 | 5% | 0.6× | 35% | 0.6× | serial Eigen; fusion target |
| make_graph | 87 | 5% | 1.4× | 29% | 1.8× | serial floor |
| sptrsv_setup | 86 | 5% | 1.2× | 50% | 1.4× | serial floor (transpose branch: −34% on grid) |
| spmv_lrm_build | 76 | 4% | 1.4× | 35% | 2.2× | serial floor |
| select | 58 | 3% | 3.5× | 59% | 4.3× | latency |
| unpermute / permute | 40 | 2% | 0.9-2.4× | 33-72% | — | copy-removal branch fixes unpermute |
| assembly | 21 | 1% | 8.9× | 10% | 9.2× | fine |

**grid_2000 — total 3520 ms (T=1: 11919 ms; 3.4×)**

| stage | ms | % total | T1→T16 | memory share | memory-free scaling | verdict |
|---|---|---|---|---|---|---|
| compute+apply_fused | 643 | 18% | 3.4× | 38% | 4.9× | mixed |
| PCG vector ops | 502 | 14% | **1.1×** | 49% | 1.2× | **serial Eigen — fusion target** |
| SpTRSV forward | 432 | 12% | 7.2× | 51% | 10.8× | bandwidth wall |
| SpMV | 396 | 11% | 3.0× | **91%** | 31× | **bandwidth wall** |
| SpTRSV backward | 388 | 11% | 5.4× | 53% | 13.7× | bandwidth wall |
| prune | 281 | 8% | 3.1× | 24% | 3.8× | latency + serial passes |
| sptrsv_setup | 256 | 7% | 1.1× | 31% | 1.0× | serial floor (transpose branch −34%) |
| unpermute / permute | 270 | 8% | 1.5-2.1× | 65-85% | 3-10× | bandwidth; copy-removal branch |
| make_graph | 107 | 3% | 2.7× | 10% | 3.4× | serial-ish |
| select | 93 | 3% | 2.7× | 63% | 5.7× | latency |
| assembly, lrm_build, merge_is | 121 | 3% | — | — | — | minor |

**Compute-vs-memory framing (same fit, whole run):**

| | T=1 | T=16 | scaling | share @T16 | how it limits scaling |
|---|---|---|---|---|---|
| iter0040 compute C (incl. serial + barrier spin) | 4.09 s | 1.17 s | 3.5× | 65% | sets the *cap*: serial stages + barriers |
| iter0040 memory M | 0.68 s | 0.62 s | 1.1× | 35% | doesn't shrink with threads: shared wall |
| iter0040 total | 4.76 s | 1.79 s | **2.7×** | | 2.7 < 3.5 because M is fixed |
| grid compute C | 9.53 s | 1.64 s | 5.8× | 49% | |
| grid memory M | 1.99 s | 1.74 s | 1.15× | 51% | |
| grid total | 11.5 s | 3.38 s | **3.4×** | | |

So on IPM the cap is roughly half memory and half our own serialization; on grids memory
dominates the T=16 time, but the *cap* (5.8×) is still ours (vector ops, sptrsv_setup,
barriers).

### Same-lock competitors (fairness note: our runs carried ~2-5% profiling overhead)

apxchol 1.88 s (iter0040 T16) **beats BoomerAMG 2.12 s**; AMGCL wins grids (1.96 vs our 3.67 s).
Scaling: apxchol 2.63×/3.28× is best-or-second in every cell; no solver exceeds 3.28× at this
operating point. The "machine envelope" claim is now same-session, same-lock, same-compiler.

### GPU (corrected accounting)

grid: GPU-resident PCG loop 3.70 ms/iter vs CPU 46 ms/iter = **12.5× per-iter** (user-facing
~6.8× — ~150 ms/solve of transfers sit outside the loop); cuSPARSE-SpTRSV build solve 0.33 s vs
CPU 2.15 s. iter0040: loop median ~9.8 ms/iter vs CPU ~16 = **~1.7× only, noisy** —
level-sync-bound as predicted. Both builds use fp32 factors (like CPU). GPU bandwidth roof
not measured this campaign.

### SDDM misclassification: bug fixed, and the fix has a cost

Root cause (confirmed by ±define flip): fp32 edge-pool quantization vs exact fp64 diagonal vs
the 1e-12 gate → all non-fp32-exact-weight Laplacians ran the SDDM branch. Post-fix BOTH
benchmark workloads classify Laplacian (m = n−1) — **iter0040's matrix is an exact Laplacian
too**, so the entire historical corpus ran the leaner no-centering branch with phantom
~1e-8·diag excess (accidental regularization). The fix (branch `fix/sddm-classification`,
exact-fp64 wdeg at ingestion, 157/157 tests) costs **+20-24% grid solve per-iter at equal
iterations** (centering-path vector passes: permute/unpermute/forward inflate) and ~+5% on
iter0040 T8ccd. Recommendation: land together with centering-into-permute fusion, or settle
algorithmically whether per-application re-centering is needed (fp-drift guard only — a
question for Rasmus).

### Lever A/Bs (locked, paired, verified)

| lever | iter0040 T16 | grid T16 | verdict |
|---|---|---|---|
| serial-copy removal (`perf/kill-serial-copies`) | solve −7…−10%/iter | −5…−8%/iter | **ship** (mechanism: unpermute cross-CCD copy collapses 194→82 ms; T8ccd neutral; small unexplained iter-draw shift at T=16 to re-check) |
| O(nnz) transpose (`perf/onnz-transpose`) | neutral | sptrsv_setup −34% (−91 ms), setup −5.9% | **ship** (bit-identical; algorithm 1.9× not 8× — bucket materialization pays bandwidth) |
| SDDM fix (`fix/sddm-classification`) | ~neutral | solve +20-24%/iter | **correctness**: land with fusion (above) |

### Measurement-method notes added this campaign

APXCHOL_PROFILE observer effect: ≤1% setup, ~2-5% iter0040 solve. Stale-binary trap: verify
binary mtime > last source commit before any A/B (caught because the null result contradicted
the model; the accident provided the A/A control). The `[apxchol] SpTRSV ... fp32` banner
requires APXCHOL_VERBOSE — absence of the banner is not evidence of fp64.

## Data & artifacts

Raw campaign text outputs were lost to a /tmp wipe (reboot) — the derived tables above were
extracted before loss and cross-verified by 8 independent re-derivations from the surviving
raw perf.data (`~/.cache/apxchol_prof/*.data`) and in-conversation records. Verifier and
synthesis reports: `~/.cache/apxchol_prof/reports/`. Workflow journal:
`~/.claude/projects/.../workflows/wf_dfc44256-718/journal.jsonl`.

## Addendum 2 — campaign 3 (2026-08-18): lever experiments, paired locked A/B

Base = main 3fac63f (includes copy-removal + O(nnz) transpose). Locked 2.5 GHz, T=16,
5 interleaved reps per variant, medians; per-iteration solve compared. Full table:
`~/.cache/apxchol_prof/results3/`, `harness2/c3_analyze.py`.

| lever | iter0040 solve/iter | iter0040 total | grid solve/iter | grid total | verdict |
|---|---|---|---|---|---|
| **PCG fusion** (30→22 vector streams, all parallel, deterministic) | **−12.3%** | **−10.3%** | **−8.4%** | **−3.9%** | ship |
| SDDM fix + `center` (correct Laplacian path, per-apply centering) | +10.6% | −3% | **+33.7%** | +21.5% | the cost of naive centering |
| SDDM fix + `center-k` (re-center every 10th application) | −0.9% | −7%* | −0.1% | ±0 | **adopt** (correctness at zero cost) |
| SDDM fix + `reg` (explicit deterministic eps=1e-8 grounding, SDDM path) | +2.3% | −5%* | −1.0% | +1.9% | viable alternative; full-rank factor |
| tail rounds `TAIL_THREADS=4` / `OMP_THRESHOLD=256` (vs same-branch center) | 0 | setup −3…−4% | 0 | setup −5…−7% | small; merge only as trivial env knob |
| SpTRSV P2P flags / counter sync (bit-identical) | +7% / +2% | 0 / −4% | +1% / −2% | +4% / 0 | **negative** — barrier count is not wall time |
| CCD auto-placement (8T on one CCD, in-library) | −3.5% | −3.0% | +9.8% | +8.3% | opt-in only; needs a workload heuristic |
| bf16 factor values (all) | 5.8× iters | — | +33% iters | — | **negative** (see memory note; diag32 variant pending) |

\* the grounding/tail branch also carries a bit-identical persistent-`incoming[]` change that
makes its setup ~5–8% faster than base on iter0040; compare within that family for the
grounding effect. Verified 3-mode convergence (independent true-residual tool): all reach
~8–9e-9 against the original singular operator; center-k leaves a null-space constant in x
(one final centring restores min-norm).

### Barrier spin — what the P2P negative does and does not mean

The P2P/counter experiment swapped the synchronization *primitive* while keeping the *work
assignment*; it measured 0…+7%. Interpretation: threads at a barrier wait for the slowest
thread of that level, which is slow because of memory or because it drew more work; the
primitive is irrelevant. This does NOT close the sync topic — it redirects it to **balance**:
(1) SpTRSV levels are split by row-index range, not by nnz → an nnz-balanced split is a
few-line test; (2) the CPU state of the art (Park ISC'14; GrowLocal 2025) reorders the
factor so co-scheduled rows are contiguous and agglomerates levels into fatter supersteps —
1.4× over the MKL-style baseline on unstructured factors; untested here; (3) elimination
rounds could pipeline next-round `prune` into idle time (substantial restructuring; the tail
knob is the cheap partial answer). Cheapest discriminator first: if nnz-balancing moves
nothing, the wait is memory, not imbalance, and (2) is not worth the effort either.

## Addendum 3 — low-precision factor storage (2026-08-18)

| variant (T=1 iterations) | grid_500 | grid_2000 | iter0040 |
|---|---|---|---|
| fp32 (default) | 40 | 50 | 44 |
| bf16 all | 52 | 68 | 260 |
| bf16 + fp32 diagonal | 49 | 62 | 132 |
| bf16 + per-column scale | 49 | 60 | 71 |
| **fp16 + per-column scale** (2 B/entry) | **42** | **51** | **44** |
| fp24 (3 B/entry) | 40 | 49 | 44 |
| fp32, drop off-diagonals < 1e-3·colmax | 40 | 50 | 63 (65% dropped) |
| **fp32, drop < 1e-4·colmax** | 40 | 50 | **44 (52% dropped)** |

Reading: the failure mode of bf16 was mantissa width (7 bits ≈ 4e-3 relative), not 16 bits per
se; fp16's 10 bits (5e-4) with a per-column scale (s_j = column max |off-diag|, fp32 diag and
scale arrays) matches fp32 quality — while flushing 19% of iter0040's off-diagonals to zero and
28% to subnormal. The isolation experiment explains why: entries below ~1e-4 of their column
max carry no preconditioning value (52% of the IPM factor!). Two levers follow: (a) fp16-scaled
storage (−22% factor bytes, F16C intrinsic mandatory: the naive `_Float16→double` compiles to
a libgcc call and is 3× slower); (b) a *compacting* threshold drop at setup (−50% factor nnz on
IPM-class matrices: bytes AND flops; ~0 on grids). Scaling granularity: per-column is natural
(a column = one vertex's clique sample; entries share scale; the CSC sweep hoists it, the CSR
sweep gathers it alongside y[col]); per-row mixes unrelated eliminations; global cannot span
IPM's ~10 decades. Format floor: the useful in-column range is ~3.5 decades (≈4 exponent bits
after scaling) and the mantissa needs ≥~9 bits — fp16 is near the practical optimum; fp8/8-bit
quantization cannot reach bf16's (already insufficient) precision. Timing A/B pending
(campaigns 5–6); bf16-scaled's unexplained 132→71 (range is not the issue for bf16) to be probed
with a power-of-two scale.

### Factor compressibility census (tools/factor_census.cpp, branch diag/factor-census)

Entropy of the stored factor at the precision we need (fp16 after per-column scaling):
grids 5.1 bits/entry (synthetic 3-weight artifact: 57% of entries equal their column max),
**iter0040 11.0 bits/entry after dropping the dead half — fp16 is within 5 bits of optimal;
no 8-bit scalar or structural format exists for IPM values.** Indices: deltas are large because
only 9% (grids) / 25% (IPM) of a column's neighbours were eliminated in the same round —
round-based elimination scatters neighbours across the ordering (structural, not an ordering
choice); entropy floor 1.3–2.4 B/entry, varint 1.9–2.8 B. Practical fixed-width target:
**24-bit absolute index + fp16 value = 5 B/entry (−37.5%)** for n < 16M; with the drop on IPM
≈2.4 B per original entry (−70%). Campaign 5 caveat: iter0040's SpTRSV is dependency-bound,
so byte cuts pay there via fewer *entries* (drop), not fewer bytes per entry; grids are where
per-entry compression pays.

### Low-precision TIMING (campaign 5, locked, paired, same-branch fp32 base)

| variant | iter0040 SpTRSV/iter | iter0040 total | grid SpTRSV/iter | grid total |
|---|---|---|---|---|
| fp16 per-column-scaled | **+7.2%** | −6.9%* | −0.5% | −0.7% |
| fp24 | +0.1% | −3.1%* | **−4.2%** | −3.7% |
| drop <1e-4 (zeroed, flops-only) | −10.2% | −2.3%* | (nothing to drop) | — |
| scheduling dynamic,256 vs static | −20% | | **+23%** | |

\* iter0040 totals are dominated by iteration-count luck (44–46); read the per-iteration columns.
Verdict: fp16 storage is iteration-neutral but NOT faster — the per-entry F16C convert +
scale gather costs about what the bytes save (IPM's SpTRSV is dependency-bound anyway; grid's
short rows make the per-entry decode a large fraction). fp24 (no scaling, cheap widen) is the
only low-precision variant with a measured win: −4% grid SpTRSV. The compacting drop is the
IPM lever (fewer entries = fewer dependent flops AND bytes) — campaign 6. Load balancing is
real but workload-dependent: dynamic chunking wins on IPM's narrow levels (−20% SpTRSV) and
loses badly on grid's wide ones (+23%) → the right fix is static-by-nnz partitioning.

### Compacting factor drop — TIMING (campaign 6, locked, paired, iter0040)

| variant | SpTRSV/iter | solve/iter | total | iters | stored entries |
|---|---|---|---|---|---|
| fp32 (base) | 9.63 ms | 16.24 ms | 1.876 s | 45 | 9.06 M |
| **drop < 1e-4·colmax (compacting)** | **4.83 (−49.8%)** | **11.63 (−28.4%)** | **1.624 (−13.4%)** | 44 | 4.64 M |
| drop < 3e-4 | 4.77 (−50.5%) | 11.49 | 1.630 | 47 | 3.98 M |
| fp16-scaled | 11.78 (+22%) | 18.58 | 1.954 | 45 | 9.06 M |
| fp16-scaled + drop 1e-4 | 6.08 | 12.94 | 1.748 | 45 | 4.64 M |

The single largest lever found: removing the 52% of IPM factor entries below 1e-4 of their
column max halves the triangular solves one-for-one (they are dependency-bound: cost tracks
dependent flops on the critical path, not bytes), at zero iteration cost. Setup-time filter,
deterministic, ~20 lines, env `APXCHOL_FACTOR_DROP` (branch perf/factor-drop @ fbbfd72).
Grids: 0 entries below threshold → exact no-op. Recommend default 1e-4 after the social-suite
sanity in the benchmark re-run. fp16-scaled is a confirmed loser on IPM (+22% SpTRSV: F16C
convert on the critical path); fp32 subnormals in the factor = 0 → FTZ irrelevant.

### Compensated drop on main (campaign 7, locked, paired) — MERGED as default 1e-4

Plain removal on the CORRECT Laplacian path costs iterations (45→67 at 1e-4): a Laplacian
factor's columns sum to zero, and deleting an entry leaves its weight as extra grounding at both
endpoints. Fix: fold each column's dropped mass into its kept off-diagonals (MILU-style,
column sums preserved) — 46→46 iters at 1e-4 AND 1e-3 (ladder neutral to 3e-3).

| iter0040 | SpTRSV/iter | solve/iter | total | iters |
|---|---|---|---|---|
| main (composed) | 9.71 ms | 15.10 | 1.777 s | 46 |
| **drop 1e-4 compensated (default)** | **4.88 (−50%)** | **10.08 (−33%)** | **1.574 (−11.4%)** | 46 |
| drop 1e-3 compensated | 4.30 (−56%) | 9.37 (−38%) | 1.478 (−16.8%) | 46 |
| drop 1e-4 plain | 5.13 | 10.17 | 1.741 | 66 |

grid_2000 / com-Amazon / coAuthorsDBLP: 0 entries below threshold → exact no-op. 1e-3 is a
candidate default pending the full-suite ladder. Composed main after all merges vs the
Aug-18 pre-merge base (iter0040 / grid_2000 totals): ≈ −14% / −7%.

## Benchmark-suite re-run, smoke tier (2026-08-18, boost-on, main 9f8246f + drop)

9 matrices × 4 apxchol configs + RCHOL/pRCHOL/BoomerAMG(+cut)/AMGCL/ParAC, `sweep_fair.py
--headline-only`, vs the stored pre-change cells (`results/cells_pre_2026_08_18`).
**Solve phase: −19…−45% on grids/PDE/IPM with identical iteration counts** (bg+tree: iter0040
0.62→0.41 s, ecology1 0.56→0.38, parabolic_fem 0.35→0.22, grid_500 0.09→0.05); social ~flat
(nothing to drop, small vectors). **Setup: not interpretable vs old cells** — competitors' setup
moved +20…60% with identical iterations (BoomerAMG, AMGCL, ParAC), i.e. environment (hot
afternoon after 3 h of campaigns; boost-on variance ±25% between consecutive reps measured on
ecology1 root+tree: 0.66/0.67/0.84 s vs a 2.07 s outlier). Two traps found and fixed:
(1) amd-pstate default governor=powersave after the kernel update ran the whole suite ~2× slow
(fixed: performance/performance); (2) the sweep is resume-safe — old cells must be moved aside.
The locked paired campaigns remain the evidence for what changed; the suite's role is a fresh
self-consistent baseline → run cold: `~/.cache/apxchol_prof/harness2/full_suite_rerun.sh`
(archives cells, CPU all families then GPU; CUDA build being refreshed).

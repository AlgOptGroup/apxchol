# Daint benchmarks

CPU scaling and apxchol-only historical comparisons from CSCS Daint live here,
separate from the laptop competitor charts in `benchmarks/latest/`.

## Current campaigns

- Current scaling and structural probes: Slurm job `4538729`, source `e58dfe8`,
  207/207 records (189 timing + 18 structural), 22/22 input checks.
- Historical A/B: Slurm job `4538730`, current `e58dfe8` against the last
  pre-campaign solver commit `877f3b92`, 162/162 bracketed full solves, 22/22
  input checks.
- Toolchain: Clang 22.1.8, Release, fp32 residual-pool and CPU factor storage.
- Allocation shape: one 288-CPU Daint node per job, split into four concurrent,
  NUMA-local 72-core ranks. The jobs consumed 898 and 1242 allocated seconds,
  respectively: 171.2 allocated CPU-hours (0.594 node-hours) in total.

The raw logs remain in checksummed campaign archives under
`/capstor/scratch/cscs/okulkov/apxchol-codex/`. This directory contains compact
CSV extracts, a [summary](summary.md), and reproducible figures:

- `scaling.csv`: all 189 current timing records.
- `historical_ab.csv`: all 162 old/current/old bracket records.
- `pivot_summary.csv`: aggregated per-pivot concentration diagnostics.
- `region_snapshots.csv`: candidate-induced component snapshots.
- `figures/setup_scaling.png`: current setup strong scaling.
- `figures/setup_t72_breakdown.png`: where T=72 setup time remains. Its
  `partition phase` includes prune + IS selection + collection; `summary.md`
  reports those scaling factors separately.
- `figures/historical_total_ratio.png`: cumulative single-RHS improvement.

Regenerate from downloaded campaign `results/` trees with:

```bash
python3 benchmarks/daint/render_campaign.py \
  --scaling /path/to/current-scaling-probes-v1/results \
  --baseline /path/to/two-week-baseline-v1/results
```

These are CPU/apxchol engineering results, not cross-machine competitor bars.
The laptop store currently has stale cells and should be regenerated before its
published comparison charts are refreshed.

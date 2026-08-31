# Daint benchmark snapshot

These historical results use one CSCS Daint GH200 node: a 72-core Grace CPU and
its Hopper GPU. The fair-solver campaign is pinned to source revision
`2b755997`; absolute times are machine-specific and must not be mixed with the
x86/RTX [laptop snapshot](../latest/).

The T=72 campaign uses three full setup-and-solve repetitions per cell and grades
every completed result by an independently recomputed true relative residual at
`1e-8`. Exact coverage, outcomes, paired geometric means, and exclusions are in
the [fair-solver summary](fair_t72_summary.md); [fair_t72.csv](fair_t72.csv) is
the portable extract.

## Fair-solver views

The renderer provides three complementary views for each matrix family:

| family | total-time heatmap | solve-only heatmap | setup + solve |
|---|---|---|---|
| grids | [total](figures/fair_t72_total_grids.png) | [solve](figures/fair_t72_solve_grids.png) | [2D](figures/fair_t72_breakdown_grids_2d.png), [3D](figures/fair_t72_breakdown_grids_3d.png) |
| LP-IPM | [total](figures/fair_t72_total_ipm.png) | [solve](figures/fair_t72_solve_ipm.png) | [breakdown](figures/fair_t72_breakdown_ipm.png) |
| SuiteSparse | [total](figures/fair_t72_total_suitesparse.png) | [solve](figures/fair_t72_solve_suitesparse.png) | [small](figures/fair_t72_breakdown_suitesparse_small.png), [giants](figures/fair_t72_breakdown_suitesparse_giants.png), [largest](figures/fair_t72_breakdown_suitesparse_giants_xl.png) |

Heatmap colour is time relative to the fastest completed CPU or GPU cell in the
same matrix column; annotations give absolute time and ratio. A capped timeout
is shown as a lower bound only on total time. Solve-only cells have no fabricated
timeout duration. Breakdown bars are linear time, with solid setup and hatched
solve segments; the family splits keep large matrices from flattening smaller
ones.

Focused views:

- [apxchol CPU/GPU crossover](figures/fair_t72_apxchol_cpu_gpu.png)
- [apxchol selector/storage ablation](figures/fair_t72_apxchol_ablation.png)

The ARM64 snapshot uses explicitly labelled portable paths where possible:
RCHOL/pRCHOL retain upstream factorization and PCG semantics without MKL;
AC/AC2 use the official ARM64 Julia build; ParAC CUDA runs natively. ParAC CPU
and CMG are omitted rather than replaced by unlike timing baselines.

Reproduce the pinned figures and summary from the committed CSV:

```bash
python3 benchmarks/daint/render_fair_t72.py
```

## Historical scaling snapshot

The earlier apxchol-only campaign is a separate, checksummed archive. It is an
explicitly historical snapshot, not current-main timing. On reproduction, its
compact [three-panel speedup figure](figures/setup_scaling.png) shows total,
PCG/solve, and setup speedup from the complete 189/189-record scaling extract
(9 matrices × 7 thread counts × 3 repetitions). The legacy
`setup_scaling.png` output path is retained so existing manifests and links do
not orphan an artifact.

Separate diagnostic views remain available for the
[T=72 setup breakdown](figures/setup_t72_breakdown.png) and the within-snapshot
[campaign-head/July-27 total ratio](figures/historical_total_ratio.png). See the
[numerical summary](summary.md), [scaling data](scaling.csv), and
[historical A/B data](historical_ab.csv) for boundaries and exact values.

Reproduce that archive from its committed extracts:

```bash
python3 benchmarks/daint/render_campaign.py --csv-input benchmarks/daint
```

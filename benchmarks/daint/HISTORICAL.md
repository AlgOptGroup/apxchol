# Historical Daint studies

These are pinned snapshots, not current-main results. [Current index](README.md).

| Study | Data and views | Boundary |
|---|---|---|
| Previous GPU trace, source641 | [540-cell snapshot](historical/gpu-trace-source641-20260910/) | Earlier timing, fill and separate memory observations; replaced trace27 values remain available |
| CPU scaling, `ea01e2ff` | [84 cells and figures](historical/cpu-scaling-ea01/) |12 matrices×7 threads; converged solves; original main-T1 references |
| GPU scaling, `ea01e2ff` | [CSV](thread_scaling_gpu.csv), [setup](figures/threads_gpu_setup_speedup.png), [solve](figures/threads_gpu_solve_speedup.png), [total](figures/threads_gpu_total_speedup.png) |84 converged points; not rerun with current CPU study |
| Fair T72, `2b755997` | [750-cell summary](fair_t72_summary.md), [CSV](fair_t72.csv), [figures](figures/) |Historical coverage excludes ParAC CPU/canonical CMG; not current availability |
| Setup diagnostic | [summary](historical_summary.md), [scaling](scaling.csv), [A/B](historical_ab.csv) |189 one-iteration scaling records; not converged-solve scaling |

Reproduce committed historical extracts:

```sh
python3 benchmarks/daint/render_fair_t72.py
python3 benchmarks/daint/render_campaign.py --csv-input benchmarks/daint
```

[Original campaign explanation](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/benchmarks/daint/HISTORICAL.md)

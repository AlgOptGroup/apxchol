# Separately sampled GPU memory

54 GPU GKS/trace-cycle q=0.8 calls were targeted out of 216 declared GPU
identities. The CSV records every identity and its actual measurement status;
other variants and competitors remain unmeasured and their entirely empty rows
are omitted from the figures. These observations come from
job 4642414 on source230 with 50 ms requested process-memory polling. These
separate observations are lower
bounds on the true whole-run peak, including CUDA context and allocator memory.
The actual query duration can exceed 50 ms, and short-lived peaks may be missed.
Short calls can provide only a few positive samples; observed differences
do not establish differences in true peak memory.
Monitored timings are not used in benchmark tables.
Colour ratios compare observed maxima, not bounds on true peak-memory ratios.

| Memory | Grids | IPM | SuiteSparse |
|---|---|---|---|
| Sampled process peak | ![GPU memory](figures/gpu_sampled_vram_grids.png) | ![GPU memory](figures/gpu_sampled_vram_ipm.png) | ![GPU memory](figures/gpu_sampled_vram_suitesparse.png) |

[All 216 identities](sampled_gpu_vram.csv) · [Source and coverage](sampled_gpu_vram_provenance.json)

[Previous source641 memory observations](historical/gpu-trace-source641-20260910/) remain historical.

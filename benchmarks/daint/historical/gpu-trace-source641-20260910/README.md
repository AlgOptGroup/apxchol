# Earlier source641 GPU trace snapshot

This preserves the previous 540-cell headline and 351-cell fill extracts before
the source230 GPU trace refresh. The 27 old trace cells and their original
`parabolic_fem` solve and `grid_3000` setup/total warnings remain recorded.
The other 513 headline cells retain the same observations in the current snapshot.

[Timing values](results.csv) · [Selection](selection_provenance.json) ·
[Coverage](coverage.json) · [Fill](fill.csv) · [Fill sources](fill_provenance.json)

[Memory values](sampled_gpu_vram.csv) and [memory provenance](sampled_gpu_vram_provenance.json)
retain the separate 54-call source641 process-memory study; polling gives lower
bounds on whole-run maxima, never benchmark timings. These are historical
measurements and must not be relabeled as source230 memory observations.

# Historical Daint setup study

Fixed checksummed extracts, not current-main performance. Times are milliseconds;
scaling uses three-repeat medians. The 189 scaling records run one iteration,
not converged solves. Historical A/B ratios divide current by the geometric mean
of old-before/old-after; below 1 is faster. Iteration deltas sum 27 matrix/seed cases.

Partition includes degree pruning, selection and collection. The tables retain
historical structural probes, not a current optimization recommendation.
All 54 historical-comparison cells converged at true residual≤1e-8; 207 scaling/probe
and 162 historical records had campaign completion/checksum receipts.

## Cumulative July-27 to current comparison

| threads | setup ratio | PCG ratio | total ratio | RSS ratio | factor nnz ratio | iteration delta |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.5384 | 0.3008 | 0.4250 | 0.8351 | 0.9788 | -43 |
| 72 | 0.4836 | 0.1935 | 0.3315 | 0.8070 | 0.9785 | -34 |

## Historical setup scaling

| threads | setup | partition phase | prune | IS selector | elimination | SpTRSV setup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x |
| 2 | 1.399x | 1.708x | 2.208x | 0.800x | 1.213x | 1.026x |
| 4 | 2.125x | 2.844x | 4.114x | 1.159x | 1.788x | 1.439x |
| 8 | 3.052x | 4.378x | 7.122x | 1.602x | 2.379x | 1.872x |
| 16 | 4.032x | 6.296x | 11.226x | 2.218x | 2.862x | 2.309x |
| 36 | 4.934x | 8.404x | 15.633x | 3.040x | 3.165x | 2.859x |
| 72 | 5.387x | 9.649x | 18.022x | 3.732x | 3.224x | 3.272x |

## Structural probe verdict

| matrix | T | singleton-tail work | imbalanced parallel work | worst parallel LPT |
|---|---:|---:|---:|---:|
| G3_circuit | 16 | 0.769% | 0.000% | 0.9985 |
| G3_circuit | 72 | 0.792% | 0.000% | 0.9951 |
| as-Skitter | 16 | 23.699% | 0.000% | 0.9951 |
| as-Skitter | 72 | 24.220% | 0.000% | 0.9943 |
| coPapersDBLP | 16 | 15.602% | 0.000% | 0.9688 |
| coPapersDBLP | 72 | 15.732% | 0.000% | 0.9653 |
| com-Amazon | 16 | 21.496% | 0.000% | 0.9991 |
| com-Amazon | 72 | 21.967% | 0.000% | 0.9984 |
| com-LiveJournal | 16 | 12.091% | 0.256% | 0.7538 |
| com-LiveJournal | 72 | 12.101% | 0.306% | 0.7291 |
| com-Orkut | 16 | 2.918% | 14.273% | 0.5795 |
| com-Orkut | 72 | 2.910% | 14.618% | 0.5757 |
| grid_500 | 16 | 3.574% | 0.000% | 0.9994 |
| grid_500 | 72 | 3.890% | 0.000% | 0.9956 |
| iter0040 | 16 | 0.072% | 0.000% | 0.9595 |
| iter0040 | 72 | 0.078% | 0.000% | 0.9652 |
| kron_g500-logn16 | 16 | 47.521% | 0.000% | 0.9556 |
| kron_g500-logn16 | 72 | 47.237% | 0.000% | 0.9475 |

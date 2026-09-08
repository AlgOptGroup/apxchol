# Coarse tree ordering: retired

The experiment retained exact weight ordering while accelerating high-degree
sorts with a stable radix prefix. It was retired on 2026-09-05: a roughly
1.3–1.4% pooled total-time improvement did not justify a second algorithm and
crossover threshold. Current sampling uses comparison sort by `(weight,
vertex)` and exact cumulative-weight lookup.

Historical Daint job 4552345 contained 900 records over five matrices,
three seeds, and T=36/72, comparing thresholds 128/256/512/1024 with the
then-current 2048. Quality counts matched across 30 matrix/thread/seed groups.
The retrospective pooled threshold-512 total ratios were 0.987 and 0.986.
Short-run stalls affected the preregistered aggregation, so the pooled view
must not silently replace its original verdict.

Approximate 16/20-bit weight grouping changed sampling or factor trajectories
and gave unstable outcomes. A three-range variant likewise did not earn its
extra implementation. Exactness alone was insufficient reason to retain a
marginal optimization.

The saved [pooled data](daint-pooled.csv), [summary data](daint-summary.csv),
and [text summary](daint-summary.txt) remain available. The historical
[bench.py](bench.py) compares existing candidate/control binaries:

```sh
python experiments/2026-08-28-tree-coarse-order/bench.py --help
```

Its default one-step solve measures setup plus one iteration, not convergence.
It forces adjacency interpretation, so use suitable adjacency inputs and
explicit candidate/control binaries. Reproducing the retired algorithm also
requires its historical source; the current binary no longer exposes it.

The [full historical record](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/experiments/2026-08-28-tree-coarse-order/README.md) supplies campaign details and qualifications.
No benchmark was rerun as part of this documentation summary.

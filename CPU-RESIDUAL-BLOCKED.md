# Deterministic blocked residual normalization

Integrated from `1a782c8d` (base `5d529ab6`). Importance sums use fixed
16,384-position blocks, summed in index order and folded in block order.
One persistent OpenMP team handles the initial sum and up to six updates.
Empty/single-block inputs retain the scalar path.

The same ordered input and floating-point environment produce identical
normalization at every thread count. Association differs from the legacy
serial fold; whole factors need not match across thread counts. Backbone
retention, conditional Bernoulli/HT sampling and final statistics are unchanged.

## Recorded validation

Daint job **4618482** passed 420 CPU tests, 40 normalization fixtures and
70 solve checks. Matched T72 setup fell **9.09% on LiveJournal** and **5.92%
on Orkut**. Orkut normalization fell 761.734→45.303 ms; complete setup fell
12.609→11.865 s.

The original broader acceptance predicate failed: grid RSS was unstable and
Orkut's total-time gain did not exceed control spread. Integration establishes
a bounded setup improvement, not a universal total-time or memory gain.

Subsequent jobs 4619381/4619429 checked the integrated CUDA build and measured
12 matrices at seven thread counts, with all 404 scaling solves converging
at true relative residual ≤`1e-8`. These results do not isolate normalization's
contribution; see [Daint](benchmarks/daint/README.md).

[Full original derivation and validation record](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/CPU-RESIDUAL-BLOCKED.md).

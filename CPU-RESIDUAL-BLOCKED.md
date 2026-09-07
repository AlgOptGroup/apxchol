# Deterministic blocked residual normalization

This private candidate starts from measured5d529ab6. It contains no collection,
static-row or exact-fixed-point variant. It accelerates the initial importance
sum and existing six normalization updates by changing their floating-point
association deliberately; the sampling formula and final statistics remain.

Fixed contiguous blocks of16,384 item positions are summed in index order,
then folded in block order. The partition includes backbone positions even
though their terms are skipped. It depends only on the immutable ordered
input, never on thread count. Empty/single-block inputs use no allocation or
OpenMP team and retain the old scalar arithmetic. Larger T1 inputs use the
same blocked association as T72. This is an implementation constant, not a
runtime knob or a chunk-size sweep.

Each block visits at most16,384*(sizeof(Item)+sizeof(double)) bytes of items
and cached importance. Orkut's recorded84,322,176items yield5,147partials,
41,176bytes. One reusable vector holds them through the initial and up-to-six
expected-value sums. All allocation precedes the parallel region. Block count
and endpoint formulas avoid size_t addition overflow; requested team is capped
atmin(omp_get_max_threads(),blocks) before conversion toint.

A single persistent OpenMP team spans all scans. Each worksharing barrier
publishes all partials before the single ordered fold; the single-region
barrier publishes scale and the continuation flag. Every thread makes the
same continuation decision. Smaller actual teams remain valid because ompfor
distributes block indices rather than assuming the requested team size.
The original target/mass/scale/expected predicates and six-pass cap remain.
No floating reduction clause, floating atomics, schedule-dependent fold,
approximate stop or new public graph method is introduced.

For fixed ordered inputs and the same FP environment, every block/fold/update
has the same arithmetic at every T. Existing preconditioner reductions do not
provide this stronger property: their per-thread chunks change withT, so their
partition was not reused. Whole factorization already changes elimination
order withT; cross-T normalization determinism is not a promise that whole
factors at differentT match.

The computed probabilities remainp=min(1,scale*sqrt(weight)), with exact
backbone retention and the same conditional Bernoulli/HT construction.
Legacy scale/probability/factor bits may differ because association changes.
Validation therefore compares the new algorithm acrossT exactly, but measures
A/B probability error and factor/iteration/fill differences rather than
requiring legacy bit identity. All original-system trueRR and full setup/total/
RSS gates remain necessary. Existing float storage/HT rounding contracts are
not replaced by a claim of exact real-arithmetic means for finite samples.

Six native test instances cover no-team boundaries, multiple blocks including
74blocks across1/2/4/72threads, repeats, a higher-precision scalar reference,
finite clipping and conditional HT expectation, uniform zero/nonfinite/
expected-underflow exits, and actual two-block graph sampling/connectivity
in both pool layouts acrossT1/4/72. Their code was independently reviewed;
compilation/execution is pending on Daint. The high-precision test explicitly
skips platforms where longdouble adds no mantissa precision; the GH200 gate
must not silently accept an unavailable reference.

The measured baseline Orkut diagnostic in4618241 attributes763.990ms to
normalization:109.738ms initial sum and654.251ms six scans. This is6.29% of
its same-call12.145522s setup. It is an upper bound on eliminable time, not a
promised gain. The former1.041s composite also contains importance preparation,
final statistics and reconstruction, all left unchanged here. LiveJournalT72
normalization134.818ms andSkitter6.416ms supply smaller endpoints; grid2000
executes none of this path. No new local numerical work was performed.

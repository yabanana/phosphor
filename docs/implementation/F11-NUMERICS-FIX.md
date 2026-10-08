# F11 numerical diagnostics reviewer correction — NON VERIFIED

No build/MSL/CPU/GPU tests executed. Git diff formatting only.

`diBRDF` no longer erases nonfinite arithmetic before its callers can see it.
Candidate targets preserve/report numeric errors; selected-sample and clustered
shade record raw-contribution, normalization/accumulation, invalid-light and
final-output faults into per-slot persistent counters BEFORE safe zero output.
The host checker reads those counters independently of the sanitized RGB texture.
Shade kernels additionally require diagnostics buffer15; graph read/write access
and a clear pass are part of the host package. New PSOs must be harvested.

`--debug-lighting-corrupt overflow` creates a finite very-bright near-point light
at a real receiver. Its finite source values overflow FP32 BRDF/inverse-square
arithmetic; zeroing output cannot make the diagnostic pass. The written CPU
negative contrasts finite double oracle, FP32 overflow and safe zero output.
Consistency checks still do NOT replace energetic brute/reference comparisons.

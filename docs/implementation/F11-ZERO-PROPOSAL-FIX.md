# F11 counted zero-domain proposals — NON VERIFIED

No tests/build/MSL/GPU execution. Fixed M normalization in CPU and GPU helpers
AND temporal/spatial prefilters. GPUDIReservoir.pad[1] now separates a counted
proposal-domain record from a selected positive endpoint. A same-epoch empty
reservoir whose degenerate/invalid endpoints legitimately contributed M can
merge that M with zero stream weight. It never dereferences an invalid light ID.
Disocclusion/view/material/light-domain invalidation still rejects the history.

The positive target floor remains for VALID endpoint support; it cannot pretend
a degenerate emitter sample was selected. Selected reservoirs keep their prior
ID/generation checks. Independent four-outcome Bernoulli tests expect0.5 and show
the prior drop-zero rule's0.625. These authored tests were NOT EXECUTED.

# F12 geometric/radiometric epoch reviewer correction — NON VERIFIED

GPUProbeGridParams.generation now denotes geometric probe state only. A new
geometry epoch clears offsets/classification/travel as before. cacheGeneration
denotes the radiometric cache/history epoch; reset invalidates radiance/atlas
history but must not clear geometric offsets.

giProbePosition ignores reset when choosing an existing offset. Classification
clears the complete state only for a mismatched geometry epoch; a radiometric
reset clears active age while preserving the offset and learned classification.
ProbeGrid.invalidateRadiance mirrors that operation and clears CPU irradiance/
moments without changing geometric generation, classification or relocation.

The written test uses an independent analytic slab intersection oracle. The
probe requires multiple0.1m steps to escape a0.9m wall while radiance changes each
frame. Radiometric invalidation preserves escape progress; the old complete
reset keeps it inside indefinitely. Explicit geometry reset still zeros offsets.

Root integration patch is docs/patches/f12-probe-epochs.patch, generated against
review branch3f9ffd9. It separates exact geometry and radiometric tuples,
passes geometric epoch to probe parameters, radiometric epoch to cache/history,
and leaves full-light/environment invalidation in the radiometric tuple.
This patch is an unapplied review artifact; no root files were edited.
No compiler/test/GPU execution occurred. ROADMAP remains unchanged.

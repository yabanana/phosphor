# F12 material-facing reviewer correction — NON VERIFIED

F9 GPURtHit.frontFacing remains the WORLD geometric winding, including negative
determinant transforms. The F12 probe classifier and GI candidate previously
used it as a material-sidedness test. An X-reflected, outside-facing single-sided
triangle has WORLD cross pointing away from the external ray while the original
material front still faces it. The old DDGI path marked that probe ray backfacing,
and the old GI candidate path discarded its radiance.

renderer/rt_geometry.h converts WORLD facing to original material facing by
XOR with INSTANCE_FLAG_MIRRORED. rt_common.h exposes a range-checked shader
overload; DDGI and GI use it only for sided material decisions. WORLD normals
remain geometric and ray offsets retain their original F9 semantics.

test_gi.cpp includes an independent double-precision triangle/cross/ray oracle
for external/internal sides under an X reflection. The old WORLD-facing-as-
material rule disagrees with that oracle as a negative control.

Written only: no compilation, MSL, test or GPU execution. Root adds rt_geometry.h
to shader dependency tracking, then tests mirrored/nonmirrored single-sided and
double-sided geometry, DDGI outside/inside classification and GI energy against
full geometry/reference. No F9 hit ABI or roadmap status changed.

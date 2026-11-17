// F15 closure round 4b (ADR-0007): the base probe-only kernel compiled with the multiple-scattering term lit by the probe radiance ("reflection_probe_only_ms"),
// selected by the engine when --specular-ms is on and the scene has no lobe guide (the pair of reflection_composite_ms).  The base kernels stay byte for byte.
#define PHOSPHOR_REFL_MS 1
#include "reflection_probes.metal"

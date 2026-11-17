// F15.1 (ticket 08): the specular-transport and composite kernels compiled with the lobe guide ("<name>_lobe",
// function constant MATERIAL_FEATURES = the scene feature union).  Same sources as reflections.metal,
// reflection_probes.metal and reflection_composite.metal; see reflection_common.h.  Linked into the material module
// (material_passes.metal) after the DI lobe kernels, which put restir_common.h in lobe mode.
#define PHOSPHOR_REFL_LOBE 1
#include "reflections.metal"
#include "reflection_probes.metal"
#include "reflection_composite.metal"

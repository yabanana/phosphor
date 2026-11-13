// One material-shading translation unit, linked first in the metallib.
// Metal toolchain 27.1 fails to relocate private argument-buffer metadata
// across the separate forward/resolve AIR modules during metal-tt (F7/F8).
// Keep the source passes separate for editing; share their compiled types,
// samplers and function constants. See docs/plans/F7-F8-EXECUTION.md.
#include "forward.metal"
#include "visibility_resolve.metal"
// F15.1 (ticket 07): the lobe-guide DI kernels (specialised by MATERIAL_FEATURES) join this module, like forward and
// resolve, for the same metal-tt relocation hazard.  Their shared helpers have distinct "_lobe" symbols (DI_FN in
// restir_common.h): same-named inline functions with different bodies in two modules would be merged at link time.
#include "restir_di_lobe.metal"
#include "light_cluster_lobe.metal"
// F15.1 (ticket 08): the specular transport and composite lobe kernels (same module reasons).
#include "reflection_lobe.metal"

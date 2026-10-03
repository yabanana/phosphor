// One material-shading translation unit, linked first in the metallib.
// Metal toolchain 27.1 fails to relocate private argument-buffer metadata
// across the separate forward/resolve AIR modules during metal-tt (F7/F8).
// Keep the source passes separate for editing; share their compiled types,
// samplers and function constants. See docs/plans/F7-F8-EXECUTION.md.
#include "forward.metal"
#include "visibility_resolve.metal"

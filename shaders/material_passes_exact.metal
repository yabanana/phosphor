// F15.2 (ticket 21): the exact-maths copy of the visibility resolve kernels, used only by scenes with F15 material features (the
// MATERIAL_FEATURES / RESOLVE_BIN constants are defined), so that a class pipeline and the all-lobes pipeline are bit-identical.
// Featureless scenes keep the kernels of material_passes.metal (their cost and images are untouched).
//
// Compiled by cmake/App.cmake with -DPHOSPHOR_EXACT_MODULE=1 -fmetal-math-mode=safe (the module flag clears air.compile.fast_math_enable,
// which the GPU compiler otherwise applies on its own; source pragmas alone leave textured scenes 1e-2 off, measured).  Everything lives in
// namespace phx_exact, so the kernel names are "phx_exact::visibility_resolve" and no inline function can be merged with the base module's
// same-named, differently compiled copy at link time.
#include <metal_stdlib>
namespace phx_exact {
#include "visibility_resolve.metal"
}

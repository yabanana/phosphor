// F15 closure (ADR-0007): the base composite kernel compiled with the multiple-scattering lobe ("reflection_composite_ms"), selected by the engine when
// --specular-ms is on and the scene has no lobe guide.  The base kernels stay byte for byte what they were (a runtime branch inside them changed the
// compiler's association of the products in a few pixels, measured): the option off keeps the old code, the option on runs this variant.
#define PHOSPHOR_REFL_MS 1
#include "reflection_composite.metal"

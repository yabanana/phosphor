// F15.1 (ticket 08): the specular denoiser kernels compiled with the lobe contract ("<name>_lobe", function constants
// DENOISE_SIGNAL (40) = specular and MATERIAL_FEATURES (22) = the scene feature union).  Same source as denoise.metal; see the
// comment at its top.  Linked into the material module (material_passes.metal) after the DI lobe kernels, which put
// restir_common.h in lobe mode.
#define PHOSPHOR_DENOISE_LOBE 1
#include "denoise.metal"

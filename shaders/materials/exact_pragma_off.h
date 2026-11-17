// F15.2 (ticket 21): bit-exact material shading.  Active ONLY in the exact module (shaders/material_passes_exact.metal, compiled with
// PHOSPHOR_EXACT_MODULE and -fmetal-math-mode=safe); in the base module this file is empty, so featureless scenes keep today's code.
#ifdef PHOSPHOR_EXACT_MODULE
#pragma METAL fp math_mode(fast)
#pragma METAL fp contract(fast)
#endif

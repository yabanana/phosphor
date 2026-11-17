// F15.2 (ticket 21): bit-exact material shading.  Active ONLY in the exact module (shaders/material_passes_exact.metal, compiled with
// PHOSPHOR_EXACT_MODULE and -fmetal-math-mode=safe); in the base module this file is empty, so featureless scenes keep today's code.
#ifdef PHOSPHOR_EXACT_MODULE
// F15.2 (ticket 21): closes the region opened by exact_math_begin.h and restores Metal's default maths for the code that follows.
// Only the inner namespace is exported: the precise using-declarations stay private to the region (exporting them would make every
// unqualified sqrt/length in the other kernels ambiguous with metal's fast overloads).
}
}
using namespace phx_math::code;
#pragma METAL fp math_mode(fast)
#pragma METAL fp contract(fast)
#endif

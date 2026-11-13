#pragma once
// ---------------------------------------------------------------------------
// F14.2 -- fog in-scattering of the DDGI field from six irradiance orientations
// (+x, -x, +y, -y, +z, -z), one colour channel at a time.  Compiled by C++
// (unit tests) and by MSL from this single source.
//
// Lambertian irradiance E(n) keeps the order-0, order-1 and even orders >= 2
// of the incident radiance L(w) (w towards the source); odd orders above 1
// vanish.  Therefore, exactly:
//   c   = (sum of the six E) / (6 pi)   the order-0 radiance (order-2 terms
//                                        cancel over the three axes),
//   b_k = 3 / (4 pi) (E(+k) - E(-k))    the order-1 vector.
// Henyey-Greenstein scattering scales order l by g^l, so with the view ray d
// (camera to froxel) the in-scattered radiance  c + g (b . d)  is exact for
// radiance up to order 1.  What remains is g^2 times the order-2 part (and
// the order-4 part leaking into c); g = 0 gives the isotropic mean that the
// former E/(6 pi) injection used.
// ---------------------------------------------------------------------------
#include "renderer/gpu_types.h"
#ifdef __METAL_VERSION__
#define PHOSPHOR_FOG_GI_FN static inline __attribute__((unused))
#else
#define PHOSPHOR_FOG_GI_FN inline
#endif

namespace phosphor {
PHOSPHOR_FOG_GI_FN float fogGiInScatter(float ePosX, float eNegX, float ePosY, float eNegY, float ePosZ, float eNegZ,
                                        float rayX, float rayY, float rayZ, float anisotropy) {
    const float pi = 3.14159265358979323846f;
    const float mean = (ePosX + eNegX + ePosY + eNegY + ePosZ + eNegZ) / (6.0f * pi);
    const float dipole = (3.0f / (4.0f * pi)) * ((ePosX - eNegX) * rayX + (ePosY - eNegY) * rayY + (ePosZ - eNegZ) * rayZ);
    // A narrow source with g > 1/3 can push the order-1 estimate below zero
    // behind it; radiance cannot be negative.  NaN passes (v < 0 is false) so
    // the injection's non-finite counter still sees it.
    const float v = mean + anisotropy * dipole;
    return v < 0.0f ? 0.0f : v;
}
} // namespace phosphor

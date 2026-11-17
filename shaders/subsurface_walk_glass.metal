// F15.1 (ticket 14, round 3): the RT subsurface walk compiled with the full RtPayload and a transmittance-aware exit shadow ray (see
// subsurface_walk.metal).  Linked with rt_alpha_generic; requested only while the scene feature union has transmission.
#define PHOSPHOR_SSS_GLASS 1
#include "subsurface_walk.metal"

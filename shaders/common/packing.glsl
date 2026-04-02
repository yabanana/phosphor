#ifndef PACKING_GLSL
#define PACKING_GLSL

// ---------------------------------------------------------------------------
// Visibility buffer packing
//
// Format: R32_UINT
//   Bits [31:22] = instanceID  (10 bits, up to 1024 instances)
//   Bits [21:10] = meshletID   (12 bits, up to 4096 meshlets per instance)
//   Bits  [9:0]  = localTriID  (10 bits, up to 1024 triangles per meshlet)
//
// Value 0xFFFFFFFF is reserved as "no geometry" (clear value).
// ---------------------------------------------------------------------------

#define VISIBILITY_CLEAR 0xFFFFFFFFu

uint encodeVisibility(uint instanceID, uint meshletID, uint localTriID) {
    return (instanceID << 22) | (meshletID << 10) | localTriID;
}

struct VisData {
    uint instanceID;
    uint meshletID;
    uint localTriID;
};

VisData decodeVisibility(uint encoded) {
    VisData d;
    d.instanceID = encoded >> 22;
    d.meshletID  = (encoded >> 10) & 0xFFFu;
    d.localTriID = encoded & 0x3FFu;
    return d;
}

#endif // PACKING_GLSL

#ifndef PACKING_GLSL
#define PACKING_GLSL

// ---------------------------------------------------------------------------
// Visibility buffer packing
//
// Format: R32_UINT
//   Bits [31:16] = instanceID (up to 65536 instances)
//   Bits [15:0]  = triangleID (up to 65536 triangles per meshlet draw)
//
// Value 0xFFFFFFFF is reserved as "no geometry" (clear value).
// ---------------------------------------------------------------------------

const uint VISIBILITY_CLEAR = 0xFFFFFFFF;

uint encodeVisibility(uint instanceID, uint triangleID) {
    return (instanceID << 16) | (triangleID & 0xFFFF);
}

uvec2 decodeVisibility(uint encoded) {
    uint instanceID = encoded >> 16;
    uint triangleID = encoded & 0xFFFF;
    return uvec2(instanceID, triangleID);
}

#endif // PACKING_GLSL

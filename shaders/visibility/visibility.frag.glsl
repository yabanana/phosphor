#version 460

#extension GL_EXT_mesh_shader : require

#include "packing.glsl"

// ---------------------------------------------------------------------------
// Visibility buffer fragment shader
//
// Writes instanceID + gl_PrimitiveID into an R32_UINT render target.
// Paired with the mesh shader which provides per-primitive instanceID.
// ---------------------------------------------------------------------------

// Per-primitive input from mesh shader (flat / not interpolated)
layout(location = 4) perprimitiveEXT flat in uint inInstanceID;

// Output: packed visibility ID
layout(location = 0) out uint outVisibility;

void main() {
    outVisibility = encodeVisibility(inInstanceID, uint(gl_PrimitiveID));
}

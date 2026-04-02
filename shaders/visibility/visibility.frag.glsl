#version 460

#extension GL_EXT_mesh_shader : require

#include "packing.glsl"

// ---------------------------------------------------------------------------
// Visibility buffer fragment shader
//
// Writes instanceID + meshletID + localTriID into an R32_UINT render target.
// Paired with the mesh shader which provides per-primitive instanceID and
// meshletID.  gl_PrimitiveID in the fragment shader is local to the mesh
// shader's SetMeshOutputsEXT — it is 0-based per workgroup (per-meshlet),
// so it IS the local triangle ID already.
// ---------------------------------------------------------------------------

// Per-primitive inputs from mesh shader (flat / not interpolated)
layout(location = 4) perprimitiveEXT flat in uint inInstanceID;
layout(location = 5) perprimitiveEXT flat in uint inMeshletID;

// Output: packed visibility ID
layout(location = 0) out uint outVisibility;

void main() {
    outVisibility = encodeVisibility(inInstanceID, inMeshletID, uint(gl_PrimitiveID));
}

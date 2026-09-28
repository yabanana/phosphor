#version 460

// ---------------------------------------------------------------------------
// Fullscreen triangle vertex shader
//
// Draws a single triangle covering the entire screen using 3 vertices
// (vertex IDs 0, 1, 2). No vertex buffer needed — positions and UVs are
// generated from gl_VertexIndex.
//
// Usage: vkCmdDraw(cmd, 3, 1, 0, 0)
// ---------------------------------------------------------------------------

layout(location = 0) out vec2 outUV;

void main() {
    // Generate fullscreen triangle coordinates from vertex index:
    //   ID 0 -> (-1, -1)  uv (0, 0)
    //   ID 1 -> ( 3, -1)  uv (2, 0)
    //   ID 2 -> (-1,  3)  uv (0, 2)
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    outUV = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}

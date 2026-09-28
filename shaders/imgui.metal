// imgui.metal -- Dear ImGui overlay, drawn at the end of the forward pass.
//
// Bindings (Metal 4 argument table):
//   buffer(0)   ImGuiVertex[]   (this frame's vertices; draws add a base vertex)
//   buffer(1)   ImGuiUniforms
//   texture(0)  the draw command's texture (font atlas)

#include <metal_stdlib>

using namespace metal;

// Mirrors ImDrawVert: packed types keep it at 20 bytes (float2 would pad to 24).
struct ImGuiVertex {
    packed_float2 position;
    packed_float2 uv;
    uint          color; // RGBA8, R in the low byte
};

struct ImGuiUniforms {
    float4x4 projection;
};

struct ImGuiVaryings {
    float4 position [[position]];
    float2 uv;
    half4  color;
};

// ImGui colours are authored in sRGB and the drawable is an *_sRGB format that
// encodes on write, so decode here to keep the UI looking as designed.
static half3 srgbToLinear(half3 c) {
    const half3 lo = c / 12.92h;
    const half3 hi = pow((c + 0.055h) / 1.055h, half3(2.4h));
    return select(hi, lo, c <= 0.04045h);
}

vertex ImGuiVaryings imgui_vs(uint vertexId                     [[vertex_id]],
                              const device ImGuiVertex* vertices [[buffer(0)]],
                              constant ImGuiUniforms& uniforms   [[buffer(1)]])
{
    const ImGuiVertex v = vertices[vertexId];
    const half4 color = unpack_unorm4x8_to_half(v.color);

    ImGuiVaryings out;
    out.position = uniforms.projection * float4(float2(v.position), 0.0, 1.0);
    out.uv       = float2(v.uv);
    out.color    = half4(srgbToLinear(color.rgb), color.a);
    return out;
}

fragment half4 imgui_fs(ImGuiVaryings in      [[stage_in]],
                        texture2d<half> image [[texture(0)]])
{
    constexpr sampler linearClamp(filter::linear, address::clamp_to_edge);
    return in.color * image.sample(linearClamp, in.uv);
}

// Indexed forward reference. The visibility resolve shares material semantics.
#include "material_shading.h"
#include "temporal_motion.h"

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float4 tangent;
    float2 uv;
    uint   materialIndex [[flat]];
    uint   mirrored [[flat]]; // 1 if INSTANCE_FLAG_MIRRORED
};

static float4x4 loadMatrix(const device float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

static float4x4 loadMatrix(constant float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

vertex VertexOut forward_vs(uint vertexId                          [[vertex_id]],
                            uint instanceId                        [[instance_id]],
                            constant FrameConstants& frame         [[buffer(0)]],
                            const device GPUVertex* vertices       [[buffer(1)]],
                            const device GPUInstance* instances    [[buffer(2)]],
                            const device uint* visible             [[buffer(6)]])
{
    // vertex_id includes the draw's base vertex and instance_id includes the
    // base instance: instance_id indexes the visible list, which names the slot.
    const device GPUVertex& v    = vertices[vertexId];
    const device GPUInstance& gi = instances[visible[instanceId]];

    const float4x4 model = loadMatrix(gi.modelMatrix);
    const float3x3 normalMatrix = float3x3(model[0].xyz, model[1].xyz, model[2].xyz);

    const float4 world = model * float4(v.px, v.py, v.pz, 1.0);

    VertexOut out;
    out.position      = loadMatrix(frame.viewProjection) * world;
    out.worldPos      = world.xyz;
    // Inverse-transpose direction, normalized after interpolation in the FS.
    out.normal = surfaceNormal(model, float3(v.nx, v.ny, v.nz));
    out.tangent =
        float4(normalMatrix * float3(v.tx, v.ty, v.tz), v.tw * ((gi.flags & INSTANCE_FLAG_MIRRORED) ? -1.0f : 1.0f));
    out.uv            = float2(v.u, v.v);
    out.materialIndex = gi.materialIndex;
    out.mirrored      = (gi.flags & INSTANCE_FLAG_MIRRORED) != 0 ? 1u : 0u;
    return out;
}

fragment half4 forward_fs(VertexOut in [[stage_in]], bool frontFacing [[front_facing]],
                          constant FrameConstants &frame [[buffer(0)]],
                          const device GPUMaterial *materials [[buffer(3)]],
                          const device GPULight *lights [[buffer(4)]],
                          const device TextureHandle *textures [[buffer(5)]]) {
#ifdef PHOSPHOR_HOT_RELOAD_PROBE
    return half4(1.0h, 0.0h, 1.0h, 1.0h);
#endif
    if (is_function_constant_defined(FC_SALT) && frame.lightCount == 0xFFFFFFFFu)
        return half4(half(float(FC_SALT)));
    SurfaceInput surface{in.worldPos, in.normal,        in.tangent,  in.uv,      dfdx(in.uv),
                         dfdy(in.uv), in.materialIndex, in.mirrored, frontFacing};
    const ShadingResult value = shadeSurface(surface, frame, materials, lights, textures);
    if (value.alpha < materials[in.materialIndex].alphaCutoff)
        discard_fragment();
    const uint debugMode = kDebugModeSpecialised ? kDebugMode : frame.debugMode;
    if (debugMode == 1)
        return half4(half3(value.normal * 0.5f + 0.5f), 1.0h);
    if (debugMode == 2)
        return half4(half3(value.baseColor), 1.0h);
    return half4(half3(tonemapACES(value.color * frame.exposure)), 1.0h);
}

struct SurfaceVertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float4 tangent;
    float2 uv;
    uint materialIndex [[flat]];
    uint mirrored [[flat]]; // 1 if INSTANCE_FLAG_MIRRORED
    float4 unjitteredClip;
    float4 previousClip;
    uint historyValid [[flat]];
};

vertex SurfaceVertexOut forward_surface_vs(uint vertexId [[vertex_id]], uint instanceId [[instance_id]],
                                           constant FrameConstants &frame [[buffer(0)]],
                                           const device GPUVertex *vertices [[buffer(1)]],
                                           const device GPUInstance *instances [[buffer(2)]],
                                           const device uint *visible [[buffer(6)]],
                                           const device GPUInstance *previous [[buffer(7)]],
                                           constant GPUTemporalParams &temporal [[buffer(8)]]) {
    // vertex_id includes the draw's base vertex and instance_id includes the
    // base instance: instance_id indexes the visible list, which names the slot.
    const device GPUVertex &v = vertices[vertexId];
    const device GPUInstance &gi = instances[visible[instanceId]];

    const float4x4 model = loadMatrix(gi.modelMatrix);
    const float3x3 normalMatrix = float3x3(model[0].xyz, model[1].xyz, model[2].xyz);

    const float4 world = model * float4(v.px, v.py, v.pz, 1.0);

    SurfaceVertexOut out;
    out.position = loadMatrix(frame.viewProjection) * world;
    out.worldPos = world.xyz;
    // Inverse-transpose direction, normalized after interpolation in the FS.
    out.normal = surfaceNormal(model, float3(v.nx, v.ny, v.nz));
    out.tangent =
        float4(normalMatrix * float3(v.tx, v.ty, v.tz), v.tw * ((gi.flags & INSTANCE_FLAG_MIRRORED) ? -1.0f : 1.0f));
    out.uv = float2(v.u, v.v);
    out.materialIndex = gi.materialIndex;
    out.mirrored = (gi.flags & INSTANCE_FLAG_MIRRORED) != 0 ? 1u : 0u;
    out.unjitteredClip = temporalMatrix(temporal.currentViewProjection) * world;
    out.previousClip = out.unjitteredClip;
    out.historyValid = 0;
    if (temporal.historyValid) {
        const device GPUInstance &old = previous[visible[instanceId]];
        out.historyValid = old.generation == gi.generation && (old.flags & INSTANCE_FLAG_VALID);
        if (out.historyValid)
            out.previousClip = temporalMatrix(temporal.previousViewProjection) * loadMatrix(old.modelMatrix) *
                               float4(v.px, v.py, v.pz, 1);
    }
    return out;
}

// F7 overflow fallback: the existing GPU-built indexed ICB writes the same
// linear lighting and guide contract as the visibility resolve.
struct ForwardSurfaceOutput {
    half4 color [[color(0)]];
    half4 normalRoughness [[color(1)]];
    half4 diffuseAlbedo [[color(2)]];
    half4 specularAlbedo [[color(3)]];
    half2 motion [[color(4)]];
    half reactive [[color(5)]];
};
fragment ForwardSurfaceOutput forward_surface_fs(SurfaceVertexOut in [[stage_in]], bool frontFacing [[front_facing]],
                                                 constant FrameConstants &frame [[buffer(0)]],
                                                 const device GPUMaterial *materials [[buffer(3)]],
                                                 const device GPULight *lights [[buffer(4)]],
                                                 const device TextureHandle *textures [[buffer(5)]],
                                                 constant GPUTemporalParams &temporal [[buffer(8)]]) {
    SurfaceInput surface{in.worldPos,
                         in.normal,
                         in.tangent,
                         in.uv,
                         dfdx(in.uv) * exp2(temporal.mipBias),
                         dfdy(in.uv) * exp2(temporal.mipBias),
                         in.materialIndex,
                         in.mirrored,
                         frontFacing};
    const ShadingResult value = shadeSurface(surface, frame, materials, lights, textures);
    if (value.alpha < materials[in.materialIndex].alphaCutoff)
        discard_fragment();
    ForwardSurfaceOutput out;
    const uint debugMode = kDebugModeSpecialised ? kDebugMode : frame.debugMode;
    out.color = half4(half3(debugMode == 1   ? value.normal * 0.5f + 0.5f
                            : debugMode == 2 ? value.baseColor
                                             : value.color),
                      1.0h);
    out.normalRoughness = half4(half3(value.normal), half(value.roughness));
    out.diffuseAlbedo = half4(half3(value.diffuseAlbedo), 1.0h);
    out.specularAlbedo = half4(half3(value.specularAlbedo), 1.0h);
    out.motion = half2(temporalMotion(in.unjitteredClip, in.previousClip, temporal, in.historyValid != 0));
    out.reactive = in.historyValid == 0 ? 1.0h : materials[in.materialIndex].alphaCutoff > 0 ? 0.75h : 0.0h;
    return out;
}

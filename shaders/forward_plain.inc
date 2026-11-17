// F15 closure (COLOR_0): the plain indexed-forward stages, instantiated twice by forward.metal (PLN_VC = 0 the original entry points, 1 the
// _vc twins with the colour interpolant).

struct PLN_OUT {
    float4 position [[position, invariant]];
    float3 worldPos;
    float3 normal;
    float4 tangent;
    float2 uv;
    uint   materialIndex [[flat]];
    uint   mirrored [[flat]]; // 1 if INSTANCE_FLAG_MIRRORED
#if PLN_VC
    float4 color; // F15 closure (COLOR_0), the _vc entry points only
#endif
};


vertex PLN_OUT PLN_NAME(forward_vs)(uint vertexId                          [[vertex_id]],
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

    PLN_OUT out;
    out.position      = loadMatrix(frame.viewProjection) * world;
    out.worldPos      = world.xyz;
    // Inverse-transpose direction, normalized after interpolation in the FS.
    out.normal = surfaceNormal(model, float3(v.nx, v.ny, v.nz));
    out.tangent =
        float4(normalMatrix * float3(v.tx, v.ty, v.tz), v.tw * ((gi.flags & INSTANCE_FLAG_MIRRORED) ? -1.0f : 1.0f));
    out.uv            = float2(v.u, v.v);
    out.materialIndex = gi.materialIndex;
    out.mirrored      = (gi.flags & INSTANCE_FLAG_MIRRORED) != 0 ? 1u : 0u;
#if PLN_VC
    out.color         = frame.vertexColors[vertexId];
#endif
    return out;
}

// Preserve FP32 through the sRGB conversion. A half output lets specialization
// introduce mixed-precision/double-rounding differences (F3 variant check).
fragment float4 PLN_NAME(forward_fs)(PLN_OUT in [[stage_in]], bool frontFacing [[front_facing]],
                           constant FrameConstants &frame [[buffer(0)]],
                           const device GPUMaterial *materials [[buffer(3)]],
                           const device GPULight *lights [[buffer(4)]],
                           const device TextureHandle *textures [[buffer(5)]]) {
#ifdef PHOSPHOR_HOT_RELOAD_PROBE
    return float4(1.0f, 0.0f, 1.0f, 1.0f);
#endif
    if (is_function_constant_defined(FC_SALT) && frame.lightCount == 0xFFFFFFFFu)
        return float4(float(FC_SALT));
    SurfaceInput surface{in.worldPos, in.normal,        in.tangent,  in.uv,      dfdx(in.uv),
                         dfdy(in.uv), in.materialIndex, in.mirrored, frontFacing};
#if PLN_VC
    surface.vertexColor = in.color;
#endif
    const ShadingResult value = shadeSurface(surface, frame, materials, lights, textures);
    if (value.alpha < materials[in.materialIndex].alphaCutoff)
        discard_fragment();
    const uint debugMode = kDebugModeSpecialised ? kDebugMode : frame.debugMode;
    if (debugMode == 1)
        return float4(value.normal * 0.5f + 0.5f, 1.0f);
    if (debugMode == 2)
        return float4(value.baseColor, 1.0f);
    return float4(tonemapACES(value.color * frame.exposure), 1.0f);
}


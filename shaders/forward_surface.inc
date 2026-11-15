// F15 closure (COLOR_0): the indexed-path surface stages, instantiated twice by forward.metal.  SFV_VC = 0: the original entry points (no vertex
// colour anywhere: byte-stable); SFV_VC = 1: the same code with the colour interpolant and the entry points suffixed _vc.  The text is the
// former tail of forward.metal with SurfaceVertexOut / the entry names parameterised and the colour lines under SFV_VC.

struct SFV_OUT {
    float4 position [[position, invariant]];
    float3 worldPos;
    float3 normal;
    float4 tangent;
    float2 uv;
    uint materialIndex [[flat]];
    uint mirrored [[flat]]; // 1 if INSTANCE_FLAG_MIRRORED
    float4 unjitteredClip;
    float4 previousClip;
    uint historyValid [[flat]];
    uint instanceSlot [[flat]];
    float volumeScale [[flat]]; // F15.1 (ticket 13): min column norm of the model matrix (KHR_materials_volume thickness in world units)
#if SFV_VC
    float4 color; // F15 closure (COLOR_0): the vertex colour (the _vc entry points only)
#endif
};

vertex SFV_OUT SFV_NAME(forward_surface_vs)(uint vertexId [[vertex_id]], uint instanceId [[instance_id]],
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

    SFV_OUT out;
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
    out.instanceSlot = visible[instanceId];
    out.volumeScale = min(length(model[0].xyz), min(length(model[1].xyz), length(model[2].xyz)));
#if SFV_VC
    out.color = frame.vertexColors[vertexId];
#endif
    if (temporal.historyValid) {
        const device GPUInstance &old = previous[visible[instanceId]];
        out.historyValid = old.generation == gi.generation && (old.flags & INSTANCE_FLAG_VALID);
        if (out.historyValid)
            out.previousClip = temporalMatrix(temporal.previousViewProjection) * loadMatrix(old.modelMatrix) *
                               float4(v.px, v.py, v.pz, 1);
    }
    return out;
}

// F10 receiver prepass: the indexed overflow ICB fills receivers BEFORE RT.
// No background V-buffer mask is ever used to shade an overflow receiver.
fragment ShadowReceiverOutput SFV_NAME(forward_shadow_receiver_fs)(
    SFV_OUT in [[stage_in]], uint primitive [[primitive_id]],
    constant FrameConstants& frame [[buffer(0)]],
    const device GPUInstance* instances [[buffer(2)]],
    const device GPUMaterial* materials [[buffer(3)]],
    const device TextureHandle* textures [[buffer(5)]],
    constant GPUTemporalParams& temporal [[buffer(8)]]) {
    const device auto& material = materials[in.materialIndex];
    float alpha = material.baseColor[3] * float(sampleOr(textures, material.baseColorTex, in.uv,
                                                             half4(1), dfdx(in.uv)*exp2(temporal.mipBias),
                                                             dfdy(in.uv)*exp2(temporal.mipBias)).a);
    {   // F15.4 (ticket 23): a layer stack's coverage (own alpha x the blended layer alpha), tested against the stack's cutoff
        const float2 sdx = dfdx(in.uv) * exp2(temporal.mipBias), sdy = dfdy(in.uv) * exp2(temporal.mipBias);
        alpha *= layerCoverage<TextureHandle, LayerRasterSamples>(material, materials, textures, in.uv, sdx, sdy, temporal.materialExtensions);
    }
#if SFV_VC
    if (alpha * in.color.a < material.alphaCutoff) discard_fragment(); // F15 closure (COLOR_0 alpha)
#endif
    if (alpha < material.alphaCutoff) discard_fragment();
    float3 n = cross(dfdx(in.worldPos), dfdy(in.worldPos));
    if (dot(n,n) < 1e-20f) discard_fragment();
    const float3 eye(frame.cameraPosition[0],frame.cameraPosition[1],frame.cameraPosition[2]);
    if(dot(n,eye-in.worldPos)<0.0f)n=-n;
    return {float4(in.worldPos, 1), half4(half3(normalize(n)), 1),
            uint4(in.instanceSlot, instances[in.instanceSlot].generation, primitive, in.materialIndex)};
}

// F15.1 (ticket 22): the same prepass for a scene with per-slot KHR_texture_transform (the base colour slot's transform in the alpha test);
// a separate entry point because metal-tt cannot lower an unspecialised descriptor whose function references a function constant.
fragment ShadowReceiverOutput SFV_NAME(forward_shadow_receiver_tt_fs)(
    SFV_OUT in [[stage_in]], uint primitive [[primitive_id]],
    constant FrameConstants& frame [[buffer(0)]],
    const device GPUInstance* instances [[buffer(2)]],
    const device GPUMaterial* materials [[buffer(3)]],
    const device TextureHandle* textures [[buffer(5)]],
    constant GPUTemporalParams& temporal [[buffer(8)]]) {
    const device auto& material = materials[in.materialIndex];
    float2 uv = in.uv, dx = dfdx(in.uv)*exp2(temporal.mipBias), dy = dfdy(in.uv)*exp2(temporal.mipBias);
    texXfUvOn(texXfForOn(frame.materialExtensions, material.flags, material.textureTransformIndex), TT_SLOT_BASE_COLOR, uv, dx, dy);
    float alpha = material.baseColor[3] * float(sampleOr(textures, material.baseColorTex, uv, half4(1), dx, dy).a);
    {   // F15.4 (ticket 23): a layer stack's coverage (own alpha x the blended layer alpha), tested against the stack's cutoff
        const float2 sdx = dfdx(in.uv) * exp2(temporal.mipBias), sdy = dfdy(in.uv) * exp2(temporal.mipBias);
        alpha *= layerCoverage<TextureHandle, LayerRasterSamples>(material, materials, textures, in.uv, sdx, sdy, temporal.materialExtensions);
    }
#if SFV_VC
    if (alpha * in.color.a < material.alphaCutoff) discard_fragment(); // F15 closure (COLOR_0 alpha)
#endif
    if (alpha < material.alphaCutoff) discard_fragment();
    float3 n = cross(dfdx(in.worldPos), dfdy(in.worldPos));
    if (dot(n,n) < 1e-20f) discard_fragment();
    const float3 eye(frame.cameraPosition[0],frame.cameraPosition[1],frame.cameraPosition[2]);
    if(dot(n,eye-in.worldPos)<0.0f)n=-n;
    return {float4(in.worldPos, 1), half4(half3(normalize(n)), 1),
            uint4(in.instanceSlot, instances[in.instanceSlot].generation, primitive, in.materialIndex)};
}

// F7 overflow fallback: the existing GPU-built indexed ICB writes the same
// linear lighting and guide contract as the visibility resolve.
fragment ForwardSurfaceOutput SFV_NAME(forward_surface_fs)(SFV_OUT in [[stage_in]], bool frontFacing [[front_facing]],
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
                         frontFacing,
                         in.volumeScale};
#if SFV_VC
    surface.vertexColor = in.color; // F15 closure (COLOR_0)
#endif
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
    out.reactive = in.historyValid == 0 ? 1.0h : (materials[in.materialIndex].alphaCutoff > 0 || (materials[in.materialIndex].flags & (MATERIAL_FEATURE_TRANSMISSION << MATERIAL_FEATURE_SHIFT)) != 0) ? 0.75h : 0.0h;
    if (kSssWrite) out.sss = float4(value.sssMaterial != 0u ? value.sssIrradiance : float3(0.0f), sssCodeOf(value.sssMaterial));
    return out;
}

fragment ForwardLitSurfaceOutput SFV_NAME(forward_surface_lit_fs)(SFV_OUT in [[stage_in]], bool frontFacing [[front_facing]],
                                                 constant FrameConstants &frame [[buffer(0)]],
                                                 const device GPUMaterial *materials [[buffer(3)]],
                                                 const device GPULight *lights [[buffer(4)]],
                                                 const device TextureHandle *textures [[buffer(5)]],
                                                 constant GPUTemporalParams &temporal [[buffer(8)]],
    constant GPUResolveLightingParams& lighting [[buffer(9)]],
    const device ForwardLightingTextureHandle* lightingTextures [[buffer(10)]]) {
    SurfaceInput surface{in.worldPos,
                         in.normal,
                         in.tangent,
                         in.uv,
                         dfdx(in.uv) * exp2(temporal.mipBias),
                         dfdy(in.uv) * exp2(temporal.mipBias),
                         in.materialIndex,
                         in.mirrored,
                         frontFacing,
                         in.volumeScale};
#if SFV_VC
    surface.vertexColor = in.color; // F15 closure (COLOR_0)
#endif
    const ShadingResult value = shadeSurface(surface, frame, materials, lights, textures, true,
        (lighting.flags & 1u) ? lightingTextures[0].tex.read(uint2(in.position.xy)).xyz : float3(1.0f), lighting.sunIndex,
        (lighting.flags & 2u) != 0, (lighting.flags & 2u) ? lightingTextures[1].tex.read(uint2(in.position.xy)).xyz : float3(0),
        (lighting.flags & 4u) != 0, (lighting.flags & 4u) ? lightingTextures[2].tex.read(uint2(in.position.xy)).xyz : float3(0),(lighting.flags & 8u)!=0,(lighting.flags & 16u)!=0,(lighting.flags & RESOLVE_EXTERNAL_DIFFUSE)!=0,
        float3(0), 0.0f, (lighting.flags & RESOLVE_GI_BACK) != 0, (lighting.flags & RESOLVE_GI_BACK) ? lightingTextures[3].tex.read(uint2(in.position.xy)).xyz : float3(0));
    if (value.alpha < materials[in.materialIndex].alphaCutoff)
        discard_fragment();
    ForwardLitSurfaceOutput out;
    const uint debugMode = kDebugModeSpecialised ? kDebugMode : frame.debugMode;
    out.color = float4(debugMode == 1   ? value.normal * 0.5f + 0.5f
                            : debugMode == 2 ? value.baseColor
                                             : value.color,
                      1.0f);
    out.normalRoughness = half4(half3(value.normal), half(value.roughness));
    out.diffuseAlbedo = half4(half3(value.diffuseAlbedo), 1.0h);
    out.specularAlbedo = half4(half3(value.specularAlbedo), 1.0h);
    out.motion = half2(temporalMotion(in.unjitteredClip, in.previousClip, temporal, in.historyValid != 0));
    out.reactive = in.historyValid == 0 ? 1.0h : (materials[in.materialIndex].alphaCutoff > 0 || (materials[in.materialIndex].flags & (MATERIAL_FEATURE_TRANSMISSION << MATERIAL_FEATURE_SHIFT)) != 0) ? 0.75h : 0.0h;
    if (kSssWrite) out.sss = float4(value.sssMaterial != 0u ? value.sssIrradiance : float3(0.0f), sssCodeOf(value.sssMaterial));
    return out;
}

// The shared body of the receiver fragment functions.
static ShadingResult diReceiverResolve(SFV_OUT in, bool frontFacing, constant FrameConstants& frame,
    const device GPUMaterial* materials, const device GPULight* lights, const device TextureHandle* textures,
    constant GPUTemporalParams& temporal) {
    SurfaceInput surface{in.worldPos,in.normal,in.tangent,in.uv,dfdx(in.uv)*exp2(temporal.mipBias),
                         dfdy(in.uv)*exp2(temporal.mipBias),in.materialIndex,in.mirrored,frontFacing,in.volumeScale};
#if SFV_VC
    surface.vertexColor = in.color; // F15 closure (COLOR_0)
#endif
    const auto value=shadeSurface(surface,frame,materials,lights,textures,true);
    if(value.alpha<materials[in.materialIndex].alphaCutoff)discard_fragment();
    return value;
}
static DIReceiverOutput diReceiverOutput(SFV_OUT in, ShadingResult value, constant GPUTemporalParams& temporal) {
    return {half4(half3(value.normal),half(value.roughness)),half4(half3(value.baseColor),half(value.metallic)),
            temporalMotion(in.unjitteredClip,in.previousClip,temporal,in.historyValid!=0),half(value.occlusion),
            packEmission(value.emission.x,value.emission.y,value.emission.z)};
}
fragment DIReceiverOutput SFV_NAME(forward_di_receiver_fs)(SFV_OUT in [[stage_in]],bool frontFacing [[front_facing]],
    constant FrameConstants& frame [[buffer(0)]],const device GPUMaterial* materials [[buffer(3)]],
    const device GPULight* lights [[buffer(4)]],const device TextureHandle* textures [[buffer(5)]],
    constant GPUTemporalParams& temporal [[buffer(8)]]) {
    const auto value=diReceiverResolve(in,frontFacing,frame,materials,lights,textures,temporal);
    return diReceiverOutput(in,value,temporal);
}

// F15.1 (ticket 06): the receiver of scenes with material features.  It also writes the lobe guide record of
// its pixel (index = DI surface width), so it must run only for the VISIBLE fragment of the pixel:
// [[early_fragment_tests]] runs the (equal, no-write) depth test before the function, so a fragment that fails
// it executes nothing and writes nothing; the alpha discard precedes the write, so a discarded fragment never
// writes either.  The buffer is a raster order group: fragments at equal depth (coplanar overlap) resolve in
// primitive order like the colour attachments (last wins).  The function writes no depth or coverage, which
// early tests require.  A separate entry point keeps SFV_NAME(forward_di_receiver_fs) (scenes without features) unchanged.
static DIReceiverOutput diReceiverGuideBody(SFV_OUT in,bool frontFacing,constant FrameConstants& frame,const device GPUMaterial* materials,
    const device GPULight* lights,const device TextureHandle* textures,constant GPUTemporalParams& temporal,
    device GPULobeGuide* lobeGuide,constant GPUDIParams& di) {
    const auto value=diReceiverResolve(in,frontFacing,frame,materials,lights,textures,temporal);
    if((di.flags&DI_LOBE_GUIDE_DELAY)!=0u) {
        // Debug control only (--debug-lobe-guide-delay): a bounded delay that is longest for the first layer of lobe-guide-coplanar (normal y 0.5) and
        // zero for the last (-0.5), so overlapping fragments complete out of submission order.  x stays inside (-1, 1): the write below never happens.
        float x=value.normal.y;
        const uint iterations=uint(clamp(value.normal.y+0.5f,0.0f,1.0f)*16384.0f);
        for(uint i=0;i<iterations;++i)x=fma(x,0.99999f,1e-7f);
        if(x==123456.0f)lobeGuide[0].reserved=1u;
    }
    if(kMaterialFeatures!=0u)lobeGuide[uint(in.position.y)*di.width+uint(in.position.x)]=lobeGuideRecord(value,di.flags);
    return diReceiverOutput(in,value,temporal);
}
[[early_fragment_tests]]
fragment DIReceiverOutput SFV_NAME(forward_di_receiver_guide_fs)(SFV_OUT in [[stage_in]],bool frontFacing [[front_facing]],
    constant FrameConstants& frame [[buffer(0)]],const device GPUMaterial* materials [[buffer(3)]],
    const device GPULight* lights [[buffer(4)]],const device TextureHandle* textures [[buffer(5)]],
    constant GPUTemporalParams& temporal [[buffer(8)]],
    device GPULobeGuide* lobeGuide [[buffer(11), raster_order_group(0)]],
    constant GPUDIParams& di [[buffer(12)]]) {
    return diReceiverGuideBody(in,frontFacing,frame,materials,lights,textures,temporal,lobeGuide,di);
}
// F15 closure (ticket 06): the NEGATIVE CONTROL of the raster order group above (--debug-lobe-guide-no-rog): the same body without the
// attribute.  Never used by the engine itself; tools/f15_lobe_guide_coplanar_check.sh shows that coplanar overlapping fragments then
// leave a guide record that disagrees with the colour attachments.
[[early_fragment_tests]]
fragment DIReceiverOutput SFV_NAME(forward_di_receiver_guide_norog_fs)(SFV_OUT in [[stage_in]],bool frontFacing [[front_facing]],
    constant FrameConstants& frame [[buffer(0)]],const device GPUMaterial* materials [[buffer(3)]],
    const device GPULight* lights [[buffer(4)]],const device TextureHandle* textures [[buffer(5)]],
    constant GPUTemporalParams& temporal [[buffer(8)]],
    device GPULobeGuide* lobeGuide [[buffer(11)]],
    constant GPUDIParams& di [[buffer(12)]]) {
    return diReceiverGuideBody(in,frontFacing,frame,materials,lights,textures,temporal,lobeGuide,di);
}


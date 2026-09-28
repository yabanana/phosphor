#version 460

#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_buffer_reference2                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_nonuniform_qualifier                 : require

#include "types.glsl"
#include "bindless.glsl"
#include "math.glsl"
#include "brdf.glsl"
#include "packing.glsl"

// ---------------------------------------------------------------------------
// ReSTIR DI — Final shading pass
//
// Per pixel: read the final reservoir, evaluate full BRDF for the selected
// light, multiply by reservoir weight W, and write direct lighting result
// to RGBA16F output.
//
// Dispatch: ceil(width/8) x ceil(height/8) x 1
// ---------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8) in;

struct Reservoir {
    uint  selectedLight;
    float weightSum;
    uint  M;
    float W;
};

// Per-pass descriptor set (set = 1)
layout(set = 1, binding = 0, r32ui)  uniform readonly  uimage2D visibilityBuffer;
layout(set = 1, binding = 1)         uniform sampler2D  depthBuffer;
layout(set = 1, binding = 2, rgba16f) uniform readonly  image2D  normalImage;

layout(set = 1, binding = 3, std430) readonly buffer ReservoirIn {
    Reservoir finalReservoirs[];
};

layout(set = 1, binding = 4, rgba16f) uniform writeonly image2D directLightOutput;

layout(set = 1, binding = 5, std140) uniform ShadeData {
    mat4 invViewProjection;
};

// ---------------------------------------------------------------------------
// Surface reconstruction (simplified: read from G-buffer-like data)
// ---------------------------------------------------------------------------

struct ShadingSurface {
    vec3  worldPos;
    vec3  normal;
    vec3  albedo;
    float metallic;
    float roughness;
};

// Fetch triangle data and interpolate attributes — same approach as
// material_resolve.comp.glsl but simplified to avoid duplication.
// We reconstruct from vis buffer + BDA scene data.

vec2 worldToScreen(vec3 worldPos, mat4 vp, vec2 resolution) {
    vec4 clip = vp * vec4(worldPos, 1.0);
    vec2 ndc  = clip.xy / clip.w;
    return (ndc * 0.5 + 0.5) * resolution;
}

ShadingSurface reconstructSurface(ivec2 pixelCoord, uint visEncoded) {
    ShadingSurface surf;

    // Decode visibility — O(1) direct meshlet lookup
    VisData vis = decodeVisibility(visEncoded);
    uint instanceIdx = vis.instanceID;
    uint foundMeshletIdx = vis.meshletID;
    uint localTriIdx = vis.localTriID;

    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);
    GPUInstance instance = loadInstance(globals.instanceAddr, instanceIdx);

    Meshlet meshlet = loadMeshlet(globals.meshletAddr, foundMeshletIdx);

    MeshletTriangleBuffer triBuffer = MeshletTriangleBuffer(globals.meshletTriangleAddr);
    uint triBase = meshlet.triangleOffset + localTriIdx * 3;
    uint localIdx0 = uint(triBuffer.indices[triBase + 0]);
    uint localIdx1 = uint(triBuffer.indices[triBase + 1]);
    uint localIdx2 = uint(triBuffer.indices[triBase + 2]);

    MeshletVertexBuffer meshletVerts = MeshletVertexBuffer(globals.meshletVertexAddr);
    uint globalIdx0 = meshletVerts.indices[meshlet.vertexOffset + localIdx0];
    uint globalIdx1 = meshletVerts.indices[meshlet.vertexOffset + localIdx1];
    uint globalIdx2 = meshletVerts.indices[meshlet.vertexOffset + localIdx2];

    GPUVertex v0 = loadVertex(globals.vertexAddr, globalIdx0);
    GPUVertex v1 = loadVertex(globals.vertexAddr, globalIdx1);
    GPUVertex v2 = loadVertex(globals.vertexAddr, globalIdx2);

    // Compute barycentrics
    vec3 wp0 = (instance.modelMatrix * vec4(v0.px, v0.py, v0.pz, 1.0)).xyz;
    vec3 wp1 = (instance.modelMatrix * vec4(v1.px, v1.py, v1.pz, 1.0)).xyz;
    vec3 wp2 = (instance.modelMatrix * vec4(v2.px, v2.py, v2.pz, 1.0)).xyz;

    vec2 resolution = vec2(pc.resolution);
    vec2 sp0 = worldToScreen(wp0, pc.viewProjection, resolution);
    vec2 sp1 = worldToScreen(wp1, pc.viewProjection, resolution);
    vec2 sp2 = worldToScreen(wp2, pc.viewProjection, resolution);

    vec3 bary = computeBarycentrics(vec2(pixelCoord) + 0.5, sp0, sp1, sp2);

    // Interpolate position
    vec3 p0 = vec3(v0.px, v0.py, v0.pz);
    vec3 p1 = vec3(v1.px, v1.py, v1.pz);
    vec3 p2 = vec3(v2.px, v2.py, v2.pz);
    vec3 localPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    surf.worldPos = (instance.modelMatrix * vec4(localPos, 1.0)).xyz;

    // Interpolate normal
    mat3 normalMatrix = transpose(inverse(mat3(instance.modelMatrix)));
    vec3 n0 = vec3(v0.nx, v0.ny, v0.nz);
    vec3 n1 = vec3(v1.nx, v1.ny, v1.nz);
    vec3 n2 = vec3(v2.nx, v2.ny, v2.nz);
    surf.normal = normalize(normalMatrix * normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z));

    // Interpolate UVs
    vec2 uv0 = vec2(v0.u, v0.v);
    vec2 uv1 = vec2(v1.u, v1.v);
    vec2 uv2 = vec2(v2.u, v2.v);
    vec2 uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    // Load material
    GPUMaterial mat = loadMaterial(globals.materialAddr, instance.materialIndex);
    vec4 baseColor = mat.baseColor * sampleBindlessLod(mat.baseColorTex, SAMPLER_LINEAR, uv, 0.0);
    surf.albedo = baseColor.rgb;

    surf.metallic  = mat.metallic;
    surf.roughness = mat.roughness;
    if (mat.metallicRoughnessTex != INVALID_TEXTURE) {
        vec4 mrSample = sampleBindlessLod(mat.metallicRoughnessTex, SAMPLER_LINEAR, uv, 0.0);
        surf.roughness *= mrSample.g;
        surf.metallic  *= mrSample.b;
    }

    return surf;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);

    if (any(greaterThanEqual(uvec2(pixelCoord), pc.resolution))) {
        return;
    }

    uint pixelIndex = pixelCoord.y * pc.resolution.x + pixelCoord.x;

    // Read visibility
    uint visEncoded = imageLoad(visibilityBuffer, pixelCoord).r;

    // Background: write black
    if (visEncoded == VISIBILITY_CLEAR) {
        imageStore(directLightOutput, pixelCoord, vec4(0.0));
        return;
    }

    // Read reservoir
    Reservoir r = finalReservoirs[pixelIndex];

    if (r.selectedLight == 0xFFFFFFFF || r.W <= 0.0) {
        imageStore(directLightOutput, pixelCoord, vec4(0.0));
        return;
    }

    // Reconstruct shading surface
    ShadingSurface surf = reconstructSurface(pixelCoord, visEncoded);

    // Load selected light
    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);
    GPULight light = loadLight(globals.lightAddr, r.selectedLight);

    // Compute light direction and attenuation
    vec3 L;
    float attenuation = 1.0;
    vec3 lightColor = light.color * light.intensity;

    if (light.type == 0) {
        L = normalize(-light.direction);
    } else if (light.type == 1) {
        vec3 toLight = light.position - surf.worldPos;
        float dist = length(toLight);
        L = toLight / max(dist, EPSILON);

        float distSq = dist * dist;
        attenuation = 1.0 / max(distSq, 0.0001);
        if (light.range > 0.0) {
            float rangeSq = light.range * light.range;
            float factor = distSq / rangeSq;
            float smooth_ = clamp(1.0 - factor * factor, 0.0, 1.0);
            attenuation *= smooth_ * smooth_;
        }
    } else {
        vec3 toLight = light.position - surf.worldPos;
        float dist = length(toLight);
        L = toLight / max(dist, EPSILON);

        float distSq = dist * dist;
        attenuation = 1.0 / max(distSq, 0.0001);
        if (light.range > 0.0) {
            float rangeSq = light.range * light.range;
            float factor = distSq / rangeSq;
            float smooth_ = clamp(1.0 - factor * factor, 0.0, 1.0);
            attenuation *= smooth_ * smooth_;
        }

        float cosAngle = dot(normalize(light.direction), -L);
        float spotFactor = clamp(
            (cosAngle - light.outerCone) / max(light.innerCone - light.outerCone, EPSILON),
            0.0, 1.0);
        attenuation *= spotFactor * spotFactor;
    }

    // View direction
    vec3 V = normalize(pc.cameraPosition.xyz - surf.worldPos);

    // Evaluate PBR BRDF
    vec3 Lo = evaluatePBR(surf.normal, V, L, surf.albedo, surf.metallic,
                          surf.roughness, lightColor * attenuation);

    // Apply reservoir weight W
    vec3 directLight = Lo * r.W;

    // Add simple ambient
    vec3 ambient = vec3(0.03) * surf.albedo;

    imageStore(directLightOutput, pixelCoord, vec4(directLight + ambient, 1.0));
}

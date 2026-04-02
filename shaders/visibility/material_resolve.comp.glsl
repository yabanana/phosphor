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
// Material resolve compute shader
//
// Reads the visibility buffer (R32_UINT) and depth buffer, reconstructs
// surface attributes per pixel via barycentric interpolation, performs
// PBR shading, and writes HDR color output.
//
// Dispatch: ceil(width/8) x ceil(height/8) x 1
// ---------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8) in;

// Input images
layout(set = 1, binding = 0, r32ui) uniform readonly uimage2D visibilityBuffer;
layout(set = 1, binding = 1)        uniform sampler2D depthBuffer;

// Output HDR color
layout(set = 1, binding = 2, rgba16f) uniform writeonly image2D hdrOutput;

// ---------------------------------------------------------------------------
// Reconstruct the 3 clip-space positions of a triangle, then compute
// screen-space barycentrics for the current pixel.
// ---------------------------------------------------------------------------

struct TriangleData {
    GPUVertex v0, v1, v2;
    vec3 barycentrics;
};

// Project a world-space position to screen-space pixel coordinates
vec2 worldToScreen(vec3 worldPos, mat4 vp, vec2 resolution) {
    vec4 clip = vp * vec4(worldPos, 1.0);
    vec2 ndc  = clip.xy / clip.w;
    // NDC [-1,1] -> screen [0, resolution]
    return (ndc * 0.5 + 0.5) * resolution;
}

bool fetchTriangleAndBarycentrics(
    SceneGlobalsBuffer globals,
    uint instanceIdx,
    uint triangleIdx,
    vec2 pixelPos,
    mat4 modelMatrix,
    out TriangleData tri
) {
    // Find which meshlet this triangle belongs to by searching the instance's meshlets.
    // The triangleIdx is the gl_PrimitiveID from the mesh shader, which is the
    // primitive index within the mesh shader output. Since each mesh shader workgroup
    // processes one meshlet and primitives are numbered per-workgroup starting at 0,
    // we need a way to find the meshlet + local triangle.
    //
    // Approach: store the meshlet index in the upper bits of instanceID? No.
    // The gl_PrimitiveID in the mesh shader is global across the entire draw.
    // We iterate over the instance's meshlets to find which one contains this triangle.

    GPUInstance instance = loadInstance(globals.instanceAddr, instanceIdx);
    MeshInfoBuffer meshInfo = MeshInfoBuffer(globals.meshInfoAddr + uint64_t(instance.meshIndex) * 32);

    // Walk meshlets belonging to this instance's mesh to find the triangle
    uint remainingTriangles = triangleIdx;
    uint foundMeshletIdx = 0;
    uint localTriIdx = 0;
    bool found = false;

    uint meshletCount = meshInfo.meshletCount;
    // Safety: clamp to a sane maximum to prevent GPU hang on corrupt data
    meshletCount = min(meshletCount, 4096u);

    for (uint m = 0; m < meshletCount; m++) {
        uint globalMeshletIdx = meshInfo.meshletOffset + m;
        Meshlet meshlet = loadMeshlet(globals.meshletAddr, globalMeshletIdx);

        if (remainingTriangles < meshlet.triangleCount) {
            foundMeshletIdx = globalMeshletIdx;
            localTriIdx = remainingTriangles;
            found = true;
            break;
        }
        remainingTriangles -= meshlet.triangleCount;
    }

    // If we didn't find the triangle (corrupt data), bail out
    if (!found) {
        return false;
    }

    // Load the meshlet
    Meshlet meshlet = loadMeshlet(globals.meshletAddr, foundMeshletIdx);

    // Read triangle vertex indices from the meshlet triangle buffer
    MeshletTriangleBuffer triBuffer = MeshletTriangleBuffer(globals.meshletTriangleAddr);
    uint triBase = meshlet.triangleOffset + localTriIdx * 3;
    uint localIdx0 = uint(triBuffer.indices[triBase + 0]);
    uint localIdx1 = uint(triBuffer.indices[triBase + 1]);
    uint localIdx2 = uint(triBuffer.indices[triBase + 2]);

    // Map to global vertex indices via the meshlet vertex buffer
    MeshletVertexBuffer meshletVerts = MeshletVertexBuffer(globals.meshletVertexAddr);
    uint globalIdx0 = meshletVerts.indices[meshlet.vertexOffset + localIdx0];
    uint globalIdx1 = meshletVerts.indices[meshlet.vertexOffset + localIdx1];
    uint globalIdx2 = meshletVerts.indices[meshlet.vertexOffset + localIdx2];

    // Load vertex data
    tri.v0 = loadVertex(globals.vertexAddr, globalIdx0);
    tri.v1 = loadVertex(globals.vertexAddr, globalIdx1);
    tri.v2 = loadVertex(globals.vertexAddr, globalIdx2);

    // Project all 3 vertices to screen space for barycentric computation
    vec3 wp0 = (modelMatrix * vec4(tri.v0.px, tri.v0.py, tri.v0.pz, 1.0)).xyz;
    vec3 wp1 = (modelMatrix * vec4(tri.v1.px, tri.v1.py, tri.v1.pz, 1.0)).xyz;
    vec3 wp2 = (modelMatrix * vec4(tri.v2.px, tri.v2.py, tri.v2.pz, 1.0)).xyz;

    vec2 resolution = vec2(pc.resolution);
    vec2 sp0 = worldToScreen(wp0, pc.viewProjection, resolution);
    vec2 sp1 = worldToScreen(wp1, pc.viewProjection, resolution);
    vec2 sp2 = worldToScreen(wp2, pc.viewProjection, resolution);

    // Compute barycentrics at this pixel
    tri.barycentrics = computeBarycentrics(pixelPos + 0.5, sp0, sp1, sp2);

    return true;
}

// ---------------------------------------------------------------------------
// Interpolate vertex attributes using barycentric coordinates
// ---------------------------------------------------------------------------

struct SurfaceAttributes {
    vec3  worldPos;
    vec3  worldNormal;
    vec4  worldTangent;
    vec2  uv;
};

SurfaceAttributes interpolateAttributes(TriangleData tri, mat4 modelMatrix) {
    SurfaceAttributes surf;

    vec3 bary = tri.barycentrics;
    mat3 normalMatrix = transpose(inverse(mat3(modelMatrix)));

    // Position
    vec3 p0 = vec3(tri.v0.px, tri.v0.py, tri.v0.pz);
    vec3 p1 = vec3(tri.v1.px, tri.v1.py, tri.v1.pz);
    vec3 p2 = vec3(tri.v2.px, tri.v2.py, tri.v2.pz);
    vec3 localPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    surf.worldPos = (modelMatrix * vec4(localPos, 1.0)).xyz;

    // Normal
    vec3 n0 = vec3(tri.v0.nx, tri.v0.ny, tri.v0.nz);
    vec3 n1 = vec3(tri.v1.nx, tri.v1.ny, tri.v1.nz);
    vec3 n2 = vec3(tri.v2.nx, tri.v2.ny, tri.v2.nz);
    vec3 localNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);
    surf.worldNormal = normalize(normalMatrix * localNormal);

    // Tangent
    vec3 t0 = vec3(tri.v0.tx, tri.v0.ty, tri.v0.tz);
    vec3 t1 = vec3(tri.v1.tx, tri.v1.ty, tri.v1.tz);
    vec3 t2 = vec3(tri.v2.tx, tri.v2.ty, tri.v2.tz);
    vec3 localTangent = normalize(t0 * bary.x + t1 * bary.y + t2 * bary.z);
    float handedness = tri.v0.tw * bary.x + tri.v1.tw * bary.y + tri.v2.tw * bary.z;
    surf.worldTangent = vec4(normalize(normalMatrix * localTangent), handedness);

    // UV
    vec2 uv0 = vec2(tri.v0.u, tri.v0.v);
    vec2 uv1 = vec2(tri.v1.u, tri.v1.v);
    vec2 uv2 = vec2(tri.v2.u, tri.v2.v);
    surf.uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    return surf;
}

// ---------------------------------------------------------------------------
// Apply normal map in tangent space
// ---------------------------------------------------------------------------

vec3 applyNormalMap(vec3 N, vec4 T, vec3 normalMapSample, float normalScale) {
    vec3 tangent   = normalize(T.xyz);
    vec3 bitangent = cross(N, tangent) * T.w;
    mat3 TBN       = mat3(tangent, bitangent, N);

    // Normal map is stored as [0,1], remap to [-1,1]
    vec3 tsNormal = normalMapSample * 2.0 - 1.0;
    tsNormal.xy *= normalScale;
    tsNormal = normalize(tsNormal);

    return normalize(TBN * tsNormal);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);

    // Bounds check
    if (any(greaterThanEqual(uvec2(pixelCoord), pc.resolution))) {
        return;
    }

    // Read visibility buffer
    uint visEncoded = imageLoad(visibilityBuffer, pixelCoord).r;

    // Background: no geometry
    if (visEncoded == VISIBILITY_CLEAR) {
        // Simple sky gradient
        vec2 uv = (vec2(pixelCoord) + 0.5) / vec2(pc.resolution);
        vec3 skyColor = mix(vec3(0.1, 0.1, 0.15), vec3(0.3, 0.5, 0.8), uv.y);
        imageStore(hdrOutput, pixelCoord, vec4(skyColor, 1.0));
        return;
    }

    // Decode visibility
    uvec2 vis = decodeVisibility(visEncoded);
    uint instanceIdx = vis.x;
    uint triangleIdx = vis.y;

    // Load scene
    SceneGlobalsBuffer globals = SceneGlobalsBuffer(pc.sceneGlobalsAddress);

    // Bounds check: make sure instanceIdx is within range
    if (instanceIdx >= globals.instanceCount) {
        imageStore(hdrOutput, pixelCoord, vec4(1.0, 0.0, 1.0, 1.0)); // magenta = error
        return;
    }

    GPUInstance instance = loadInstance(globals.instanceAddr, instanceIdx);

    // Fetch triangle vertices and compute barycentrics
    TriangleData tri;
    bool triOk = fetchTriangleAndBarycentrics(
        globals, instanceIdx, triangleIdx,
        vec2(pixelCoord), instance.modelMatrix, tri
    );
    if (!triOk) {
        imageStore(hdrOutput, pixelCoord, vec4(1.0, 0.0, 1.0, 1.0)); // magenta = error
        return;
    }

    // Interpolate surface attributes
    SurfaceAttributes surf = interpolateAttributes(tri, instance.modelMatrix);

    // Load material
    GPUMaterial mat = loadMaterial(globals.materialAddr, instance.materialIndex);

    // Sample textures via bindless descriptors
    vec4 baseColor = mat.baseColor * sampleBindlessLod(mat.baseColorTex, SAMPLER_LINEAR, surf.uv, 0.0);

    // Alpha test
    if (baseColor.a < mat.alphaCutoff) {
        imageStore(hdrOutput, pixelCoord, vec4(0.0));
        return;
    }

    vec3 N = surf.worldNormal;

    // Normal mapping
    if (mat.normalTex != INVALID_TEXTURE) {
        vec3 normalSample = sampleBindlessLod(mat.normalTex, SAMPLER_LINEAR, surf.uv, 0.0).rgb;
        N = applyNormalMap(N, surf.worldTangent, normalSample, mat.normalScale);
    }

    // Metallic-roughness
    float metallic  = mat.metallic;
    float roughness = mat.roughness;
    if (mat.metallicRoughnessTex != INVALID_TEXTURE) {
        vec4 mrSample = sampleBindlessLod(mat.metallicRoughnessTex, SAMPLER_LINEAR, surf.uv, 0.0);
        roughness *= mrSample.g; // green channel = roughness
        metallic  *= mrSample.b; // blue channel = metallic
    }

    // Occlusion
    float occlusion = 1.0;
    if (mat.occlusionTex != INVALID_TEXTURE) {
        occlusion = sampleBindlessLod(mat.occlusionTex, SAMPLER_LINEAR, surf.uv, 0.0).r;
        occlusion = mix(1.0, occlusion, mat.occlusionStrength);
    }

    // Emissive
    vec3 emissive = mat.emissive;
    if (mat.emissiveTex != INVALID_TEXTURE) {
        emissive *= sampleBindlessLod(mat.emissiveTex, SAMPLER_LINEAR, surf.uv, 0.0).rgb;
    }

    // View direction
    vec3 V = normalize(pc.cameraPosition.xyz - surf.worldPos);

    // ---------------------------------------------------------------------------
    // Lighting — simple analytical lights for now (full ReSTIR/DDGI comes later)
    // ---------------------------------------------------------------------------

    vec3 Lo = vec3(0.0);

    for (uint i = 0; i < pc.lightCount; i++) {
        GPULight light = loadLight(globals.lightAddr, i);

        vec3 L;
        vec3 lightColor = light.color * light.intensity;
        float attenuation = 1.0;

        if (light.type == 0) {
            // Directional light
            L = normalize(-light.direction);
        } else if (light.type == 1) {
            // Point light
            vec3 toLight = light.position - surf.worldPos;
            float dist = length(toLight);
            L = toLight / dist;

            // Distance attenuation with range cutoff
            float distSq = dist * dist;
            attenuation = 1.0 / max(distSq, 0.0001);
            if (light.range > 0.0) {
                float rangeSq = light.range * light.range;
                float factor = distSq / rangeSq;
                float smoothFactor = clamp(1.0 - factor * factor, 0.0, 1.0);
                attenuation *= smoothFactor * smoothFactor;
            }
        } else {
            // Spot light
            vec3 toLight = light.position - surf.worldPos;
            float dist = length(toLight);
            L = toLight / dist;

            float distSq = dist * dist;
            attenuation = 1.0 / max(distSq, 0.0001);
            if (light.range > 0.0) {
                float rangeSq = light.range * light.range;
                float factor = distSq / rangeSq;
                float smoothFactor = clamp(1.0 - factor * factor, 0.0, 1.0);
                attenuation *= smoothFactor * smoothFactor;
            }

            // Spot cone attenuation
            float cosAngle = dot(normalize(light.direction), -L);
            float spotFactor = clamp((cosAngle - light.outerCone) /
                                      (light.innerCone - light.outerCone), 0.0, 1.0);
            attenuation *= spotFactor * spotFactor;
        }

        Lo += evaluatePBR(N, V, L, baseColor.rgb, metallic, roughness, lightColor * attenuation);
    }

    // Ambient (very simple, will be replaced by DDGI)
    vec3 ambient = vec3(0.03) * baseColor.rgb * occlusion;

    vec3 color = ambient + Lo + emissive;

    imageStore(hdrOutput, pixelCoord, vec4(color, 1.0));
}

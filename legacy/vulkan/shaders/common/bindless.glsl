#ifndef BINDLESS_GLSL
#define BINDLESS_GLSL

#extension GL_EXT_nonuniform_qualifier : require

// ---------------------------------------------------------------------------
// Descriptor set 0 — Bindless resources
// ---------------------------------------------------------------------------

layout(set = 0, binding = 0) uniform texture2D bindless_textures[];
layout(set = 0, binding = 1) uniform sampler   bindless_samplers[3];
// Sampler indices: 0 = linear (trilinear + aniso), 1 = nearest, 2 = shadow (comparison)

layout(set = 0, binding = 2) buffer BindlessBuffers {
    uint data[];
} bindless_buffers[];

// ---------------------------------------------------------------------------
// Sampler index constants
// ---------------------------------------------------------------------------

const uint SAMPLER_LINEAR  = 0;
const uint SAMPLER_NEAREST = 1;
const uint SAMPLER_SHADOW  = 2;

// ---------------------------------------------------------------------------
// Invalid texture sentinel
// ---------------------------------------------------------------------------

const uint INVALID_TEXTURE = 0xFFFFFFFF;

// ---------------------------------------------------------------------------
// Sample a bindless texture with the specified sampler
// ---------------------------------------------------------------------------

vec4 sampleBindless(uint texIndex, uint samplerIndex, vec2 uv) {
    if (texIndex == INVALID_TEXTURE) {
        return vec4(1.0);
    }
    return texture(
        sampler2D(bindless_textures[nonuniformEXT(texIndex)],
                  bindless_samplers[samplerIndex]),
        uv
    );
}

// ---------------------------------------------------------------------------
// Sample with explicit LOD (useful in compute shaders)
// ---------------------------------------------------------------------------

vec4 sampleBindlessLod(uint texIndex, uint samplerIndex, vec2 uv, float lod) {
    if (texIndex == INVALID_TEXTURE) {
        return vec4(1.0);
    }
    return textureLod(
        sampler2D(bindless_textures[nonuniformEXT(texIndex)],
                  bindless_samplers[samplerIndex]),
        uv,
        lod
    );
}

// ---------------------------------------------------------------------------
// Fetch a texel at integer coordinates (no filtering)
// ---------------------------------------------------------------------------

vec4 fetchBindless(uint texIndex, ivec2 coord, int lod) {
    if (texIndex == INVALID_TEXTURE) {
        return vec4(1.0);
    }
    return texelFetch(
        sampler2D(bindless_textures[nonuniformEXT(texIndex)],
                  bindless_samplers[SAMPLER_NEAREST]),
        coord,
        lod
    );
}

#endif // BINDLESS_GLSL

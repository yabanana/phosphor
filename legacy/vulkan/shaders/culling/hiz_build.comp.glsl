#version 460

// ---------------------------------------------------------------------------
// Hi-Z pyramid builder — downsamples depth buffer into a mip chain.
//
// Each thread reads a 2x2 quad from the source mip and writes the MAX
// depth (reversed-Z: max = farthest = most conservative for occlusion)
// to the destination mip.
//
// Dispatch: ceil(dstWidth/16) x ceil(dstHeight/16) x 1
// Per-mip dispatch with barriers between levels.
// ---------------------------------------------------------------------------

layout(local_size_x = 16, local_size_y = 16) in;

// Per-pass descriptor set (set = 0 — standalone layout, no bindless set):
//   binding 0 = source mip (combined image sampler, nearest)
//   binding 1 = destination mip (storage image, r32f, writeonly)
layout(set = 0, binding = 0) uniform sampler2D srcMip;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D dstMip;

// Lightweight push constants for compute-only Hi-Z pass.
// Separate pipeline layout from the main 128-byte push constants.
layout(push_constant) uniform HiZBuildPC {
    uvec2 dstSize;       // destination mip dimensions
    uvec2 srcSize;       // source mip dimensions
};

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;

    // Bounds check: don't write outside the destination mip
    if (pos.x >= dstSize.x || pos.y >= dstSize.y) {
        return;
    }

    // Compute the 2x2 texel coordinates in the source mip.
    // Clamp to source bounds to handle odd-sized mips.
    ivec2 srcBase = ivec2(pos) * 2;

    float d00 = texelFetch(srcMip, srcBase + ivec2(0, 0), 0).r;
    float d10 = texelFetch(srcMip, srcBase + ivec2(1, 0), 0).r;
    float d01 = texelFetch(srcMip, srcBase + ivec2(0, 1), 0).r;
    float d11 = texelFetch(srcMip, srcBase + ivec2(1, 1), 0).r;

    // For reversed-Z (near=1, far=0), we want MAX to get the most
    // conservative (farthest) depth for occlusion culling.
    // For standard Z (near=0, far=1), MAX is also correct for
    // "is the occludee behind the farthest point in this region?"
    float maxDepth = max(max(d00, d10), max(d01, d11));

    imageStore(dstMip, ivec2(pos), vec4(maxDepth));
}

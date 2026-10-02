// F6-S3 negative control: a WRONG Hi-Z reduction (one texel instead of the
// 2x2 minimum).  It must mismatch the CPU pyramid on random depth; a run in
// which it matches proves nothing about the real backends.
#include <metal_stdlib>
using namespace metal;

struct HiZParams {
    uint srcSize[2];
    uint dstSize[2];
    uint dstLevel;
    uint levels;
    uint pad[2];
};

kernel void s3_reduce_point(constant HiZParams& p [[buffer(0)]], texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.dstSize[0] || gid.y >= p.dstSize[1]) return;
    const uint2 s = min(gid * 2u, uint2(p.srcSize[0] - 1u, p.srcSize[1] - 1u));
    dst.write(src.read(s, p.dstLevel - 1u), gid, p.dstLevel);
}

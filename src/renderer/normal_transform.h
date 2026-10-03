#pragma once
namespace phosphor {
struct SurfaceNormal {
    float x, y, z;
};
// Cofactor matrix times sign(det) is a positive scalar multiple of the
// inverse transpose. Keeping that common scale until fragment normalization
// preserves interpolation without an unstable division by a small determinant.
inline SurfaceNormal transformSurfaceNormal(float ax, float ay, float az, float bx, float by, float bz, float cx,
                                            float cy, float cz, float nx, float ny, float nz) {
    const float ux = by * cz - bz * cy, uy = bz * cx - bx * cz, uz = bx * cy - by * cx;
    const float vx = cy * az - cz * ay, vy = cz * ax - cx * az, vz = cx * ay - cy * ax;
    const float wx = ay * bz - az * by, wy = az * bx - ax * bz, wz = ax * by - ay * bx;
    const float sign = ax * ux + ay * uy + az * uz < 0.0f ? -1.0f : 1.0f;
    SurfaceNormal result{(ux * nx + vx * ny + wx * nz) * sign, (uy * nx + vy * ny + wy * nz) * sign,
                         (uz * nx + vz * ny + wz * nz) * sign};
    if (result.x == 0 && result.y == 0 && result.z == 0)
        return {nx, ny, nz};
    return result;
}
} // namespace phosphor

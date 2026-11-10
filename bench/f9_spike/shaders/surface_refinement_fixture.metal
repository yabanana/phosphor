#include "rt_common.h"
kernel void surface_refinement_fixture(const device GPURtRay* rays [[buffer(0)]],const device GPURtHit* hits [[buffer(1)]],
const device GPUInstance* instances [[buffer(2)]],const device GPURtMesh* meshes [[buffer(3)]],
const device GPUVertex* vertices [[buffer(4)]],const device uint* indices [[buffer(5)]],
device GPURtHit* output [[buffer(6)]],uint tid [[thread_position_in_grid]]) {
 output[tid]=rtRefineSurfaceHit(rays[tid],hits[tid],instances,1,meshes,vertices,indices,1);
}

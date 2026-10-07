#include "atmosphere_common.h"

kernel void volume_lut_stamp(constant GPUAtmosphereParams& atmosphere [[buffer(0)]],constant GPUVolumeDiagnosticParams& p [[buffer(17)]],
    device uint* produced [[buffer(18)]],uint tid [[thread_position_in_grid]]) {
    if(tid==0)produced[p.stampKind]=p.stampKind==2?atmosphere.skyRevision:atmosphere.parameterRevision;
}
kernel void volume_homogeneous_fog(constant GPUFogParams& p [[buffer(0)]],constant GPUVolumeDiagnosticParams& d [[buffer(17)]],
    device GPUFogCell* cells [[buffer(1)]],uint3 cell [[thread_position_in_grid]]) {
    if(any(cell>=uint3(p.gridX,p.gridY,p.gridZ)))return;
    const float3 camera=atmoVec(p.cameraPosition),ray=atmoPixelRay(cell.xy,uint2(p.gridX,p.gridY),p.inverseViewProjection,camera);
    const float z=p.nearDistance*pow(p.farDistance/p.nearDistance,(float(cell.z)+0.5f)/float(p.gridZ));
    const float cosine=max(1e-5f,-(atmoMatrix(p.view)*float4(ray,0)).z);const float3 point=camera+ray*(z/cosine);
    GPUFogCell value{};value.extinction=d.fixtureExtinction*(d.corruption==VOLUME_CORRUPT_UNITS?1000.0f:1.0f);
    float3 source=float3(d.fixtureSource[0],d.fixtureSource[1],d.fixtureSource[2]);
    if(d.corruption==VOLUME_CORRUPT_UNITS)source*=1000;if(d.corruption==VOLUME_CORRUPT_LIGHT)source*=8;
    value.source[0]=source.x;value.source[1]=source.y;value.source[2]=source.z;
    value.worldPosition[0]=point.x;value.worldPosition[1]=point.y;value.worldPosition[2]=point.z;value.viewDepth=z;
    value.viewID=p.viewID;value.generation=p.generation;value.age=1;value.valid=1;cells[(cell.z*p.gridY+cell.y)*p.gridX+cell.x]=value;
}
kernel void volume_foreign_fog_history(constant GPUVolumeDiagnosticParams& p [[buffer(17)]],device GPUFogCell* cells [[buffer(18)]],uint i [[thread_position_in_grid]]) {
    if(i<p.fogX*p.fogY*p.fogZ&&cells[i].valid)cells[i].viewID^=0x10000u;
}
kernel void volume_foreign_cloud_history(constant GPUVolumeDiagnosticParams& p [[buffer(17)]],device GPUCloudHistory* cells [[buffer(18)]],uint i [[thread_position_in_grid]]) {
    if(i<p.fogIndices[0]&&cells[i].valid)cells[i].viewID^=0x10000u;
}
kernel void volume_numeric_collect(constant GPUVolumeDiagnosticParams& p [[buffer(17)]],
    const device uint* globalProduced [[buffer(18)]],const device uint* skyProduced [[buffer(19)]],
    device GPUVolumeNumericSample* records [[buffer(20)]],const device GPUFogIntegrated* fog [[buffer(21)]],
    const device GPUFogCell* cells [[buffer(22)]],texture2d<float,access::read> trans [[texture(0)]],
    texture2d<float,access::read> multi [[texture(1)]],texture2d<float,access::read> sky [[texture(2)]],
    texture2d<float,access::read> solar [[texture(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.sampleCount)return;GPUVolumeNumericSample r{};
    if(i<2){r.kind=VOLUME_NUMERIC_TRANS;r.x=i==0?0:p.transWidth/3;r.y=i==0?0:p.transHeight/4;
        const float4 value=trans.read(uint2(r.x,r.y));r.value[0]=value.x;r.value[1]=value.y;r.value[2]=value.z;r.value[3]=1;r.producedRevision=globalProduced[0];}
    else if(i==2){r.kind=VOLUME_NUMERIC_MULTI;r.x=p.multiWidth*3/4;r.y=p.multiHeight/4;
        const float4 value=multi.read(uint2(r.x,r.y));r.value[0]=value.x;r.value[1]=value.y;r.value[2]=value.z;r.value[3]=1;r.producedRevision=globalProduced[1];}
    else if(i<5){r.kind=VOLUME_NUMERIC_SKY;r.x=i==3?0:p.skyWidth/2;r.y=i==3?0:p.skyHeight/4;
        const float4 value=sky.read(uint2(r.x,r.y));r.value[0]=value.x;r.value[1]=value.y;r.value[2]=value.z;r.value[3]=1;r.producedRevision=skyProduced[2];}
    else if(i<8){r.kind=VOLUME_NUMERIC_SOLAR;r.x=i-5;const float4 value=solar.read(uint2(r.x,0));
        r.value[0]=value.x;r.value[1]=value.y;r.value[2]=value.z;r.value[3]=1;r.producedRevision=skyProduced[2];}
    else {r.kind=VOLUME_NUMERIC_FOG;r.x=p.fogIndices[i-8];const GPUFogIntegrated value=fog[r.x];
        r.value[0]=value.radiance[0];r.value[1]=value.radiance[1];r.value[2]=value.radiance[2];r.value[3]=value.transmittance;r.producedRevision=cells[r.x].generation;}
    records[i]=r;
}
kernel void volume_solar_wide_probe(constant GPUAtmosphereParams& p [[buffer(0)]],texture2d<float,access::write> output [[texture(5)]],uint i [[thread_position_in_grid]]) {
    if(i>=3)return;const float3 sun=normalize(atmoVec(p.sunDirection));
    const float3 ray=i==0?sun:i==1?-sun:normalize(cross(abs(sun.z)<0.9f?float3(0,0,1):float3(1,0,0),sun));
    output.write(float4(atmoSolarDisk(ray,p),1),uint2(i,0));
}

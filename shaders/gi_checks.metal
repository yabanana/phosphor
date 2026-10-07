#include <metal_stdlib>
#include "renderer/gpu_types.h"
using namespace metal;
using namespace phosphor;

// DEBUG ONLY, WRITTEN / NON VERIFIED. No RT/F11 helper dependency and no IFT.
// gi_check_clear: counters buffer0, dispatch8 lanes.
// gi_check_state: grid params0/state1/cache2/reservoir3/counters4/check params5,
//                 indirect IRRADIANCE texture0.
// gi_corrupt_state: same params0/state1/cache2/reservoir3/check params5,
//                   dispatch1, deliberately no counters binding.
// Validate after final signal writers. Corrupt after those writers and before
// validation, so a normal update cannot silently overwrite the negative.
//
// Counters, exactly eight u32 (one object error per lane):
// 0 inspectedPixels, 1 inspectedProbes, 2 inspectedCacheLanes,
// 3 outputErrors, 4 probeErrors, 5 cacheErrors,
// 6 reservoirErrors, 7 duplicateCacheKeys (each current key pair once).
// PASS requires positive inspection counts for actual work and counters3..7=0.
// The corruption selector is NEVER inspected by gi_check_state.

inline void giCheckAdd(device atomic_uint* counters,uint field) {
    atomic_fetch_add_explicit(counters+field,1u,memory_order_relaxed);
}
inline uint giCheckHashWord(uint x) {
    x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;return x^(x>>16u);
}
inline uint giCheckKeyHash(GPURadianceCacheEntry k) {
    return giCheckHashWord(k.cellX)^giCheckHashWord(k.cellY+0x9e3779b9u)^
        giCheckHashWord(k.cellZ+0x85ebca6bu)^giCheckHashWord(k.normalBin+0xc2b2ae35u)^
        giCheckHashWord(k.directionBin+0x27d4eb2fu);
}
inline bool giCheckSameKey(GPURadianceCacheEntry a,GPURadianceCacheEntry b) {
    return a.cellX==b.cellX && a.cellY==b.cellY && a.cellZ==b.cellZ &&
        a.normalBin==b.normalBin && a.directionBin==b.directionBin;
}
inline bool giCheckCurrentEpoch(GPURadianceCacheEntry e,constant GPUProbeGridParams& p) {
    // This is an independent expectation, not a call to giCacheLookup/Current.
    // reset frames do not accept cache hits but their freshly written table must
    // still have a unique key layout, hence reset is intentionally absent here.
    return e.state==1u && e.samples>0u && e.generation==p.cacheGeneration &&
        e.geometryRevision==p.geometryRevision && e.lightRevision==p.lightRevision &&
        e.materialRevision==p.materialRevision && p.frameIndex-e.lastFrame<=p.cacheMaxAge;
}
inline bool giCheckUnit(float3 n) {
    return all(isfinite(n)) && abs(dot(n,n)-1.0f)<=2e-3f;
}
inline bool giCheckConfig(constant GPUProbeGridParams& p,constant GPUGiCheckParams& q) {
    const float3 spacing(p.spacing[0],p.spacing[1],p.spacing[2]);
    return q.width==p.width && q.height==p.height && q.mode==p.mode &&
        q.width>0u && q.height>0u && q.width<=32768u && q.height<=32768u &&
        q.mode>=GI_MODE_DDGI && q.mode<=GI_MODE_RESTIR &&
        p.countX>=2u && p.countY>=2u && p.countZ>=2u &&
        p.countX<=256u && p.countY<=256u && p.countZ<=256u &&
        ulong(p.countX)*ulong(p.countY)*ulong(p.countZ)<=65536ul &&
        all(isfinite(spacing)) && all(spacing>0.0f) &&
        isfinite(p.maxRelocation) && p.maxRelocation>=0.0f && p.maxRelocation<=0.49f &&
        p.cacheCapacity>0u && p.cacheCapacity<=(1u<<24u) &&
        p.cacheProbeLimit>0u && p.cacheProbeLimit<=p.cacheCapacity &&
        p.cacheMaxAge<0x80000000u;
}

kernel void gi_check_clear(device atomic_uint* counters [[buffer(0)]],uint tid [[thread_position_in_grid]]) {
    if(tid<8u) atomic_store_explicit(counters+tid,0u,memory_order_relaxed);
}

kernel void gi_check_state(constant GPUProbeGridParams& p [[buffer(0)]],
                           const device GPUProbeState* states [[buffer(1)]],
                           const device GPURadianceCacheEntry* cache [[buffer(2)]],
                           const device GPUGiReservoir* reservoirs [[buffer(3)]],
                           device atomic_uint* counters [[buffer(4)]],
                           constant GPUGiCheckParams& q [[buffer(5)]],
                           texture2d<float,access::read> indirectIrradiance [[texture(0)]],
                           uint tid [[thread_position_in_grid]]) {
    if(!giCheckConfig(p,q)) {
        if(tid==0u) {giCheckAdd(counters,3u);giCheckAdd(counters,4u);}
        return; // No out-of-range resource read for a malformed debug binding.
    }
    const uint pixels=q.width*q.height,probeCount=p.countX*p.countY*p.countZ;
    if(tid<pixels) {
        giCheckAdd(counters,0u);
        if(indirectIrradiance.get_width()!=q.width || indirectIrradiance.get_height()!=q.height) {
            giCheckAdd(counters,3u);
        } else {
            float4 E=indirectIrradiance.read(uint2(tid%q.width,tid/q.width));
            if(!all(isfinite(E)) || any(E.xyz<0.0f) || E.w<0.0f) giCheckAdd(counters,3u);
        }
        // DDGI never initializes reservoir buffers. In that mode do not even
        // fetch one element; validation must not manufacture an undefined read.
        if(q.mode!=GI_MODE_DDGI) {
            GPUGiReservoir r=reservoirs[tid];
            float3 L(r.radiance[0],r.radiance[1],r.radiance[2]);
            float3 y(r.position[0],r.position[1],r.position[2]);
            float3 x(r.sourcePosition[0],r.sourcePosition[1],r.sourcePosition[2]);
            bool bad=!all(isfinite(L)) || any(L<0.0f) || !all(isfinite(y)) || !all(isfinite(x)) ||
                !isfinite(r.W) || r.W<0.0f || !isfinite(r.target) || r.target<0.0f ||
                !isfinite(r.weightSum) || r.weightSum<0.0f || !isfinite(r.proposalArea) || r.proposalArea<0.0f ||
                !isfinite(r.proposalSolidAngle) || r.proposalSolidAngle<0.0f ||
                r.proposalSolidAngle>1.0f/M_PI_F+1e-5f ||
                !isfinite(r.sourceProposalArea) || r.sourceProposalArea<0.0f ||
                (r.flags&~GI_SAMPLE_VALID)!=0u ||
                (q.mode==GI_MODE_CACHE?r.M>1u:r.M>161u);
            if(r.flags&GI_SAMPLE_VALID) {
                bad=bad || r.M==0u || r.W<=0.0f || r.target<=0.0f || r.weightSum<=0.0f ||
                    r.proposalArea<=0.0f || r.sourceProposalArea<=0.0f || r.proposalSolidAngle<=0.0f ||
                    r.geometryRevision!=p.geometryRevision || r.lightRevision!=p.lightRevision ||
                    r.materialRevision!=p.materialRevision || r.viewRevision!=p.viewRevision ||
                    r.slot>=p.slotCount || r.age>32u ||
                    !giCheckUnit(float3(r.normal[0],r.normal[1],r.normal[2])) ||
                    !giCheckUnit(float3(r.sourceNormal[0],r.sourceNormal[1],r.sourceNormal[2]));
                // Independent estimator invariant, including merged M cap:
                // W = sumWeights / (M * target_selected).
                float expected=(r.M>0u && r.target>0.0f)?r.weightSum/(float(r.M)*r.target):0.0f;
                bad=bad || !isfinite(expected) || expected<=0.0f ||
                    abs(r.W-expected)>8e-5f*max(abs(expected),1e-20f);
                float pdfTolerance=8e-5f*max(r.proposalArea,1e-20f);
                bad=bad || abs(r.sourceProposalArea-r.proposalArea)>pdfTolerance;
                // Independently rebuild the selected AREA target from stored
                // current receiver and secondary endpoint. This is intentionally
                // not a call to giAreaIntegrand or the reservoir helper.
                float3 delta=y-x;float distance2=dot(delta,delta),expectedTarget=0.0f;
                if(isfinite(distance2) && distance2>1e-12f) {
                    float3 wi=delta*rsqrt(distance2);
                    float cosX=max(0.0f,dot(float3(r.sourceNormal[0],r.sourceNormal[1],r.sourceNormal[2]),wi));
                    float cosY=max(0.0f,dot(float3(r.normal[0],r.normal[1],r.normal[2]),-wi));
                    expectedTarget=dot(L,float3(0.2126f,0.7152f,0.0722f))*cosX*cosY/(M_PI_F*distance2);
                }
                bad=bad || !isfinite(expectedTarget) || expectedTarget<=0.0f ||
                    abs(r.target-expectedTarget)>8e-5f*max(expectedTarget,1e-20f);
            } else {
                // A zero-contribution path may have M>0 and a finite proposal.
                // It must not carry a positive final normalization into shading.
                bad=bad || r.W!=0.0f;
            }
            if(bad) giCheckAdd(counters,6u);
        }
    }
    if(tid<probeCount) {
        giCheckAdd(counters,1u);
        GPUProbeState s=states[tid];
        float3 offset(s.offset[0],s.offset[1],s.offset[2]),spacing(p.spacing[0],p.spacing[1],p.spacing[2]);
        bool active=s.state==GI_PROBE_ACTIVE,inactive=s.state==GI_PROBE_INACTIVE;
        bool bad=!all(isfinite(offset)) || !isfinite(s.relocationTravel) || s.relocationTravel<0.0f ||
            s.generation!=p.generation || !(active||inactive) ||
            (active?s.age==0u:s.age!=0u) ||
            length(offset/spacing)>p.maxRelocation+2e-4f;
        if(bad) giCheckAdd(counters,4u);
    }
    if(tid<p.cacheCapacity) {
        giCheckAdd(counters,2u);
        GPURadianceCacheEntry e=cache[tid];
        bool bad=e.state>1u;
        if(e.state==1u) {
            float3 L(e.radiance[0],e.radiance[1],e.radiance[2]);
            bad=bad || e.normalBin>=256u || e.directionBin>=256u || e.samples==0u || e.samples>64u ||
                !all(isfinite(L)) || any(L<0.0f);
            // Every inserted key must reside within its own bounded probe window,
            // even when old/expired. Arbitrary key tampering cannot pass by merely
            // leaving a finite RGB in a lane that the true hash could never reach.
            uint start=giCheckKeyHash(e)%p.cacheCapacity;
            uint windowOffset=(tid+p.cacheCapacity-start)%p.cacheCapacity;
            bad=bad || windowOffset>=p.cacheProbeLimit;
            if(giCheckCurrentEpoch(e,p)) {
                for(uint i=0u;i<p.cacheProbeLimit;++i) {
                    uint other=(start+i)%p.cacheCapacity;
                    if(other<=tid) continue;
                    GPURadianceCacheEntry candidate=cache[other];
                    if(giCheckCurrentEpoch(candidate,p) && giCheckSameKey(e,candidate)) giCheckAdd(counters,7u);
                }
            }
        }
        // Expired/prior-epoch lanes are legitimate storage and must be rejected
        // by lookup; they are not themselves accepted radiance or a check failure.
        // No hit provenance is in the reservoir ABI, so this raw-state check
        // cannot claim to prove a returned cache hit's age/epoch by itself.
        if(bad) giCheckAdd(counters,5u);
    }
}

kernel void gi_corrupt_state(constant GPUProbeGridParams& p [[buffer(0)]],
                             device GPUProbeState* states [[buffer(1)]],
                             device GPURadianceCacheEntry* cache [[buffer(2)]],
                             device GPUGiReservoir* reservoirs [[buffer(3)]],
                             constant GPUGiCheckParams& q [[buffer(5)]],
                             uint tid [[thread_position_in_grid]]) {
    if(tid!=0u) return;
    if(q.corruption==GI_CORRUPT_CACHE && p.cacheCapacity>0u) {
        GPURadianceCacheEntry e{};
        e.state=1u;e.samples=1u;e.generation=p.cacheGeneration;e.lastFrame=p.frameIndex;
        e.geometryRevision=p.geometryRevision;e.lightRevision=p.lightRevision;e.materialRevision=p.materialRevision;
        e.radiance[0]=e.radiance[1]=e.radiance[2]=1.0f;
        e.normalBin=256u; // Impossible oct16 key, not a validator's forced flag.
        cache[0]=e;
    } else if(q.corruption==GI_CORRUPT_PROBE && p.countX && p.countY && p.countZ) {
        GPUProbeState s=states[0];s.generation=p.generation;s.state=GI_PROBE_ACTIVE;s.age=1u;
        s.offset[0]=p.spacing[0]*(p.maxRelocation+1.0f); // Violates actual WORLD offset bound.
        states[0]=s;
    } else if(q.corruption==GI_CORRUPT_PDF && q.mode!=GI_MODE_DDGI && q.width && q.height) {
        GPUGiReservoir r{};r.position[1]=1.0f;r.normal[1]=-1.0f;r.sourceNormal[1]=1.0f;
        r.radiance[0]=r.radiance[1]=r.radiance[2]=1.0f;
        r.M=1u;r.target=1.0f;r.weightSum=1.0f;r.W=2.0f;
        r.flags=GI_SAMPLE_VALID;r.proposalSolidAngle=1.0f/M_PI_F;
        r.proposalArea=0.0f;r.sourceProposalArea=0.0f; // Positive contribution with no proposal support.
        r.geometryRevision=p.geometryRevision;r.lightRevision=p.lightRevision;
        r.materialRevision=p.materialRevision;r.viewRevision=p.viewRevision;
        reservoirs[0]=r;
    }
}
